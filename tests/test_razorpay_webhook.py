"""Tests for real-time Razorpay webhook verification, parsing, and ingestion.

Covers:
- HMAC-SHA256 signature verification (valid, invalid, tampered)
- WebhookEvent normalization across payment, subscription, and order events
- Error extraction and classification mapping from webhook payloads
- FastAPI endpoint POST /api/recovery/webhook integration
"""

from __future__ import annotations

import hashlib
import hmac
import json
import pytest
from starlette.testclient import TestClient

from app.main import create_app
from recovery.failure_classifier import FailureCategory
from recovery.razorpay_client import (
    RazorpayAPIError,
    RazorpayClient,
    RazorpayConfig,
    WebhookEvent,
    parse_webhook,
    verify_webhook_signature,
)


def _generate_signature(body: str | bytes, secret: str) -> str:
    body_bytes = body.encode("utf-8") if isinstance(body, str) else body
    return hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Signature Verification Tests
# ---------------------------------------------------------------------------

class TestWebhookSignatureVerification:
    def test_valid_signature(self):
        secret = "test_webhook_secret_123"
        body = '{"event": "payment.failed"}'
        sig = _generate_signature(body, secret)
        assert verify_webhook_signature(body, sig, secret) is True

    def test_invalid_signature_fails(self):
        secret = "test_webhook_secret_123"
        body = '{"event": "payment.failed"}'
        assert verify_webhook_signature(body, "invalid_signature_hex", secret) is False

    def test_tampered_payload_fails(self):
        secret = "test_webhook_secret_123"
        body = '{"event": "payment.failed"}'
        sig = _generate_signature(body, secret)
        tampered_body = '{"event": "payment.captured"}'
        assert verify_webhook_signature(tampered_body, sig, secret) is False

    def test_empty_signature_or_secret_fails(self):
        assert verify_webhook_signature("{}", "", "secret") is False
        assert verify_webhook_signature("{}", "sig", "") is False


# ---------------------------------------------------------------------------
# Webhook Parsing Tests
# ---------------------------------------------------------------------------

class TestWebhookParsing:
    def test_parse_payment_failed_payload(self):
        secret = "secret_abc"
        payload_dict = {
            "entity": "event",
            "account_id": "acc_123",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_test_001",
                        "amount": 450000,
                        "currency": "INR",
                        "status": "failed",
                        "order_id": "order_test_999",
                        "error_code": "BAD_REQUEST_ERROR",
                        "error_description": "Payment failed due to incorrect OTP",
                        "error_source": "customer",
                        "error_step": "payment_authentication",
                        "error_reason": "incorrect_otp",
                        "email": "customer@example.com",
                        "contact": "+919876543210",
                        "created_at": 1700000000,
                    }
                }
            },
            "created_at": 1700000000,
        }
        body = json.dumps(payload_dict)
        sig = _generate_signature(body, secret)

        event = parse_webhook(body, signature=sig, secret=secret, verify=True)

        assert isinstance(event, WebhookEvent)
        assert event.event == "payment.failed"
        assert event.entity_type == "payment"
        assert event.entity_id == "pay_test_001"
        assert event.amount_paise == 450000
        assert event.currency == "INR"
        assert event.error_code == "BAD_REQUEST_ERROR"
        assert event.error_reason == "incorrect_otp"
        assert event.customer_email == "customer@example.com"
        assert event.customer_contact == "+919876543210"

    def test_parse_subscription_halted_payload(self):
        payload_dict = {
            "event": "subscription.halted",
            "payload": {
                "subscription": {
                    "entity": {
                        "id": "sub_test_halted_01",
                        "amount": 99900,
                        "status": "halted",
                        "created_at": 1700000100,
                    }
                }
            },
        }
        body = json.dumps(payload_dict)
        event = parse_webhook(body, verify=False)

        assert event.event == "subscription.halted"
        assert event.entity_type == "subscription"
        assert event.entity_id == "sub_test_halted_01"
        assert event.amount_paise == 99900

    def test_parse_invalid_json_raises_error(self):
        with pytest.raises(RazorpayAPIError, match="Invalid JSON"):
            parse_webhook("invalid-json{", verify=False)

    def test_parse_signature_mismatch_raises_error(self):
        body = '{"event": "payment.failed"}'
        with pytest.raises(RazorpayAPIError, match="signature verification failed"):
            parse_webhook(body, signature="wrong_sig", secret="secret_123", verify=True)


# ---------------------------------------------------------------------------
# API Route Ingestion Tests
# ---------------------------------------------------------------------------

class TestWebhookAPIRoute:
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_ingest_webhook_payment_failed(self, client):
        payload_dict = {
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_live_fail_1",
                        "amount": 50000,
                        "currency": "INR",
                        "status": "failed",
                        "error_code": "GATEWAY_ERROR",
                        "error_description": "Network timeout occurred",
                        "error_reason": "timeout",
                        "email": "user@example.com",
                    }
                }
            },
        }
        response = client.post("/api/recovery/webhook", json=payload_dict)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processed"
        assert data["event"] == "payment.failed"
        assert data["entity_id"] == "pay_live_fail_1"
        assert data["amount_inr"] == 500.0
        assert data["classification"]["category"] == FailureCategory.NETWORK_ERROR.value
        assert data["recommended_action"]["action_type"] == "retry_now"

    def test_ingest_webhook_signature_enforcement(self, client, monkeypatch):
        monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "super_secret_key")
        body = json.dumps({"event": "payment.captured", "payload": {}})

        # Missing signature should fail
        response = client.post(
            "/api/recovery/webhook",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

        # Valid signature should pass
        valid_sig = _generate_signature(body, "super_secret_key")
        response_ok = client.post(
            "/api/recovery/webhook",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": valid_sig,
            },
        )
        assert response_ok.status_code == 200
