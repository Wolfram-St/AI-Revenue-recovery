"""Tests for Promise-to-Pay (PTP) tracker, state machine, and dynamic dunning pause.

Covers:
- Promise registration & validation
- State transitions (PTP_ACTIVE -> PTP_FULFILLED, PTP_ACTIVE -> PTP_BROKEN)
- Grace period cutoff boundary checks
- Dynamic dunning hold check (should_pause_dunning)
- Parameterised SQL INSERT generation matching db/schema.sql
- In-memory PTPRegistry lookups
- FastAPI PTP endpoints and webhook auto-fulfillment
"""

from __future__ import annotations

import datetime
import json
import pytest
from starlette.testclient import TestClient

from app.main import create_app
from recovery.ptp_tracker import (
    PTPChannel,
    PTPStatus,
    PromiseToPay,
    build_sql_insert_for_ptp,
    default_ptp_registry,
    evaluate_ptp_status,
    fulfill_promise,
    register_promise,
    should_pause_dunning,
)


class TestPTPStateMachine:
    def test_register_promise_valid(self):
        due = datetime.datetime(2026, 9, 5, 17, 0, tzinfo=datetime.timezone.utc)
        ptp = register_promise(
            recovery_case_id="case_101",
            customer_id="cust_abc",
            promised_amount_inr=4500.0,
            promised_date=due,
            channel_source=PTPChannel.WHATSAPP,
            concession_applied={"waiver_pct": 5, "waiver_inr": 225},
            transcript_snippet="Haan, main 5 September tak payment clear kar doonga.",
        )

        assert ptp.status == PTPStatus.PTP_ACTIVE
        assert ptp.promised_amount_inr == 4500.0
        assert ptp.channel_source == PTPChannel.WHATSAPP
        assert ptp.grace_period_hours == 4
        assert ptp.concession_applied["waiver_pct"] == 5

    def test_register_promise_negative_amount_fails(self):
        with pytest.raises(ValueError, match="must be positive"):
            register_promise(
                recovery_case_id="case_102",
                customer_id="cust_abc",
                promised_amount_inr=-100.0,
                promised_date="2026-09-05T17:00:00Z",
            )

    def test_active_window_evaluation_remains_active(self):
        due = datetime.datetime(2026, 9, 5, 17, 0, tzinfo=datetime.timezone.utc)
        ptp = register_promise(
            recovery_case_id="case_103",
            customer_id="cust_abc",
            promised_amount_inr=2000.0,
            promised_date=due,
            grace_period_hours=4,
        )

        # 2 hours after due date is within the 4h grace period
        check_time = due + datetime.timedelta(hours=2)
        evaluated = evaluate_ptp_status(ptp, current_time=check_time)
        assert evaluated.status == PTPStatus.PTP_ACTIVE
        assert evaluated.broken_at is None

    def test_expired_grace_period_transitions_to_broken(self):
        due = datetime.datetime(2026, 9, 5, 17, 0, tzinfo=datetime.timezone.utc)
        ptp = register_promise(
            recovery_case_id="case_104",
            customer_id="cust_abc",
            promised_amount_inr=2000.0,
            promised_date=due,
            grace_period_hours=4,
        )

        # 4.5 hours after due date is past the 4h grace period
        check_time = due + datetime.timedelta(hours=4, minutes=30)
        evaluated = evaluate_ptp_status(ptp, current_time=check_time)
        assert evaluated.status == PTPStatus.PTP_BROKEN
        assert evaluated.broken_at is not None

    def test_fulfill_promise_success(self):
        due = datetime.datetime(2026, 9, 5, 17, 0, tzinfo=datetime.timezone.utc)
        ptp = register_promise(
            recovery_case_id="case_105",
            customer_id="cust_abc",
            promised_amount_inr=3000.0,
            promised_date=due,
        )

        fulfilled = fulfill_promise(ptp, paid_amount_inr=3000.0)
        assert fulfilled.status == PTPStatus.PTP_FULFILLED
        assert fulfilled.fulfilled_at is not None

    def test_fulfill_promise_partial_amount_raises_error(self):
        due = datetime.datetime(2026, 9, 5, 17, 0, tzinfo=datetime.timezone.utc)
        ptp = register_promise(
            recovery_case_id="case_106",
            customer_id="cust_abc",
            promised_amount_inr=3000.0,
            promised_date=due,
        )

        with pytest.raises(ValueError, match="less than promised amount"):
            fulfill_promise(ptp, paid_amount_inr=1500.0)


class TestDynamicDunningPauser:
    def test_active_ptp_pauses_dunning(self):
        due = datetime.datetime(2026, 9, 5, 17, 0, tzinfo=datetime.timezone.utc)
        ptp = register_promise(
            recovery_case_id="case_107",
            customer_id="cust_abc",
            promised_amount_inr=1000.0,
            promised_date=due,
        )

        check_time = due - datetime.timedelta(days=1)
        paused, reason = should_pause_dunning(ptp, current_time=check_time)
        assert paused is True
        assert "Dunning paused" in reason

    def test_broken_ptp_unpauses_dunning_with_escalation(self):
        due = datetime.datetime(2026, 9, 5, 17, 0, tzinfo=datetime.timezone.utc)
        ptp = register_promise(
            recovery_case_id="case_108",
            customer_id="cust_abc",
            promised_amount_inr=1000.0,
            promised_date=due,
            grace_period_hours=4,
        )

        check_time = due + datetime.timedelta(hours=6)
        paused, reason = should_pause_dunning(ptp, current_time=check_time)
        assert paused is False
        assert "Promise broken" in reason
        assert "Escalating" in reason


class TestPTPSQLPersistence:
    def test_build_sql_insert_for_ptp(self):
        due = datetime.datetime(2026, 9, 5, 17, 0, tzinfo=datetime.timezone.utc)
        ptp = register_promise(
            recovery_case_id="case_sql_1",
            customer_id="cust_sql_1",
            promised_amount_inr=5000.0,
            promised_date=due,
            channel_source=PTPChannel.VOICE_AGENT,
        )

        query, params = build_sql_insert_for_ptp(ptp)
        assert "INSERT INTO promise_to_pay" in query
        assert len(params) == 14

        ptp_id, case_id, cust_id, amount, p_date, grace, status, channel, concession, snippet, fulfilled, broken, created, updated = params
        assert case_id == "case_sql_1"
        assert cust_id == "cust_sql_1"
        assert amount == 5000.0
        assert status == "PTP_ACTIVE"
        assert channel == "VOICE_AGENT"


class TestPTPAPIEndpoints:
    @pytest.fixture
    def client(self):
        default_ptp_registry.clear()
        app = create_app()
        return TestClient(app)

    def test_register_and_get_ptp(self, client):
        req = {
            "recovery_case_id": "case_api_1",
            "customer_id": "cust_api_1",
            "promised_amount_inr": 7500.0,
            "promised_date": "2026-09-10T12:00:00Z",
            "channel_source": "VOICE_AGENT",
            "concession_applied": {"waiver_pct": 5},
            "transcript_snippet": "Will pay on the 10th.",
        }

        resp = client.post("/api/recovery/ptp/register", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["recovery_case_id"] == "case_api_1"
        assert data["status"] == "PTP_ACTIVE"
        assert data["promised_amount_inr"] == 7500.0

        # Retrieve
        resp_get = client.get("/api/recovery/ptp/cases/case_api_1")
        assert resp_get.status_code == 200
        assert resp_get.json()["ptp_id"] == data["ptp_id"]

    def test_webhook_payment_captured_fulfills_ptp(self, client):
        # Register an active PTP for cust_payer_1
        client.post(
            "/api/recovery/ptp/register",
            json={
                "recovery_case_id": "case_auto_fulfill",
                "customer_id": "cust_payer_1",
                "promised_amount_inr": 2500.0,
                "promised_date": "2026-09-15T12:00:00Z",
            },
        )

        # Ingest payment.captured webhook for 250000 paise (₹2500.0)
        payload = {
            "event": "payment.captured",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_ptp_done_1",
                        "amount": 250000,
                        "currency": "INR",
                        "status": "captured",
                        "customer_id": "cust_payer_1",
                    }
                }
            },
        }
        resp = client.post("/api/recovery/webhook", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ptp_fulfillment"] is not None
        assert data["ptp_fulfillment"]["status"] == "PTP_FULFILLED"
