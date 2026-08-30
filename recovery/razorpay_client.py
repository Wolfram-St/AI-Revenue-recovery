"""Razorpay API client wrapper for revenue recovery operations.

Provides a typed, exception-safe interface over the Razorpay Python SDK for
the specific operations needed by the recovery agent:

- Fetch failed payments within a time window
- Fetch pending / halted subscriptions
- Fetch individual payment or subscription details
- Create retry orders for failed payments
- Cryptographically verify and parse real-time Razorpay webhooks

All monetary values use **paise** (integer) internally, matching the Razorpay
API convention. Public convenience methods accept INR floats and convert
immediately via ``int(round(inr * 100))``.

The module is deliberately thin: it wraps SDK calls, normalises errors into
``RazorpayAPIError``, and returns typed models or plain dicts. It performs no business
logic, no database writes, and no I/O beyond SDK calls.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any

import razorpay
from razorpay.errors import (
    BadRequestError,
    GatewayError,
    ServerError,
    SignatureVerificationError,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RazorpayAPIError(Exception):
    """Raised when a Razorpay API call fails or webhook verification fails."""

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RazorpayConfig:
    """Credentials and secrets for the Razorpay API.

    Attributes
    ----------
    key_id:
        Razorpay Key ID.
    key_secret:
        Razorpay Key Secret.
    webhook_secret:
        Optional webhook secret for verifying inbound webhook HMAC signatures.
    """

    key_id: str
    key_secret: str
    webhook_secret: str | None = None


# ---------------------------------------------------------------------------
# Webhook Models & Verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WebhookEvent:
    """Normalized, typed representation of a Razorpay webhook event.

    Attributes
    ----------
    event:
        Webhook event name (e.g. ``"payment.failed"``, ``"payment.captured"``, ``"subscription.halted"``).
    event_id:
        Unique webhook delivery identifier (if present).
    entity_type:
        Type of entity affected (e.g. ``"payment"``, ``"subscription"``, ``"order"``, ``"dispute"``).
    entity_id:
        Primary identifier of the entity (e.g. ``"pay_xxx"``, ``"sub_xxx"``).
    amount_paise:
        Amount in paise.
    currency:
        ISO currency code (default ``"INR"``).
    status:
        Entity status (e.g. ``"failed"``, ``"captured"``, ``"halted"``).
    error_code:
        Error code if payment/mandate failed (e.g. ``"BAD_REQUEST_ERROR"``, ``"GATEWAY_ERROR"``).
    error_description:
        Human-readable error description from bank/gateway.
    error_source:
        Origin of the failure (e.g. ``"bank"``, ``"issuer"``, ``"customer"``).
    error_step:
        Transaction step where failure occurred (e.g. ``"payment_authorization"``).
    error_reason:
        Granular reason code (e.g. ``"incorrect_otp"``, ``"insufficient_funds"``).
    customer_id:
        Razorpay customer ID if attached.
    customer_email:
        Customer email for communication.
    customer_contact:
        Customer phone number.
    order_id:
        Associated order ID if present.
    subscription_id:
        Associated subscription ID if present.
    created_at:
        Unix timestamp when event was created.
    raw_payload:
        Original unmodified JSON payload.
    """

    event: str
    event_id: str | None
    entity_type: str
    entity_id: str
    amount_paise: int
    currency: str
    status: str
    error_code: str | None = None
    error_description: str | None = None
    error_source: str | None = None
    error_step: str | None = None
    error_reason: str | None = None
    customer_id: str | None = None
    customer_email: str | None = None
    customer_contact: str | None = None
    order_id: str | None = None
    subscription_id: str | None = None
    created_at: int = 0
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_webhook_signature(body: bytes | str, signature: str, secret: str) -> bool:
    """Verify Razorpay webhook signature using HMAC SHA256.

    Parameters
    ----------
    body:
        Raw request body as bytes or string.
    signature:
        Value of the ``X-Razorpay-Signature`` header.
    secret:
        Webhook secret configured in Razorpay dashboard.

    Returns
    -------
    bool
        True if the signature is valid, False otherwise.
    """
    if not signature or not secret:
        return False

    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    else:
        body_bytes = body

    computed = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)


def parse_webhook(
    body: bytes | str,
    signature: str | None = None,
    secret: str | None = None,
    verify: bool = True,
) -> WebhookEvent:
    """Verify and parse a raw webhook payload into a typed ``WebhookEvent``.

    Parameters
    ----------
    body:
        Raw request body as bytes or string.
    signature:
        Value of the ``X-Razorpay-Signature`` header.
    secret:
        Webhook secret (if verifying).
    verify:
        If True and secret is provided, verifies HMAC signature before parsing.

    Returns
    -------
    WebhookEvent
        Typed and normalized webhook event.

    Raises
    ------
    RazorpayAPIError
        If signature verification fails or payload is malformed JSON.
    """
    if verify and secret:
        if not signature or not verify_webhook_signature(body, signature, secret):
            raise RazorpayAPIError("Webhook signature verification failed")

    raw_str = body.decode("utf-8") if isinstance(body, bytes) else body
    try:
        data = json.loads(raw_str)
    except json.JSONDecodeError as exc:
        raise RazorpayAPIError(f"Invalid JSON webhook payload: {exc}") from exc

    event_name = data.get("event", "unknown")
    event_id = data.get("event_id")
    payload = data.get("payload", {})

    entity_type = "unknown"
    entity: dict[str, Any] = {}
    for key in ("payment", "subscription", "order", "dispute", "refund"):
        if key in payload:
            entity_type = key
            entity = payload[key].get("entity", {})
            break

    amount_paise = int(entity.get("amount") or 0)
    currency = entity.get("currency", "INR")
    status = entity.get("status", "unknown")
    entity_id = entity.get("id", "")

    return WebhookEvent(
        event=event_name,
        event_id=event_id,
        entity_type=entity_type,
        entity_id=entity_id,
        amount_paise=amount_paise,
        currency=currency,
        status=status,
        error_code=entity.get("error_code"),
        error_description=entity.get("error_description"),
        error_source=entity.get("error_source"),
        error_step=entity.get("error_step"),
        error_reason=entity.get("error_reason"),
        customer_id=entity.get("customer_id"),
        customer_email=entity.get("email"),
        customer_contact=entity.get("contact"),
        order_id=entity.get("order_id"),
        subscription_id=entity.get("subscription_id"),
        created_at=int(entity.get("created_at") or data.get("created_at") or 0),
        raw_payload=data,
    )


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------

class RazorpayClient:
    """Thin wrapper around the Razorpay Python SDK.

    Parameters
    ----------
    config:
        Razorpay API credentials and optional webhook secret.

    Examples
    --------
    >>> client = RazorpayClient(RazorpayConfig("rzp_test_xxx", "secret"))
    >>> failed = client.fetch_failed_payments(from_ts=1700000000, to_ts=1700100000)
    """

    def __init__(self, config: RazorpayConfig) -> None:
        self._config = config
        self._client = razorpay.Client(auth=(config.key_id, config.key_secret))
        self._client.set_app_details({"title": "RecoverAI", "version": "0.1.0"})

    @property
    def config(self) -> RazorpayConfig:
        return self._config

    # ---- internal helpers --------------------------------------------------

    def _handle_api_error(self, exc: Exception) -> RazorpayAPIError:
        """Convert SDK exceptions into RazorpayAPIError."""
        if isinstance(exc, (BadRequestError, GatewayError, ServerError)):
            return RazorpayAPIError(str(exc))
        if isinstance(exc, SignatureVerificationError):
            return RazorpayAPIError(f"Signature verification failed: {exc}")
        return RazorpayAPIError(str(exc))

    # ---- webhook ingestion -------------------------------------------------

    def verify_and_parse_webhook(
        self,
        body: bytes | str,
        signature: str | None = None,
        *,
        verify: bool = True,
    ) -> WebhookEvent:
        """Verify signature and parse incoming webhook using client configuration."""
        return parse_webhook(
            body=body,
            signature=signature,
            secret=self._config.webhook_secret,
            verify=verify,
        )

    # ---- payments ----------------------------------------------------------

    def fetch_failed_payments(
        self,
        *,
        from_ts: int | None = None,
        to_ts: int | None = None,
        count: int = 100,
        skip: int = 0,
    ) -> list[dict]:
        """Fetch payments with ``status=failed`` from the Razorpay API.

        Parameters
        ----------
        from_ts:
            Start of the time window as a Unix timestamp (inclusive).
        to_ts:
            End of the time window as a Unix timestamp (inclusive).
        count:
            Maximum number of payments to return (max 100 per page).
        skip:
            Number of records to skip for pagination.

        Returns
        -------
        list[dict]
            List of payment objects (each a dict matching the Razorpay schema).
        """
        data: dict = {"status": "failed", "count": min(count, 100), "skip": skip}
        if from_ts is not None:
            data["from"] = from_ts
        if to_ts is not None:
            data["to"] = to_ts

        try:
            # Suppress the deprecation warning from the SDK — we use .all()
            # internally; callers should not see SDK internals.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                response: dict = self._client.payment.all(data)  # type: ignore[union-attr]
        except (BadRequestError, GatewayError, ServerError) as exc:
            raise self._handle_api_error(exc) from exc

        return response.get("items", [])  # type: ignore[no-any-return]

    def fetch_payment(self, payment_id: str) -> dict:
        """Fetch a single payment by ID.

        Parameters
        ----------
        payment_id:
            Razorpay payment ID (e.g. ``"pay_G8VQzjPLoAvm6D"``).

        Returns
        -------
        dict
            Payment object.
        """
        try:
            return self._client.payment.fetch(payment_id)  # type: ignore[no-any-return]
        except (BadRequestError, GatewayError, ServerError) as exc:
            raise self._handle_api_error(exc) from exc

    def fetch_payments_for_order(self, order_id: str) -> list[dict]:
        """Fetch all payments associated with an order.

        Parameters
        ----------
        order_id:
            Razorpay order ID.

        Returns
        -------
        list[dict]
            List of payment objects.
        """
        try:
            response: dict = self._client.order.fetch_payments(order_id)  # type: ignore[union-attr]
        except (BadRequestError, GatewayError, ServerError) as exc:
            raise self._handle_api_error(exc) from exc
        return response.get("items", [])  # type: ignore[no-any-return]

    # ---- subscriptions -----------------------------------------------------

    def fetch_pending_subscriptions(self, count: int = 100) -> list[dict]:
        """Fetch subscriptions in ``pending`` state (failed auto-charge, retries ongoing).

        Parameters
        ----------
        count:
            Maximum number of subscriptions to return.

        Returns
        -------
        list[dict]
            List of subscription objects.
        """
        data: dict = {"status": "pending", "count": min(count, 100)}
        try:
            response: dict = self._client.subscription.all(data)  # type: ignore[union-attr]
        except (BadRequestError, GatewayError, ServerError) as exc:
            raise self._handle_api_error(exc) from exc
        return response.get("items", [])  # type: ignore[no-any-return]

    def fetch_halted_subscriptions(self, count: int = 100) -> list[dict]:
        """Fetch subscriptions in ``halted`` state (all retries exhausted).

        Parameters
        ----------
        count:
            Maximum number of subscriptions to return.

        Returns
        -------
        list[dict]
            List of subscription objects.
        """
        data: dict = {"status": "halted", "count": min(count, 100)}
        try:
            response: dict = self._client.subscription.all(data)  # type: ignore[union-attr]
        except (BadRequestError, GatewayError, ServerError) as exc:
            raise self._handle_api_error(exc) from exc
        return response.get("items", [])  # type: ignore[no-any-return]

    def fetch_subscription(self, subscription_id: str) -> dict:
        """Fetch a single subscription by ID.

        Parameters
        ----------
        subscription_id:
            Razorpay subscription ID (e.g. ``"sub_00000000000001"``).

        Returns
        -------
        dict
            Subscription object.
        """
        try:
            return self._client.subscription.fetch(subscription_id)  # type: ignore[no-any-return]
        except (BadRequestError, GatewayError, ServerError) as exc:
            raise self._handle_api_error(exc) from exc

    # ---- orders (for retry creation) --------------------------------------

    def create_retry_order(
        self,
        *,
        amount_paise: int,
        currency: str = "INR",
        receipt: str = "",
        notes: dict[str, str] | None = None,
    ) -> dict:
        """Create a new order for retrying a failed payment.

        Parameters
        ----------
        amount_paise:
            Amount in paise (e.g. ``50000`` for ₹500.00).
        currency:
            ISO 4217 currency code (default ``"INR"``).
        receipt:
            Merchant receipt identifier.
        notes:
            Arbitrary key-value notes attached to the order.

        Returns
        -------
        dict
            Order object with ``id``, ``amount_due``, ``status``, etc.
        """
        if amount_paise <= 0:
            raise ValueError(f"amount_paise must be positive, got {amount_paise}")
        data: dict = {
            "amount": amount_paise,
            "currency": currency,
            "receipt": receipt,
        }
        if notes:
            data["notes"] = notes
        try:
            return self._client.order.create(data)  # type: ignore[no-any-return]
        except (BadRequestError, GatewayError, ServerError) as exc:
            raise self._handle_api_error(exc) from exc

    # ---- convenience: INR float → paise -----------------------------------

    @staticmethod
    def inr_to_paise(inr: float) -> int:
        """Convert INR float to integer paise (e.g. ``500.0`` → ``50000``)."""
        return int(round(inr * 100))

    @staticmethod
    def paise_to_inr(paise: int) -> float:
        """Convert integer paise to INR float (e.g. ``50000`` → ``500.0``)."""
        return paise / 100.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_razorpay_client_from_env() -> RazorpayClient:
    """Create a ``RazorpayClient`` from environment variables.

    Expected variables:

    - ``RAZORPAY_KEY_ID`` — Razorpay test/live key ID (e.g. ``"rzp_test_xxx"``)
    - ``RAZORPAY_KEY_SECRET`` — Razorpay key secret
    - ``RAZORPAY_WEBHOOK_SECRET`` — (Optional) Razorpay webhook secret

    Raises
    ------
    ValueError
        If key_id or key_secret environment variable is missing or empty.
    """
    key_id = os.environ.get("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()
    webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "").strip() or None

    missing = []
    if not key_id:
        missing.append("RAZORPAY_KEY_ID")
    if not key_secret:
        missing.append("RAZORPAY_KEY_SECRET")
    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in your .env file or environment."
        )

    return RazorpayClient(
        RazorpayConfig(
            key_id=key_id,
            key_secret=key_secret,
            webhook_secret=webhook_secret,
        )
    )
