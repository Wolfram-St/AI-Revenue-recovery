"""Failure root-cause classifier for Razorpay payment failures.

Maps Razorpay error codes, error sources, error steps, and error reasons
to a closed set of ``FailureCategory`` members.  The classification is
deterministic and pure — identical inputs always produce the same category.

The classifier operates on the raw Razorpay payment dict (as returned by
``RazorpayClient.fetch_failed_payments``) and does **not** call any external
service.

Taxonomy
--------
The root-cause categories are deliberately coarse to support downstream
strategy selection without overfitting to transient bank-specific error
strings:

- ``CARD_ISSUE`` — expired, blocked, lost/stolen, invalid card
- ``INSUFFICIENT_FUNDS`` — balance too low
- ``AUTHENTICATION_FAILURE`` — OTP / 3DS / UPI PIN failure
- ``NETWORK_ERROR`` — gateway timeout, connectivity
- ``MANDATE_ISSUE`` — mandate cancelled, paused, or expired
- ``LIMIT_EXCEEDED`` — per-transaction or daily limit
- ``DO_NOT_HONOR`` — generic bank decline (no specific reason)
- ``FRAUD_SUSPECTED`` — fraud risk detected by bank or Razorpay
- ``BUSINESS_ERROR`` — merchant-side configuration issue
- ``UNKNOWN`` — unclassifiable failure
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureCategory(str, Enum):
    """Closed set of payment failure root causes."""

    CARD_ISSUE = "card_issue"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    AUTHENTICATION_FAILURE = "authentication_failure"
    NETWORK_ERROR = "network_error"
    MANDATE_ISSUE = "mandate_issue"
    LIMIT_EXCEEDED = "limit_exceeded"
    DO_NOT_HONOR = "do_not_honor"
    FRAUD_SUSPECTED = "fraud_suspected"
    BUSINESS_ERROR = "business_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassificationResult:
    """Immutable result of classifying a payment failure."""

    category: FailureCategory
    confidence: float  # 0.0–1.0
    reason: str  # human-readable explanation
    raw_error_code: str | None
    raw_error_reason: str | None


# ---------------------------------------------------------------------------
# Rule tables — (substring_match_set → category)
# Ordered by specificity; first match wins.
# ---------------------------------------------------------------------------

_REASON_RULES: list[tuple[frozenset[str], FailureCategory]] = [
    # Card-specific
    (frozenset({"expired", "card expired"}), FailureCategory.CARD_ISSUE),
    (frozenset({"blocked", "card blocked", "do not honor"}), FailureCategory.CARD_ISSUE),
    (frozenset({"lost", "stolen"}), FailureCategory.CARD_ISSUE),
    (frozenset({"invalid card", "invalid card number"}), FailureCategory.CARD_ISSUE),
    (frozenset({"declined", "generic decline"}), FailureCategory.DO_NOT_HONOR),

    # Funds
    (frozenset({"insufficient", "insufficient funds", "not enough"}), FailureCategory.INSUFFICIENT_FUNDS),

    # Authentication
    (frozenset({"otp", "incorrect otp", "otp failure"}), FailureCategory.AUTHENTICATION_FAILURE),
    (frozenset({"3ds", "authentication failed", "authentication failure"}), FailureCategory.AUTHENTICATION_FAILURE),
    (frozenset({"upi pin", "incorrect upi pin"}), FailureCategory.AUTHENTICATION_FAILURE),

    # Network / gateway
    (frozenset({"timeout", "timed out", "gateway timeout"}), FailureCategory.NETWORK_ERROR),
    (frozenset({"network", "connectivity", "connection refused"}), FailureCategory.NETWORK_ERROR),

    # Mandate
    (frozenset({"mandate cancelled", "mandate paused", "mandate expired"}), FailureCategory.MANDATE_ISSUE),
    (frozenset({"max_amount exceeded"}), FailureCategory.MANDATE_ISSUE),

    # Limits
    (frozenset({"daily limit", "transaction limit", "per transaction limit"}), FailureCategory.LIMIT_EXCEEDED),

    # Fraud
    (frozenset({"fraud", "suspected fraud", "risk"}), FailureCategory.FRAUD_SUSPECTED),

    # Business / merchant
    (frozenset({"business error", "configuration", "setup error"}), FailureCategory.BUSINESS_ERROR),
]

_CODE_RULES: list[tuple[frozenset[str], FailureCategory]] = [
    (frozenset({"BAD_REQUEST_ERROR"}), FailureCategory.BUSINESS_ERROR),
    (frozenset({"GATEWAY_ERROR"}), FailureCategory.NETWORK_ERROR),
    (frozenset({"SERVER_ERROR"}), FailureCategory.NETWORK_ERROR),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_failure(payment: dict) -> ClassificationResult:
    """Classify a Razorpay payment failure dict into a root-cause category.

    Parameters
    ----------
    payment:
        A payment object as returned by the Razorpay API.  Expected keys
        (all optional — missing keys are treated as empty strings):

        - ``error_code`` (e.g. ``"BAD_REQUEST_ERROR"``)
        - ``error_description`` (e.g. ``"Payment processing failed because of incorrect OTP"``)
        - ``error_source`` (e.g. ``"customer"``)
        - ``error_step`` (e.g. ``"payment_authentication"``)
        - ``error_reason`` (e.g. ``"incorrect_otp"``)
        - ``method`` (e.g. ``"card"``, ``"upi"``)
        - ``bank`` (e.g. ``"HDFC"``)

    Returns
    -------
    ClassificationResult
        The classified root cause with confidence and explanation.
    """
    error_code = str(payment.get("error_code") or "").strip()
    error_description = str(payment.get("error_description") or "").lower()
    error_source = str(payment.get("error_source") or "").lower()
    error_step = str(payment.get("error_step") or "").lower()
    error_reason = str(payment.get("error_reason") or "").lower()
    method = str(payment.get("method") or "").lower()

    # Combine all text fields for substring matching
    searchable = " ".join(filter(None, [
        error_description,
        error_source,
        error_step,
        error_reason,
    ]))

    # 1. Try error_reason first (most specific)
    if error_reason:
        for keywords, category in _REASON_RULES:
            if any(kw in error_reason for kw in keywords):
                return ClassificationResult(
                    category=category,
                    confidence=0.9,
                    reason=f"Error reason '{error_reason}' matches {category.value}",
                    raw_error_code=error_code or None,
                    raw_error_reason=error_reason or None,
                )

    # 2. Try error_code
    if error_code:
        for keywords, category in _CODE_RULES:
            if error_code in keywords:
                return ClassificationResult(
                    category=category,
                    confidence=0.85,
                    reason=f"Error code '{error_code}' maps to {category.value}",
                    raw_error_code=error_code,
                    raw_error_reason=error_reason or None,
                )

    # 3. Try description/step/source substring matching
    if searchable:
        for keywords, category in _REASON_RULES:
            if any(kw in searchable for kw in keywords):
                return ClassificationResult(
                    category=category,
                    confidence=0.7,
                    reason=f"Text match in '{searchable[:80]}' → {category.value}",
                    raw_error_code=error_code or None,
                    raw_error_reason=error_reason or None,
                )

    # 4. Heuristic: method-specific fallbacks
    if method == "upi" and error_step == "payment_authentication":
        return ClassificationResult(
            category=FailureCategory.AUTHENTICATION_FAILURE,
            confidence=0.5,
            reason="UPI authentication step failed (no specific reason given)",
            raw_error_code=error_code or None,
            raw_error_reason=error_reason or None,
        )

    if method == "card" and not searchable:
        return ClassificationResult(
            category=FailureCategory.DO_NOT_HONOR,
            confidence=0.4,
            reason="Card payment failed with no specific error details",
            raw_error_code=error_code or None,
            raw_error_reason=error_reason or None,
        )

    # 5. Unknown
    return ClassificationResult(
        category=FailureCategory.UNKNOWN,
        confidence=0.1,
        reason="Unable to determine root cause from available error data",
        raw_error_code=error_code or None,
        raw_error_reason=error_reason or None,
    )
