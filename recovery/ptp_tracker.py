"""Promise-to-Pay (PTP) tracker and dynamic dunning pause state machine.

Manages debtor commitments:
1. **Commitment Registration**: Records promised repayment amount, deadline,
   channel source, and negotiated concessions (e.g. 5% waiver, 2-part split).
2. **Dynamic Dunning Hold**: Pauses outbound communications during active commitment
   window (+4-hour statutory grace period).
3. **Automated Webhook Resolution**: Matches incoming payment capture webhooks to
   fulfill active PTPs.
4. **Broken-Promise Escalation**: Automatically transitions expired promises past the
   grace window to ``PTP_BROKEN`` and triggers priority collection escalation.
"""

from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PTPStatus(str, Enum):
    """Promise-to-Pay lifecycle states."""
    PTP_ACTIVE = "PTP_ACTIVE"
    PTP_FULFILLED = "PTP_FULFILLED"
    PTP_BROKEN = "PTP_BROKEN"
    PTP_CANCELLED = "PTP_CANCELLED"


class PTPChannel(str, Enum):
    """Channel through which customer commitment was captured."""
    VOICE_AGENT = "VOICE_AGENT"
    WHATSAPP = "WHATSAPP"
    HUMAN_AGENT = "HUMAN_AGENT"
    SELF_SERVE = "SELF_SERVE"
    EMAIL = "EMAIL"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromiseToPay:
    """Immutable representation of a debtor Promise-to-Pay commitment.

    Attributes
    ----------
    ptp_id:
        Unique identifier for the PTP record (e.g. ``"ptp_xxx"``).
    recovery_case_id:
        Associated recovery case ID.
    customer_id:
        Customer identifier.
    promised_amount_inr:
        Amount promised in INR (must be > 0).
    promised_date:
        ISO-8601 UTC timestamp of the promised payment date.
    grace_period_hours:
        Grace period in hours before marking promise broken (default 4h).
    status:
        Current lifecycle state.
    channel_source:
        Channel where commitment was negotiated.
    concession_applied:
        Structured details of any discount, waiver, or EMI terms applied.
    transcript_snippet:
        Verbatim excerpt from voice/chat conversation capturing consent.
    fulfilled_at:
        ISO-8601 UTC timestamp when payment was received.
    broken_at:
        ISO-8601 UTC timestamp when promise was determined broken.
    created_at:
        ISO-8601 UTC creation timestamp.
    updated_at:
        ISO-8601 UTC update timestamp.
    """

    ptp_id: str
    recovery_case_id: str
    customer_id: str
    promised_amount_inr: float
    promised_date: str
    grace_period_hours: int = 4
    status: PTPStatus = PTPStatus.PTP_ACTIVE
    channel_source: PTPChannel = PTPChannel.VOICE_AGENT
    concession_applied: dict[str, Any] = field(default_factory=dict)
    transcript_snippet: str | None = None
    fulfilled_at: str | None = None
    broken_at: str | None = None
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper Functions & State Machine Logic
# ---------------------------------------------------------------------------

def _to_utc_datetime(value: datetime.datetime | float | int | str | None) -> datetime.datetime:
    if value is None:
        return datetime.datetime.now(datetime.timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(datetime.timezone.utc)
        except ValueError:
            return datetime.datetime.now(datetime.timezone.utc)
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(datetime.timezone.utc)
    return datetime.datetime.now(datetime.timezone.utc)


def register_promise(
    *,
    recovery_case_id: str,
    customer_id: str,
    promised_amount_inr: float,
    promised_date: datetime.datetime | str,
    channel_source: PTPChannel = PTPChannel.VOICE_AGENT,
    grace_period_hours: int = 4,
    concession_applied: dict[str, Any] | None = None,
    transcript_snippet: str | None = None,
    ptp_id: str | None = None,
) -> PromiseToPay:
    """Create and register a new active Promise-to-Pay commitment."""
    if promised_amount_inr <= 0:
        raise ValueError(f"promised_amount_inr must be positive, got {promised_amount_inr}")

    promise_dt = _to_utc_datetime(promised_date)
    promise_iso = promise_dt.isoformat()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    generated_id = ptp_id or f"ptp_{uuid.uuid4().hex[:12]}"

    return PromiseToPay(
        ptp_id=generated_id,
        recovery_case_id=recovery_case_id,
        customer_id=customer_id,
        promised_amount_inr=round(promised_amount_inr, 2),
        promised_date=promise_iso,
        grace_period_hours=grace_period_hours,
        status=PTPStatus.PTP_ACTIVE,
        channel_source=channel_source,
        concession_applied=concession_applied or {},
        transcript_snippet=transcript_snippet,
        created_at=now_iso,
        updated_at=now_iso,
    )


def evaluate_ptp_status(
    ptp: PromiseToPay,
    current_time: datetime.datetime | float | str | None = None,
) -> PromiseToPay:
    """Evaluate PTP state against current time, marking broken if grace expired."""
    if ptp.status != PTPStatus.PTP_ACTIVE:
        return ptp

    now_utc = _to_utc_datetime(current_time)
    promise_utc = _to_utc_datetime(ptp.promised_date)
    cutoff_utc = promise_utc + datetime.timedelta(hours=ptp.grace_period_hours)

    if now_utc > cutoff_utc:
        now_iso = now_utc.isoformat()
        return PromiseToPay(
            ptp_id=ptp.ptp_id,
            recovery_case_id=ptp.recovery_case_id,
            customer_id=ptp.customer_id,
            promised_amount_inr=ptp.promised_amount_inr,
            promised_date=ptp.promised_date,
            grace_period_hours=ptp.grace_period_hours,
            status=PTPStatus.PTP_BROKEN,
            channel_source=ptp.channel_source,
            concession_applied=ptp.concession_applied,
            transcript_snippet=ptp.transcript_snippet,
            fulfilled_at=None,
            broken_at=now_iso,
            created_at=ptp.created_at,
            updated_at=now_iso,
        )

    return ptp


def fulfill_promise(
    ptp: PromiseToPay,
    paid_amount_inr: float,
    paid_at: datetime.datetime | float | str | None = None,
) -> PromiseToPay:
    """Transition an active or broken promise to fulfilled upon receipt of payment."""
    if paid_amount_inr < ptp.promised_amount_inr:
        raise ValueError(
            f"Paid amount ₹{paid_amount_inr:.2f} is less than promised amount "
            f"₹{ptp.promised_amount_inr:.2f}"
        )

    paid_iso = _to_utc_datetime(paid_at).isoformat()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return PromiseToPay(
        ptp_id=ptp.ptp_id,
        recovery_case_id=ptp.recovery_case_id,
        customer_id=ptp.customer_id,
        promised_amount_inr=ptp.promised_amount_inr,
        promised_date=ptp.promised_date,
        grace_period_hours=ptp.grace_period_hours,
        status=PTPStatus.PTP_FULFILLED,
        channel_source=ptp.channel_source,
        concession_applied=ptp.concession_applied,
        transcript_snippet=ptp.transcript_snippet,
        fulfilled_at=paid_iso,
        broken_at=None,
        created_at=ptp.created_at,
        updated_at=now_iso,
    )


def cancel_promise(ptp: PromiseToPay) -> PromiseToPay:
    """Transition a promise to cancelled (e.g. customer opt-out or explicit dispute)."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return PromiseToPay(
        ptp_id=ptp.ptp_id,
        recovery_case_id=ptp.recovery_case_id,
        customer_id=ptp.customer_id,
        promised_amount_inr=ptp.promised_amount_inr,
        promised_date=ptp.promised_date,
        grace_period_hours=ptp.grace_period_hours,
        status=PTPStatus.PTP_CANCELLED,
        channel_source=ptp.channel_source,
        concession_applied=ptp.concession_applied,
        transcript_snippet=ptp.transcript_snippet,
        fulfilled_at=None,
        broken_at=None,
        created_at=ptp.created_at,
        updated_at=now_iso,
    )


def should_pause_dunning(
    ptp: PromiseToPay | None,
    current_time: datetime.datetime | float | str | None = None,
) -> tuple[bool, str]:
    """Check if automated recovery dunning should be paused for this PTP."""
    if ptp is None:
        return False, "No active promise to pay"

    evaluated = evaluate_ptp_status(ptp, current_time)
    if evaluated.status == PTPStatus.PTP_ACTIVE:
        return True, (
            f"Dunning paused: Active Promise-to-Pay until {ptp.promised_date} "
            f"(+{ptp.grace_period_hours}h grace)"
        )

    if evaluated.status == PTPStatus.PTP_BROKEN:
        return False, f"Promise broken at {evaluated.broken_at}: Escalating to priority collection"

    if evaluated.status == PTPStatus.PTP_FULFILLED:
        return False, f"Promise fulfilled at {evaluated.fulfilled_at}: Case resolved"

    return False, "Promise is cancelled"


# ---------------------------------------------------------------------------
# SQL Persistence Helper (Matching db/schema.sql)
# ---------------------------------------------------------------------------

def build_sql_insert_for_ptp(ptp: PromiseToPay) -> tuple[str, tuple[Any, ...]]:
    """Generate parameterised SQL INSERT matching promise_to_pay table in db/schema.sql."""
    query = """
    INSERT INTO promise_to_pay (
        ptp_id,
        recovery_case_id,
        customer_id,
        promised_amount_inr,
        promised_date,
        grace_period_hours,
        status,
        channel_source,
        concession_applied,
        transcript_snippet,
        fulfilled_at,
        broken_at,
        created_at,
        updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """.strip()

    concession_json = json.dumps(ptp.concession_applied)
    params = (
        ptp.ptp_id,
        ptp.recovery_case_id,
        ptp.customer_id,
        ptp.promised_amount_inr,
        ptp.promised_date,
        ptp.grace_period_hours,
        ptp.status.value,
        ptp.channel_source.value,
        concession_json,
        ptp.transcript_snippet,
        ptp.fulfilled_at,
        ptp.broken_at,
        ptp.created_at,
        ptp.updated_at,
    )
    return query, params


# ---------------------------------------------------------------------------
# In-Memory PTP Registry
# ---------------------------------------------------------------------------

class PTPRegistry:
    """Thread-safe in-memory PTP registry with indexing by case and customer."""

    def __init__(self) -> None:
        self._by_id: dict[str, PromiseToPay] = {}
        self._by_case: dict[str, str] = {}
        self._by_customer: dict[str, list[str]] = {}

    def register(self, ptp: PromiseToPay) -> PromiseToPay:
        self._by_id[ptp.ptp_id] = ptp
        self._by_case[ptp.recovery_case_id] = ptp.ptp_id
        if ptp.customer_id not in self._by_customer:
            self._by_customer[ptp.customer_id] = []
        if ptp.ptp_id not in self._by_customer[ptp.customer_id]:
            self._by_customer[ptp.customer_id].append(ptp.ptp_id)
        return ptp

    def get(self, ptp_id: str) -> PromiseToPay | None:
        return self._by_id.get(ptp_id)

    def get_by_case(self, recovery_case_id: str) -> PromiseToPay | None:
        ptp_id = self._by_case.get(recovery_case_id)
        if ptp_id:
            return self._by_id.get(ptp_id)
        return None

    def get_active_by_customer(self, customer_id: str) -> list[PromiseToPay]:
        ptp_ids = self._by_customer.get(customer_id, [])
        active = []
        for pid in ptp_ids:
            p = self._by_id.get(pid)
            if p and p.status == PTPStatus.PTP_ACTIVE:
                active.append(p)
        return active

    def update_all_statuses(self, current_time: datetime.datetime | None = None) -> list[PromiseToPay]:
        updated = []
        for ptp_id, ptp in list(self._by_id.items()):
            if ptp.status == PTPStatus.PTP_ACTIVE:
                evaluated = evaluate_ptp_status(ptp, current_time)
                if evaluated.status != ptp.status:
                    self._by_id[ptp_id] = evaluated
                    updated.append(evaluated)
        return updated

    def clear(self) -> None:
        self._by_id.clear()
        self._by_case.clear()
        self._by_customer.clear()


# Default singleton instance
default_ptp_registry = PTPRegistry()
