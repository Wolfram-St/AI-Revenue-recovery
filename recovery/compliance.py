"""Regulatory compliance and consumer protection safeguards for revenue recovery.

Implements statutory and regulatory constraints under Indian frameworks:
1. **TRAI Calling & Messaging Hours**: Direct communication permitted only
   between 08:00 and 19:00 IST (UTC+05:30).
2. **Contact Frequency Ceilings**: Maximum 1 outreach per 24 hours, and maximum
   3 outreaches per 7-day window per customer across all automated channels.
3. **Mandatory 72-Hour Cooling-Off Period**: Outreach blocked for 72 hours following
   customer refusal, dispute escalation, or failed contact threshold.
4. **Active Promise-to-Pay (PTP) Dunning Freeze**: Direct dunning frozen during
   active PTP commitment windows (+4-hour grace period).
5. **Customer Dispute Suspension**: Instant freeze on all automated recovery while
   an account or invoice is under dispute.

Every check is pure, deterministic, and returns a structured ``ComplianceCheckResult``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Regulatory Constants
# ---------------------------------------------------------------------------

TRAI_CALL_START_HOUR_IST = 8  # 08:00 AM IST
TRAI_CALL_END_HOUR_IST = 19   # 07:00 PM IST
MAX_CONTACTS_PER_24H = 1
MAX_CONTACTS_PER_7D = 3
MANDATORY_COOLING_OFF_HOURS = 72.0
PTP_GRACE_PERIOD_HOURS = 4.0

IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
IST_TZ = datetime.timezone(IST_OFFSET, name="IST")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComplianceCheckResult:
    """Result of a single regulatory compliance check.

    Attributes
    ----------
    compliant:
        True if the action satisfies regulatory rules, False if violated.
    rule_id:
        Identifier for the rule (e.g. ``"TRAI-001"``, ``"RBI-001"``).
    rule_name:
        Short descriptive name.
    reason:
        Human-readable explanation of the check or violation.
    action_override:
        If non-compliant, the required action override (e.g. ``"STOP"``).
    metadata:
        Additional contextual metrics (e.g. current IST hour, attempt count).
    """

    compliant: bool
    rule_id: str
    rule_name: str
    reason: str
    action_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComplianceAudit:
    """Aggregate audit report evaluating all regulatory rules for an attempt."""

    all_passed: bool
    checks: tuple[ComplianceCheckResult, ...]
    blocking_rule: ComplianceCheckResult | None = None

    @property
    def authorized_action_override(self) -> str | None:
        if self.blocking_rule is not None:
            return self.blocking_rule.action_override
        return None

    @property
    def failure_reasons(self) -> list[str]:
        return [c.reason for c in self.checks if not c.compliant]


# ---------------------------------------------------------------------------
# Helper: Timestamp Coercion
# ---------------------------------------------------------------------------

def _to_utc_datetime(value: datetime.datetime | float | int | str | None) -> datetime.datetime:
    """Coerce any timestamp format to an explicit UTC datetime."""
    if value is None:
        return datetime.datetime.now(datetime.timezone.utc)

    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)

    if isinstance(value, str):
        # Handle ISO-8601 strings
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


# ---------------------------------------------------------------------------
# Check 1: TRAI Calling & Messaging Hours (08:00 - 19:00 IST)
# ---------------------------------------------------------------------------

def check_trai_calling_hours(
    timestamp: datetime.datetime | float | int | str | None = None,
) -> ComplianceCheckResult:
    """Validate that communication is within permitted TRAI hours (08:00 to 19:00 IST).

    Under TRAI Telecom Commercial Communications Customer Preference Regulations,
    direct promotional or dunning calls/messages are prohibited between 19:00 and 08:00 IST.
    """
    utc_dt = _to_utc_datetime(timestamp)
    ist_dt = utc_dt.astimezone(IST_TZ)
    hour = ist_dt.hour
    minute = ist_dt.minute

    # Valid range: 08:00 <= time < 19:00
    is_valid = TRAI_CALL_START_HOUR_IST <= hour < TRAI_CALL_END_HOUR_IST

    if is_valid:
        return ComplianceCheckResult(
            compliant=True,
            rule_id="TRAI-001",
            rule_name="trai_calling_hours_window",
            reason=f"Current time ({hour:02d}:{minute:02d} IST) is within TRAI permitted window (08:00 - 19:00 IST)",
            action_override=None,
            metadata={"ist_hour": hour, "ist_minute": minute, "ist_time": ist_dt.isoformat()},
        )

    return ComplianceCheckResult(
        compliant=False,
        rule_id="TRAI-001",
        rule_name="trai_calling_hours_window",
        reason=f"Communication attempted outside TRAI permitted window ({hour:02d}:{minute:02d} IST is outside 08:00 - 19:00 IST)",
        action_override="STOP",
        metadata={"ist_hour": hour, "ist_minute": minute, "ist_time": ist_dt.isoformat()},
    )


# ---------------------------------------------------------------------------
# Check 2: Contact Frequency Ceilings (Max 1/24h, Max 3/7d)
# ---------------------------------------------------------------------------

def check_contact_frequency(
    contact_timestamps: Sequence[datetime.datetime | float | int | str],
    current_time: datetime.datetime | float | int | str | None = None,
    *,
    max_24h: int = MAX_CONTACTS_PER_24H,
    max_7d: int = MAX_CONTACTS_PER_7D,
) -> ComplianceCheckResult:
    """Enforce contact frequency ceilings under RBI fair practices guidelines."""
    now_utc = _to_utc_datetime(current_time)
    t_24h_ago = now_utc - datetime.timedelta(hours=24)
    t_7d_ago = now_utc - datetime.timedelta(days=7)

    parsed_history = [_to_utc_datetime(ts) for ts in contact_timestamps]
    count_24h = sum(1 for ts in parsed_history if t_24h_ago <= ts <= now_utc)
    count_7d = sum(1 for ts in parsed_history if t_7d_ago <= ts <= now_utc)

    if count_24h >= max_24h:
        return ComplianceCheckResult(
            compliant=False,
            rule_id="RBI-001",
            rule_name="contact_frequency_24h_ceiling",
            reason=f"Contact frequency ceiling reached: {count_24h} contacts in last 24h (max allowed: {max_24h})",
            action_override="STOP",
            metadata={"count_24h": count_24h, "max_24h": max_24h},
        )

    if count_7d >= max_7d:
        return ComplianceCheckResult(
            compliant=False,
            rule_id="RBI-002",
            rule_name="contact_frequency_7d_ceiling",
            reason=f"Contact frequency ceiling reached: {count_7d} contacts in last 7 days (max allowed: {max_7d})",
            action_override="STOP",
            metadata={"count_7d": count_7d, "max_7d": max_7d},
        )

    return ComplianceCheckResult(
        compliant=True,
        rule_id="RBI-001",
        rule_name="contact_frequency_within_limits",
        reason=f"Contact frequency within bounds ({count_24h}/24h, {count_7d}/7d)",
        action_override=None,
        metadata={"count_24h": count_24h, "count_7d": count_7d},
    )


# ---------------------------------------------------------------------------
# Check 3: Mandatory 72-Hour Cooling-Off Period
# ---------------------------------------------------------------------------

def check_cooling_off_period(
    last_refusal_time: datetime.datetime | float | int | str | None,
    current_time: datetime.datetime | float | int | str | None = None,
    *,
    cooldown_hours: float = MANDATORY_COOLING_OFF_HOURS,
) -> ComplianceCheckResult:
    """Enforce mandatory cooling-off period after customer refusal or grievance."""
    if last_refusal_time is None:
        return ComplianceCheckResult(
            compliant=True,
            rule_id="RBI-003",
            rule_name="cooling_off_inactive",
            reason="No customer refusal or grievance logged; cooling-off inactive",
            action_override=None,
        )

    now_utc = _to_utc_datetime(current_time)
    refusal_utc = _to_utc_datetime(last_refusal_time)
    elapsed_hours = (now_utc - refusal_utc).total_seconds() / 3600.0

    if elapsed_hours < cooldown_hours:
        remaining_hours = round(cooldown_hours - elapsed_hours, 1)
        return ComplianceCheckResult(
            compliant=False,
            rule_id="RBI-003",
            rule_name="mandatory_cooling_off_active",
            reason=f"Mandatory {int(cooldown_hours)}h cooling-off period active ({remaining_hours}h remaining)",
            action_override="STOP",
            metadata={"elapsed_hours": elapsed_hours, "remaining_hours": remaining_hours},
        )

    return ComplianceCheckResult(
        compliant=True,
        rule_id="RBI-003",
        rule_name="cooling_off_expired",
        reason=f"Cooling-off period expired ({elapsed_hours:.1f}h elapsed >= {cooldown_hours}h)",
        action_override=None,
        metadata={"elapsed_hours": elapsed_hours},
    )


# ---------------------------------------------------------------------------
# Check 4: Active Promise-to-Pay (PTP) Dunning Freeze
# ---------------------------------------------------------------------------

def check_promise_to_pay_active(
    ptp_status: str | None,
    promised_date: datetime.datetime | float | int | str | None = None,
    current_time: datetime.datetime | float | int | str | None = None,
    *,
    grace_hours: float = PTP_GRACE_PERIOD_HOURS,
) -> ComplianceCheckResult:
    """Suspend automated dunning while an active debtor commitment is within window."""
    if ptp_status != "PTP_ACTIVE" or promised_date is None:
        return ComplianceCheckResult(
            compliant=True,
            rule_id="PTP-001",
            rule_name="ptp_not_active",
            reason="No active Promise-to-Pay commitment",
            action_override=None,
        )

    now_utc = _to_utc_datetime(current_time)
    promise_utc = _to_utc_datetime(promised_date)
    deadline_with_grace = promise_utc + datetime.timedelta(hours=grace_hours)

    if now_utc <= deadline_with_grace:
        return ComplianceCheckResult(
            compliant=False,
            rule_id="PTP-001",
            rule_name="ptp_active_freeze",
            reason=f"Active Promise-to-Pay in effect (Due: {promise_utc.isoformat()}, Grace: +{grace_hours}h) — automated dunning suspended",
            action_override="STOP",
            metadata={"promised_date": promise_utc.isoformat(), "grace_hours": grace_hours},
        )

    return ComplianceCheckResult(
        compliant=True,
        rule_id="PTP-001",
        rule_name="ptp_grace_expired",
        reason="Promise-to-Pay deadline and grace period expired without payment fulfillment",
        action_override=None,
    )


# ---------------------------------------------------------------------------
# Check 5: Active Customer Dispute Suspension
# ---------------------------------------------------------------------------

def check_dispute_active(dispute_status: str | None) -> ComplianceCheckResult:
    """Suspend automated recovery operations when an account or charge is disputed."""
    if dispute_status and dispute_status.upper() in ("UNDER_REVIEW", "CONFIRMED", "DISPUTED"):
        return ComplianceCheckResult(
            compliant=False,
            rule_id="RBI-004",
            rule_name="dispute_active_freeze",
            reason=f"Account under active dispute ({dispute_status}) — recovery operations suspended",
            action_override="STOP",
            metadata={"dispute_status": dispute_status},
        )

    return ComplianceCheckResult(
        compliant=True,
        rule_id="RBI-004",
        rule_name="dispute_inactive",
        reason="No active dispute on account",
        action_override=None,
    )


# ---------------------------------------------------------------------------
# Master Evaluator
# ---------------------------------------------------------------------------

def evaluate_all_compliance_rules(
    context: dict[str, Any],
    current_time: datetime.datetime | float | int | str | None = None,
) -> ComplianceAudit:
    """Evaluate full suite of TRAI/RBI compliance checks for a recovery attempt.

    Evaluation order:
    1. Active Dispute (RBI-004)
    2. Active Promise-to-Pay (PTP-001)
    3. Mandatory Cooling-Off Period (RBI-003)
    4. TRAI Calling Hours (TRAI-001)
    5. Contact Frequency Limits (RBI-001 / RBI-002)
    """
    now = current_time or context.get("event_timestamp") or datetime.datetime.now(datetime.timezone.utc)

    checks: list[ComplianceCheckResult] = []

    # 1. Dispute Check
    checks.append(check_dispute_active(context.get("dispute_status")))

    # 2. PTP Check
    checks.append(
        check_promise_to_pay_active(
            ptp_status=context.get("ptp_status"),
            promised_date=context.get("ptp_promised_date"),
            current_time=now,
        )
    )

    # 3. Cooling-off Check
    checks.append(
        check_cooling_off_period(
            last_refusal_time=context.get("last_refusal_timestamp"),
            current_time=now,
        )
    )

    # 4. TRAI Calling Hours Check (if direct contact required)
    checks.append(check_trai_calling_hours(timestamp=now))

    # 5. Frequency Check
    contact_history = context.get("contact_history_timestamps", ())
    checks.append(
        check_contact_frequency(
            contact_timestamps=contact_history,
            current_time=now,
        )
    )

    blocking = next((c for c in checks if not c.compliant), None)
    return ComplianceAudit(
        all_passed=blocking is None,
        checks=tuple(checks),
        blocking_rule=blocking,
    )
