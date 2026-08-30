"""Mandate retry sequencer for recurring payment recovery (UPI Autopay, e-NACH, SI).

Orchestrates recurring debit retry presentation windows:
1. **Temporal Salary-Cycle Heuristic**: Schedules debits during calendar days 28-31
   and 1-5 when customer account liquidity is highest.
2. **Smart NPCI Batch Hour Routing**: Presents mandates during early morning
   off-peak hours (06:30 IST) to minimize bank switch congestion and timeouts.
3. **RBI 24-Hour Pre-Debit Notification Scheduling**: Automatically schedules
   required statutory pre-debit alerts 24 hours in advance.
4. **Progressive Backoff Ladder**: Enforces a 3-attempt escalation limit before
   transferring the mandate to human review.
"""

from __future__ import annotations

import calendar
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any

from recovery.failure_classifier import FailureCategory


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

SALARY_DAYS: frozenset[int] = frozenset({28, 29, 30, 31, 1, 2, 3, 4, 5})
MAX_MANDATE_ATTEMPTS: int = 3

IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
IST_TZ = datetime.timezone(IST_OFFSET, name="IST")


class MandateType(str, Enum):
    """Recurring mandate payment methods."""
    UPI_AUTOPAY = "upi_autopay"
    E_NACH = "e_nach"
    CARD_RECURRING = "card_recurring"


class BatchSlot(str, Enum):
    """NPCI and sponsor bank clearing batch presentation windows."""
    OFF_PEAK_MORNING_0630 = "06:30_IST"
    AFTERNOON_CLEARING_1430 = "14:30_IST"
    EVENING_WINDOW_1800 = "18:00_IST"


_SLOT_HOURS: dict[BatchSlot, tuple[int, int]] = {
    BatchSlot.OFF_PEAK_MORNING_0630: (6, 30),
    BatchSlot.AFTERNOON_CLEARING_1430: (14, 30),
    BatchSlot.EVENING_WINDOW_1800: (18, 0),
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MandateContext:
    """Context of a failed recurring mandate attempt.

    Attributes
    ----------
    mandate_id:
        Unique mandate / subscription ID (e.g. ``"sub_xxx"``).
    customer_id:
        Customer ID.
    amount_paise:
        Amount to be debited in integer paise.
    mandate_type:
        Type of mandate (UPI Autopay, e-NACH, or Card Recurring).
    failure_category:
        Diagnostic classification of the failure.
    attempt_number:
        Current attempt number (1-indexed).
    last_failed_at:
        Timestamp when the last debit failed.
    max_attempts:
        Maximum permitted automated attempts before escalation.
    """

    mandate_id: str
    customer_id: str
    amount_paise: int
    mandate_type: MandateType
    failure_category: FailureCategory
    attempt_number: int = 1
    last_failed_at: datetime.datetime | float | int | str | None = None
    max_attempts: int = MAX_MANDATE_ATTEMPTS


@dataclass(frozen=True)
class MandateRetrySchedule:
    """Calculated schedule for presenting a mandate retry.

    Attributes
    ----------
    mandate_id:
        Associated mandate ID.
    mandate_type:
        Type of mandate.
    attempt_number:
        Attempt number this schedule is for.
    scheduled_retry_utc:
        ISO-8601 UTC timestamp for presenting the debit order.
    scheduled_retry_ist:
        ISO-8601 string in Indian Standard Time (IST).
    pre_debit_notification_utc:
        Mandatory 24h prior notification timestamp in UTC.
    batch_slot:
        The chosen bank clearing batch slot.
    is_salary_window:
        Whether the retry falls within the corporate salary window (28-5).
    sequencing_rationale:
        Technical explanation of the scheduling decision.
    requires_human_escalation:
        Whether automated retries are exhausted and manual review is required.
    """

    mandate_id: str
    mandate_type: MandateType
    attempt_number: int
    scheduled_retry_utc: str
    scheduled_retry_ist: str
    pre_debit_notification_utc: str
    batch_slot: BatchSlot
    is_salary_window: bool
    sequencing_rationale: str
    requires_human_escalation: bool


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _to_ist_datetime(value: datetime.datetime | float | int | str | None) -> datetime.datetime:
    """Coerce input to an explicit IST timezone-aware datetime."""
    if value is None:
        return datetime.datetime.now(IST_TZ)

    if isinstance(value, (int, float)):
        utc_dt = datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)
        return utc_dt.astimezone(IST_TZ)

    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(IST_TZ)
        except ValueError:
            return datetime.datetime.now(IST_TZ)

    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(IST_TZ)

    return datetime.datetime.now(IST_TZ)


def is_in_salary_window(dt: datetime.datetime | float | int | str) -> bool:
    """Check if the given date is in the primary salary cycle window (Days 28-31 or 1-5)."""
    ist_dt = _to_ist_datetime(dt)
    return ist_dt.day in SALARY_DAYS


def find_next_salary_date(from_dt: datetime.datetime | float | int | str) -> datetime.date:
    """Find the next upcoming calendar date that falls within the salary cycle.

    If currently within the salary window, returns the next day or target day.
    If mid-month (e.g. Day 10), returns Day 28 of the current month.
    """
    ist_dt = _to_ist_datetime(from_dt)
    current_date = ist_dt.date()
    day = current_date.day

    # If mid-month (6 <= day <= 27), target 28th of current month
    if 6 <= day <= 27:
        return datetime.date(current_date.year, current_date.month, 28)

    # If within 28-31, step to tomorrow (or 1st of next month)
    if day >= 28:
        last_day_of_month = calendar.monthrange(current_date.year, current_date.month)[1]
        if day < last_day_of_month:
            return current_date + datetime.timedelta(days=1)
        # Advance to 1st of next month
        if current_date.month == 12:
            return datetime.date(current_date.year + 1, 1, 1)
        return datetime.date(current_date.year, current_date.month + 1, 1)

    # If within 1-4, step to tomorrow
    if 1 <= day < 5:
        return current_date + datetime.timedelta(days=1)

    # If day == 5, next salary cycle begins on 28th
    return datetime.date(current_date.year, current_date.month, 28)


def calculate_optimal_presentation_slot(
    target_date: datetime.date,
    slot: BatchSlot = BatchSlot.OFF_PEAK_MORNING_0630,
) -> datetime.datetime:
    """Calculate the presentation timestamp in IST for a specific date and clearing slot."""
    hour, minute = _SLOT_HOURS[slot]
    return datetime.datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=IST_TZ,
    )


# ---------------------------------------------------------------------------
# Main Sequencer Engine
# ---------------------------------------------------------------------------

def sequence_mandate_retry(ctx: MandateContext) -> MandateRetrySchedule:
    """Calculate the optimal retry presentation schedule for a recurring mandate failure.

    Applies failure diagnostics, salary cycles, NPCI batch hours, and RBI 24h notification rules.
    """
    last_ist = _to_ist_datetime(ctx.last_failed_at)
    next_attempt = ctx.attempt_number + 1

    # Escalation check
    if next_attempt > ctx.max_attempts:
        return MandateRetrySchedule(
            mandate_id=ctx.mandate_id,
            mandate_type=ctx.mandate_type,
            attempt_number=ctx.attempt_number,
            scheduled_retry_utc=last_ist.astimezone(datetime.timezone.utc).isoformat(),
            scheduled_retry_ist=last_ist.isoformat(),
            pre_debit_notification_utc=last_ist.astimezone(datetime.timezone.utc).isoformat(),
            batch_slot=BatchSlot.OFF_PEAK_MORNING_0630,
            is_salary_window=is_in_salary_window(last_ist),
            sequencing_rationale=(
                f"Maximum mandate attempts ({ctx.max_attempts}) reached. "
                "Automated presentation terminated; escalated to Human Review."
            ),
            requires_human_escalation=True,
        )

    # Determine target presentation date and batch slot based on failure category
    category = ctx.failure_category

    if category == FailureCategory.NETWORK_ERROR:
        # Transient network error -> Schedule for next morning off-peak batch slot
        target_date = (last_ist + datetime.timedelta(days=1)).date()
        slot = BatchSlot.OFF_PEAK_MORNING_0630
        rationale = (
            "Transient gateway/network error: scheduled for next early morning off-peak "
            "clearing slot (06:30 IST) to avoid bank switch congestion."
        )
    elif category == FailureCategory.INSUFFICIENT_FUNDS:
        # Funds issue -> Apply temporal salary-cycle heuristic
        if is_in_salary_window(last_ist) and last_ist.day < 5:
            # Already in salary window (e.g. Day 2), retry in 2 days within window
            target_date = (last_ist + datetime.timedelta(days=2)).date()
            rationale = (
                f"Insufficient funds during active salary window (Day {last_ist.day}): "
                "scheduled follow-up debit 2 days later during liquidity window."
            )
        else:
            # Mid-month failure -> schedule on upcoming 28th or 1st
            target_date = find_next_salary_date(last_ist)
            rationale = (
                f"Insufficient funds on Day {last_ist.day}: scheduled for next salary "
                f"cycle arrival date ({target_date.isoformat()} at 06:30 IST)."
            )
        slot = BatchSlot.OFF_PEAK_MORNING_0630
    elif category == FailureCategory.LIMIT_EXCEEDED:
        # Limit exceeded -> Schedule for next calendar day morning
        target_date = (last_ist + datetime.timedelta(days=1)).date()
        slot = BatchSlot.OFF_PEAK_MORNING_0630
        rationale = (
            "Daily limit exceeded: scheduled presentation for next calendar day morning "
            "after bank transaction limits reset."
        )
    else:
        # Generic decline or unclassified mandate issue -> Next morning slot
        target_date = (last_ist + datetime.timedelta(days=2)).date()
        slot = BatchSlot.OFF_PEAK_MORNING_0630
        rationale = (
            f"Mandate failure ({category.value}): scheduled for standard 48h retry "
            "during early morning clearing window."
        )

    # Calculate exact timestamps
    presentation_ist = calculate_optimal_presentation_slot(target_date, slot)
    presentation_utc = presentation_ist.astimezone(datetime.timezone.utc)

    # RBI 24-Hour Pre-Debit Notification Requirement
    notification_utc = presentation_utc - datetime.timedelta(hours=24)

    return MandateRetrySchedule(
        mandate_id=ctx.mandate_id,
        mandate_type=ctx.mandate_type,
        attempt_number=next_attempt,
        scheduled_retry_utc=presentation_utc.isoformat(),
        scheduled_retry_ist=presentation_ist.isoformat(),
        pre_debit_notification_utc=notification_utc.isoformat(),
        batch_slot=slot,
        is_salary_window=is_in_salary_window(presentation_ist),
        sequencing_rationale=rationale,
        requires_human_escalation=False,
    )
