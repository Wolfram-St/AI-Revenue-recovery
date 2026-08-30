"""Tests for Mandate Retry Sequencer (UPI Autopay, e-NACH, Recurring Cards).

Covers:
- Salary cycle window detection (Days 28-31, 1-5)
- Next salary cycle date calculation
- NPCI batch hour slot mapping (06:30 IST)
- RBI 24-hour pre-debit notification timing
- Category-specific mandate scheduling heuristics
- Attempt escalation ladder and human review transition
"""

from __future__ import annotations

import datetime
import pytest

from recovery.failure_classifier import FailureCategory
from recovery.mandate_sequencer import (
    BatchSlot,
    MandateContext,
    MandateRetrySchedule,
    MandateType,
    calculate_optimal_presentation_slot,
    find_next_salary_date,
    is_in_salary_window,
    sequence_mandate_retry,
    IST_TZ,
)


class TestSalaryCycleDetection:
    def test_salary_window_days(self):
        # Days 1-5 and 28-31 should be in salary window
        assert is_in_salary_window(datetime.datetime(2026, 8, 1, tzinfo=IST_TZ)) is True
        assert is_in_salary_window(datetime.datetime(2026, 8, 5, tzinfo=IST_TZ)) is True
        assert is_in_salary_window(datetime.datetime(2026, 8, 28, tzinfo=IST_TZ)) is True
        assert is_in_salary_window(datetime.datetime(2026, 8, 31, tzinfo=IST_TZ)) is True

    def test_mid_month_days_not_in_salary_window(self):
        assert is_in_salary_window(datetime.datetime(2026, 8, 10, tzinfo=IST_TZ)) is False
        assert is_in_salary_window(datetime.datetime(2026, 8, 15, tzinfo=IST_TZ)) is False
        assert is_in_salary_window(datetime.datetime(2026, 8, 25, tzinfo=IST_TZ)) is False

    def test_find_next_salary_date_from_mid_month(self):
        # Day 14 should schedule 28th of current month
        from_dt = datetime.datetime(2026, 8, 14, tzinfo=IST_TZ)
        next_date = find_next_salary_date(from_dt)
        assert next_date == datetime.date(2026, 8, 28)

    def test_find_next_salary_date_from_month_end(self):
        # Aug 31 (month end) should advance to Sep 1
        from_dt = datetime.datetime(2026, 8, 31, tzinfo=IST_TZ)
        next_date = find_next_salary_date(from_dt)
        assert next_date == datetime.date(2026, 9, 1)


class TestBatchHourRouting:
    def test_calculate_optimal_slot_0630_ist(self):
        target = datetime.date(2026, 9, 1)
        slot_dt = calculate_optimal_presentation_slot(target, BatchSlot.OFF_PEAK_MORNING_0630)
        assert slot_dt.hour == 6
        assert slot_dt.minute == 30
        assert slot_dt.tzinfo == IST_TZ

        # Verify UTC conversion (06:30 IST == 01:00 UTC)
        utc_dt = slot_dt.astimezone(datetime.timezone.utc)
        assert utc_dt.hour == 1
        assert utc_dt.minute == 0


class TestMandateRetrySequencerLogic:
    def test_network_error_schedules_next_morning_slot(self):
        fail_time = datetime.datetime(2026, 8, 10, 15, 0, tzinfo=IST_TZ)
        ctx = MandateContext(
            mandate_id="sub_net_fail_01",
            customer_id="cust_001",
            amount_paise=150000,
            mandate_type=MandateType.UPI_AUTOPAY,
            failure_category=FailureCategory.NETWORK_ERROR,
            attempt_number=1,
            last_failed_at=fail_time,
        )

        schedule = sequence_mandate_retry(ctx)
        assert schedule.attempt_number == 2
        assert schedule.batch_slot == BatchSlot.OFF_PEAK_MORNING_0630
        assert schedule.requires_human_escalation is False

        # Should be scheduled for Aug 11 at 06:30 IST
        assert "2026-08-11T06:30:00+05:30" in schedule.scheduled_retry_ist

        # Pre-debit notification must be exactly 24h prior (Aug 10 at 01:00 UTC)
        retry_utc = datetime.datetime.fromisoformat(schedule.scheduled_retry_utc)
        notify_utc = datetime.datetime.fromisoformat(schedule.pre_debit_notification_utc)
        assert (retry_utc - notify_utc) == datetime.timedelta(hours=24)

    def test_insufficient_funds_schedules_to_salary_window(self):
        fail_time = datetime.datetime(2026, 8, 12, 11, 0, tzinfo=IST_TZ) # Mid month
        ctx = MandateContext(
            mandate_id="sub_funds_01",
            customer_id="cust_002",
            amount_paise=299900,
            mandate_type=MandateType.E_NACH,
            failure_category=FailureCategory.INSUFFICIENT_FUNDS,
            attempt_number=1,
            last_failed_at=fail_time,
        )

        schedule = sequence_mandate_retry(ctx)
        assert schedule.attempt_number == 2
        assert "2026-08-28T06:30:00+05:30" in schedule.scheduled_retry_ist
        assert schedule.is_salary_window is True
        assert schedule.requires_human_escalation is False

    def test_max_retries_triggers_human_escalation(self):
        fail_time = datetime.datetime(2026, 8, 30, 10, 0, tzinfo=IST_TZ)
        ctx = MandateContext(
            mandate_id="sub_exhausted_01",
            customer_id="cust_003",
            amount_paise=500000,
            mandate_type=MandateType.CARD_RECURRING,
            failure_category=FailureCategory.CARD_ISSUE,
            attempt_number=3, # Already at attempt 3
            max_attempts=3,
            last_failed_at=fail_time,
        )

        schedule = sequence_mandate_retry(ctx)
        assert schedule.requires_human_escalation is True
        assert "Automated presentation terminated; escalated to Human Review" in schedule.sequencing_rationale
