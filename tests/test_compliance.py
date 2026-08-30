"""Tests for TRAI/RBI regulatory compliance safeguards and frequency ceilings.

Covers:
- TRAI calling and messaging hours (08:00 - 19:00 IST)
- RBI contact frequency ceilings (max 1/24h, max 3/7d)
- Mandatory 72-hour cooling-off periods after refusal
- Promise-to-Pay (PTP) active dunning freezes
- Account dispute freeze
- Integration with Recovery Strategy engine
"""

from __future__ import annotations

import datetime
import pytest

from recovery.compliance import (
    IST_TZ,
    check_cooling_off_period,
    check_contact_frequency,
    check_dispute_active,
    check_promise_to_pay_active,
    check_trai_calling_hours,
    evaluate_all_compliance_rules,
)
from recovery.failure_classifier import ClassificationResult, FailureCategory
from recovery.recovery_strategy import (
    RecoveryActionType,
    StrategyContext,
    select_recovery_action,
)


class TestTRAICallingHours:
    def test_calling_hours_within_window(self):
        # 10:30 AM IST is within 08:00-19:00 IST
        ist_dt = datetime.datetime(2026, 8, 30, 10, 30, tzinfo=IST_TZ)
        res = check_trai_calling_hours(ist_dt)
        assert res.compliant is True
        assert res.action_override is None

    def test_calling_hours_early_morning_blocked(self):
        # 07:59 AM IST is outside 08:00-19:00 IST
        ist_dt = datetime.datetime(2026, 8, 30, 7, 59, tzinfo=IST_TZ)
        res = check_trai_calling_hours(ist_dt)
        assert res.compliant is False
        assert res.action_override == "STOP"
        assert "TRAI permitted window" in res.reason

    def test_calling_hours_late_night_blocked(self):
        # 20:15 IST (08:15 PM) is outside 08:00-19:00 IST
        ist_dt = datetime.datetime(2026, 8, 30, 20, 15, tzinfo=IST_TZ)
        res = check_trai_calling_hours(ist_dt)
        assert res.compliant is False
        assert res.action_override == "STOP"

    def test_calling_hours_utc_conversion(self):
        # 03:00 UTC == 08:30 IST (Valid)
        utc_dt = datetime.datetime(2026, 8, 30, 3, 0, tzinfo=datetime.timezone.utc)
        res = check_trai_calling_hours(utc_dt)
        assert res.compliant is True


class TestContactFrequencyCeilings:
    def test_first_contact_within_limits(self):
        now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
        res = check_contact_frequency([], current_time=now)
        assert res.compliant is True

    def test_exceeding_24h_ceiling_blocked(self):
        now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
        past_contacts = [
            now - datetime.timedelta(hours=4),
        ]
        res = check_contact_frequency(past_contacts, current_time=now, max_24h=1)
        assert res.compliant is False
        assert res.action_override == "STOP"
        assert "last 24h" in res.reason

    def test_exceeding_7d_ceiling_blocked(self):
        now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
        past_contacts = [
            now - datetime.timedelta(days=1),
            now - datetime.timedelta(days=3),
            now - datetime.timedelta(days=5),
        ]
        res = check_contact_frequency(past_contacts, current_time=now, max_24h=10, max_7d=3)
        assert res.compliant is False
        assert res.action_override == "STOP"
        assert "last 7 days" in res.reason


class TestCoolingOffPeriod:
    def test_no_refusal_cooling_off_inactive(self):
        now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
        res = check_cooling_off_period(None, current_time=now)
        assert res.compliant is True

    def test_recent_refusal_blocks_outreach(self):
        now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
        refusal_time = now - datetime.timedelta(hours=24)  # 24h ago < 72h
        res = check_cooling_off_period(refusal_time, current_time=now, cooldown_hours=72)
        assert res.compliant is False
        assert res.action_override == "STOP"
        assert "cooling-off period active" in res.reason

    def test_expired_cooling_off_passes(self):
        now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
        refusal_time = now - datetime.timedelta(hours=75)  # 75h ago >= 72h
        res = check_cooling_off_period(refusal_time, current_time=now, cooldown_hours=72)
        assert res.compliant is True


class TestPromiseToPayFreeze:
    def test_ptp_active_within_window_blocks(self):
        now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
        promise_date = now + datetime.timedelta(days=2)
        res = check_promise_to_pay_active("PTP_ACTIVE", promise_date, current_time=now)
        assert res.compliant is False
        assert res.action_override == "STOP"
        assert "Active Promise-to-Pay" in res.reason

    def test_ptp_grace_expired_passes(self):
        now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
        promise_date = now - datetime.timedelta(hours=5)  # 5h ago > 4h grace
        res = check_promise_to_pay_active("PTP_ACTIVE", promise_date, current_time=now, grace_hours=4)
        assert res.compliant is True


class TestDisputeFreeze:
    def test_dispute_under_review_blocks(self):
        res = check_dispute_active("UNDER_REVIEW")
        assert res.compliant is False
        assert res.action_override == "STOP"

    def test_dispute_none_passes(self):
        res = check_dispute_active("NONE")
        assert res.compliant is True


class TestStrategyComplianceIntegration:
    def test_strategy_stops_on_cooling_off_violation(self):
        now = datetime.datetime(2026, 8, 30, 6, 0, tzinfo=datetime.timezone.utc) # 11:30 AM IST (valid hour)
        refusal = now - datetime.timedelta(hours=10)

        cls = ClassificationResult(
            category=FailureCategory.NETWORK_ERROR,
            confidence=0.9,
            reason="Timeout",
            raw_error_code=None,
            raw_error_reason=None,
        )
        ctx = StrategyContext(
            classification=cls,
            amount_paise=50000,
            timestamp=now,
            last_refusal_timestamp=refusal,
            enforce_compliance=True,
        )
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.STOP
        assert "Regulatory compliance block [RBI-003]" in action.reason
