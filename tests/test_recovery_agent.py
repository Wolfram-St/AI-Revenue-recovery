"""Tests for the recovery agent pipeline.

Covers:
- Failure classification accuracy
- Strategy selection correctness
- Executor simulation mode
- Audit trail integrity
- Full pipeline end-to-end
- Budget enforcement
- Retry-cap enforcement
"""

from __future__ import annotations

import pytest

from recovery.failure_classifier import (
    classify_failure,
    FailureCategory,
    ClassificationResult,
)
from recovery.recovery_strategy import (
    RecoveryAction,
    RecoveryActionType,
    StrategyContext,
    select_recovery_action,
)
from recovery.recovery_executor import (
    RecoveryExecutor,
    RecoveryTarget,
    ExecutionResult,
)
from recovery.audit_trail import AuditTrail, AuditEntry
from recovery.recovery_agent import (
    AgentConfig,
    BatchResult,
    RecoveryAgent,
    generate_synthetic_failures,
)


# ---------------------------------------------------------------------------
# Failure classifier tests
# ---------------------------------------------------------------------------

class TestFailureClassifier:
    """Verify classification accuracy for common Razorpay error patterns."""

    def test_incorrect_otp_classified_as_auth_failure(self):
        result = classify_failure({
            "error_reason": "incorrect_otp",
            "method": "card",
        })
        assert result.category == FailureCategory.AUTHENTICATION_FAILURE
        assert result.confidence >= 0.8

    def test_timeout_classified_as_network_error(self):
        result = classify_failure({
            "error_reason": "timeout",
            "method": "card",
        })
        assert result.category == FailureCategory.NETWORK_ERROR

    def test_insufficient_funds_classified(self):
        result = classify_failure({
            "error_reason": "insufficient_funds",
            "method": "card",
        })
        assert result.category == FailureCategory.INSUFFICIENT_FUNDS

    def test_expired_card_classified_as_card_issue(self):
        result = classify_failure({
            "error_reason": "expired_card",
            "method": "card",
        })
        assert result.category == FailureCategory.CARD_ISSUE

    def test_fraud_classified(self):
        result = classify_failure({
            "error_reason": "suspected_fraud",
            "method": "card",
        })
        assert result.category == FailureCategory.FRAUD_SUSPECTED

    def test_unknown_failure_classified(self):
        result = classify_failure({})
        assert result.category == FailureCategory.UNKNOWN

    def test_description_text_match(self):
        result = classify_failure({
            "error_description": "Payment failed due to incorrect OTP",
        })
        assert result.category == FailureCategory.AUTHENTICATION_FAILURE

    def test_gateway_error_maps_to_network(self):
        result = classify_failure({
            "error_code": "GATEWAY_ERROR",
        })
        assert result.category == FailureCategory.NETWORK_ERROR

    def test_bad_request_maps_to_business_error(self):
        result = classify_failure({
            "error_code": "BAD_REQUEST_ERROR",
        })
        assert result.category == FailureCategory.BUSINESS_ERROR


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestRecoveryStrategy:
    """Verify strategy selection correctness."""

    def _make_ctx(
        self,
        category: FailureCategory,
        *,
        retry_count: int = 0,
        max_retries: int = 3,
        budget: int = 100_000,
        hr: int = 10,
    ) -> StrategyContext:
        cls = ClassificationResult(
            category=category,
            confidence=0.8,
            reason="test",
            raw_error_code=None,
            raw_error_reason=None,
        )
        return StrategyContext(
            classification=cls,
            amount_paise=50000,
            retry_count=retry_count,
            max_retries=max_retries,
            remaining_budget_paise=budget,
            hr_capacity_remaining=hr,
        )

    def test_network_error_triggers_retry_now(self):
        ctx = self._make_ctx(FailureCategory.NETWORK_ERROR)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.RETRY_NOW

    def test_insufficient_funds_triggers_retry_later(self):
        ctx = self._make_ctx(FailureCategory.INSUFFICIENT_FUNDS)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.RETRY_LATER

    def test_card_issue_triggers_change_method(self):
        ctx = self._make_ctx(FailureCategory.CARD_ISSUE)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.CHANGE_PAYMENT_METHOD

    def test_fraud_triggers_escalate(self):
        ctx = self._make_ctx(FailureCategory.FRAUD_SUSPECTED)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.ESCALATE_TO_HUMAN

    def test_retries_exhausted_escalates(self):
        ctx = self._make_ctx(FailureCategory.NETWORK_ERROR, retry_count=3, max_retries=3)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.ESCALATE_TO_HUMAN

    def test_retries_exhausted_fraud_stops(self):
        ctx = self._make_ctx(FailureCategory.FRAUD_SUSPECTED, retry_count=3, max_retries=3)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.STOP

    def test_budget_exhausted_escalates(self):
        ctx = self._make_ctx(FailureCategory.NETWORK_ERROR, budget=0)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.ESCALATE_TO_HUMAN

    def test_budget_exhausted_fraud_stops(self):
        # Fraud with retries exhausted AND no HR capacity → STOP
        ctx = self._make_ctx(FailureCategory.FRAUD_SUSPECTED, budget=0, hr=0, retry_count=3, max_retries=3)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.STOP

    def test_fraud_always_escalates_even_with_budget(self):
        # Fraud with no retries → ESCALATE (fraud always escalates)
        ctx = self._make_ctx(FailureCategory.FRAUD_SUSPECTED, budget=0, hr=0)
        action = select_recovery_action(ctx)
        assert action.action_type == RecoveryActionType.ESCALATE_TO_HUMAN


# ---------------------------------------------------------------------------
# Executor tests
# ---------------------------------------------------------------------------

class TestRecoveryExecutor:
    """Verify executor behaviour in simulation mode."""

    def test_simulation_retry_now_succeeds(self):
        executor = RecoveryExecutor(simulation=True)
        action = RecoveryAction(RecoveryActionType.RETRY_NOW, "test", 0, 50000)
        target = RecoveryTarget(payment_id="pay_test", amount_paise=50000)
        result = executor.execute(action, target)
        assert result.success is True
        assert result.new_order_id is not None
        assert result.cost_paise == 50000

    def test_simulation_retry_later_no_cost(self):
        executor = RecoveryExecutor(simulation=True)
        action = RecoveryAction(RecoveryActionType.RETRY_LATER, "test", 0, 0)
        target = RecoveryTarget(payment_id="pay_test", amount_paise=50000)
        result = executor.execute(action, target)
        assert result.success is True
        assert result.cost_paise == 0

    def test_simulation_escalate(self):
        executor = RecoveryExecutor(simulation=True)
        action = RecoveryAction(RecoveryActionType.ESCALATE_TO_HUMAN, "test", 0, 0)
        target = RecoveryTarget(payment_id="pay_test")
        result = executor.execute(action, target)
        assert result.success is True

    def test_simulation_stop(self):
        executor = RecoveryExecutor(simulation=True)
        action = RecoveryAction(RecoveryActionType.STOP, "test", 0, 0)
        target = RecoveryTarget(payment_id="pay_test")
        result = executor.execute(action, target)
        assert result.success is True


# ---------------------------------------------------------------------------
# Audit trail tests
# ---------------------------------------------------------------------------

class TestAuditTrail:
    """Verify audit trail recording and aggregation."""

    def test_record_entry(self):
        trail = AuditTrail()
        cls = ClassificationResult(FailureCategory.NETWORK_ERROR, 0.8, "test", None, None)
        action = RecoveryAction(RecoveryActionType.RETRY_NOW, "test", 0, 50000)
        exec_r = ExecutionResult(True, RecoveryActionType.RETRY_NOW, cost_paise=50000)

        entry = trail.record(
            payment_id="pay_1",
            amount_paise=50000,
            classification=cls,
            action=action,
            execution=exec_r,
        )
        assert entry.entry_id == "audit_000001"
        assert entry.payment_id == "pay_1"
        assert trail.count == 1

    def test_summary(self):
        trail = AuditTrail()
        cls = ClassificationResult(FailureCategory.NETWORK_ERROR, 0.8, "test", None, None)
        action = RecoveryAction(RecoveryActionType.RETRY_NOW, "test", 0, 50000)
        exec_r = ExecutionResult(True, RecoveryActionType.RETRY_NOW, cost_paise=50000)
        trail.record(payment_id="pay_1", amount_paise=50000, classification=cls, action=action, execution=exec_r)

        summary = trail.summary()
        assert summary["total_entries"] == 1
        assert summary["total_amount_inr"] == 500.0
        assert summary["total_recovered_inr"] == 500.0

    def test_clear(self):
        trail = AuditTrail()
        cls = ClassificationResult(FailureCategory.UNKNOWN, 0.1, "test", None, None)
        action = RecoveryAction(RecoveryActionType.STOP, "test", 0, 0)
        exec_r = ExecutionResult(True, RecoveryActionType.STOP)
        trail.record(payment_id="pay_1", amount_paise=1000, classification=cls, action=action, execution=exec_r)
        trail.clear()
        assert trail.count == 0


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestRecoveryAgentPipeline:
    """End-to-end pipeline tests."""

    def test_synthetic_generation_deterministic(self):
        f1 = generate_synthetic_failures(count=10)
        f2 = generate_synthetic_failures(count=10)
        assert len(f1) == len(f2)
        assert [f["id"] for f in f1] == [f["id"] for f in f2]

    def test_batch_run_completes(self):
        agent = RecoveryAgent.simulation()
        failures = generate_synthetic_failures(count=10)
        result = agent.run_batch(failures)
        assert result.total_processed == 10
        assert len(result.audit_entries) == 10

    def test_budget_enforced(self):
        config = AgentConfig(budget_limit_paise=10_000, simulation=True)  # Rs.100
        agent = RecoveryAgent.simulation(config=config)
        failures = generate_synthetic_failures(count=20)
        result = agent.run_batch(failures)
        # Budget should not be exceeded
        assert result.budget_used_paise <= 10_000

    def test_result_serializable(self):
        agent = RecoveryAgent.simulation()
        failures = generate_synthetic_failures(count=5)
        result = agent.run_batch(failures)
        d = result.to_dict()
        assert "total_processed" in d
        assert "recovered_inr" in d
        assert "audit_summary" in d

    def test_audit_entries_have_ids(self):
        agent = RecoveryAgent.simulation()
        failures = generate_synthetic_failures(count=3)
        result = agent.run_batch(failures)
        for entry in result.audit_entries:
            assert entry.entry_id.startswith("audit_")
            assert entry.payment_id.startswith("pay_")
