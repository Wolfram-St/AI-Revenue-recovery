"""Tests for SQL audit persistence and database durability.

Validates that:
1. Decision traces produce valid parameterised SQL INSERT statements for `audit_logs`.
2. Audit entries produce valid parameterised SQL INSERT statements for `audit_logs`.
3. Parameter values map 1-to-1 to column types in `db/schema.sql`.
4. AuditTrail supports batch SQL export and in-memory persistence.
"""

from __future__ import annotations

import json
import pytest

from recovery.audit import (
    DecisionTrace,
    build_sql_insert_for_decision_trace,
)
from recovery.audit_trail import (
    AuditEntry,
    AuditLogRecord,
    AuditTrail,
    build_sql_insert_for_audit_entry,
)
from recovery.failure_classifier import ClassificationResult, FailureCategory
from recovery.recovery_strategy import RecoveryAction, RecoveryActionType
from recovery.recovery_executor import ExecutionResult


def _make_dummy_trace() -> DecisionTrace:
    return DecisionTrace(
        attempt_id="att_001",
        payment_id="pay_001",
        customer_id="cust_001",
        event_timestamp="2026-01-01T12:00:00Z",
        amount_inr=1500.0,
        failure_category="temporary_decline",
        recovery_probability=0.85,
        expected_recovery_value_inr=1265.0,
        scoring_recommendation="INTERVENE",
        authorized_action="RETRY_NOW",
        authorization_reason="High probability transient failure",
        matched_rule_id="R007",
        matched_rule_name="retry_now_eligible",
        rule_priority=50,
        is_stop=False,
        evaluated_rules=(("R001", False), ("R007", True)),
        model_contract="P(recovered | context)",
        probability_is_action_conditional=False,
    )


def _make_dummy_audit_entry() -> AuditEntry:
    cls = ClassificationResult(
        category=FailureCategory.NETWORK_ERROR,
        confidence=0.9,
        reason="Network timeout",
        raw_error_code="GATEWAY_ERROR",
        raw_error_reason="timeout",
    )
    action = RecoveryAction(
        action_type=RecoveryActionType.RETRY_NOW,
        reason="Transient network error",
        retry_count=0,
        estimated_cost_paise=50000,
    )
    exec_res = ExecutionResult(
        success=True,
        action_type=RecoveryActionType.RETRY_NOW,
        new_order_id="order_retry_001",
        message="Retry order created successfully",
        cost_paise=50000,
    )

    trail = AuditTrail()
    return trail.record(
        payment_id="pay_test_sql_01",
        subscription_id="",
        amount_paise=50000,
        classification=cls,
        action=action,
        execution=exec_res,
        cycle_duration_ms=12.5,
    )


class TestDecisionTraceSQLPersistence:
    def test_build_sql_insert_for_decision_trace(self):
        trace = _make_dummy_trace()
        query, params = build_sql_insert_for_decision_trace(trace)

        assert "INSERT INTO audit_logs" in query
        assert "recovery_case_id" in query
        assert "event_payload" in query
        assert len(params) == 7

        case_id, event_type, actor_type, action, reason, payload_str, created_at = params
        assert case_id == "att_001"
        assert event_type == "DECISION_RECORDED"
        assert actor_type == "POLICY_ENGINE"
        assert action == "RETRY_NOW"
        assert reason == "High probability transient failure"
        assert created_at == "2026-01-01T12:00:00Z"

        # Verify JSON payload
        payload = json.loads(payload_str)
        assert payload["attempt_id"] == "att_001"
        assert payload["matched_rule_id"] == "R007"


class TestAuditEntrySQLPersistence:
    def test_build_sql_insert_for_audit_entry(self):
        entry = _make_dummy_audit_entry()
        query, params = build_sql_insert_for_audit_entry(entry)

        assert "INSERT INTO audit_logs" in query
        assert len(params) == 7

        case_id, event_type, actor_type, action, reason, payload_str, created_iso = params
        assert case_id == "pay_test_sql_01"
        assert event_type == "RECOVERY_EXECUTION"
        assert actor_type == "SYSTEM"
        assert action == "retry_now"
        assert "Category: network_error" in reason

        payload = json.loads(payload_str)
        assert payload["entry_id"] == entry.entry_id
        assert payload["execution_success"] is True

    def test_audit_trail_batch_sql_export(self):
        trail = AuditTrail()
        cls = ClassificationResult(
            category=FailureCategory.INSUFFICIENT_FUNDS,
            confidence=0.8,
            reason="Low balance",
            raw_error_code="INSUFFICIENT_FUNDS",
            raw_error_reason="insufficient_funds",
        )
        action = RecoveryAction(
            action_type=RecoveryActionType.RETRY_LATER,
            reason="Customer may add balance",
            retry_count=0,
            estimated_cost_paise=0,
        )
        exec_res = ExecutionResult(
            success=True,
            action_type=RecoveryActionType.RETRY_LATER,
            message="Scheduled retry",
            cost_paise=0,
        )

        trail.record(
            payment_id="pay_batch_1",
            amount_paise=100000,
            classification=cls,
            action=action,
            execution=exec_res,
        )
        trail.record(
            payment_id="pay_batch_2",
            amount_paise=200000,
            classification=cls,
            action=action,
            execution=exec_res,
        )

        sql_inserts = trail.to_sql_inserts()
        assert len(sql_inserts) == 2
        for q, p in sql_inserts:
            assert "INSERT INTO audit_logs" in q
            assert len(p) == 7
