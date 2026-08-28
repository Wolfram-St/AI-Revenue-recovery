"""Tests for Day 7 portfolio audit interfaces and contracts (Task 1)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from ml.portfolio_audit import (
    PortfolioEntry,
    PortfolioSummary,
    PortfolioAllocation,
)


class TestPortfolioEntry:
    """Tests for PortfolioEntry dataclass."""

    def test_portfolio_entry_required_fields(self):
        fields = {f.name for f in dataclasses.fields(PortfolioEntry)}
        expected = {
            "attempt_id",
            "payment_id",
            "row_index",
            "optimizer_recommendation",
            "no_intervention_reason",
            "gross_incremental_value_by_arm",
            "action_cost_by_arm",
            "net_incremental_value_by_arm",
            "selected_gross_incremental_value_inr",
            "selected_action_cost_inr",
            "selected_action_cost_paise",
            "selected_net_incremental_value_inr",
            "optimizer_sort_rank",
            "authorized_action",
            "authorization_reason",
            "matched_rule_id",
            "policy_overrode_recommendation",
        }
        assert fields == expected

    def test_portfolio_entry_is_frozen(self):
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="RETRY_NOW",
            authorization_reason="Test reason",
            matched_rule_id="R007",
            policy_overrode_recommendation=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.optimizer_recommendation = "STOP"

    def test_portfolio_entry_no_intervention_reason_can_be_none(self):
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="RETRY_NOW",
            authorization_reason="Test reason",
            matched_rule_id="R007",
            policy_overrode_recommendation=False,
        )
        assert entry.no_intervention_reason is None

    def test_portfolio_entry_no_intervention_reason_can_be_string(self):
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="NO_INTERVENTION",
            no_intervention_reason="non_positive_value",
            gross_incremental_value_by_arm={},
            action_cost_by_arm={},
            net_incremental_value_by_arm={},
            selected_gross_incremental_value_inr=None,
            selected_action_cost_inr=None,
            selected_action_cost_paise=None,
            selected_net_incremental_value_inr=None,
            optimizer_sort_rank=None,
            authorized_action="STOP",
            authorization_reason="Test reason",
            matched_rule_id=None,
            policy_overrode_recommendation=False,
        )
        assert entry.no_intervention_reason == "non_positive_value"

    def test_portfolio_entry_selected_fields_can_be_none_for_no_intervention(self):
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="NO_INTERVENTION",
            no_intervention_reason="non_positive_value",
            gross_incremental_value_by_arm={},
            action_cost_by_arm={},
            net_incremental_value_by_arm={},
            selected_gross_incremental_value_inr=None,
            selected_action_cost_inr=None,
            selected_action_cost_paise=None,
            selected_net_incremental_value_inr=None,
            optimizer_sort_rank=None,
            authorized_action="STOP",
            authorization_reason="Test reason",
            matched_rule_id=None,
            policy_overrode_recommendation=False,
        )
        assert entry.selected_gross_incremental_value_inr is None
        assert entry.selected_action_cost_inr is None
        assert entry.selected_action_cost_paise is None
        assert entry.selected_net_incremental_value_inr is None
        assert entry.optimizer_sort_rank is None

    def test_portfolio_entry_distinguishes_optimizer_recommendation_and_authorized_action(self):
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="STOP",
            authorization_reason="Hard declines should not be repeatedly retried automatically.",
            matched_rule_id="R003",
            policy_overrode_recommendation=True,
        )
        assert entry.optimizer_recommendation == "RETRY_NOW"
        assert entry.authorized_action == "STOP"
        assert entry.policy_overrode_recommendation is True


class TestPortfolioSummary:
    """Tests for PortfolioSummary dataclass."""

    def test_portfolio_summary_required_fields(self):
        fields = {f.name for f in dataclasses.fields(PortfolioSummary)}
        expected = {
            "total_rows",
            "pre_screen_stopped_count",
            "invalid_prediction_count",
            "optimizer_allocated_count",
            "no_intervention_count",
            "eligible_candidate_count",
            "budget_limit_inr",
            "budget_limit_paise",
            "budget_allocated_inr",
            "budget_allocated_paise",
            "budget_remaining_inr",
            "budget_remaining_paise",
            "human_review_capacity_limit",
            "human_review_allocated_count",
            "post_policy_net_authorized_count",
            "total_policy_overrides",
            "total_policy_stop_overrides",
            "optimizer_objective_value_inr",
            "optimizer_status",
            "action_recommendation_counts",
            "action_authorized_counts",
        }
        assert fields == expected

    def test_portfolio_summary_is_frozen(self):
        summary = PortfolioSummary(
            total_rows=100,
            pre_screen_stopped_count=10,
            invalid_prediction_count=5,
            optimizer_allocated_count=50,
            no_intervention_count=35,
            eligible_candidate_count=85,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=500.0,
            budget_allocated_paise=50000,
            budget_remaining_inr=500.0,
            budget_remaining_paise=50000,
            human_review_capacity_limit=10,
            human_review_allocated_count=5,
            post_policy_net_authorized_count=45,
            total_policy_overrides=5,
            total_policy_stop_overrides=3,
            optimizer_objective_value_inr=45000.0,
            optimizer_status="success",
            action_recommendation_counts={"RETRY_NOW": 30, "HUMAN_REVIEW": 20},
            action_authorized_counts={"RETRY_NOW": 28, "STOP": 2, "HUMAN_REVIEW": 15},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.total_rows = 200

    def test_portfolio_summary_budget_fields_in_both_inr_and_paise(self):
        summary = PortfolioSummary(
            total_rows=100,
            pre_screen_stopped_count=10,
            invalid_prediction_count=5,
            optimizer_allocated_count=50,
            no_intervention_count=35,
            eligible_candidate_count=85,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=500.0,
            budget_allocated_paise=50000,
            budget_remaining_inr=500.0,
            budget_remaining_paise=50000,
            human_review_capacity_limit=10,
            human_review_allocated_count=5,
            post_policy_net_authorized_count=45,
            total_policy_overrides=5,
            total_policy_stop_overrides=3,
            optimizer_objective_value_inr=45000.0,
            optimizer_status="success",
            action_recommendation_counts={},
            action_authorized_counts={},
        )
        assert summary.budget_limit_inr == 1000.0
        assert summary.budget_limit_paise == 100000
        assert summary.budget_allocated_inr == 500.0
        assert summary.budget_allocated_paise == 50000
        assert summary.budget_remaining_inr == 500.0
        assert summary.budget_remaining_paise == 50000

    def test_portfolio_summary_none_for_unconstrained_budget(self):
        summary = PortfolioSummary(
            total_rows=100,
            pre_screen_stopped_count=10,
            invalid_prediction_count=5,
            optimizer_allocated_count=50,
            no_intervention_count=35,
            eligible_candidate_count=85,
            budget_limit_inr=None,
            budget_limit_paise=None,
            budget_allocated_inr=500.0,
            budget_allocated_paise=50000,
            budget_remaining_inr=None,
            budget_remaining_paise=None,
            human_review_capacity_limit=None,
            human_review_allocated_count=5,
            post_policy_net_authorized_count=45,
            total_policy_overrides=5,
            total_policy_stop_overrides=3,
            optimizer_objective_value_inr=45000.0,
            optimizer_status="success",
            action_recommendation_counts={},
            action_authorized_counts={},
        )
        assert summary.budget_limit_inr is None
        assert summary.budget_limit_paise is None
        assert summary.budget_remaining_inr is None
        assert summary.budget_remaining_paise is None

    def test_portfolio_summary_row_count_invariant_fields_present(self):
        summary = PortfolioSummary(
            total_rows=100,
            pre_screen_stopped_count=10,
            invalid_prediction_count=5,
            optimizer_allocated_count=50,
            no_intervention_count=35,
            eligible_candidate_count=85,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=500.0,
            budget_allocated_paise=50000,
            budget_remaining_inr=500.0,
            budget_remaining_paise=50000,
            human_review_capacity_limit=10,
            human_review_allocated_count=5,
            post_policy_net_authorized_count=45,
            total_policy_overrides=5,
            total_policy_stop_overrides=3,
            optimizer_objective_value_inr=45000.0,
            optimizer_status="success",
            action_recommendation_counts={},
            action_authorized_counts={},
        )
        assert summary.total_rows == (
            summary.pre_screen_stopped_count +
            summary.invalid_prediction_count +
            summary.optimizer_allocated_count +
            summary.no_intervention_count
        )


class TestPortfolioAllocation:
    """Tests for PortfolioAllocation dataclass and JSON serialization."""

    def test_portfolio_allocation_required_fields(self):
        fields = {f.name for f in dataclasses.fields(PortfolioAllocation)}
        expected = {"entries", "summary", "metadata"}
        assert fields == expected

    def test_portfolio_allocation_is_frozen(self):
        from ml.portfolio_audit import PortfolioEntry, PortfolioSummary
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="RETRY_NOW",
            authorization_reason="Test",
            matched_rule_id="R007",
            policy_overrode_recommendation=False,
        )
        summary = PortfolioSummary(
            total_rows=1,
            pre_screen_stopped_count=0,
            invalid_prediction_count=0,
            optimizer_allocated_count=1,
            no_intervention_count=0,
            eligible_candidate_count=1,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=10.0,
            budget_allocated_paise=1000,
            budget_remaining_inr=990.0,
            budget_remaining_paise=99000,
            human_review_capacity_limit=10,
            human_review_allocated_count=0,
            post_policy_net_authorized_count=1,
            total_policy_overrides=0,
            total_policy_stop_overrides=0,
            optimizer_objective_value_inr=90.0,
            optimizer_status="success",
            action_recommendation_counts={"RETRY_NOW": 1},
            action_authorized_counts={"RETRY_NOW": 1},
        )
        allocation = PortfolioAllocation(
            entries=(entry,),
            summary=summary,
            metadata={"test": True},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            allocation.entries = ()

    def test_portfolio_allocation_to_json_returns_string(self):
        from ml.portfolio_audit import PortfolioEntry, PortfolioSummary
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="RETRY_NOW",
            authorization_reason="Test",
            matched_rule_id="R007",
            policy_overrode_recommendation=False,
        )
        summary = PortfolioSummary(
            total_rows=1,
            pre_screen_stopped_count=0,
            invalid_prediction_count=0,
            optimizer_allocated_count=1,
            no_intervention_count=0,
            eligible_candidate_count=1,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=10.0,
            budget_allocated_paise=1000,
            budget_remaining_inr=990.0,
            budget_remaining_paise=99000,
            human_review_capacity_limit=10,
            human_review_allocated_count=0,
            post_policy_net_authorized_count=1,
            total_policy_overrides=0,
            total_policy_stop_overrides=0,
            optimizer_objective_value_inr=90.0,
            optimizer_status="success",
            action_recommendation_counts={"RETRY_NOW": 1},
            action_authorized_counts={"RETRY_NOW": 1},
        )
        allocation = PortfolioAllocation(
            entries=(entry,),
            summary=summary,
            metadata={"test": True},
        )
        json_str = allocation.to_json()
        assert isinstance(json_str, str)

    def test_portfolio_allocation_to_json_deterministic(self):
        from ml.portfolio_audit import PortfolioEntry, PortfolioSummary
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="RETRY_NOW",
            authorization_reason="Test",
            matched_rule_id="R007",
            policy_overrode_recommendation=False,
        )
        summary = PortfolioSummary(
            total_rows=1,
            pre_screen_stopped_count=0,
            invalid_prediction_count=0,
            optimizer_allocated_count=1,
            no_intervention_count=0,
            eligible_candidate_count=1,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=10.0,
            budget_allocated_paise=1000,
            budget_remaining_inr=990.0,
            budget_remaining_paise=99000,
            human_review_capacity_limit=10,
            human_review_allocated_count=0,
            post_policy_net_authorized_count=1,
            total_policy_overrides=0,
            total_policy_stop_overrides=0,
            optimizer_objective_value_inr=90.0,
            optimizer_status="success",
            action_recommendation_counts={"RETRY_NOW": 1},
            action_authorized_counts={"RETRY_NOW": 1},
        )
        allocation = PortfolioAllocation(
            entries=(entry,),
            summary=summary,
            metadata={"test": True},
        )
        json_str_1 = allocation.to_json()
        json_str_2 = allocation.to_json()
        assert json_str_1 == json_str_2

    def test_portfolio_allocation_to_json_sorted_keys(self):
        from ml.portfolio_audit import PortfolioEntry, PortfolioSummary
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="RETRY_NOW",
            authorization_reason="Test",
            matched_rule_id="R007",
            policy_overrode_recommendation=False,
        )
        summary = PortfolioSummary(
            total_rows=1,
            pre_screen_stopped_count=0,
            invalid_prediction_count=0,
            optimizer_allocated_count=1,
            no_intervention_count=0,
            eligible_candidate_count=1,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=10.0,
            budget_allocated_paise=1000,
            budget_remaining_inr=990.0,
            budget_remaining_paise=99000,
            human_review_capacity_limit=10,
            human_review_allocated_count=0,
            post_policy_net_authorized_count=1,
            total_policy_overrides=0,
            total_policy_stop_overrides=0,
            optimizer_objective_value_inr=90.0,
            optimizer_status="success",
            action_recommendation_counts={"RETRY_NOW": 1},
            action_authorized_counts={"RETRY_NOW": 1},
        )
        allocation = PortfolioAllocation(
            entries=(entry,),
            summary=summary,
            metadata={"test": True},
        )
        json_str = allocation.to_json()
        parsed = json.loads(json_str)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_portfolio_allocation_to_json_compact_separators(self):
        from ml.portfolio_audit import PortfolioEntry, PortfolioSummary
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="RETRY_NOW",
            authorization_reason="Test",
            matched_rule_id="R007",
            policy_overrode_recommendation=False,
        )
        summary = PortfolioSummary(
            total_rows=1,
            pre_screen_stopped_count=0,
            invalid_prediction_count=0,
            optimizer_allocated_count=1,
            no_intervention_count=0,
            eligible_candidate_count=1,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=10.0,
            budget_allocated_paise=1000,
            budget_remaining_inr=990.0,
            budget_remaining_paise=99000,
            human_review_capacity_limit=10,
            human_review_allocated_count=0,
            post_policy_net_authorized_count=1,
            total_policy_overrides=0,
            total_policy_stop_overrides=0,
            optimizer_objective_value_inr=90.0,
            optimizer_status="success",
            action_recommendation_counts={"RETRY_NOW": 1},
            action_authorized_counts={"RETRY_NOW": 1},
        )
        allocation = PortfolioAllocation(
            entries=(entry,),
            summary=summary,
            metadata={"test": True},
        )
        json_str = allocation.to_json()
        assert ": " not in json_str
        assert ", " not in json_str

    def test_portfolio_allocation_to_json_rejects_nan(self):
        from ml.portfolio_audit import PortfolioEntry, PortfolioSummary
        import math
        entry = PortfolioEntry(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            optimizer_recommendation="RETRY_NOW",
            no_intervention_reason=None,
            gross_incremental_value_by_arm={"RETRY_NOW": 100.0},
            action_cost_by_arm={"RETRY_NOW": 10.0},
            net_incremental_value_by_arm={"RETRY_NOW": 90.0},
            selected_gross_incremental_value_inr=100.0,
            selected_action_cost_inr=10.0,
            selected_action_cost_paise=1000,
            selected_net_incremental_value_inr=90.0,
            optimizer_sort_rank=1,
            authorized_action="RETRY_NOW",
            authorization_reason="Test",
            matched_rule_id="R007",
            policy_overrode_recommendation=False,
        )
        summary = PortfolioSummary(
            total_rows=1,
            pre_screen_stopped_count=0,
            invalid_prediction_count=0,
            optimizer_allocated_count=1,
            no_intervention_count=0,
            eligible_candidate_count=1,
            budget_limit_inr=1000.0,
            budget_limit_paise=100000,
            budget_allocated_inr=10.0,
            budget_allocated_paise=1000,
            budget_remaining_inr=990.0,
            budget_remaining_paise=99000,
            human_review_capacity_limit=10,
            human_review_allocated_count=0,
            post_policy_net_authorized_count=1,
            total_policy_overrides=0,
            total_policy_stop_overrides=0,
            optimizer_objective_value_inr=math.nan,
            optimizer_status="success",
            action_recommendation_counts={"RETRY_NOW": 1},
            action_authorized_counts={"RETRY_NOW": 1},
        )
        allocation = PortfolioAllocation(
            entries=(entry,),
            summary=summary,
            metadata={"test": True},
        )
        with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
            allocation.to_json()