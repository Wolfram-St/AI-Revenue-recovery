"""Tests for Day 7 portfolio optimizer interfaces and contracts (Task 1)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from ml.portfolio_optimizer import (
    OptimizerConfig,
    OPTIMIZER_FORBIDDEN_COLUMNS,
    PortfolioOptimizationError,
    PortfolioProblemTooLargeError,
)
from ml.features import FORBIDDEN_FEATURES


class TestOptimizerConfigValidation:
    """Tests for OptimizerConfig validation and immutability."""

    def test_optimizer_config_defaults(self):
        config = OptimizerConfig()
        assert config.budget_limit_inr is None
        assert config.human_review_capacity is None
        assert config.max_supported_rows == 1000
        assert config.max_supported_budget_units == 500
        assert config.max_supported_hr_capacity == 200

    def test_optimizer_config_custom_values(self):
        config = OptimizerConfig(
            budget_limit_inr=5000.0,
            human_review_capacity=10,
            max_supported_rows=2000,
            max_supported_budget_units=1000,
            max_supported_hr_capacity=100,
        )
        assert config.budget_limit_inr == 5000.0
        assert config.human_review_capacity == 10
        assert config.max_supported_rows == 2000
        assert config.max_supported_budget_units == 1000
        assert config.max_supported_hr_capacity == 100

    def test_negative_budget_limit_raises_value_error(self):
        with pytest.raises(ValueError, match="budget_limit_inr"):
            OptimizerConfig(budget_limit_inr=-100.0)

    def test_negative_human_review_capacity_raises_value_error(self):
        with pytest.raises(ValueError, match="human_review_capacity"):
            OptimizerConfig(human_review_capacity=-1)

    def test_optimizer_config_is_frozen(self):
        config = OptimizerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.budget_limit_inr = 1000.0


class TestPortfolioOptimizationExceptions:
    """Tests for portfolio optimization exception hierarchy."""

    def test_portfolio_optimization_error_is_exception(self):
        assert issubclass(PortfolioOptimizationError, Exception)

    def test_portfolio_problem_too_large_error_inheritance(self):
        assert issubclass(PortfolioProblemTooLargeError, PortfolioOptimizationError)
        assert issubclass(PortfolioProblemTooLargeError, Exception)

    def test_portfolio_problem_too_large_error_can_be_raised(self):
        with pytest.raises(PortfolioProblemTooLargeError):
            raise PortfolioProblemTooLargeError("Problem exceeds supported dimensions")

    def test_portfolio_problem_too_large_error_caught_as_parent(self):
        try:
            raise PortfolioProblemTooLargeError("Too large")
        except PortfolioOptimizationError:
            pass
        else:
            pytest.fail("PortfolioProblemTooLargeError should be caught as PortfolioOptimizationError")


class TestOptimizerForbiddenColumns:
    """Tests for OPTIMIZER_FORBIDDEN_COLUMNS completeness."""

    def test_optimizer_forbidden_columns_is_frozenset(self):
        assert isinstance(OPTIMIZER_FORBIDDEN_COLUMNS, frozenset)

    def test_optimizer_forbidden_columns_includes_day1_forbidden_features(self):
        # OPTIMIZER_FORBIDDEN_COLUMNS should include the post-decision parts of FORBIDDEN_FEATURES
        # but not identifiers or timestamp (allowed in optimizer input for audit tracing)
        from ml.features import IDENTIFIER_COLUMNS, TIME_COLUMN, LABEL_COLUMNS
        day1_post_decision_forbidden = set(LABEL_COLUMNS) | {"recovery_time_hours", "recovery_action", "action_outcome", "recovered_amount_inr"}
        assert day1_post_decision_forbidden.issubset(OPTIMIZER_FORBIDDEN_COLUMNS)

    def test_optimizer_forbidden_columns_includes_day4_outcome_fields(self):
        day4_outcome_fields = {
            "simulated_recovered",
            "simulated_recovered_amount_inr",
            "treatment_timestamp",
            "outcome_timestamp",
        }
        assert day4_outcome_fields.issubset(OPTIMIZER_FORBIDDEN_COLUMNS)

    def test_optimizer_forbidden_columns_includes_day5_ground_truth_fields(self):
        day5_ground_truth_fields = {
            "base_recovery_propensity",
            "action_effect_logit",
            "propensity_under_assignment",
        }
        assert day5_ground_truth_fields.issubset(OPTIMIZER_FORBIDDEN_COLUMNS)

    def test_optimizer_forbidden_columns_includes_day4_assignment_fields(self):
        day4_assignment_fields = {
            "assignment_probability",
            "arm_source",
            "assigned_action",
        }
        assert day4_assignment_fields.issubset(OPTIMIZER_FORBIDDEN_COLUMNS)

    def test_optimizer_forbidden_columns_excludes_decision_time_features(self):
        from ml.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        decision_time_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        assert decision_time_features.isdisjoint(OPTIMIZER_FORBIDDEN_COLUMNS)

    def test_optimizer_forbidden_columns_excludes_identifiers_and_timestamp(self):
        allowed_identifiers = {"attempt_id", "payment_id", "customer_id", "event_timestamp"}
        assert allowed_identifiers.isdisjoint(OPTIMIZER_FORBIDDEN_COLUMNS)


class TestDeterministicJsonSerializationContract:
    """Tests for deterministic JSON serialization contract."""

    def test_json_dumps_sort_keys_true(self):
        import json
        data = {"b": 1, "a": 2}
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert serialized == '{"a":2,"b":1}'

    def test_json_dumps_separators_compact(self):
        import json
        data = {"a": 1, "b": 2}
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert " " not in serialized

    def test_json_dumps_allow_nan_false(self):
        import json
        import math
        data = {"value": math.nan}
        with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def test_json_dumps_rejects_infinity(self):
        import json
        data = {"value": float("inf")}
        with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def test_json_dumps_rejects_negative_infinity(self):
        import json
        data = {"value": float("-inf")}
        with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TestCandidatePairDataclass:
    """Tests for CandidatePair dataclass (defined in Task 2 but contract verified here)."""

    def test_candidate_pair_imports_cleanly(self):
        from ml.portfolio_optimizer import CandidatePair
        assert CandidatePair is not None

    def test_candidate_pair_is_frozen_dataclass(self):
        from ml.portfolio_optimizer import CandidatePair
        pair = CandidatePair(
            attempt_id="ATT-000001",
            payment_id="PAY-000001",
            row_index=0,
            arm="RETRY_NOW",
            gross_incremental_value_inr=100.0,
            action_cost_inr=10.0,
            action_cost_paise=1000,
            net_incremental_value_inr=90.0,
            p_hat_arm=0.8,
            p_hat_control=0.5,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            pair.arm = "RETRY_LATER"

    def test_candidate_pair_required_fields(self):
        from ml.portfolio_optimizer import CandidatePair
        fields = {f.name for f in dataclasses.fields(CandidatePair)}
        expected = {
            "attempt_id", "payment_id", "row_index", "arm",
            "gross_incremental_value_inr", "action_cost_inr", "action_cost_paise",
            "net_incremental_value_inr", "p_hat_arm", "p_hat_control",
        }
        assert fields == expected