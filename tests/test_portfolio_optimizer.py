"""Tests for Day 7 portfolio optimizer interfaces and contracts (Task 1 & 2)."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from ml.portfolio_optimizer import (
    OptimizerConfig,
    OPTIMIZER_FORBIDDEN_COLUMNS,
    PortfolioOptimizationError,
    PortfolioProblemTooLargeError,
    CandidatePair,
    build_candidate_universe,
    _validate_candidate_frame,
    _pre_screen_policy,
    TREATED_ARMS,
)
from ml.features import FORBIDDEN_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from recovery.policy import load_policy_config
from recovery.scoring import RETRY_INTERVENTION_COST_INR, UNKNOWN_CATEGORY_RISK_FRACTION


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
        # but not identifiers, timestamp, or label columns (allowed in optimizer input for audit tracing)
        from ml.features import IDENTIFIER_COLUMNS, TIME_COLUMN, LABEL_COLUMNS
        day1_post_decision_forbidden = {"recovery_time_hours", "recovery_action", "action_outcome", "recovered_amount_inr"}
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


class TestCandidateConstruction:
    """Tests for Task 2: leakage-safe candidate construction."""

    def setup_method(self):
        """Create a minimal valid candidate frame for testing."""
        self.policy = load_policy_config("config/business_rules.yaml")
        # We'll create a minimal frame with required columns
        self.base_frame = pd.DataFrame({
            "attempt_id": ["ATT-000001", "ATT-000002", "ATT-000003"],
            "payment_id": ["PAY-000001", "PAY-000002", "PAY-000003"],
            "customer_id": ["CUS-000001", "CUS-000002", "CUS-000003"],
            "event_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "amount_inr": [1000.0, 2000.0, 3000.0],
            "failure_category": ["temporary_decline", "network_timeout", "unknown"],
            "attempt_number": [1, 2, 1],
            "customer_tenure_days": [30, 60, 90],
            "successful_payment_count": [5, 10, 15],
            "failed_payment_count": [1, 2, 0],
            "historical_recovery_count": [0, 1, 2],
            "customer_opted_out": [0, 0, 0],
            "fraud_risk": [0, 0, 0],
            "payment_method": ["upi", "card", "upi"],
            "failure_code": ["T001", "T002", "T003"],
            "issuer_response": ["00", "01", "02"],
            "device_type": ["android", "ios", "web"],
            "country": ["IN", "IN", "IN"],
            "recovered": [0, 1, 0],  # Dummy column required by build_feature_matrix
        })

    def _make_mock_bundle(self, probs_per_arm):
        """Create a mock ActionModelBundle that returns fixed probabilities."""
        from ml.action_model import ActionModelBundle, ARM_ORDER
        
        class MockModel:
            def __init__(self, prob):
                self.prob = prob
            def predict_proba(self, X):
                n = len(X)
                # Return shape (n, 2) with column 1 = probability of class 1
                return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])
        
        models = {arm: MockModel(probs_per_arm.get(arm, 0.5)) for arm in ARM_ORDER}
        return ActionModelBundle(models=models, arms=ARM_ORDER, metadata={})

    def test_forbidden_column_rejection(self):
        """Input frame with simulated_recovered raises ValueError."""
        frame = self.base_frame.copy()
        frame["simulated_recovered"] = [1, 0, 1]
        
        # Create a mock bundle
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.6, "RETRY_LATER": 0.5, "REQUEST_UPDATE": 0.4, "HUMAN_REVIEW": 0.7})
        
        with pytest.raises(ValueError, match="simulated_recovered"):
            _validate_candidate_frame(frame)

    def test_missing_required_context(self):
        """Missing amount_inr or failure_category raises ValueError."""
        frame = self.base_frame.drop(columns=["amount_inr"])
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.6, "RETRY_LATER": 0.5, "REQUEST_UPDATE": 0.4, "HUMAN_REVIEW": 0.7})
        
        with pytest.raises(ValueError, match="amount_inr"):
            _validate_candidate_frame(frame)
        
        frame2 = self.base_frame.drop(columns=["failure_category"])
        with pytest.raises(ValueError, match="failure_category"):
            _validate_candidate_frame(frame2)

    def test_malformed_monetary_float_rejection(self):
        """amount_inr = 10.12345 raises PortfolioOptimizationError without silent rounding."""
        frame = self.base_frame.copy()
        frame.loc[0, "amount_inr"] = 10.12345
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.6, "RETRY_LATER": 0.5, "REQUEST_UPDATE": 0.4, "HUMAN_REVIEW": 0.7})
        
        with pytest.raises(PortfolioOptimizationError):
            _validate_candidate_frame(frame)

    def test_invalid_prediction_nan_handling(self):
        """NaN probability places row in INVALID_PREDICTION bucket, 0 budget consumed, 0 candidate pairs."""
        # This test will be for build_candidate_universe
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": float('nan'), "RETRY_LATER": 0.5, "REQUEST_UPDATE": 0.4, "HUMAN_REVIEW": 0.7})
        
        candidates, entries, metadata = build_candidate_universe(self.base_frame, bundle, self.policy)
        
        # ATT-000001 should be in INVALID_PREDICTION due to NaN RETRY_NOW
        assert "ATT-000001" in entries
        assert entries["ATT-000001"].no_intervention_reason == "invalid_prediction"
        # Should not appear in candidates
        attempt_ids_in_candidates = {c.attempt_id for c in candidates}
        assert "ATT-000001" not in attempt_ids_in_candidates

    def test_invalid_prediction_out_of_bounds_handling(self):
        """Probability 1.2 or -0.1 places row in INVALID_PREDICTION without silent clipping."""
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 1.2, "RETRY_LATER": 0.5, "REQUEST_UPDATE": 0.4, "HUMAN_REVIEW": 0.7})
        
        candidates, entries, metadata = build_candidate_universe(self.base_frame, bundle, self.policy)
        
        assert "ATT-000001" in entries
        assert entries["ATT-000001"].no_intervention_reason == "invalid_prediction"
        attempt_ids_in_candidates = {c.attempt_id for c in candidates}
        assert "ATT-000001" not in attempt_ids_in_candidates
        
        # Test negative probability
        bundle2 = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": -0.1, "RETRY_LATER": 0.5, "REQUEST_UPDATE": 0.4, "HUMAN_REVIEW": 0.7})
        candidates2, entries2, metadata2 = build_candidate_universe(self.base_frame, bundle2, self.policy)
        assert "ATT-000001" in entries2
        assert entries2["ATT-000001"].no_intervention_reason == "invalid_prediction"

    def test_pre_allocation_context_only_stop_prescreen(self):
        """Rows matching R001-R004 placed in PRE_SCREEN_STOPPED with authorized_action == 'STOP'."""
        # Create frame with R001 (customer_opted_out), R002 (fraud_risk), R003 (hard_decline), R004 (attempt_number >= 4)
        frame = pd.DataFrame({
            "attempt_id": ["ATT-000001", "ATT-000002", "ATT-000003", "ATT-000004"],
            "payment_id": ["PAY-000001", "PAY-000002", "PAY-000003", "PAY-000004"],
            "customer_id": ["CUS-000001", "CUS-000002", "CUS-000003", "CUS-000004"],
            "event_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "amount_inr": [1000.0, 2000.0, 3000.0, 4000.0],
            "failure_category": ["temporary_decline", "temporary_decline", "hard_decline", "temporary_decline"],
            "attempt_number": [1, 2, 1, 4],  # ATT-000004 has attempt_number=4 (R004)
            "customer_tenure_days": [30, 60, 90, 120],
            "successful_payment_count": [5, 10, 15, 20],
            "failed_payment_count": [1, 2, 0, 3],
            "historical_recovery_count": [0, 1, 2, 1],
            "customer_opted_out": [1, 0, 0, 0],  # ATT-000001 opted out (R001)
            "fraud_risk": [0, 1, 0, 0],  # ATT-000002 fraud risk (R002)
            "payment_method": ["upi", "card", "upi", "upi"],
            "failure_code": ["T001", "T002", "T003", "T004"],
            "issuer_response": ["00", "01", "02", "03"],
            "device_type": ["android", "ios", "web", "android"],
            "country": ["IN", "IN", "IN", "IN"],
            "recovered": [0, 0, 0, 0],  # Dummy column required by build_feature_matrix
        })
        
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.8, "RETRY_LATER": 0.6, "REQUEST_UPDATE": 0.5, "HUMAN_REVIEW": 0.7})
        
        candidates, entries, metadata = build_candidate_universe(frame, bundle, self.policy)
        
        # All 4 rows should be pre-screened STOPPED
        assert "ATT-000001" in entries  # R001
        assert "ATT-000002" in entries  # R002
        assert "ATT-000003" in entries  # R003
        assert "ATT-000004" in entries  # R004
        
        for aid in ["ATT-000001", "ATT-000002", "ATT-000003", "ATT-000004"]:
            assert entries[aid].no_intervention_reason == "policy_pre_screen"
            assert entries[aid].authorized_action == "STOP"
            assert entries[aid].optimizer_recommendation == "NO_INTERVENTION"
        
        # No candidates should be created for these rows
        attempt_ids_in_candidates = {c.attempt_id for c in candidates}
        assert len(attempt_ids_in_candidates) == 0

    def test_pre_allocation_ignores_probability_rules(self):
        """Probability-dependent rules (R006-R008) do NOT pre-screen rows when probabilities are un-injected."""
        # Create frame that would match R006 (high value + low prob), R007 (temp decline + high prob), R008 (low prob)
        # But without probability injection, these should NOT fire during pre-screening
        frame = pd.DataFrame({
            "attempt_id": ["ATT-000001", "ATT-000002", "ATT-000003"],
            "payment_id": ["PAY-000001", "PAY-000002", "PAY-000003"],
            "customer_id": ["CUS-000001", "CUS-000002", "CUS-000003"],
            "event_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "amount_inr": [30000.0, 1000.0, 5000.0],  # ATT-000001 > 25000 (R006 condition)
            "failure_category": ["temporary_decline", "temporary_decline", "temporary_decline"],
            "attempt_number": [2, 2, 2],
            "customer_tenure_days": [30, 60, 90],
            "successful_payment_count": [5, 10, 15],
            "failed_payment_count": [1, 2, 0],
            "historical_recovery_count": [0, 1, 2],
            "customer_opted_out": [0, 0, 0],
            "fraud_risk": [0, 0, 0],
            "payment_method": ["upi", "card", "upi"],
            "failure_code": ["T001", "T002", "T003"],
            "issuer_response": ["00", "01", "02"],
            "device_type": ["android", "ios", "web"],
            "country": ["IN", "IN", "IN"],
            "recovered": [0, 0, 0],  # Dummy column required by build_feature_matrix
        })
        
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.8, "RETRY_LATER": 0.6, "REQUEST_UPDATE": 0.5, "HUMAN_REVIEW": 0.7})
        
        candidates, entries, metadata = build_candidate_universe(frame, bundle, self.policy)
        
        # None should be pre-screened because R006, R007, R008 need recovery_probability
        # which is NOT injected during pre-screening
        pre_screened = [aid for aid, e in entries.items() if e.no_intervention_reason == "policy_pre_screen"]
        assert len(pre_screened) == 0, f"Expected no pre-screened rows, got {pre_screened}"
        
        # All rows should be eligible for candidate generation
        assert len(candidates) > 0

    def test_gross_vs_net_value_calculation(self):
        """Verify gross_incremental_value_inr, action_cost_inr, action_cost_paise, net_incremental_value_inr."""
        # Use a simple frame with one row, known probabilities
        frame = pd.DataFrame({
            "attempt_id": ["ATT-000001"],
            "payment_id": ["PAY-000001"],
            "customer_id": ["CUS-000001"],
            "event_timestamp": pd.to_datetime(["2026-01-01"]),
            "amount_inr": [1000.0],
            "failure_category": ["temporary_decline"],
            "attempt_number": [1],
            "customer_tenure_days": [30],
            "successful_payment_count": [5],
            "failed_payment_count": [1],
            "historical_recovery_count": [0],
            "customer_opted_out": [0],
            "fraud_risk": [0],
            "payment_method": ["upi"],
            "failure_code": ["T001"],
            "issuer_response": ["00"],
            "device_type": ["android"],
            "country": ["IN"],
            "recovered": [0],  # Dummy column required by build_feature_matrix
        })
        
        # CONTROL=0.3, RETRY_NOW=0.8 -> incremental = 0.5 * 1000 = 500
        # No risk penalty (not unknown category)
        # Gross = 500, Cost = 10, Cost_paise = 1000, Net = 490
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.8, "RETRY_LATER": 0.6, "REQUEST_UPDATE": 0.5, "HUMAN_REVIEW": 0.7})
        
        candidates, entries, metadata = build_candidate_universe(frame, bundle, self.policy)
        
        # Should have candidate for RETRY_NOW
        assert len(candidates) >= 1
        retry_now_candidates = [c for c in candidates if c.arm == "RETRY_NOW"]
        assert len(retry_now_candidates) == 1
        
        c = retry_now_candidates[0]
        # Gross = (0.8 - 0.3) * 1000 - 0 = 500
        assert abs(c.gross_incremental_value_inr - 500.0) < 0.001
        # Action cost = 10.0
        assert abs(c.action_cost_inr - 10.0) < 0.001
        # Action cost paise = 1000
        assert c.action_cost_paise == 1000
        # Net = 500 - 10 = 490
        assert abs(c.net_incremental_value_inr - 490.0) < 0.001
        assert c.p_hat_arm == 0.8
        assert c.p_hat_control == 0.3

    def test_positive_net_value_gate(self):
        """Candidates with positive gross value but negative net value are excluded."""
        # amount_inr = 10, CONTROL=0.5, RETRY_NOW=0.55 -> gross = 0.05 * 10 = 0.5
        # cost = 10 -> net = -9.5 (negative, should be excluded)
        frame = pd.DataFrame({
            "attempt_id": ["ATT-000001"],
            "payment_id": ["PAY-000001"],
            "customer_id": ["CUS-000001"],
            "event_timestamp": pd.to_datetime(["2026-01-01"]),
            "amount_inr": [10.0],
            "failure_category": ["temporary_decline"],
            "attempt_number": [1],
            "customer_tenure_days": [30],
            "successful_payment_count": [5],
            "failed_payment_count": [1],
            "historical_recovery_count": [0],
            "customer_opted_out": [0],
            "fraud_risk": [0],
            "payment_method": ["upi"],
            "failure_code": ["T001"],
            "issuer_response": ["00"],
            "device_type": ["android"],
            "country": ["IN"],
            "recovered": [0],  # Dummy column required by build_feature_matrix
        })
        
        bundle = self._make_mock_bundle({"CONTROL": 0.5, "RETRY_NOW": 0.55, "RETRY_LATER": 0.52, "REQUEST_UPDATE": 0.51, "HUMAN_REVIEW": 0.53})
        
        candidates, entries, metadata = build_candidate_universe(frame, bundle, self.policy)
        
        # No candidates should be created because all net values <= 0
        assert len(candidates) == 0
        # Row should be in entries with no_intervention_reason = "non_positive_value"
        assert "ATT-000001" in entries
        assert entries["ATT-000001"].no_intervention_reason == "non_positive_value"

    def test_risk_penalty_calculation(self):
        """Risk penalty applied for unknown category."""
        frame = pd.DataFrame({
            "attempt_id": ["ATT-000001"],
            "payment_id": ["PAY-000001"],
            "customer_id": ["CUS-000001"],
            "event_timestamp": pd.to_datetime(["2026-01-01"]),
            "amount_inr": [1000.0],
            "failure_category": ["unknown"],  # Unknown category -> 5% risk penalty
            "attempt_number": [1],
            "customer_tenure_days": [30],
            "successful_payment_count": [5],
            "failed_payment_count": [1],
            "historical_recovery_count": [0],
            "customer_opted_out": [0],
            "fraud_risk": [0],
            "payment_method": ["upi"],
            "failure_code": ["T001"],
            "issuer_response": ["00"],
            "device_type": ["android"],
            "country": ["IN"],
            "recovered": [0],  # Dummy column required by build_feature_matrix
        })
        
        # CONTROL=0.3, RETRY_NOW=0.8 -> incremental = 0.5 * 1000 = 500
        # Risk penalty = 0.05 * 1000 = 50
        # Gross = 500 - 50 = 450
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.8, "RETRY_LATER": 0.6, "REQUEST_UPDATE": 0.5, "HUMAN_REVIEW": 0.7})
        
        candidates, entries, metadata = build_candidate_universe(frame, bundle, self.policy)
        
        retry_now_candidates = [c for c in candidates if c.arm == "RETRY_NOW"]
        assert len(retry_now_candidates) == 1
        
        c = retry_now_candidates[0]
        # Gross = (0.8 - 0.3) * 1000 - 50 = 450
        assert abs(c.gross_incremental_value_inr - 450.0) < 0.001
        assert abs(c.net_incremental_value_inr - 440.0) < 0.001  # 450 - 10

    def test_all_treated_arms_evaluated(self):
        """All 4 TREATED_ARMS are evaluated for each eligible row."""
        frame = self.base_frame.copy()
        frame["recovered"] = [0, 0, 0]  # Dummy column required by build_feature_matrix
        bundle = self._make_mock_bundle({"CONTROL": 0.2, "RETRY_NOW": 0.8, "RETRY_LATER": 0.7, "REQUEST_UPDATE": 0.6, "HUMAN_REVIEW": 0.9})
        
        candidates, entries, metadata = build_candidate_universe(frame, bundle, self.policy)
        
        # 3 rows * 4 arms = 12 candidates (all have positive net value with these probs)
        arms_in_candidates = {c.arm for c in candidates}
        assert arms_in_candidates == set(TREATED_ARMS)
        
        # Each row should have 4 candidates
        for aid in ["ATT-000001", "ATT-000002", "ATT-000003"]:
            row_candidates = [c for c in candidates if c.attempt_id == aid]
            assert len(row_candidates) == 4

    def test_deterministic_candidate_ordering(self):
        """Identical input produces identical candidate ordering."""
        frame = self.base_frame.copy()
        frame["recovered"] = [0, 0, 0]  # Dummy column required by build_feature_matrix
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.8, "RETRY_LATER": 0.6, "REQUEST_UPDATE": 0.5, "HUMAN_REVIEW": 0.7})
        
        candidates1, _, _ = build_candidate_universe(frame, bundle, self.policy)
        candidates2, _, _ = build_candidate_universe(frame, bundle, self.policy)
        
        assert len(candidates1) == len(candidates2)
        for c1, c2 in zip(candidates1, candidates2):
            assert c1.attempt_id == c2.attempt_id
            assert c1.arm == c2.arm
            assert c1.net_incremental_value_inr == c2.net_incremental_value_inr

    def test_valid_monetary_float_passes(self):
        """Valid 2-decimal floats pass validation and convert to paise correctly."""
        frame = self.base_frame.copy()
        frame["recovered"] = [0, 0, 0]  # Dummy column required by build_feature_matrix
        frame.loc[0, "amount_inr"] = 1000.50  # Valid 2-decimal
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": 0.8, "RETRY_LATER": 0.6, "REQUEST_UPDATE": 0.5, "HUMAN_REVIEW": 0.7})
        
        # Should not raise
        _validate_candidate_frame(frame)
        
        candidates, entries, metadata = build_candidate_universe(frame, bundle, self.policy)
        assert len(candidates) > 0

    def test_inf_prediction_handling(self):
        """+Inf and -Inf probabilities are treated as invalid predictions."""
        frame = self.base_frame.copy()
        frame["recovered"] = [0, 0, 0]  # Dummy column required by build_feature_matrix
        # +Inf
        bundle = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": float('inf'), "RETRY_LATER": 0.6, "REQUEST_UPDATE": 0.5, "HUMAN_REVIEW": 0.7})
        candidates, entries, metadata = build_candidate_universe(self.base_frame, bundle, self.policy)
        assert "ATT-000001" in entries
        assert entries["ATT-000001"].no_intervention_reason == "invalid_prediction"
        
        # -Inf
        bundle2 = self._make_mock_bundle({"CONTROL": 0.3, "RETRY_NOW": float('-inf'), "RETRY_LATER": 0.6, "REQUEST_UPDATE": 0.5, "HUMAN_REVIEW": 0.7})
        candidates2, entries2, metadata2 = build_candidate_universe(self.base_frame, bundle2, self.policy)
        assert "ATT-000001" in entries2
        assert entries2["ATT-000001"].no_intervention_reason == "invalid_prediction"


class TestCandidateRanking:
    """Tests for Task 3: deterministic global-pair ranking."""

    def _make_candidates(self, candidate_data):
        """Helper to create CandidatePair objects for testing."""
        from ml.portfolio_optimizer import CandidatePair
        candidates = []
        for i, data in enumerate(candidate_data):
            candidates.append(CandidatePair(
                attempt_id=data.get("attempt_id", f"ATT-{i:06d}"),
                payment_id=data.get("payment_id", f"PAY-{i:06d}"),
                row_index=data.get("row_index", i),
                arm=data["arm"],
                gross_incremental_value_inr=data.get("gross_incremental_value_inr", 100.0),
                action_cost_inr=data.get("action_cost_inr", 10.0),
                action_cost_paise=data.get("action_cost_paise", 1000),
                net_incremental_value_inr=data["net_incremental_value_inr"],
                p_hat_arm=data.get("p_hat_arm", 0.5),
                p_hat_control=data.get("p_hat_control", 0.3),
            ))
        return candidates

    def test_primary_sort_net_value_descending(self):
        """Higher net_incremental_value_inr ranked earlier."""
        from ml.portfolio_optimizer import sort_key_candidate_pair, rank_candidate_pairs
        
        candidates = self._make_candidates([
            {"arm": "RETRY_NOW", "net_incremental_value_inr": 100.0},
            {"arm": "RETRY_LATER", "net_incremental_value_inr": 300.0},
            {"arm": "REQUEST_UPDATE", "net_incremental_value_inr": 200.0},
        ])
        
        ranked = rank_candidate_pairs(candidates)
        
        assert ranked[0].net_incremental_value_inr == 300.0
        assert ranked[1].net_incremental_value_inr == 200.0
        assert ranked[2].net_incremental_value_inr == 100.0

    def test_secondary_sort_attempt_id_ascending(self):
        """Equal net values broken deterministically by attempt_id ascending."""
        from ml.portfolio_optimizer import rank_candidate_pairs
        
        candidates = self._make_candidates([
            {"attempt_id": "ATT-000003", "arm": "RETRY_NOW", "net_incremental_value_inr": 200.0},
            {"attempt_id": "ATT-000001", "arm": "RETRY_LATER", "net_incremental_value_inr": 200.0},
            {"attempt_id": "ATT-000002", "arm": "REQUEST_UPDATE", "net_incremental_value_inr": 200.0},
        ])
        
        ranked = rank_candidate_pairs(candidates)
        
        assert ranked[0].attempt_id == "ATT-000001"
        assert ranked[1].attempt_id == "ATT-000002"
        assert ranked[2].attempt_id == "ATT-000003"

    def test_tertiary_sort_arm_order_ascending(self):
        """Same row with equal net values across multiple arms broken by ARM_ORDER index."""
        from ml.portfolio_optimizer import rank_candidate_pairs
        from ml.action_model import ARM_ORDER
        
        # Same attempt_id, same net value, different arms
        candidates = self._make_candidates([
            {"attempt_id": "ATT-000001", "arm": "HUMAN_REVIEW", "net_incremental_value_inr": 100.0},
            {"attempt_id": "ATT-000001", "arm": "RETRY_NOW", "net_incremental_value_inr": 100.0},
            {"attempt_id": "ATT-000001", "arm": "REQUEST_UPDATE", "net_incremental_value_inr": 100.0},
            {"attempt_id": "ATT-000001", "arm": "RETRY_LATER", "net_incremental_value_inr": 100.0},
        ])
        
        ranked = rank_candidate_pairs(candidates)
        
        # ARM_ORDER: RETRY_NOW(1), RETRY_LATER(2), REQUEST_UPDATE(3), HUMAN_REVIEW(4)
        assert ranked[0].arm == "RETRY_NOW"
        assert ranked[1].arm == "RETRY_LATER"
        assert ranked[2].arm == "REQUEST_UPDATE"
        assert ranked[3].arm == "HUMAN_REVIEW"

    def test_ranking_permutation_invariance(self):
        """Shuffling input candidate list produces byte-identical sorted output."""
        from ml.portfolio_optimizer import rank_candidate_pairs
        import random
        
        base_candidates = self._make_candidates([
            {"attempt_id": "ATT-000001", "arm": "RETRY_NOW", "net_incremental_value_inr": 300.0},
            {"attempt_id": "ATT-000002", "arm": "RETRY_LATER", "net_incremental_value_inr": 200.0},
            {"attempt_id": "ATT-000001", "arm": "HUMAN_REVIEW", "net_incremental_value_inr": 100.0},
            {"attempt_id": "ATT-000003", "arm": "REQUEST_UPDATE", "net_incremental_value_inr": 400.0},
        ])
        
        # Rank multiple times with shuffled input
        results = []
        for _ in range(10):
            shuffled = base_candidates.copy()
            random.shuffle(shuffled)
            ranked = rank_candidate_pairs(shuffled)
            results.append(tuple((c.attempt_id, c.arm, c.net_incremental_value_inr) for c in ranked))
        
        # All results should be identical
        assert all(r == results[0] for r in results)
        
        # Verify correct order: highest net value first
        assert results[0][0] == ("ATT-000003", "REQUEST_UPDATE", 400.0)
        assert results[0][1] == ("ATT-000001", "RETRY_NOW", 300.0)
        assert results[0][2] == ("ATT-000002", "RETRY_LATER", 200.0)
        assert results[0][3] == ("ATT-000001", "HUMAN_REVIEW", 100.0)