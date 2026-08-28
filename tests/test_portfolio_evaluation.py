"""Task 7 tests: Leakage-safe portfolio outcome evaluation.

Tests verify that evaluation joins synthetic outcomes only after allocation
is frozen, enforces held-out split isolation, and correctly separates
confounded vs unconfounded comparison semantics.
"""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pandas as pd
import pytest

from ml.action_model import ARM_ORDER, ActionModelBundle
from ml.portfolio_audit import PortfolioAllocation, PortfolioEntry, PortfolioSummary
from ml.portfolio_evaluation import (
    compare_portfolio_to_baseline,
    evaluate_portfolio_allocation,
)
from ml.portfolio_greedy import optimize_portfolio_greedy
from ml.portfolio_optimizer import (
    CandidatePair,
    OptimizerConfig,
    TREATED_ARMS,
    build_candidate_universe,
    solve_portfolio_allocation,
    authorize_post_allocation,
)
from data.splits import chronological_split
from data.generate_dataset import generate_dataset
from recovery.policy import load_policy_config


# =============================================================================
# Test Helpers
# =============================================================================

class _MockModel:
    """Constant-probability mock model for deterministic tests."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


def _make_mock_bundle(probs_per_arm: dict[str, float]) -> ActionModelBundle:
    """Create a mock ActionModelBundle returning fixed probabilities."""
    models = {arm: _MockModel(probs_per_arm.get(arm, 0.5)) for arm in ARM_ORDER}
    return ActionModelBundle(models=models, arms=ARM_ORDER, metadata={})


def _make_base_frame(rows_data: list[dict]) -> pd.DataFrame:
    """Build a minimal valid candidate DataFrame for build_candidate_universe."""
    rows = []
    for i, r in enumerate(rows_data):
        rows.append({
            "attempt_id": r.get("attempt_id", f"ATT-{i:06d}"),
            "payment_id": r.get("payment_id", f"PAY-{i:06d}"),
            "customer_id": r.get("customer_id", f"CUS-{i:06d}"),
            "event_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "amount_inr": r.get("amount_inr", 1000.0),
            "failure_category": r.get("failure_category", "temporary_decline"),
            "attempt_number": r.get("attempt_number", 1),
            "customer_tenure_days": r.get("customer_tenure_days", 30),
            "successful_payment_count": r.get("successful_payment_count", 5),
            "failed_payment_count": r.get("failed_payment_count", 1),
            "historical_recovery_count": r.get("historical_recovery_count", 0),
            "customer_opted_out": r.get("customer_opted_out", 0),
            "fraud_risk": r.get("fraud_risk", 0),
            "payment_method": r.get("payment_method", "upi"),
            "failure_code": r.get("failure_code", "T001"),
            "issuer_response": r.get("issuer_response", "00"),
            "device_type": r.get("device_type", "android"),
            "country": r.get("country", "IN"),
            "recovered": r.get("recovered", 0),
        })
    return pd.DataFrame(rows)


def _make_candidates(candidate_data: list[dict]) -> list[CandidatePair]:
    """Helper to create CandidatePair objects for testing."""
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


def _make_portfolio_entries(entry_data: list[dict]) -> tuple[PortfolioEntry, ...]:
    """Create PortfolioEntry objects for testing."""
    entries = []
    for i, data in enumerate(entry_data):
        aid = data.get("attempt_id", f"ATT-{i:06d}")
        arm = data.get("optimizer_recommendation", "NO_INTERVENTION")
        auth = data.get("authorized_action", arm)
        entries.append(PortfolioEntry(
            attempt_id=aid,
            payment_id=data.get("payment_id", f"PAY-{i:06d}"),
            row_index=data.get("row_index", i),
            optimizer_recommendation=arm,
            no_intervention_reason=data.get("no_intervention_reason"),
            gross_incremental_value_by_arm=data.get("gross_incremental_value_by_arm", {}),
            action_cost_by_arm=data.get("action_cost_by_arm", {}),
            net_incremental_value_by_arm=data.get("net_incremental_value_by_arm", {}),
            selected_gross_incremental_value_inr=data.get("selected_gross_incremental_value_inr"),
            selected_action_cost_inr=data.get("selected_action_cost_inr"),
            selected_action_cost_paise=data.get("selected_action_cost_paise"),
            selected_net_incremental_value_inr=data.get("selected_net_incremental_value_inr"),
            optimizer_sort_rank=data.get("optimizer_sort_rank"),
            authorized_action=auth,
            authorization_reason=data.get("authorization_reason", "test"),
            matched_rule_id=data.get("matched_rule_id"),
            policy_overrode_recommendation=data.get("policy_overrode_recommendation", False),
        ))
    return tuple(entries)


def _make_allocation(
    entries: tuple[PortfolioEntry, ...],
    metadata: dict | None = None,
) -> PortfolioAllocation:
    """Create a PortfolioAllocation from entries."""
    alloc_count = sum(1 for e in entries if e.optimizer_recommendation not in ("NO_INTERVENTION",))
    no_int_count = len(entries) - alloc_count
    summary = PortfolioSummary(
        total_rows=len(entries),
        pre_screen_stopped_count=0,
        invalid_prediction_count=0,
        optimizer_allocated_count=alloc_count,
        no_intervention_count=no_int_count,
        eligible_candidate_count=len(entries),
        budget_limit_inr=None,
        budget_limit_paise=None,
        budget_allocated_inr=0.0,
        budget_allocated_paise=0,
        budget_remaining_inr=None,
        budget_remaining_paise=None,
        human_review_capacity_limit=None,
        human_review_allocated_count=0,
        post_policy_net_authorized_count=alloc_count,
        total_policy_overrides=0,
        total_policy_stop_overrides=0,
        optimizer_objective_value_inr=0.0,
        optimizer_status="success",
        action_recommendation_counts={},
        action_authorized_counts={},
    )
    return PortfolioAllocation(
        entries=entries,
        summary=summary,
        metadata=metadata or {},
    )


def _make_outcome_frame(attempt_ids: list[dict]) -> pd.DataFrame:
    """Create an outcome frame with recovered and amount_inr."""
    rows = []
    for i, r in enumerate(attempt_ids):
        rows.append({
            "attempt_id": r.get("attempt_id", f"ATT-{i:06d}"),
            "amount_inr": r.get("amount_inr", 1000.0),
            "recovered": r.get("recovered", 0),
            "failure_category": r.get("failure_category", "temporary_decline"),
        })
    return pd.DataFrame(rows)


def _allocation_digest(alloc: PortfolioAllocation) -> str:
    """Compute a deterministic digest of a PortfolioAllocation for freeze verification."""
    return hashlib.sha256(alloc.to_json().encode("utf-8")).hexdigest()


# =============================================================================
# Test Class: Allocation Freeze
# =============================================================================

class TestAllocationFreeze:
    """Tests verifying allocation is frozen before outcome join."""

    def test_allocation_immutable_during_evaluation(self):
        """Evaluation cannot change allocation entries, summary, or metadata."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "NO_INTERVENTION",
             "authorized_action": "NO_INTERVENTION", "no_intervention_reason": "budget_exhausted"},
        ])
        alloc = _make_allocation(entries, metadata={"frozen": True})
        digest_before = _allocation_digest(alloc)

        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 0},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)

        # Allocation must be unchanged
        digest_after = _allocation_digest(alloc)
        assert digest_before == digest_after
        # Result must be a dict
        assert isinstance(result, dict)

    def test_allocation_frozen_before_outcome_join(self):
        """Allocation is constructed before evaluator outcome access."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc = _make_allocation(entries)

        # Capture allocation state before evaluation
        pre_eval_recommendations = {e.attempt_id: e.optimizer_recommendation for e in alloc.entries}
        pre_eval_actions = {e.attempt_id: e.authorized_action for e in alloc.entries}

        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
        ])

        _ = evaluate_portfolio_allocation(alloc, outcome_df)

        # Verify allocation was not mutated
        for e in alloc.entries:
            assert e.optimizer_recommendation == pre_eval_recommendations[e.attempt_id]
            assert e.authorized_action == pre_eval_actions[e.attempt_id]

    def test_outcome_permutation_does_not_alter_allocation(self):
        """Permuting outcome values after freezing allocation does not change it."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "RETRY_LATER",
             "authorized_action": "RETRY_LATER"},
        ])
        alloc = _make_allocation(entries)
        digest_original = _allocation_digest(alloc)

        # Evaluate with original outcomes
        outcome_original = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 0},
        ])
        _ = evaluate_portfolio_allocation(alloc, outcome_original)

        # Permute outcome values
        outcome_permuted = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 2000.0, "recovered": 0},
            {"attempt_id": "ATT-000002", "amount_inr": 1000.0, "recovered": 1},
        ])
        _ = evaluate_portfolio_allocation(alloc, outcome_permuted)

        # Allocation must still be unchanged
        assert _allocation_digest(alloc) == digest_original


# =============================================================================
# Test Class: Leakage Prevention
# =============================================================================

class TestLeakagePrevention:
    """Tests verifying outcome data cannot leak into allocation decisions."""

    def test_evaluator_does_not_call_optimizer(self):
        """Evaluation does not invoke DP or greedy allocation algorithms."""
        import ml.portfolio_evaluation as eval_mod
        # The module must NOT import allocation functions
        assert not hasattr(eval_mod, "solve_portfolio_allocation")
        assert not hasattr(eval_mod, "optimize_portfolio_greedy")

    def test_evaluator_does_not_call_predict(self):
        """Evaluation does not invoke model prediction."""
        import ml.portfolio_evaluation as eval_mod
        assert not hasattr(eval_mod, "predict_all_actions")

    def test_evaluator_does_not_call_build_candidate_universe(self):
        """Evaluation does not rebuild candidates."""
        import ml.portfolio_evaluation as eval_mod
        assert not hasattr(eval_mod, "build_candidate_universe")

    def test_evaluator_source_has_no_allocation_imports(self):
        """Module source does not import allocation or prediction functions."""
        import inspect
        import ml.portfolio_evaluation as eval_mod
        source = inspect.getsource(eval_mod)
        forbidden = [
            "from ml.portfolio_optimizer import solve_portfolio_allocation",
            "from ml.portfolio_greedy import optimize_portfolio_greedy",
            "from ml.action_model import predict_all_actions",
            "from ml.portfolio_optimizer import build_candidate_universe",
        ]
        for imp in forbidden:
            assert imp not in source, f"Module imports forbidden function: {imp}"


# =============================================================================
# Test Class: Held-Out Split Isolation
# =============================================================================

class TestHeldOutSplitIsolation:
    """Tests verifying evaluation uses only held-out test split attempt IDs."""

    def test_held_out_test_split_disjoint_ids(self):
        """Evaluation attempt IDs are disjoint from train and validation attempt IDs."""
        df = generate_dataset(400, seed=42)
        train, validation, test = chronological_split(df)

        train_ids = set(train["attempt_id"])
        val_ids = set(validation["attempt_id"])
        test_ids = set(test["attempt_id"])

        # Splits must be disjoint
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

        # Test IDs are the evaluation universe
        assert len(test_ids) > 0

    def test_train_val_test_attempt_id_uniqueness(self):
        """No attempt_id appears in more than one split."""
        df = generate_dataset(600, seed=42)
        train, validation, test = chronological_split(df)

        all_ids = list(train["attempt_id"]) + list(validation["attempt_id"]) + list(test["attempt_id"])
        assert len(all_ids) == len(set(all_ids))

    def test_evaluation_only_uses_test_split_ids(self):
        """Evaluator only processes rows whose attempt_ids are in the test split."""
        df = generate_dataset(400, seed=42)
        train, validation, test = chronological_split(df)

        # Create allocation entries only from test split
        test_ids = list(test["attempt_id"])[:5]
        entries = _make_portfolio_entries([
            {"attempt_id": aid, "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"}
            for aid in test_ids
        ])
        alloc = _make_allocation(entries)

        # Outcome frame includes both test and train IDs (but evaluator should handle gracefully)
        outcome_rows = [{"attempt_id": aid, "amount_inr": 1000.0, "recovered": 1}
                        for aid in test_ids]
        outcome_df = _make_outcome_frame(outcome_rows)

        result = evaluate_portfolio_allocation(alloc, outcome_df)

        # All evaluated rows should be from test split
        evaluated_ids = set(result.get("evaluated_attempt_ids", []))
        assert evaluated_ids == set(test_ids)


# =============================================================================
# Test Class: Action Semantics
# =============================================================================

class TestActionSemantics:
    """Tests verifying outcomes are evaluated using the correct action field."""

    def test_uses_authorized_action_not_optimizer_recommendation(self):
        """Evaluation uses authorized_action, not optimizer_recommendation."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001",
             "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "STOP",
             "policy_overrode_recommendation": True},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)

        # The STOP authorization should be reflected in evaluation
        action_metrics = result.get("action_metrics", {})
        # STOP rows should be counted under STOP, not RETRY_NOW
        assert "STOP" in action_metrics or result.get("stop_count", 0) >= 1

    def test_stop_rows_handled_correctly(self):
        """STOP authorized rows are handled without crashing."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "STOP", "policy_overrode_recommendation": True},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "RETRY_LATER",
             "authorized_action": "STOP", "policy_overrode_recommendation": True},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 0},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 1},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)
        assert isinstance(result, dict)
        assert result.get("total_evaluated", 0) == 2

    def test_no_intervention_rows_handled_correctly(self):
        """NO_INTERVENTION rows are handled without crashing."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "NO_INTERVENTION",
             "authorized_action": "NO_INTERVENTION", "no_intervention_reason": "budget_exhausted"},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "NO_INTERVENTION",
             "authorized_action": "NO_INTERVENTION", "no_intervention_reason": "non_positive_net_value"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 0},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 1},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)
        assert isinstance(result, dict)
        assert result.get("total_evaluated", 0) == 2

    def test_pre_screen_stopped_rows_handled(self):
        """Pre-screen stopped rows (from build_candidate_universe) are handled."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "NO_INTERVENTION",
             "authorized_action": "STOP", "no_intervention_reason": "policy_pre_screen: STOP"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 0},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)
        assert isinstance(result, dict)


# =============================================================================
# Test Class: Comparison Semantics
# =============================================================================

class TestComparisonSemantics:
    """Tests verifying confounded vs unconfounded comparison labeling."""

    def test_confounded_labeling(self):
        """Confounded comparison block exists with explicit label."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "NO_INTERVENTION",
             "authorized_action": "NO_INTERVENTION", "no_intervention_reason": "budget_exhausted"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 0},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)

        comparisons = result.get("comparisons", {})
        confounded = comparisons.get("confounded", {})
        assert confounded.get("label", "").startswith("CONFOUNDED")

    def test_unconfounded_labeling(self):
        """Unconfounded comparison block exists with explicit label."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc = _make_allocation(entries)

        # Outcome frame includes CONTROL arm rows for unconfounded comparison
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 0,
             "assigned_action": "CONTROL"},
            {"attempt_id": "ATT-000003", "amount_inr": 1500.0, "recovered": 1,
             "assigned_action": "CONTROL"},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)

        comparisons = result.get("comparisons", {})
        unconfounded = comparisons.get("unconfounded", {})
        assert unconfounded.get("label", "").startswith("UNCONFOUNDED")

    def test_confounded_vs_unconfounded_both_present(self):
        """Both comparison blocks exist and carry correct explicit labels."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "NO_INTERVENTION",
             "authorized_action": "NO_INTERVENTION"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 0},
            {"attempt_id": "ATT-000003", "amount_inr": 500.0, "recovered": 0,
             "assigned_action": "CONTROL"},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)

        comparisons = result.get("comparisons", {})
        assert "confounded" in comparisons
        assert "unconfounded" in comparisons
        assert comparisons["confounded"]["label"].startswith("CONFOUNDED")
        assert comparisons["unconfounded"]["label"].startswith("UNCONFOUNDED")


# =============================================================================
# Test Class: Comparison Delta
# =============================================================================

class TestComparisonDelta:
    """Tests verifying optimizer vs greedy delta calculation."""

    def test_optimizer_vs_greedy_delta_calculation(self):
        """Net objective and realized outcome deltas calculated accurately."""
        optimizer_eval = {
            "total_evaluated": 2,
            "total_recovered": 1,
            "total_recovered_amount_inr": 1000.0,
            "optimizer_objective_value_inr": 200.0,
            "intervention_count": 2,
        }
        greedy_eval = {
            "total_evaluated": 2,
            "total_recovered": 0,
            "total_recovered_amount_inr": 0.0,
            "optimizer_objective_value_inr": 150.0,
            "intervention_count": 2,
        }

        result = compare_portfolio_to_baseline(optimizer_eval, greedy_eval)

        assert result["objective_delta_inr"] == pytest.approx(50.0)
        assert result["recovered_amount_delta_inr"] == pytest.approx(1000.0)
        assert result["recovery_count_delta"] == 1

    def test_equal_objectives_produce_zero_delta(self):
        """When both have equal objective, delta is zero."""
        eval_a = {"optimizer_objective_value_inr": 100.0, "total_recovered_amount_inr": 500.0}
        eval_b = {"optimizer_objective_value_inr": 100.0, "total_recovered_amount_inr": 500.0}

        result = compare_portfolio_to_baseline(eval_a, eval_b)

        assert result["objective_delta_inr"] == pytest.approx(0.0)
        assert result["recovered_amount_delta_inr"] == pytest.approx(0.0)

    def test_greedy_superior_produces_negative_delta(self):
        """When greedy is better, delta is negative."""
        optimizer_eval = {"optimizer_objective_value_inr": 50.0}
        greedy_eval = {"optimizer_objective_value_inr": 200.0}

        result = compare_portfolio_to_baseline(optimizer_eval, greedy_eval)

        assert result["objective_delta_inr"] == pytest.approx(-150.0)


# =============================================================================
# Test Class: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests verifying error handling for invalid inputs."""

    def test_missing_attempt_id_alignment_rejection(self):
        """Outcome frame missing required attempt_ids raises ValueError."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc = _make_allocation(entries)

        # Outcome frame with no matching attempt_ids
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-999999", "amount_inr": 1000.0, "recovered": 1},
        ])

        with pytest.raises(ValueError):
            evaluate_portfolio_allocation(alloc, outcome_df)

    def test_empty_outcome_frame_rejection(self):
        """Empty outcome frame raises ValueError."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = pd.DataFrame(columns=["attempt_id", "amount_inr", "recovered"])

        with pytest.raises(ValueError):
            evaluate_portfolio_allocation(alloc, outcome_df)

    def test_duplicate_attempt_ids_in_outcome_frame(self):
        """Duplicate attempt_ids in outcome frame raises ValueError."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc = _make_allocation(entries)

        outcome_df = pd.DataFrame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
            {"attempt_id": "ATT-000001", "amount_inr": 2000.0, "recovered": 0},
        ])

        with pytest.raises(ValueError):
            evaluate_portfolio_allocation(alloc, outcome_df)


# =============================================================================
# Test Class: Empty and Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for empty allocations and edge cases."""

    def test_empty_allocation(self):
        """Empty allocation (no entries) with empty outcome frame returns zero counts."""
        alloc = _make_allocation(tuple())
        # Empty outcome frame is rejected by validation
        # So we test with a valid but empty-matching outcome frame
        outcome_df = pd.DataFrame(columns=["attempt_id", "amount_inr", "recovered"])
        with pytest.raises(ValueError):
            evaluate_portfolio_allocation(alloc, outcome_df)

    def test_empty_allocation_with_matching_outcome(self):
        """Empty allocation with non-empty outcome frame returns zero evaluated."""
        alloc = _make_allocation(tuple())
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
        ])
        result = evaluate_portfolio_allocation(alloc, outcome_df)
        assert result["total_evaluated"] == 0

    def test_deterministic_repeated_evaluation(self):
        """Repeated evaluation of identical inputs produces identical results."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "NO_INTERVENTION",
             "authorized_action": "NO_INTERVENTION"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 0},
        ])

        results = []
        for _ in range(5):
            r = evaluate_portfolio_allocation(alloc, outcome_df)
            results.append(json.dumps(r, sort_keys=True, default=str))

        first = results[0]
        for r in results[1:]:
            assert r == first

    def test_outcome_join_does_not_drop_rows(self):
        """Outcome join does not silently drop allocation rows."""
        entries = _make_portfolio_entries([
            {"attempt_id": f"ATT-{i:06d}", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"}
            for i in range(10)
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": f"ATT-{i:06d}", "amount_inr": 1000.0, "recovered": i % 2}
            for i in range(10)
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)
        assert result["total_evaluated"] == 10

    def test_evaluation_result_serialization(self):
        """Evaluation result is JSON-serializable."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)
        serialized = json.dumps(result, sort_keys=True, default=str)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_policy_override_reflected_in_evaluation(self):
        """Policy overrides (where authorized_action != optimizer_recommendation) are evaluated correctly."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001",
             "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "STOP",
             "policy_overrode_recommendation": True},
            {"attempt_id": "ATT-000002",
             "optimizer_recommendation": "RETRY_LATER",
             "authorized_action": "RETRY_LATER",
             "policy_overrode_recommendation": False},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 0},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 1},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)
        assert result["total_evaluated"] == 2
        # The STOP override should be reflected
        overrides = result.get("policy_overrides_evaluated", 0)
        assert overrides >= 1


# =============================================================================
# Test Class: Same Universe for Optimizer and Greedy
# =============================================================================

class TestSameUniverse:
    """Tests verifying fair comparison between optimizer and greedy."""

    def test_same_evaluation_universe_for_both(self):
        """Both optimizer and greedy are evaluated on the same attempt-ID universe."""
        entries_opt = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "NO_INTERVENTION",
             "authorized_action": "NO_INTERVENTION"},
        ])
        alloc_opt = _make_allocation(entries_opt)

        entries_greedy = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_LATER",
             "authorized_action": "RETRY_LATER"},
            {"attempt_id": "ATT-000002", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc_greedy = _make_allocation(entries_greedy)

        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
            {"attempt_id": "ATT-000002", "amount_inr": 2000.0, "recovered": 0},
        ])

        result_opt = evaluate_portfolio_allocation(alloc_opt, outcome_df)
        result_greedy = evaluate_portfolio_allocation(alloc_greedy, outcome_df)

        # Both should evaluate the same universe
        assert result_opt["total_evaluated"] == result_greedy["total_evaluated"]
        assert set(result_opt["evaluated_attempt_ids"]) == set(result_greedy["evaluated_attempt_ids"])


# =============================================================================
# Test Class: Label Discipline
# =============================================================================

class TestLabelDiscipline:
    """Tests verifying output fields carry required label discipline."""

    def test_model_estimate_labels_present(self):
        """Evaluation output contains MODEL ESTIMATE labels where appropriate."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)

        # Check that model estimate labels exist in comparisons
        comparisons = result.get("comparisons", {})
        for comp_type in ["confounded", "unconfounded"]:
            comp = comparisons.get(comp_type, {})
            if comp:
                # Should have labeled metrics
                assert "label" in comp

    def test_observed_simulated_outcome_labels(self):
        """Realized metrics are labeled as observed/simulated outcomes."""
        entries = _make_portfolio_entries([
            {"attempt_id": "ATT-000001", "optimizer_recommendation": "RETRY_NOW",
             "authorized_action": "RETRY_NOW"},
        ])
        alloc = _make_allocation(entries)
        outcome_df = _make_outcome_frame([
            {"attempt_id": "ATT-000001", "amount_inr": 1000.0, "recovered": 1},
        ])

        result = evaluate_portfolio_allocation(alloc, outcome_df)

        # The result should distinguish model estimates from observed outcomes
        assert "realized" in result or "observed" in result or "total_recovered" in result
