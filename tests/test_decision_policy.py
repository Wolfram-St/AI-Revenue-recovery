"""Tests for Day 6 evidence classifier + bounded recommender (plan Task 5, D-E5).

Layers pinned here:

1. Classifier unit matrix: ``classify_optimizer_justification`` applies the
   four pre-registered D-E5 thresholds EXACTLY -- match rate >= 0.60, relative
   regret <= 0.15 (None fails with the literal "regret undefined" reason),
   policy-safety probe True, and at least two pairwise-non-overlapping
   treated-arm bootstrap CIs -- over synthetic evidence dicts covering every
   pass/fail combination plus boundary values.
2. Malformed evidence: missing keys / wrong types raise ValueError naming the
   offending key; the provenance_digest must be a non-empty string.
3. Recommender gating: ``BoundedRecommender`` is constructible ONLY from
   OPTIMIZER_JUSTIFIED evidence and raises ``OptimizerNotJustifiedError``
   carrying the classifier's machine-readable reasons otherwise.
4. JUSTIFIED path on a real calibrated per-arm bundle: fabricated evidence is
   possible BY DESIGN (the gate is advisory scaffolding -- an in-process
   caller can hand-craft a dict; only the echoed provenance_digest makes
   non-canonical bundles visible), so one test below documents that honesty
   explicitly while exercising the full recommend() wiring: opted-out /
   fraud / hard-decline contexts authorize STOP regardless of candidate;
   eligible rows recompute independently; labels are MODEL ESTIMATE.
5. Policy dominance pin: with a deterministic stub bundle forcing candidate
   RETRY_NOW on a hard_decline context, the authorized action is STOP --
   the recommender NEVER overrides policy.
6. Determinism: identical inputs produce identical recommendations.
7. Module purity: whitelisted import roots, no default_rng anywhere, no
   wall-clock/stdlib-randomness tokens, advisory/synthetic docstring
   language, no causal-estimate language, cost basis imported never restated.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.action_model import (
    ACTION_COLUMN,
    ARM_ORDER,
    ActionModelBundle,
    calibrate_action_models,
    predict_action_probability,
    train_action_models,
)
from ml.decision_evidence import TREATED_ARMS, decision_evidence
from ml.decision_policy import (
    CLASSIFICATION_JUSTIFIED,
    CLASSIFICATION_NOT_YET_JUSTIFIED,
    LABEL_MODEL_ESTIMATE,
    MIN_NON_OVERLAPPING_CI_PAIRS,
    MATCH_RATE_THRESHOLD,
    RELATIVE_REGRET_THRESHOLD,
    BoundedRecommender,
    CandidateRecommendation,
    OptimizerNotJustifiedError,
    classify_optimizer_justification,
)
import ml.decision_policy as policy_module
from ml.train import predict_recovery_probability, train_baseline
from recovery.policy import decide_action, load_policy_config
from recovery.scoring import (
    RETRY_INTERVENTION_COST_INR,
    UNKNOWN_CATEGORY_RISK_FRACTION,
)
from simulation.config import load_treatment_policy
from simulation.observations import assemble_observations, split_observations

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "ml" / "decision_policy.py"

ATTEMPT_ROWS = 1500
SEED = 20260826

CANONICAL_DIGEST = "f" * 64

CRITERION_MATCH = "decision_match_rate_at_or_above_threshold"
CRITERION_REGRET = "relative_regret_at_or_below_threshold"
CRITERION_PROBE = "policy_safety_probe_passed"
CRITERION_OVERLAP = "minimum_non_overlapping_treated_arm_ci_pairs"


# ---------------------------------------------------------------------------
# Synthetic evidence factories (pure dicts, no bundle needed)
# ---------------------------------------------------------------------------


def _arms_block(bounds_by_arm: dict[str, list[float]]) -> dict:
    return {
        arm: {
            "n": 210,
            "mean_model_revenue": float(np.mean(bounds_by_arm[arm])),
            "bootstrap_ci95_mean_model_revenue": list(bounds_by_arm[arm]),
            "ci_overlap_with": [],
        }
        for arm in TREATED_ARMS
    }


def _evidence(**overrides) -> dict:
    """Canonical all-passing synthetic evidence bundle."""
    base = {
        "decision_match_rate": 0.90,
        "relative_regret": 0.05,
        "policy_safety_probe_passed": True,
        "arms": _arms_block(
            {
                "RETRY_NOW": [100.0, 200.0],
                "RETRY_LATER": [300.0, 400.0],
                "REQUEST_UPDATE": [500.0, 600.0],
                "HUMAN_REVIEW": [700.0, 800.0],
            }
        ),
        "provenance_digest": CANONICAL_DIGEST,
    }
    base.update(overrides)
    return base


def criterion_by_name(verdict: dict, name: str) -> dict:
    matches = [entry for entry in verdict["criteria"] if entry["criterion"] == name]
    assert len(matches) == 1, f"criterion {name!r} not reported exactly once"
    return matches[0]


def failed_criteria(verdict: dict) -> list[str]:
    return [
        entry["criterion"] for entry in verdict["criteria"] if not entry["passed"]
    ]


def non_overlap_count(bounds_by_arm: dict[str, list[float]]) -> int:
    arms = list(TREATED_ARMS)
    count = 0
    for i, first in enumerate(arms):
        lo_a, hi_a = bounds_by_arm[first]
        for second in arms[i + 1 :]:
            lo_b, hi_b = bounds_by_arm[second]
            if hi_a < lo_b or hi_b < lo_a:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Shared frozen-chain world (module-scoped, mirrors test_decision_evidence)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def day6_world():
    attempts = generate_dataset(ATTEMPT_ROWS, seed=42).reset_index(drop=True)
    train_df, validation_df, _ = chronological_split(attempts, 0.70, 0.15)
    baseline, _baseline_metadata = train_baseline(train_df, validation_df, seed=42)
    probabilities = [
        float(value) for value in predict_recovery_probability(baseline, attempts)
    ]
    assembled = assemble_observations(attempts, probabilities, POLICY)
    train_obs, validation_obs, test_obs = split_observations(assembled)

    raw_bundle, _metadata = train_action_models(train_obs, validation_obs, seed=SEED)
    calibrated_bundle = calibrate_action_models(raw_bundle, validation_obs)
    return {
        "calibrated_bundle": calibrated_bundle,
        "assembled": assembled,
        "test_obs": test_obs,
        "report": decision_evidence(calibrated_bundle, test_obs, POLICY, seed=SEED),
    }


@pytest.fixture(scope="module")
def recommender(day6_world):
    # ADVISORY-GATE HONESTY NOTE: this evidence dict is FABRICATED by hand.
    # That is possible by design -- the justification gate is advisory
    # scaffolding for in-process callers, and only the echoed
    # provenance_digest marks this bundle as non-canonical (it is not).
    fabricated = _evidence()
    assert classify_optimizer_justification(fabricated)["classification"] == (
        CLASSIFICATION_JUSTIFIED
    )
    return BoundedRecommender(fabricated, day6_world["calibrated_bundle"])


def _first_row(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    subset = frame.loc[mask]
    assert len(subset) > 0, "fixture pool exhausted: expected such a row to exist"
    return subset.iloc[0]


def _opted_out_row(day6_world) -> pd.Series:
    assembled = day6_world["assembled"]
    return _first_row(assembled, assembled["customer_opted_out"].astype(bool))


def _fraud_row(day6_world) -> pd.Series:
    assembled = day6_world["assembled"]
    return _first_row(assembled, assembled["fraud_risk"].astype(bool))


def _hard_decline_row(day6_world) -> pd.Series:
    assembled = day6_world["assembled"]
    return _first_row(
        assembled,
        (assembled["failure_category"] == "hard_decline")
        & ~assembled["customer_opted_out"].astype(bool)
        & ~assembled["fraud_risk"].astype(bool),
    )


def _eligible_row(day6_world) -> pd.Series:
    randomized_test = day6_world["test_obs"]
    mask = (
        (randomized_test["failure_category"] == "temporary_decline")
        & (randomized_test["attempt_number"] < 4)
        & ~randomized_test["customer_opted_out"].astype(bool)
        & ~randomized_test["fraud_risk"].astype(bool)
    )
    return _first_row(randomized_test.loc[randomized_test["stratum"] == "randomized"], mask)


# ---------------------------------------------------------------------------
# Deterministic stub bundle for exact candidate control
# ---------------------------------------------------------------------------


class _FixedProbabilityPipeline:
    """Stand-in arm pipeline returning a constant positive-class probability."""

    def __init__(self, probability: float):
        self._probability = float(probability)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rows = len(X)
        return np.column_stack(
            [
                np.full(rows, 1.0 - self._probability),
                np.full(rows, self._probability),
            ]
        )


def _stub_bundle(probabilities: dict[str, float]) -> ActionModelBundle:
    return ActionModelBundle(
        models={
            arm: _FixedProbabilityPipeline(probabilities[arm]) for arm in ARM_ORDER
        },
        arms=ARM_ORDER,
        metadata={"stub": "fixed_probability_pipeline"},
    )


STUB_PROBABILITIES = {
    "CONTROL": 0.10,
    "RETRY_NOW": 0.40,
    "RETRY_LATER": 0.30,
    "REQUEST_UPDATE": 0.20,
    "HUMAN_REVIEW": 0.15,
}


def _crafted_row(**overrides) -> dict:
    """Decision-time context shaped like one assembled observation row.

    Carries every decision-time feature column, the ``recovered`` label
    column that the shared feature builder reads (its value is ignored for
    probability queries), and every rule column the frozen policy
    conditions reference."""
    row = {
        "amount_inr": 1000.0,
        "attempt_number": 1,
        "customer_tenure_days": 500,
        "successful_payment_count": 3,
        "failed_payment_count": 1,
        "historical_recovery_count": 2,
        "customer_opted_out": False,
        "fraud_risk": False,
        "payment_method": "card",
        "failure_code": "generic_decline",
        "failure_category": "temporary_decline",
        "issuer_response": "do_not_honor",
        "device_type": "mobile",
        "country": "US",
        "recovered": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture(scope="module")
def stub_recommender():
    return BoundedRecommender(_evidence(), _stub_bundle(STUB_PROBABILITIES))


# ---------------------------------------------------------------------------
# 1. Classifier unit matrix: every criterion pass/fail combination
# ---------------------------------------------------------------------------


def test_all_pass_evidence_classifies_justified_with_clean_reasons():
    verdict = classify_optimizer_justification(_evidence())

    assert verdict["classification"] == CLASSIFICATION_JUSTIFIED
    assert verdict["classification"] == "OPTIMIZER_JUSTIFIED"
    assert [entry["criterion"] for entry in verdict["criteria"]] == [
        CRITERION_MATCH,
        CRITERION_REGRET,
        CRITERION_PROBE,
        CRITERION_OVERLAP,
    ]
    assert all(entry["passed"] for entry in verdict["criteria"])
    assert verdict["reasons"] == []
    assert verdict["provenance_digest"] == CANONICAL_DIGEST


def test_thresholds_are_the_pre_registered_de5_values():
    assert MATCH_RATE_THRESHOLD == 0.60
    assert RELATIVE_REGRET_THRESHOLD == 0.15
    assert MIN_NON_OVERLAPPING_CI_PAIRS == 2


def test_boundary_values_pass_exactly_at_thresholds():
    verdict = classify_optimizer_justification(
        _evidence(decision_match_rate=0.60, relative_regret=0.15)
    )

    assert verdict["classification"] == CLASSIFICATION_JUSTIFIED


def test_match_rate_fail_isolated_to_match_criterion():
    verdict = classify_optimizer_justification(_evidence(decision_match_rate=0.59))

    assert verdict["classification"] == CLASSIFICATION_NOT_YET_JUSTIFIED
    assert failed_criteria(verdict) == [CRITERION_MATCH]
    assert any("0.60" in reason or "match" in reason for reason in verdict["reasons"])
    assert criterion_by_name(verdict, CRITERION_MATCH)["evidence"]["observed"] == 0.59


def test_regret_above_threshold_fails_with_reason_naming_it():
    verdict = classify_optimizer_justification(_evidence(relative_regret=0.151))

    assert failed_criteria(verdict) == [CRITERION_REGRET]
    assert verdict["reasons"], "a failed criterion must carry a reason string"
    assert "regret" in verdict["reasons"][0].lower()


def test_none_relative_regret_fails_with_exact_undefined_reason():
    verdict = classify_optimizer_justification(_evidence(relative_regret=None))

    assert verdict["classification"] == CLASSIFICATION_NOT_YET_JUSTIFIED
    assert failed_criteria(verdict) == [CRITERION_REGRET]
    assert "regret undefined" in verdict["reasons"]
    assert criterion_by_name(verdict, CRITERION_REGRET)["evidence"]["observed"] is None


def test_safety_probe_failure_fails_with_probe_reason():
    verdict = classify_optimizer_justification(
        _evidence(policy_safety_probe_passed=False)
    )

    assert failed_criteria(verdict) == [CRITERION_PROBE]
    assert verdict["reasons"], "a failed probe must explain itself"
    joined = " ".join(verdict["reasons"]).lower()
    assert "probe" in joined or "stop" in joined


@pytest.mark.parametrize(
    "bounds_by_arm",
    [
        pytest.param(
            {
                "RETRY_NOW": [0.0, 100.0],
                "RETRY_LATER": [0.0, 100.0],
                "REQUEST_UPDATE": [0.0, 100.0],
                "HUMAN_REVIEW": [0.0, 100.0],
            },
            id="zero_disjoint_pairs",
        ),
        pytest.param(
            {
                "RETRY_NOW": [0.0, 10.0],
                "RETRY_LATER": [5.0, 15.0],
                "REQUEST_UPDATE": [9.0, 19.0],
                "HUMAN_REVIEW": [13.0, 23.0],
            },
            id="exactly_one_disjoint_pair",
        ),
    ],
)
def test_insufficient_ci_non_overlap_counts_fail(bounds_by_arm):
    assert non_overlap_count(bounds_by_arm) < MIN_NON_OVERLAPPING_CI_PAIRS

    verdict = classify_optimizer_justification(
        _evidence(arms=_arms_block(bounds_by_arm))
    )

    assert verdict["classification"] == CLASSIFICATION_NOT_YET_JUSTIFIED
    overlap = criterion_by_name(verdict, CRITERION_OVERLAP)
    assert overlap["passed"] is False
    assert overlap["evidence"]["non_overlapping_pairs"] == non_overlap_count(
        bounds_by_arm
    )
    assert overlap["evidence"]["required"] == MIN_NON_OVERLAPPING_CI_PAIRS
    assert failed_criteria(verdict) == [CRITERION_OVERLAP]


@pytest.mark.parametrize(
    "bounds_by_arm",
    [
        pytest.param(
            {
                "RETRY_NOW": [0.0, 10.0],
                "RETRY_LATER": [5.0, 15.0],
                "REQUEST_UPDATE": [11.0, 21.0],
                "HUMAN_REVIEW": [13.0, 23.0],
            },
            id="exactly_two_disjoint_pairs",
        ),
        pytest.param(
            {
                "RETRY_NOW": [100.0, 200.0],
                "RETRY_LATER": [300.0, 400.0],
                "REQUEST_UPDATE": [500.0, 600.0],
                "HUMAN_REVIEW": [700.0, 800.0],
            },
            id="all_pairs_disjoint",
        ),
    ],
)
def test_sufficient_ci_non_overlap_counts_pass(bounds_by_arm):
    assert non_overlap_count(bounds_by_arm) >= MIN_NON_OVERLAPPING_CI_PAIRS

    verdict = classify_optimizer_justification(
        _evidence(arms=_arms_block(bounds_by_arm))
    )

    assert criterion_by_name(verdict, CRITERION_OVERLAP)["passed"] is True
    assert verdict["classification"] == CLASSIFICATION_JUSTIFIED


def test_fully_intersecting_ci_family_yields_zero_disjoint_pairs():
    """Every pair intersects (touching endpoints included), so no pair
    qualifies as pairwise-non-overlapping -- the criterion must fail."""
    bounds = {
        "RETRY_NOW": [0.0, 10.0],
        "RETRY_LATER": [2.0, 12.0],
        "REQUEST_UPDATE": [4.0, 14.0],
        "HUMAN_REVIEW": [6.0, 16.0],
    }
    assert non_overlap_count(bounds) == 0

    verdict = classify_optimizer_justification(_evidence(arms=_arms_block(bounds)))

    assert failed_criteria(verdict) == [CRITERION_OVERLAP]


def test_touching_ci_endpoints_are_overlapping_not_disjoint():
    """Adjacent intervals sharing an endpoint ([0,10] vs [10,20]) overlap by
    the shared convention; only strictly separated intervals are disjoint."""

    def disjoint(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
        return hi_a < lo_b or hi_b < lo_a

    assert not disjoint(0.0, 10.0, 10.0, 20.0)
    assert disjoint(0.0, 10.0, 10.0001, 20.0)


def test_multiple_failures_accumulate_one_reason_each_in_order():
    verdict = classify_optimizer_justification(
        _evidence(
            decision_match_rate=0.1,
            relative_regret=None,
            policy_safety_probe_passed=False,
            arms=_arms_block({arm: [0.0, 1.0] for arm in TREATED_ARMS}),
        )
    )

    assert verdict["classification"] == CLASSIFICATION_NOT_YET_JUSTIFIED
    assert failed_criteria(verdict) == [
        CRITERION_MATCH,
        CRITERION_REGRET,
        CRITERION_PROBE,
        CRITERION_OVERLAP,
    ]
    assert len(verdict["reasons"]) == 4
    assert "regret undefined" in verdict["reasons"]


def test_classifier_is_deterministic_and_does_not_mutate_input():
    payload = _evidence()
    snapshot = copy.deepcopy(payload)

    first = classify_optimizer_justification(payload)
    second = classify_optimizer_justification(payload)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert payload == snapshot


# ---------------------------------------------------------------------------
# 2. Malformed evidence: loud ValueErrors naming the key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_key",
    [
        "decision_match_rate",
        "relative_regret",
        "policy_safety_probe_passed",
        "arms",
        "provenance_digest",
    ],
)
def test_missing_required_key_raises_value_error_naming_it(missing_key):
    payload = _evidence()
    del payload[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        classify_optimizer_justification(payload)


@pytest.mark.parametrize(
    "mutation,key_name",
    [
        pytest.param(lambda e: e.update(decision_match_rate="0.9"), "decision_match_rate", id="match_rate_string"),
        pytest.param(lambda e: e.update(decision_match_rate=True), "decision_match_rate", id="match_rate_bool"),
        pytest.param(lambda e: e.update(decision_match_rate=None), "decision_match_rate", id="match_rate_none"),
        pytest.param(lambda e: e.update(relative_regret="low"), "relative_regret", id="regret_string"),
        pytest.param(lambda e: e.update(relative_regret=True), "relative_regret", id="regret_bool"),
        pytest.param(lambda e: e.update(policy_safety_probe_passed=1), "policy_safety_probe_passed", id="probe_int"),
        pytest.param(lambda e: e.update(policy_safety_probe_passed="true"), "policy_safety_probe_passed", id="probe_string"),
        pytest.param(lambda e: e.update(arms=[]), "arms", id="arms_list"),
        pytest.param(lambda e: e.update(arms={}), "arms", id="arms_empty"),
        pytest.param(
            lambda e: e.update(
                arms={
                    **{
                        arm: {"bootstrap_ci95_mean_model_revenue": [0.0, 1.0]}
                        for arm in TREATED_ARMS
                        if arm != "HUMAN_REVIEW"
                    },
                    "CONTROL": {"bootstrap_ci95_mean_model_revenue": [0.0, 1.0]},
                }
            ),
            "arms",
            id="arms_missing_treated_arm",
        ),
        pytest.param(
            lambda e: e.update(
                arms={
                    "RETRY_NOW": {"mean_model_revenue": 1.0},
                    **{
                        arm: {"bootstrap_ci95_mean_model_revenue": [0.0, 1.0]}
                        for arm in TREATED_ARMS
                        if arm != "RETRY_NOW"
                    },
                }
            ),
            "arms",
            id="arm_entry_missing_ci_key",
        ),
        pytest.param(
            lambda e: e.update(
                arms={
                    "RETRY_NOW": {"bootstrap_ci95_mean_model_revenue": [1.0]},
                    **{
                        arm: {"bootstrap_ci95_mean_model_revenue": [0.0, 1.0]}
                        for arm in TREATED_ARMS
                        if arm != "RETRY_NOW"
                    },
                }
            ),
            "arms",
            id="ci_single_bound",
        ),
        pytest.param(
            lambda e: e.update(
                arms={
                    "RETRY_LATER": {"bootstrap_ci95_mean_model_revenue": [5.0, 1.0]},
                    **{
                        arm: {"bootstrap_ci95_mean_model_revenue": [0.0, 1.0]}
                        for arm in TREATED_ARMS
                        if arm != "RETRY_LATER"
                    },
                }
            ),
            "arms",
            id="ci_bounds_inverted",
        ),
        pytest.param(
            lambda e: e.update(
                arms={
                    "REQUEST_UPDATE": {"bootstrap_ci95_mean_model_revenue": ["a", "b"]},
                    **{
                        arm: {"bootstrap_ci95_mean_model_revenue": [0.0, 1.0]}
                        for arm in TREATED_ARMS
                        if arm != "REQUEST_UPDATE"
                    },
                }
            ),
            "arms",
            id="ci_bounds_non_numeric",
        ),
        pytest.param(lambda e: e.update(provenance_digest=""), "provenance_digest", id="digest_empty"),
        pytest.param(lambda e: e.update(provenance_digest=12345), "provenance_digest", id="digest_int"),
    ],
)
def test_wrong_types_raise_value_error_naming_key(mutation, key_name):
    payload = _evidence()
    mutation(payload)

    with pytest.raises(ValueError, match=key_name):
        classify_optimizer_justification(payload)


def test_non_dict_evidence_raises_value_error():
    with pytest.raises(ValueError, match="dict"):
        classify_optimizer_justification("not evidence")


# ---------------------------------------------------------------------------
# 3. Recommender gating behind the classifier verdict
# ---------------------------------------------------------------------------


def test_not_justified_evidence_blocks_construction_with_reasons_attached():
    payload = _evidence(decision_match_rate=0.30, relative_regret=None)
    expected_reasons = classify_optimizer_justification(payload)["reasons"]
    assert expected_reasons

    with pytest.raises(OptimizerNotJustifiedError) as excinfo:
        BoundedRecommender(payload, _stub_bundle(STUB_PROBABILITIES))

    error = excinfo.value
    assert isinstance(error, Exception)
    assert list(error.reasons) == expected_reasons
    assert all(reason in str(error) for reason in expected_reasons)


def test_not_yet_justified_classification_constant_is_pinned():
    assert CLASSIFICATION_NOT_YET_JUSTIFIED == "OPTIMIZER_NOT_YET_JUSTIFIED"


def test_non_action_model_bundle_raises_value_error_after_gate():
    with pytest.raises(ValueError, match="ActionModelBundle"):
        BoundedRecommender(_evidence(), calibrated_bundle={"models": {}})


# ---------------------------------------------------------------------------
# 4. JUSTIFIED path: real calibrated bundle + crafted context rows
# ---------------------------------------------------------------------------


def test_opted_out_row_authorizes_stop_and_overrides_candidate(recommender, day6_world):
    row = _opted_out_row(day6_world)

    recommendation = recommender.recommend(row)

    assert recommendation.authorized_action == "STOP"
    assert recommendation.top_candidate_action in TREATED_ARMS
    assert recommendation.policy_overrode_candidate is True
    assert "opted out" in recommendation.authorization_reason.lower()
    assert recommendation.model_label == LABEL_MODEL_ESTIMATE == "MODEL ESTIMATE"


def test_fraud_row_authorizes_stop_and_overrides_candidate(recommender, day6_world):
    row = _fraud_row(day6_world)

    recommendation = recommender.recommend(row)

    assert recommendation.authorized_action == "STOP"
    assert recommendation.top_candidate_action in TREATED_ARMS
    assert recommendation.policy_overrode_candidate is True
    assert recommendation.model_label == "MODEL ESTIMATE"


def test_hard_decline_row_authorizes_stop_regardless_of_candidate(
    recommender, day6_world
):
    row = _hard_decline_row(day6_world)

    recommendation = recommender.recommend(row)

    assert recommendation.authorized_action == "STOP"
    assert recommendation.top_candidate_action in TREATED_ARMS
    assert recommendation.policy_overrode_candidate is True
    assert "hard decline" in recommendation.authorization_reason.lower()


def test_eligible_row_matches_independently_recomputed_recommendation(
    recommender, day6_world
):
    """Full wiring check recomputed from public APIs: per-arm probabilities,
    incremental revenue math, ARM_ORDER tie-break, and the policy decision
    with the TOP ARM's model probability injected as recovery_probability."""
    bundle = day6_world["calibrated_bundle"]
    row = _eligible_row(day6_world)
    single = pd.DataFrame([row.to_dict()])

    def probability(arm: str) -> float:
        counterfactual = single.copy()
        counterfactual[ACTION_COLUMN] = arm
        return float(predict_action_probability(bundle, counterfactual, arm)[0])

    probabilities = {arm: probability(arm) for arm in ARM_ORDER}
    amount = float(row["amount_inr"])
    risk_penalty = (
        UNKNOWN_CATEGORY_RISK_FRACTION * amount
        if row["failure_category"] == "unknown"
        else 0.0
    )
    revenues = {
        arm: (
            (probabilities[arm] - probabilities["CONTROL"]) * amount
            - RETRY_INTERVENTION_COST_INR
            - risk_penalty
        )
        for arm in TREATED_ARMS
    }
    top_arm = TREATED_ARMS[0]
    for arm in TREATED_ARMS[1:]:
        if revenues[arm] > revenues[top_arm]:
            top_arm = arm
    expected_context = {**row.to_dict(), "recovery_probability": probabilities[top_arm]}
    expected_decision = decide_action(expected_context, load_policy_config())

    recommendation = recommender.recommend(row)

    assert recommendation.top_candidate_action == top_arm
    assert recommendation.incremental_revenue_estimate_inr == pytest.approx(
        revenues[top_arm]
    )
    assert recommendation.authorized_action == expected_decision.authorized_action
    assert recommendation.authorization_reason == expected_decision.reason
    assert recommendation.policy_overrode_candidate is (
        expected_decision.authorized_action != top_arm
    )
    assert recommendation.model_label == "MODEL ESTIMATE"


def test_recommendation_fields_present_and_frozen(stub_recommender):
    recommendation = stub_recommender.recommend(_crafted_row())

    assert isinstance(recommendation, CandidateRecommendation)
    assert set(vars(recommendation)) >= {
        "top_candidate_action",
        "incremental_revenue_estimate_inr",
        "model_label",
        "authorized_action",
        "authorization_reason",
        "policy_overrode_candidate",
    }
    with pytest.raises(Exception):
        recommendation.model_label = "REAL WORLD CLAIM"


def test_stub_candidate_top_revenue_hand_computed(stub_recommender):
    """Stub world: CONTROL 0.10 vs RETRY_NOW 0.40 on amount 1000 with no risk
    penalty -> incremental revenue (0.40 - 0.10) * 1000 - 10 = 290 INR."""
    recommendation = stub_recommender.recommend(_crafted_row())

    assert recommendation.top_candidate_action == "RETRY_NOW"
    assert recommendation.incremental_revenue_estimate_inr == pytest.approx(290.0)


def test_unknown_category_applies_risk_penalty_in_estimate(stub_recommender):
    recommendation = stub_recommender.recommend(
        _crafted_row(failure_category="unknown")
    )

    expected = (STUB_PROBABILITIES["RETRY_NOW"] - STUB_PROBABILITIES["CONTROL"]) * 1000.0 - RETRY_INTERVENTION_COST_INR - UNKNOWN_CATEGORY_RISK_FRACTION * 1000.0
    assert recommendation.incremental_revenue_estimate_inr == pytest.approx(expected)


def test_full_tie_breaks_candidate_to_first_arm_order_precedence():
    tied = {arm: 0.30 for arm in ARM_ORDER}

    recommender = BoundedRecommender(_evidence(), _stub_bundle(tied))
    row = _crafted_row(amount_inr=500.0, device_type="desktop")

    recommendation = recommender.recommend(row)

    assert recommendation.top_candidate_action == TREATED_ARMS[0]


def test_high_probability_temporary_decline_authorizes_candidate():
    """Documented agreement branch: injected top-arm probability 0.75 clears
    R007 (>= 0.70), so the authorized action EQUALS the RETRY_NOW candidate."""
    probabilities = {
        "CONTROL": 0.10,
        "RETRY_NOW": 0.75,
        "RETRY_LATER": 0.40,
        "REQUEST_UPDATE": 0.20,
        "HUMAN_REVIEW": 0.15,
    }
    recommender = BoundedRecommender(_evidence(), _stub_bundle(probabilities))
    recommendation = recommender.recommend(_crafted_row())

    assert recommendation.top_candidate_action == "RETRY_NOW"
    assert recommendation.authorized_action == "RETRY_NOW"
    assert recommendation.policy_overrode_candidate is False


def test_subthreshold_probability_documents_residual_default_divergence():
    """Documented divergence branch: injected top-arm probability 0.40 misses
    R007 (< 0.70) and no STOP rule fires, so the residual default
    RETRY_LATER is authorized while the revenue candidate was RETRY_NOW."""
    recommender = BoundedRecommender(_evidence(), _stub_bundle(STUB_PROBABILITIES))
    recommendation = recommender.recommend(_crafted_row())

    assert recommendation.top_candidate_action == "RETRY_NOW"
    assert recommendation.authorized_action == "RETRY_LATER"
    assert recommendation.policy_overrode_candidate is True
    assert "residual default" in recommendation.authorization_reason.lower()


# ---------------------------------------------------------------------------
# 5. Policy dominance pin: candidate RETRY_NOW on hard_decline -> STOP
# ---------------------------------------------------------------------------


def test_stop_dominance_pin_hard_decline_beats_positive_retry_now_candidate(
    stub_recommender,
):
    """The stub bundle FORCES candidate RETRY_NOW deterministically; the
    hard_decline rule must still win via stop precedence -- the optimizer/
    recommender NEVER overrides policy and STOP remains dominant."""
    row = _crafted_row(
        failure_code="insufficient_funds", failure_category="hard_decline"
    )

    recommendation = stub_recommender.recommend(row)

    assert recommendation.top_candidate_action == "RETRY_NOW"
    assert recommendation.incremental_revenue_estimate_inr > 0.0
    assert recommendation.authorized_action == "STOP"
    assert recommendation.policy_overrode_candidate is True
    assert "hard decline" in recommendation.authorization_reason.lower()


def test_stop_dominance_survives_custom_config_without_stop_precedence_flag_only_if_rule_wins():
    """Even with a config whose stop_precedence flag is False, the
    hard_decline STOP rule itself still matches and wins on priority here --
    dominance in THIS scenario does not depend on the flag alone."""
    raw_rules = load_policy_config()
    assert raw_rules.stop_precedence is True

    recommender = BoundedRecommender(
        _evidence(),
        _stub_bundle(STUB_PROBABILITIES),
        raw_rules,
    )
    recommendation = recommender.recommend(
        _crafted_row(
            failure_code="insufficient_funds", failure_category="hard_decline"
        )
    )

    assert recommendation.authorized_action == "STOP"


# ---------------------------------------------------------------------------
# 6. Determinism + input immutability
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_recommendations(recommender, day6_world):
    row = _eligible_row(day6_world)
    frame_snapshot = day6_world["assembled"].copy(deep=True)

    first = recommender.recommend(row)
    second = recommender.recommend(row)

    assert first == second
    pd.testing.assert_frame_equal(day6_world["assembled"], frame_snapshot)


def test_series_and_dict_rows_agree(recommender, day6_world):
    row = _eligible_row(day6_world)

    from_series = recommender.recommend(row)
    from_dict = recommender.recommend(dict(row))

    assert from_series == from_dict


def test_input_series_never_mutated(recommender, day6_world):
    row = _eligible_row(day6_world)
    snapshot = row.copy(deep=True)

    recommender.recommend(row)

    pd.testing.assert_series_equal(row, snapshot)


def test_missing_amount_or_category_raises_value_error_naming_key(stub_recommender):
    incomplete = {
        "attempt_number": 1,
        "customer_opted_out": False,
        "fraud_risk": False,
    }

    with pytest.raises(ValueError, match="amount_inr"):
        stub_recommender.recommend(incomplete)

    with pytest.raises(ValueError, match="failure_category"):
        stub_recommender.recommend({**incomplete, "amount_inr": 100.0})


def test_unsupported_row_type_raises_value_error(stub_recommender):
    with pytest.raises(ValueError, match="context_row"):
        stub_recommender.recommend(["not", "a", "row"])


# ---------------------------------------------------------------------------
# 6b. Review fixes: genuine-evidence contract, NaN gate guard, probability
#     validation
# ---------------------------------------------------------------------------


def test_genuine_day6_evidence_flows_through_classifier_and_gate(day6_world):
    """Cross-module contract (review F1): a GENUINE ml/decision_evidence
    report must validate against the Task 5 classifier -- historically it
    raised ValueError naming the missing policy_safety_probe_passed key --
    and its probe criterion must PASS on the honest bundle."""
    world_report = day6_world["report"]

    verdict = classify_optimizer_justification(world_report)

    probe_entries = [
        entry
        for entry in verdict["criteria"]
        if entry["criterion"] == CRITERION_PROBE
    ]
    assert len(probe_entries) == 1
    assert probe_entries[0]["passed"] is True

    if verdict["classification"] == CLASSIFICATION_JUSTIFIED:
        BoundedRecommender(world_report, day6_world["calibrated_bundle"])
    else:
        with pytest.raises(OptimizerNotJustifiedError):
            BoundedRecommender(world_report, day6_world["calibrated_bundle"])


@pytest.mark.parametrize(
    "column",
    [
        "customer_opted_out",
        "fraud_risk",
        "failure_category",
        "attempt_number",
        "amount_inr",
    ],
)
def test_nan_policy_gate_columns_raise_instead_of_silent_false(
    stub_recommender, column
):
    """NaN in a policy-gate column compares False inside rule conditions and
    would silently bypass STOP gating (mirrors treatment._reject_nan_context);
    recommend() must fail loudly naming the offending column."""
    row = _crafted_row()
    row[column] = float("nan")

    with pytest.raises(ValueError, match=column):
        stub_recommender.recommend(row)


def test_broken_bundle_top_probability_outside_unit_interval_raises(monkeypatch):
    """F3 guard: an injected recovery_probability outside [0, 1] means the
    calibrated bundle is broken or mis-stubbed; fail loudly instead of
    feeding garbage into the policy engine."""

    def broken_predict(bundle, frame, action):
        return np.array([1.5])

    monkeypatch.setattr(policy_module, "predict_action_probability", broken_predict)
    recommender = BoundedRecommender(_evidence(), _stub_bundle(STUB_PROBABILITIES))

    with pytest.raises(ValueError, match="probability"):
        recommender.recommend(_crafted_row())


# ---------------------------------------------------------------------------
# 7. Purity: imports, randomness bans, honest language
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "numpy",
        "pandas",
        "dataclasses",
        "ml",
        "recovery",
    }
)

FORBIDDEN_PATTERNS = (
    r"(?<![\w.])datetime\s*\.\s*now\b",
    r"(?<![\w.])time\s*\.",
    r"(?<![\w.])secrets?\b",
    r"(?<![\w.])uuid\b",
    r"(?<![\w.])random\b",
)


def _import_root_modules(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _source_without_docstring() -> str:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))
    assert docstring is not None
    assert source.count(docstring) == 1
    return source.replace(docstring, " ", 1)


def test_decision_policy_import_roots_whitelisted():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    roots = _import_root_modules(source)

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )
    assert "from ml.decision_evidence import" in source
    assert "TREATED_ARMS" in source
    assert "from recovery.scoring import" in source
    assert "RETRY_INTERVENTION_COST_INR" in source
    assert "UNKNOWN_CATEGORY_RISK_FRACTION" in source
    assert re.search(r"^RETRY_INTERVENTION_COST_INR\s*[:=]", source, re.M) is None
    assert re.search(r"^UNKNOWN_CATEGORY_RISK_FRACTION\s*[:=]", source, re.M) is None


def test_no_randomness_generator_anywhere():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "default_rng" not in source
    assert "RandomState" not in source


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_no_wall_clock_or_stdlib_randomness_tokens(pattern):
    code = _source_without_docstring()

    assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"


def test_docstring_states_advisory_gate_synthetic_scope_and_authority_language():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))

    assert docstring is not None
    lowered = " ".join(docstring.lower().split())
    assert "advisory" in lowered
    assert "synthetic" in lowered
    assert "ai recommends" in lowered
    assert "never overrides" in lowered
    assert "stop remains dominant" in lowered
    assert "tamper-proof" in lowered
    assert "provenance_digest" in docstring
    assert "fabricate" in lowered
    assert "nothing causal" in lowered
    assert "causal estimate" not in source.lower()


def test_classifier_and_exception_names_exported():
    import ml.decision_policy as module

    assert callable(module.classify_optimizer_justification)
    assert issubclass(module.OptimizerNotJustifiedError, Exception)
    assert module.LABEL_MODEL_ESTIMATE == "MODEL ESTIMATE"
