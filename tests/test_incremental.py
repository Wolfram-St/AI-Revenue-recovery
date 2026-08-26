"""Tests for Day 5 incremental recovery / incremental revenue reporting (Task 5).

Four layers are pinned here:

1. HAND-CHECKED MATH: a crafted duck-typed bundle (``types.SimpleNamespace``
   wrapping constant-probability pseudo-models, {CONTROL .30, RETRY_NOW .50,
   RETRY_LATER .40, REQUEST_UPDATE .35, HUMAN_REVIEW .25}) makes every arm
   contrast exactly computable (.20/.10/.05/-.05), and the SIMULATED GROUND
   TRUTH twin is cross-checked against an independent hand-written
   Gauss-Hermite k=20 quadrature on a uniform-context frame where every row
   shares one known integrated propensity.

2. PAIRED-ROW IDENTITY: a recording pseudo-model captures a checksum of the
   exact feature matrix handed to EACH arm's model, proving all five arms
   scored the SAME randomized rows; an order-stable ``attempt_id``
   fingerprint (pandas SipHash sum) is embedded in the report and verified
   against an independent recomputation plus a shuffle-invariance probe.

3. REVENUE ACCOUNTING (D-M6): per-row IncrementalRevenue =
   IncRec x amount - RETRY_INTERVENTION_COST_INR - unknown-category risk
   penalty, with the Day 2 scoring constants IMPORTED (never restated);
   hand-checked rows make the unknown penalty visible, exercise 2-decimal
   rounding, and exhibit an honestly NEGATIVE projected arm (HUMAN_REVIEW
   at -.05 contrast vs the flat retry cost).

4. DISCIPLINE: exact label vocabulary (MODEL ESTIMATE / SIMULATED GROUND
   TRUTH), the exact uniform-retry-cost disclosure sentence, model-vs-truth
   structural parity, determinism, loud ValueErrors naming missing columns,
   zeroed-but-valid empty behavior, input purity (never mutates frames), and
   module purity (import whitelist, ZERO randomness derivations, no wall
   clock, no forbidden tokens, synthetic-world-only language with "causal
   estimate" absent from the entire source).
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml.action_model import (
    ACTION_COLUMN,
    ARM_ORDER,
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
)
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from simulation.config import load_treatment_policy

from ml.incremental import (
    COST_SIMPLIFICATION_NOTE,
    LABEL_MODEL_ESTIMATE,
    LABEL_SIMULATED_GROUND_TRUTH,
    incremental_recovery_table,
    incremental_recovery_truth,
    incremental_revenue_table,
)

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "ml" / "incremental.py"

SIGMA = float(POLICY.noise_sigma_logit)

# Constant per-arm probabilities for the duck-typed bundle: contrasts are
# exactly RETRY_NOW +.20, RETRY_LATER +.10, REQUEST_UPDATE +.05,
# HUMAN_REVIEW -.05 versus CONTROL .30.
CONSTANT_PROBABILITIES = {
    "CONTROL": 0.30,
    "RETRY_NOW": 0.50,
    "RETRY_LATER": 0.40,
    "REQUEST_UPDATE": 0.35,
    "HUMAN_REVIEW": 0.25,
}

EXPECTED_MEAN_DIFFERENCES = {
    arm: round(CONSTANT_PROBABILITIES[arm] - CONSTANT_PROBABILITIES["CONTROL"], 4)
    for arm in CONSTANT_PROBABILITIES
}


# ---------------------------------------------------------------------------
# Shared fixture builders (balanced_context pattern from test_action_evaluation)
# ---------------------------------------------------------------------------


def context_frame(n_rows: int) -> pd.DataFrame:
    """Full 14-feature decision-time context + attempt_id, arm-cycle free."""
    positions = np.arange(n_rows)
    categories = (
        "temporary_decline",
        "payment_method_issue",
        "authentication_required",
        "unknown",
        "hard_decline",
    )
    methods = ("card", "upi", "netbanking", "wallet")
    devices = ("android", "ios", "web")
    codes = ("B01", "B02", "B03", "B04", "B05")
    issuers = ("do_not_honor", "insufficient_funds", "expired_card", "none")
    attempt_number = (positions % 4) + 1
    frame = pd.DataFrame(
        {
            "attempt_id": [f"ATT-{position:06d}" for position in positions],
            "amount_inr": 3000.0 + (positions % 7) * 250.0,
            "attempt_number": attempt_number,
            "customer_tenure_days": (positions % 360).astype(float),
            "successful_payment_count": positions % 5,
            "failed_payment_count": attempt_number - 1,
            "historical_recovery_count": positions % 7,
            "customer_opted_out": (positions % 13) == 12,
            "fraud_risk": (positions % 47) == 46,
            "payment_method": np.array(methods, dtype=object)[positions % 4],
            "failure_code": np.array(codes, dtype=object)[positions % 5],
            "failure_category": np.array(categories, dtype=object)[positions % 5],
            "issuer_response": np.array(issuers, dtype=object)[positions % 4],
            "device_type": np.array(devices, dtype=object)[positions % 3],
            "country": "IN",
            # build_feature_matrix structurally returns this Day-1 label next
            # to X; predictions never consume it, but assembled observations
            # always carry it, so the fixture mirrors that contract.
            "recovered": (positions % 3 == 0).astype(int),
            ACTION_COLUMN: np.array(ARM_ORDER, dtype=object)[(positions // 5) % 5],
        }
    )
    frame[STRATUM_COLUMN] = STRATUM_RANDOMIZED
    return frame


def uniform_context(n_rows: int, category: str = "temporary_decline") -> pd.DataFrame:
    """n copies of ONE decision-time row (unique attempt_ids, same features).

    Every row shares one base logit, so the noise-integrated ground-truth
    propensity is a single hand-computable scalar for every arm and each
    per-row difference collapses to one constant.
    """
    single = context_frame(1)
    single["failure_category"] = category
    frame = pd.concat([single] * n_rows, ignore_index=True)
    frame["attempt_id"] = [f"UNI-{position:04d}" for position in range(n_rows)]
    return frame


class _ConstantPredictor:
    """Duck-typed pipeline returning one constant positive-class probability."""

    def __init__(self, probability: float):
        self._probability = float(probability)

    def predict_proba(self, X):  # noqa: ANN001 - test double signature
        rows = len(X)
        probability = self._probability
        return np.tile([1.0 - probability, probability], (rows, 1))


class _RecordingPredictor(_ConstantPredictor):
    """Constant predictor that ALSO records a checksum of every feature
    matrix it is handed -- the paired-row identity evidence."""

    def __init__(self, probability: float):
        super().__init__(probability)
        self.seen: list[tuple[int, int]] = []

    def predict_proba(self, X):
        fingerprint = (
            len(X),
            int(pd.util.hash_pandas_object(X, index=False).sum()),
        )
        self.seen.append(fingerprint)
        return super().predict_proba(X)


def fake_bundle(recording: bool = False) -> SimpleNamespace:
    """Minimal duck-typed ActionModelBundle with constant per-arm models."""
    predictors = {
        arm: (_RecordingPredictor if recording else _ConstantPredictor)(
            CONSTANT_PROBABILITIES[arm]
        )
        for arm in ARM_ORDER
    }
    return SimpleNamespace(
        models={arm: predictor for arm, predictor in predictors.items()},
        predictors=predictors,
        arms=ARM_ORDER,
        metadata={"fixture": "constant"},
    )


def _nan_safe_json(result: dict) -> str:
    return json.dumps(result, sort_keys=True, allow_nan=True)


# ---------------------------------------------------------------------------
# 1. Hand-computed model table on the constant-probability bundle
# ---------------------------------------------------------------------------


def test_model_table_hand_computed_contrasts_and_structure():
    bundle = fake_bundle()
    frame = context_frame(25)

    result = incremental_recovery_table(bundle, frame)

    assert result["label"] == LABEL_MODEL_ESTIMATE == "MODEL ESTIMATE"
    assert tuple(result["arms_covered"]) == (
        "RETRY_NOW",
        "RETRY_LATER",
        "REQUEST_UPDATE",
        "HUMAN_REVIEW",
    )
    assert "CONTROL" not in result["arms"]
    assert result["n_randomized_test_rows"] == 25
    lowered = result["note"].lower()
    assert "naive" in lowered
    assert "randomized" in lowered
    assert "synthetic" in lowered
    assert "unconfounded" in lowered
    assert "no production claim" in lowered
    for arm in result["arms_covered"]:
        entry = result["arms"][arm]
        expected = EXPECTED_MEAN_DIFFERENCES[arm]
        assert entry["n"] == 25
        assert entry["mean_probability_arm"] == pytest.approx(
            CONSTANT_PROBABILITIES[arm], abs=1e-9
        )
        assert entry["mean_probability_control"] == pytest.approx(0.30, abs=1e-9)
        assert entry["mean_difference"] == pytest.approx(expected, abs=1e-9)
        assert entry["paired_mean_difference"] == pytest.approx(expected, abs=1e-9)
        # Constant contrasts: the median equals the mean exactly.
        assert entry["paired_median_difference"] == pytest.approx(expected, abs=1e-9)
        assert len(entry["per_row_differences"]) == 25
        assert all(
            difference == pytest.approx(
                CONSTANT_PROBABILITIES[arm] - CONSTANT_PROBABILITIES["CONTROL"],
                abs=1e-12,
            )
            for difference in entry["per_row_differences"]
        )


# ---------------------------------------------------------------------------
# 2. Paired-row identity: identical rows through every arm's model
# ---------------------------------------------------------------------------


def test_every_arm_model_scores_the_identical_feature_matrix():
    bundle = fake_bundle(recording=True)
    frame = context_frame(40)

    incremental_recovery_table(bundle, frame)

    fingerprints = list(bundle.predictors["CONTROL"].seen)
    assert len(fingerprints) == 1
    assert fingerprints[0][0] == 40
    for arm in ARM_ORDER:
        assert bundle.predictors[arm].seen == fingerprints, (
            f"{arm} was not evaluated on the identical rows as CONTROL"
        )


def _expected_fingerprint(randomized: pd.DataFrame) -> int:
    return int(
        pd.util.hash_pandas_object(
            randomized["attempt_id"].astype(str), index=False
        ).sum()
    )


def test_row_fingerprint_embedded_matches_independent_hash_and_shuffle():
    bundle = fake_bundle()
    frame = context_frame(60)
    randomized = frame.loc[frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]

    result = incremental_recovery_table(bundle, frame)

    identity = result["row_identity"]
    assert identity is not None
    assert identity["column"] == "attempt_id"
    assert identity["n"] == 60
    assert identity["fingerprint"] == _expected_fingerprint(randomized)

    generator = np.random.default_rng(7)
    shuffled = randomized.iloc[generator.permutation(len(randomized))]
    assert identity["fingerprint"] == _expected_fingerprint(shuffled), (
        "fingerprint must describe the ROW SET, not the row order"
    )


def test_row_identity_none_when_attempt_id_absent():
    bundle = fake_bundle()
    frame = context_frame(10).drop(columns=["attempt_id"])

    result = incremental_recovery_table(bundle, frame)

    assert result["row_identity"] is None


# ---------------------------------------------------------------------------
# 3. SIMULATED GROUND TRUTH twin vs manual Gauss-Hermite quadrature
# ---------------------------------------------------------------------------


def _sigmoid_scalar(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def analytic_base_logit(frame: pd.DataFrame) -> np.ndarray:
    terms = POLICY.base_propensity_terms
    return (
        float(terms.intercept)
        + frame["failure_category"].map(terms.category_effects).to_numpy(dtype=float)
        + float(terms.successful_payment_count_log1p)
        * np.log1p(frame["successful_payment_count"].to_numpy(dtype=float))
        + float(terms.historical_recovery_count_min5)
        * np.minimum(frame["historical_recovery_count"].to_numpy(dtype=float), 5.0)
        + float(terms.attempt_number_prior_offset)
        * np.maximum(frame["attempt_number"].to_numpy(dtype=float) - 1.0, 0.0)
        + float(terms.fraud_risk) * frame["fraud_risk"].astype(float).to_numpy()
        + float(terms.amount_log1p_per_k)
        * np.log1p(frame["amount_inr"].to_numpy(dtype=float) / 1000.0)
        + float(terms.method_upi) * (frame["payment_method"] == "upi").astype(float).to_numpy()
        + float(terms.device_android) * (frame["device_type"] == "android").astype(float).to_numpy()
    )


def _configured_interaction(action: str, column: str, equals_value: str) -> float:
    return math.fsum(
        float(rule.effect_logit)
        for rule in POLICY.interactions
        if rule.action == action
        and rule.column == column
        and rule.equals_value == equals_value
    )


def _manual_integrated_propensity(base_logit: float, effect_logit: float) -> float:
    """Independent k=20 Gauss-Hermite quadrature (scalar math.exp formula)."""
    nodes, weights = np.polynomial.hermite.hermgauss(20)
    total_logit = base_logit + effect_logit
    return math.fsum(
        (float(weight) / math.sqrt(math.pi))
        * _sigmoid_scalar(total_logit + math.sqrt(2.0) * SIGMA * float(node))
        for node, weight in zip(nodes, weights)
    )


def test_truth_table_matches_manual_quadrature_on_uniform_context():
    """Every row of the uniform frame shares ONE base logit, so the reported
    arm contrast must equal the hand-integrated difference q_RETRY_NOW -
    q_CONTROL to display precision (4 dp => +-5e-5 slack)."""
    frame = uniform_context(12)
    base_logit = float(analytic_base_logit(frame)[0])
    q_control = _manual_integrated_propensity(
        base_logit, float(POLICY.main_effects_logit["CONTROL"])
    )
    q_retry_now = _manual_integrated_propensity(
        base_logit,
        float(POLICY.main_effects_logit["RETRY_NOW"])
        + _configured_interaction("RETRY_NOW", "failure_category", "temporary_decline"),
    )
    q_human_review = _manual_integrated_propensity(
        base_logit, float(POLICY.main_effects_logit["HUMAN_REVIEW"])
    )
    expected_retry_now = q_retry_now - q_control
    expected_review = q_human_review - q_control

    result = incremental_recovery_truth(POLICY, frame)

    assert result["label"] == LABEL_SIMULATED_GROUND_TRUTH == "SIMULATED GROUND TRUTH"
    assert "known by construction" in result["note"]
    retry_entry = result["arms"]["RETRY_NOW"]
    assert retry_entry["n"] == 12
    # 4-dp rounding bounds the displayed value within half a unit of 1e-4.
    assert abs(retry_entry["mean_difference"] - expected_retry_now) <= 5.1e-5
    assert abs(retry_entry["paired_mean_difference"] - expected_retry_now) <= 5.1e-5
    assert abs(retry_entry["paired_median_difference"] - expected_retry_now) <= 5.1e-5
    review_entry = result["arms"]["HUMAN_REVIEW"]
    assert abs(review_entry["mean_difference"] - expected_review) <= 5.1e-5
    assert abs(
        review_entry["mean_probability_arm"] - q_human_review
    ) <= 5.1e-5
    assert abs(review_entry["mean_probability_control"] - q_control) <= 5.1e-5


def test_truth_table_structure_mirrors_model_table_exactly():
    frame = context_frame(30)

    model_table = incremental_recovery_table(fake_bundle(), frame)
    truth_table = incremental_recovery_truth(POLICY, frame)

    assert set(model_table) == set(truth_table)
    assert tuple(model_table["arms_covered"]) == tuple(truth_table["arms_covered"])
    for arm in model_table["arms_covered"]:
        assert (
            set(model_table["arms"][arm].keys()) == set(truth_table["arms"][arm].keys())
        )
    assert model_table["label"] == "MODEL ESTIMATE"
    assert truth_table["label"] == "SIMULATED GROUND TRUTH"
    assert model_table["note"] != truth_table["note"]


# ---------------------------------------------------------------------------
# 4. Incremental revenue accounting (D-M6)
# ---------------------------------------------------------------------------

# Four randomized rows with hand-friendly economics. Expected per-row
# IncrementalRevenue for RETRY_NOW (contrast +.20, cost 10.0, risk 5% iff
# unknown):
#   row 0: amount 1000.00, temporary_decline -> 200 - 10 -   0 = 190.00
#   row 1: amount 2000.00, unknown           -> 400 - 10 - 100 = 290.00
#   row 2: amount  800.00, hard_decline      -> 160 - 10 -   0 = 150.00
#   row 3: amount  333.30, unknown           ->  66.66 - 10 - 16.665 = 39.995 -> 40.00 (2dp)
REVENUE_AMOUNTS = (1000.00, 2000.00, 800.00, 333.30)
REVENUE_CATEGORIES = ("temporary_decline", "unknown", "hard_decline", "unknown")


def revenue_fixture() -> pd.DataFrame:
    frame = context_frame(4).iloc[[0, 1, 2, 3]].reset_index(drop=True)
    frame["amount_inr"] = REVENUE_AMOUNTS
    frame["failure_category"] = REVENUE_CATEGORIES
    frame[STRATUM_COLUMN] = STRATUM_RANDOMIZED
    return frame


def test_revenue_table_hand_computed_unknown_penalty_negative_arm_and_rounding():
    bundle = fake_bundle()
    frame = revenue_fixture()

    recovery = incremental_recovery_table(bundle, frame)
    revenue = incremental_revenue_table(recovery, frame)

    assert revenue["label"] == "MODEL ESTIMATE"
    assert revenue["intervention_cost_inr"] == pytest.approx(10.0)
    assert revenue["unknown_category_risk_fraction"] == pytest.approx(0.05)
    assert revenue["arms_covered"] == recovery["arms_covered"]

    retry_now = revenue["arms"]["RETRY_NOW"]
    expected_rows = [
        0.20 * amount - 10.0 - (0.05 * amount if cat == "unknown" else 0.0)
        for amount, cat in zip(REVENUE_AMOUNTS, REVENUE_CATEGORIES)
    ]
    assert retry_now["n"] == 4
    assert retry_now["risk_penalty_applied_rows"] == 2
    assert retry_now["mean_incremental_revenue_per_case_inr"] == pytest.approx(
        round(float(np.mean(expected_rows)), 2), abs=1e-9
    )
    assert retry_now["total_projected_incremental_revenue_inr"] == pytest.approx(
        round(float(np.sum(expected_rows)), 2), abs=1e-9
    )

    # Unknown-category penalty VISIBLY applied: the identical-contrast arm
    # would earn 160.00/320.00 more per matching row without the 5% haircut.
    request_update = revenue["arms"]["REQUEST_UPDATE"]
    assert request_update["mean_incremental_revenue_per_case_inr"] < retry_now[
        "mean_incremental_revenue_per_case_inr"
    ]

    # Honest NEGATIVE projection: HUMAN_REVIEW contrast -.05 cannot pay the
    # flat retry cost on these amounts.
    review = revenue["arms"]["HUMAN_REVIEW"]
    assert review["mean_incremental_revenue_per_case_inr"] < 0.0
    assert review["total_projected_incremental_revenue_inr"] < 0.0
    expected_review_rows = [
        -0.05 * amount - 10.0 - (0.05 * amount if cat == "unknown" else 0.0)
        for amount, cat in zip(REVENUE_AMOUNTS, REVENUE_CATEGORIES)
    ]
    assert review["total_projected_incremental_revenue_inr"] == pytest.approx(
        round(float(np.sum(expected_review_rows)), 2), abs=1e-9
    )


def test_revenue_rounding_reports_two_decimals():
    """A row engineered to land on a third decimal pins the 2-dp display."""
    frame = revenue_fixture()
    bundle = fake_bundle()
    recovery = incremental_recovery_table(bundle, frame)
    revenue = incremental_revenue_table(recovery, frame)

    for arm in revenue["arms_covered"]:
        for field in (
            "mean_incremental_revenue_per_case_inr",
            "total_projected_incremental_revenue_inr",
        ):
            value = revenue["arms"][arm][field]
            assert value == round(value, 2), f"{arm}.{field} not 2dp: {value}"
    # Row 3 alone: 0.20*333.30 - 10 - 0.05*333.30 = 39.995 -> 40.00.
    assert revenue["arms"]["RETRY_NOW"]["total_projected_incremental_revenue_inr"] == (
        pytest.approx(round(190.0 + 290.0 + 150.0 + 39.995, 2), abs=1e-9)
    )


def test_revenue_from_truth_table_inherits_label_and_flags_is_truth():
    frame = revenue_fixture()

    truth = incremental_recovery_truth(POLICY, frame)
    revenue_truth = incremental_revenue_table(truth, frame, is_truth=True)

    assert revenue_truth["label"] == "SIMULATED GROUND TRUTH"
    assert revenue_truth["is_truth_input"] is True
    # Same accounting applied to truth contrasts stays hand-verifiable: every
    # uniform row shares one contrast, so mean == total / n for RETRY_NOW.
    retry_now = revenue_truth["arms"]["RETRY_NOW"]
    assert retry_now["n"] == 4
    implied_total = retry_now["mean_incremental_revenue_per_case_inr"] * 4
    assert retry_now["total_projected_incremental_revenue_inr"] == pytest.approx(
        implied_total, abs=0.03
    )


# ---------------------------------------------------------------------------
# 5. Labels, disclosure sentence, determinism, purity of inputs
# ---------------------------------------------------------------------------


def test_exact_cost_simplification_sentence_present_in_revenue_table():
    frame = revenue_fixture()
    recovery = incremental_recovery_table(fake_bundle(), frame)

    revenue = incremental_revenue_table(recovery, frame)
    truth = incremental_recovery_truth(POLICY, frame)
    revenue_truth = incremental_revenue_table(truth, frame, is_truth=True)

    assert revenue["cost_simplification_note"] == (
        "A single retry-cost constant is applied uniformly to all treated arms "
        "including REQUEST_UPDATE and HUMAN_REVIEW, whose true economics differ."
    )
    # F2: the verbatim D-M6 disclosure rides on BOTH label variants.
    assert revenue_truth["cost_simplification_note"] == (
        "A single retry-cost constant is applied uniformly to all treated arms "
        "including REQUEST_UPDATE and HUMAN_REVIEW, whose true economics differ."
    )
    assert revenue_truth["cost_simplification_note"] == COST_SIMPLIFICATION_NOTE
    assert revenue_truth["label"] == "SIMULATED GROUND TRUTH"
    assert "optimization target" in revenue["note"]
    lowered = revenue["note"].lower()
    assert "nothing causal" in lowered or "non-causal" in lowered


def test_determinism_two_calls_byte_identical_all_three_tables():
    bundle = fake_bundle()
    frame = context_frame(35)

    assert _nan_safe_json(incremental_recovery_table(bundle, frame)) == _nan_safe_json(
        incremental_recovery_table(bundle, frame)
    )
    assert _nan_safe_json(incremental_recovery_truth(POLICY, frame)) == _nan_safe_json(
        incremental_recovery_truth(POLICY, frame)
    )
    recovery = incremental_recovery_table(bundle, frame)
    assert _nan_safe_json(incremental_revenue_table(recovery, frame)) == _nan_safe_json(
        incremental_revenue_table(recovery, frame)
    )


def test_calls_never_mutate_inputs():
    bundle = fake_bundle()
    frame = context_frame(20)
    snapshot = frame.copy(deep=True)

    recovery = incremental_recovery_table(bundle, frame)
    incremental_recovery_truth(POLICY, frame)
    incremental_revenue_table(recovery, frame)

    pd.testing.assert_frame_equal(frame, snapshot)


# ---------------------------------------------------------------------------
# 6. Loud failures and empty-frame behavior
# ---------------------------------------------------------------------------


def test_missing_columns_raise_value_error_naming_offenders():
    bundle = fake_bundle()
    frame = context_frame(15)

    with pytest.raises(ValueError) as excinfo:
        incremental_recovery_table(bundle, frame.drop(columns=[STRATUM_COLUMN]))
    assert "stratum" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        incremental_recovery_table(bundle, frame.drop(columns=["device_type"]))
    assert "device_type" in str(excinfo.value)
    assert "build_feature_matrix" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        incremental_recovery_truth(POLICY, frame.drop(columns=["attempt_number"]))
    assert "attempt_number" in str(excinfo.value)

    recovery = incremental_recovery_table(bundle, frame)
    with pytest.raises(ValueError) as excinfo:
        incremental_revenue_table(recovery, frame.drop(columns=["amount_inr"]))
    assert "amount_inr" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        incremental_revenue_table(
            recovery, frame.drop(columns=["failure_category"])
        )
    assert "failure_category" in str(excinfo.value)


def test_revenue_row_alignment_mismatch_raises_value_error():
    """Vector-length guard: an id-less frame bypasses the F1 fingerprint
    cross-check (nothing to compare) so the per-arm difference-count
    mismatch itself must raise, naming the arm and both counts."""
    bundle = fake_bundle()
    frame = revenue_fixture().drop(columns=["attempt_id"])
    recovery = incremental_recovery_table(bundle, frame)
    assert recovery["row_identity"] is None

    with pytest.raises(ValueError) as excinfo:
        incremental_revenue_table(recovery, frame.iloc[:-1])

    message = str(excinfo.value)
    assert "RETRY_NOW" in message
    assert "4" in message and "3" in message


def test_revenue_row_identity_cross_check_accepts_matching_frame():
    """F1: a revenue projection on the SAME frame the table was built from
    passes the fingerprint cross-check silently."""
    bundle = fake_bundle()
    frame = revenue_fixture()
    recovery = incremental_recovery_table(bundle, frame)

    revenue = incremental_revenue_table(recovery, frame)

    assert revenue["arms"]["RETRY_NOW"]["n"] == 4


def test_revenue_row_identity_mismatch_same_count_raises_loudly():
    """F1 regression: same row COUNT but a different attempt_id set must
    fail loudly -- count equality alone cannot prove row alignment."""
    bundle = fake_bundle()
    frame = revenue_fixture()
    recovery = incremental_recovery_table(bundle, frame)
    other = revenue_fixture()
    other["attempt_id"] = [f"OTH-{position:04d}" for position in range(4)]
    other["amount_inr"] = [10.0, 20.0, 30.0, 40.0]
    assert len(other) == len(frame)

    with pytest.raises(ValueError) as excinfo:
        incremental_revenue_table(recovery, other)

    message = str(excinfo.value)
    assert (
        "recovery table was computed on a different row set than the "
        "provided context frame" in message
    )


def test_revenue_table_without_label_key_raises_value_error():
    frame = revenue_fixture()
    recovery = incremental_recovery_table(fake_bundle(), frame)
    stripped = {key: value for key, value in recovery.items() if key != "label"}

    with pytest.raises(ValueError) as excinfo:
        incremental_revenue_table(stripped, frame)

    assert "label" in str(excinfo.value)


def test_revenue_table_with_empty_arms_raises_value_error_naming_problem():
    frame = revenue_fixture()

    with pytest.raises(ValueError) as excinfo:
        incremental_revenue_table({"label": "MODEL ESTIMATE", "arms": {}}, frame)

    message = str(excinfo.value)
    assert "missing treated arms" in message
    for arm in ("RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW"):
        assert arm in message


def test_incremental_recovery_table_rejects_bundle_without_models():
    frame = context_frame(5)

    with pytest.raises(ValueError) as excinfo:
        incremental_recovery_table(object(), frame)

    message = str(excinfo.value)
    assert "models" in message
    assert "object" in message


def test_empty_frames_yield_zeroed_valid_structures_with_labels():
    bundle = fake_bundle()
    empty = context_frame(0)

    model_empty = incremental_recovery_table(bundle, empty)
    truth_empty = incremental_recovery_truth(POLICY, empty)

    for table, expected_label in (
        (model_empty, "MODEL ESTIMATE"),
        (truth_empty, "SIMULATED GROUND TRUTH"),
    ):
        assert table["label"] == expected_label
        assert table["n_randomized_test_rows"] == 0
        assert table["row_identity"] is None
        assert set(table["arms"]) == set(table["arms_covered"])
        for arm in table["arms_covered"]:
            entry = table["arms"][arm]
            assert entry["n"] == 0
            assert entry["mean_difference"] == 0.0
            assert entry["paired_mean_difference"] == 0.0
            assert entry["paired_median_difference"] == 0.0
            assert entry["per_row_differences"] == []

    revenue_empty = incremental_revenue_table(model_empty, empty)
    assert revenue_empty["label"] == "MODEL ESTIMATE"
    assert revenue_empty["n_rows"] == 0
    for arm in revenue_empty["arms_covered"]:
        entry = revenue_empty["arms"][arm]
        assert entry["n"] == 0
        assert entry["mean_incremental_revenue_per_case_inr"] == 0.0
        assert entry["total_projected_incremental_revenue_inr"] == 0.0
        assert entry["risk_penalty_applied_rows"] == 0


def test_all_safety_censored_frame_behaves_like_empty():
    bundle = fake_bundle()
    frame = context_frame(10)
    frame[STRATUM_COLUMN] = "safety_censored"

    result = incremental_recovery_table(bundle, frame)

    assert result["n_randomized_test_rows"] == 0
    assert all(result["arms"][arm]["n"] == 0 for arm in result["arms_covered"])


# ---------------------------------------------------------------------------
# 7. Module purity: imports, randomness, language discipline
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "numpy", "pandas", "ml", "simulation", "recovery"}
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


def test_incremental_import_roots_whitelisted_constants_imported_not_restated():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )
    # D-M6: the Day 2 scoring constants MUST be imported from recovery.scoring
    # -- restating 10.0 / 0.05 locally would fork the cost basis.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "from recovery.scoring import" in source
    assert "RETRY_INTERVENTION_COST_INR" in source
    assert "UNKNOWN_CATEGORY_RISK_FRACTION" in source
    assert re.search(r"^RETRY_INTERVENTION_COST_INR\s*[:=]", source, re.M) is None
    assert re.search(r"^UNKNOWN_CATEGORY_RISK_FRACTION\s*[:=]", source, re.M) is None


def test_zero_randomness_derivations_and_no_forbidden_tokens():
    code = _source_without_docstring()

    assert "default_rng" not in code, (
        "incremental reporting draws NO randomness of its own; the module must "
        "contain zero default_rng derivations"
    )
    for pattern in FORBIDDEN_PATTERNS:
        assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"


def test_docstring_language_pins_synthetic_only_noncausal_not_optimization():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))

    assert docstring is not None
    assert "synthetic" in docstring.lower()
    assert "SYNTHETIC" in docstring
    assert "MODEL ESTIMATE" in docstring
    assert "SIMULATED GROUND TRUTH" in docstring
    assert "nothing causal" in docstring.lower()
    assert "optimization target" in docstring.lower()
    assert "causal estimate" not in source.lower()
