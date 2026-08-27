"""Tests for Day 6 decision-quality evidence + uncertainty (plan Task 4, D-E4).

Layers pinned here:

1. Pure decision core: ``_decision_core`` consumes plain per-arm probability
   frames + amounts + categories and is hand-checked against constant-
   probability fixtures (perfect agreement -> match rate 1.0 / zero regret;
   unknown-category risk penalty arithmetic; adversarial mis-ordering ->
   match rate < 0.5 with a visible heavy regret tail p99 > p50; denominator
   guard when every truth revenue is nonpositive).
2. Evidence bundle contract: ``decision_evidence`` over a real calibrated
   per-arm bundle on the frozen-chain randomized test segment embeds the
   labels OBSERVED SIMULATED OUTCOME / SIMULATED GROUND TRUTH, restricts the
   candidate set to the four treated arms with ARM_ORDER tie-breaking,
   reports seeded stratified-bootstrap CIs whose pairwise overlap matrix is
   symmetric with the diagonal excluded, guards the relative-regret
   denominator, and reproduces byte-identically across two calls.
3. Provenance digest: sorted-json SHA256 self-hash over the report minus the
   digest field itself; identical inputs reproduce it, perturbed inputs do
   not.
4. Module purity: whitelisted import roots (hashlib/json are REQUIRED by the
   D-E5 provenance-digest contract; recovery.scoring supplies the Day-2 cost
   basis imported, never restated), exactly ONE named-seed default_rng for
   the bootstrap, no wall-clock/stdlib-randomness tokens, no causal language.
"""

from __future__ import annotations

import ast
import json
import math
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
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    calibrate_action_models,
    predict_action_probability,
    train_action_models,
)
from ml.decision_evidence import (
    TREATED_ARMS,
    _assigned_arm_strata_blocks,
    _binomial_ci95,
    _decision_core,
    _provenance_digest,
    _seed_variance_block,
    decision_evidence,
    policy_safety_probe,
)
from ml.decision_policy import classify_optimizer_justification
from ml.train import predict_recovery_probability, train_baseline
from recovery.scoring import (
    RETRY_INTERVENTION_COST_INR,
    UNKNOWN_CATEGORY_RISK_FRACTION,
)
from simulation.config import load_treatment_policy
from simulation.observations import assemble_observations, split_observations
from simulation.outcomes import ground_truth_propensity

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "ml" / "decision_evidence.py"

SEED = 20260826
ATTEMPT_ROWS = 1500

TOP_KEYS = frozenset(
    {
        "label",
        "truth_label",
        "scope_note",
        "candidate_arms",
        "candidate_set_note",
        "decision_rule_note",
        "cost_simplification_note",
        "seed",
        "bundle_kind",
        "bootstrap",
        "n_randomized_test_rows",
        "decision_match_rate",
        "decision_match_count",
        "decision_match_rate_ci95",
        "decision_match_rate_ci95_note",
        "relative_regret",
        "relative_regret_reason",
        "absolute_regret_inr",
        "expected_best_truth_revenue_inr",
        "regret_quantiles",
        "arms",
        "uncertainty_inventory",
        "policy_safety_probe_passed",
        "policy_safety_probe_details",
        "provenance_digest",
    }
)

ARM_KEYS = frozenset(
    {
        "n",
        "mean_model_revenue",
        "mean_truth_revenue",
        "bootstrap_ci95_mean_model_revenue",
        "ci_overlap_with",
    }
)

INVENTORY_KEYS = frozenset(
    {
        "per_arm_n",
        "calibration_status",
        "propensity_overlap_note",
        "propensity_range_overlap_by_arm",
        "seed_variance",
    }
)

QUANTILE_KEYS = frozenset({"p50", "p90", "p99"})


def probability_frame(
    control: np.ndarray, treated: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Assemble an n x ARM_ORDER probability frame from raw arrays."""
    columns = {"CONTROL": np.asarray(control, dtype=float)}
    for arm in TREATED_ARMS:
        columns[arm] = np.asarray(treated[arm], dtype=float)
    return pd.DataFrame(columns, columns=list(ARM_ORDER))


# ---------------------------------------------------------------------------
# Shared frozen-chain world (module-scoped: one train/calibrate/report pass)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def day6_world():
    """Frozen-chain world + calibrated per-arm bundle + one evidence report."""
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

    snapshot = test_obs.copy(deep=True)
    report = decision_evidence(calibrated_bundle, test_obs, POLICY, seed=SEED)
    return {
        "raw_bundle": raw_bundle,
        "calibrated_bundle": calibrated_bundle,
        "test_obs": test_obs,
        "snapshot": snapshot,
        "report": report,
    }


# ---------------------------------------------------------------------------
# 1. Hand-computed constant-probability fixtures over the pure core
# ---------------------------------------------------------------------------

CONSTANT_CONTROL = 0.10
CONSTANT_TREATED = {
    "RETRY_NOW": 0.30,
    "RETRY_LATER": 0.20,
    "REQUEST_UPDATE": 0.15,
    "HUMAN_REVIEW": 0.12,
}


def _constant_frames(rows: int, category: str):
    model_frame = probability_frame(
        np.full(rows, CONSTANT_CONTROL),
        {arm: np.full(rows, value) for arm, value in CONSTANT_TREATED.items()},
    )
    truth_frame = model_frame.copy()
    amounts = np.array([1000.0, 2000.0])[:rows]
    categories = np.full(rows, category, dtype=object)
    return model_frame, truth_frame, amounts, categories


def test_constant_probability_fixture_hand_computed_perfect_agreement():
    """Every row: RETRY_NOW maximizes both model and truth incremental
    revenue; hand-checked per-arm means {290, 140, 65, 20} INR; match rate
    exactly 1.0 with zero regret everywhere."""
    model_frame, truth_frame, amounts, categories = _constant_frames(
        2, "temporary_decline"
    )

    core = _decision_core(model_frame, truth_frame, amounts, categories)

    assert core["match_rate"] == pytest.approx(1.0)
    assert core["match_count"] == 2
    expected_argmax = np.array(["RETRY_NOW", "RETRY_NOW"], dtype=object)
    assert list(core["model_argmax_names"]) == list(expected_argmax)
    assert list(core["truth_argmax_names"]) == list(expected_argmax)
    # Hand math: diff .20/.10/.05/.02 x amounts [1000, 2000] - cost 10.
    assert core["model_revenue_by_arm"]["RETRY_NOW"][0] == pytest.approx(190.0)
    assert core["model_revenue_by_arm"]["RETRY_NOW"][1] == pytest.approx(390.0)
    assert core["mean_model_revenue"]["RETRY_NOW"] == pytest.approx(290.0)
    assert core["mean_model_revenue"]["RETRY_LATER"] == pytest.approx(140.0)
    assert core["mean_model_revenue"]["REQUEST_UPDATE"] == pytest.approx(65.0)
    assert core["mean_model_revenue"]["HUMAN_REVIEW"] == pytest.approx(20.0)
    assert core["mean_truth_revenue"]["RETRY_NOW"] == pytest.approx(290.0)
    assert core["absolute_regret_inr"] == pytest.approx(0.0)
    assert core["relative_regret"] == pytest.approx(0.0)
    assert core["expected_best_truth_revenue_inr"] == pytest.approx(290.0)
    assert core["regret_quantiles"] == {"p50": 0.0, "p90": 0.0, "p99": 0.0}


def test_unknown_category_risk_penalty_hand_computed_in_core():
    """failure_category 'unknown' applies UNKNOWN_CATEGORY_RISK_FRACTION *
    amount to EVERY arm's revenue identically: RETRY_NOW rows become
    140 / 290 INR (means 215) and REQUEST_UPDATE collapses to -10 per row."""
    model_frame, truth_frame, amounts, categories = _constant_frames(2, "unknown")

    core = _decision_core(model_frame, truth_frame, amounts, categories)

    assert core["model_revenue_by_arm"]["RETRY_NOW"][0] == pytest.approx(140.0)
    assert core["model_revenue_by_arm"]["RETRY_NOW"][1] == pytest.approx(290.0)
    assert core["mean_model_revenue"]["RETRY_NOW"] == pytest.approx(215.0)
    assert core["mean_model_revenue"]["RETRY_LATER"] == pytest.approx(65.0)
    assert core["mean_model_revenue"]["REQUEST_UPDATE"] == pytest.approx(-10.0)
    assert core["mean_model_revenue"]["HUMAN_REVIEW"] == pytest.approx(-55.0)
    # Argmax unchanged: the risk term is arm-independent per row.
    assert list(core["model_argmax_names"]) == ["RETRY_NOW", "RETRY_NOW"]


def test_perfect_separation_fixture_match_rate_one_and_zero_regret():
    """Model probabilities EQUAL noise-integrated truth probabilities with a
    distinct strict winner per row cycling through all four treated arms:
    argmaxes coincide row-for-row -> match rate 1.0, all regrets zero."""
    rows = 4
    base = np.full(rows, 0.10)
    columns = {}
    for position, arm in enumerate(TREATED_ARMS):
        values = base.copy()
        values[position] = 0.70
        columns[arm] = values
    frame = probability_frame(np.full(rows, 0.05), columns)

    core = _decision_core(
        frame, frame.copy(), np.full(rows, 500.0), np.full(rows, "card", dtype=object)
    )

    assert core["match_rate"] == pytest.approx(1.0)
    assert list(core["model_argmax_names"]) == list(TREATED_ARMS)
    assert list(core["truth_argmax_names"]) == list(TREATED_ARMS)
    assert core["absolute_regret_inr"] == pytest.approx(0.0)
    assert core["regret_quantiles"] == {"p50": 0.0, "p90": 0.0, "p99": 0.0}


def test_adversarial_misordering_fixture_low_match_rate_heavy_tail():
    """Model systematically promotes the SECOND-best arm on the seven largest
    amounts: match rate 3/10 = 0.3 (< 0.5), absolute regret 73.5 INR exactly,
    quantiles p50=82.5 / p90=136.5 / p99=148.65 -- p99 strictly above p50."""
    rows = 10
    amounts = np.array([100.0 * (i + 1) for i in range(rows)])
    truth_treated = {arm: np.full(rows, value) for arm, value in {
        "RETRY_NOW": 0.35,
        "RETRY_LATER": 0.20,
        "REQUEST_UPDATE": 0.12,
        "HUMAN_REVIEW": 0.13,
    }.items()}
    model_treated = {arm: values.copy() for arm, values in truth_treated.items()}
    # Rows 3..9: swap RETRY_NOW <-> RETRY_LATER so the model's best is the
    # truth's second-best exactly there (amounts ascend with row index).
    model_treated["RETRY_NOW"][3:] = 0.20
    model_treated["RETRY_LATER"][3:] = 0.35
    model_frame = probability_frame(np.full(rows, 0.05), model_treated)
    truth_frame = probability_frame(np.full(rows, 0.05), truth_treated)
    categories = np.full(rows, "temporary_decline", dtype=object)

    core = _decision_core(model_frame, truth_frame, amounts, categories)

    assert core["match_rate"] == pytest.approx(0.3)
    assert core["match_count"] == 3
    # Per-row regret on swapped rows = (.30 - .15) * amount = .15 * amount.
    assert core["absolute_regret_inr"] == pytest.approx(73.5)
    assert core["relative_regret"] == pytest.approx(735.0 / 1550.0)
    assert core["regret_quantiles"]["p50"] == pytest.approx(82.5)
    assert core["regret_quantiles"]["p90"] == pytest.approx(136.5)
    assert core["regret_quantiles"]["p99"] == pytest.approx(148.65)
    assert core["regret_quantiles"]["p99"] > core["regret_quantiles"]["p50"]


def test_denominator_guard_all_nonpositive_truth_revenues_reports_undefined():
    """Truth propensities BELOW control for every treated arm make every
    truth revenue negative (diff * amount - cost < 0): the relative-regret
    denominator E[best truth revenue] <= 0 must yield an undefined ratio +
    reason string -- no crash, no division by zero."""
    rows = 3
    treated = {arm: np.full(rows, 0.05) for arm in TREATED_ARMS}
    frame = probability_frame(np.full(rows, 0.90), treated)

    core = _decision_core(
        frame.copy(),
        frame,
        np.full(rows, 800.0),
        np.full(rows, "card", dtype=object),
    )

    assert core["relative_regret"] is None
    assert isinstance(core["relative_regret_reason"], str)
    assert "denominator" in core["relative_regret_reason"]
    assert "undefined" in core["relative_regret_reason"]
    assert core["expected_best_truth_revenue_inr"] <= 0.0
    # Model mirrors truth ordering: decisions still agree perfectly.
    assert core["match_rate"] == pytest.approx(1.0)
    assert core["absolute_regret_inr"] == pytest.approx(0.0)


def test_decision_core_ties_break_to_first_arm_order_precedence():
    """Exactly tied top revenues must resolve to the earlier ARM_ORDER arm on
    BOTH sides (RETRY_NOW beats RETRY_LATER under a full tie)."""
    rows = 2
    treated = {arm: np.full(rows, 0.30) for arm in TREATED_ARMS}
    frame = probability_frame(np.full(rows, 0.10), treated)

    core = _decision_core(
        frame.copy(),
        frame,
        np.full(rows, 100.0),
        np.full(rows, "card", dtype=object),
    )

    assert list(core["model_argmax_names"]) == ["RETRY_NOW", "RETRY_NOW"]
    assert list(core["truth_argmax_names"]) == ["RETRY_NOW", "RETRY_NOW"]


# ---------------------------------------------------------------------------
# 2. Evidence-bundle contract on the real calibrated bundle
# ---------------------------------------------------------------------------


def test_report_top_level_contract_labels_candidate_set(day6_world):
    report = day6_world["report"]

    assert set(report) == TOP_KEYS
    assert report["label"] == "OBSERVED SIMULATED OUTCOME"
    assert report["truth_label"] == "SIMULATED GROUND TRUTH"
    assert tuple(report["candidate_arms"]) == TREATED_ARMS
    assert "CONTROL" not in report["arms"]
    assert set(report["arms"]) == set(TREATED_ARMS)
    assert report["seed"] == SEED
    assert report["bundle_kind"] == "calibrated"
    assert report["bootstrap"]["replications"] == 500
    assert report["bootstrap"]["confidence_level"] == 0.95
    assert set(report["arms"][TREATED_ARMS[0]]) == ARM_KEYS
    assert set(report["uncertainty_inventory"]) == INVENTORY_KEYS
    assert set(report["regret_quantiles"]) == QUANTILE_KEYS
    # Candidate-set + tie-break discipline stated in the emitted notes.
    assert "CONTROL" in report["candidate_set_note"]
    assert "ARM_ORDER" in report["decision_rule_note"]
    assert "nothing causal" in report["scope_note"].lower()


def test_report_metrics_recomputed_independently_on_identical_rows(day6_world):
    """Integration honesty: match rate, regret, and per-arm means recomputed
    in the TEST from raw predictions + truth replay must equal the report
    (same randomized rows, same constants, same ARM_ORDER tie-break)."""
    bundle = day6_world["calibrated_bundle"]
    frame = day6_world["test_obs"]
    report = day6_world["report"]

    randomized = frame.loc[frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]
    amounts = randomized["amount_inr"].to_numpy(dtype=float)
    unknown_mask = randomized["failure_category"].to_numpy() == "unknown"

    def revenues(probabilities):
        table = {}
        for arm in TREATED_ARMS:
            difference = probabilities(arm) - probabilities("CONTROL")
            table[arm] = (
                difference * amounts
                - RETRY_INTERVENTION_COST_INR
                - unknown_mask * UNKNOWN_CATEGORY_RISK_FRACTION * amounts
            )
        return table

    model_revenue = revenues(
        lambda arm: np.asarray(
            predict_action_probability(bundle, randomized, arm), dtype=float
        )
    )
    truth_revenue = revenues(
        lambda arm: ground_truth_propensity(randomized, POLICY, arm)
    )

    def argmax_names(table):
        names = []
        for row in range(len(randomized)):
            best_arm = TREATED_ARMS[0]
            best_value = table[TREATED_ARMS[0]][row]
            for arm in TREATED_ARMS[1:]:
                if table[arm][row] > best_value:
                    best_value = table[arm][row]
                    best_arm = arm
            names.append(best_arm)
        return np.array(names, dtype=object)

    model_names = argmax_names(model_revenue)
    truth_names = argmax_names(truth_revenue)
    matches = float((model_names == truth_names).mean())
    regrets = np.array(
        [
            truth_revenue[truth_names[row]][row]
            - truth_revenue[model_names[row]][row]
            for row in range(len(randomized))
        ],
        dtype=float,
    )

    assert report["n_randomized_test_rows"] == len(randomized)
    assert report["decision_match_rate"] == pytest.approx(matches, abs=1e-12)
    assert report["absolute_regret_inr"] == pytest.approx(regrets.mean(), abs=1e-9)
    for arm in TREATED_ARMS:
        entry = report["arms"][arm]
        assert entry["n"] == len(randomized)
        assert entry["mean_model_revenue"] == pytest.approx(
            float(model_revenue[arm].mean()), abs=1e-9
        )
        assert entry["mean_truth_revenue"] == pytest.approx(
            float(truth_revenue[arm].mean()), abs=1e-9
        )


def test_binomial_ci_contains_match_rate_point_estimate(day6_world):
    report = day6_world["report"]

    low, high = report["decision_match_rate_ci95"]
    assert low <= report["decision_match_rate"] <= high


def test_binomial_ci_clamped_to_unit_interval_with_note(day6_world):
    """F3: the Wald interval overshoots [0, 1] at extreme rates (rate 0.9
    on n=10 gives upper ~1.086); bounds must be clamped into [0, 1] and the
    emitted note must record the clamping."""
    unclipped_high = 0.9 + 1.959963984540054 * math.sqrt(0.09 / 10)
    assert unclipped_high > 1.0

    low, high = _binomial_ci95(matches=9, n_rows=10, rate=0.9)

    assert high == pytest.approx(1.0)
    assert 0.0 <= low < 1.0

    report = day6_world["report"]
    wrapped_low, wrapped_high = report["decision_match_rate_ci95"]
    assert 0.0 <= wrapped_low <= wrapped_high <= 1.0
    note = report["decision_match_rate_ci95_note"]
    assert "clamped" in note
    assert "[0, 1]" in note


def test_bootstrap_cis_bracket_means_overlap_symmetric_diagonal_excluded(day6_world):
    report = day6_world["report"]

    bounds = {}
    for arm in TREATED_ARMS:
        entry = report["arms"][arm]
        low, high = entry["bootstrap_ci95_mean_model_revenue"]
        bounds[arm] = (low, high)
        assert low <= entry["mean_model_revenue"] <= high
    for arm in TREATED_ARMS:
        overlaps = report["arms"][arm]["ci_overlap_with"]
        assert arm not in overlaps
        for other in overlaps:
            assert other in TREATED_ARMS
            lo_a, hi_a = bounds[arm]
            lo_b, hi_b = bounds[other]
            overlapped = lo_a <= hi_b and lo_b <= hi_a
            assert overlapped, f"{arm}/{other} listed without overlapping CIs"
        # Symmetry: b in overlap(a) iff a in overlap(b).
        for other in TREATED_ARMS:
            if other == arm:
                continue
            assert (other in overlaps) == (arm in report["arms"][other]["ci_overlap_with"])


def test_n_randomized_counts_only_randomized_stratum_rows(day6_world):
    frame = day6_world["test_obs"]
    report = day6_world["report"]

    expected = int((frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED).sum())
    assert report["n_randomized_test_rows"] == expected
    assert expected > 0
    inventory = report["uncertainty_inventory"]
    assert set(inventory["per_arm_n"]) == set(TREATED_ARMS)
    assert all(count == expected for count in inventory["per_arm_n"].values())


def test_strata_blocks_are_true_filtered_positions_on_interleaved_chain(
    day6_world,
):
    """F1 regression: strata blocks must hold each arm's TRUE filtered row
    positions (np.flatnonzero over the arm mask), not fabricated contiguous
    count ranges. The real-chain randomized segment is interleaved by the
    multinomial assignment, so a per-block single-arm check fails under any
    count-based fabrication."""
    frame = day6_world["test_obs"]
    randomized = frame.loc[frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]
    assigned = randomized[ACTION_COLUMN].to_numpy()

    blocks = _assigned_arm_strata_blocks(randomized)

    assert len(blocks) == len(ARM_ORDER)
    for arm, block in zip(ARM_ORDER, blocks):
        assert block.size > 0
        block_arms = np.unique(assigned[block])
        assert len(block_arms) == 1
        assert block_arms[0] == arm
    # Blocks PARTITION all randomized rows exactly once.
    partition = np.sort(np.concatenate(blocks))
    np.testing.assert_array_equal(partition, np.arange(len(randomized)))


def test_calibration_status_calibrated_for_shipped_bundle_warning_for_raw(
    day6_world,
):
    report = day6_world["report"]
    assert report["uncertainty_inventory"]["calibration_status"] == "calibrated"

    raw_report = decision_evidence(
        day6_world["raw_bundle"], day6_world["test_obs"], POLICY, seed=SEED
    )

    status = raw_report["uncertainty_inventory"]["calibration_status"].lower()
    assert "warning" in status
    assert "calibrat" in status


def test_propensity_overlap_note_and_per_arm_ranges_present(day6_world):
    inventory = day6_world["report"]["uncertainty_inventory"]

    assert isinstance(inventory["propensity_overlap_note"], str)
    assert len(inventory["propensity_overlap_note"]) > 0
    assert set(inventory["propensity_range_overlap_by_arm"]) == set(TREATED_ARMS)
    assert all(
        isinstance(value, bool)
        for value in inventory["propensity_range_overlap_by_arm"].values()
    )


# ---------------------------------------------------------------------------
# 3. Seed variance inventory
# ---------------------------------------------------------------------------


def stability_runs():
    return [
        {
            "decision_match_rate": 0.8,
            "relative_regret": 0.10,
            "arms": {
                arm: {"mean_model_revenue": 100.0} for arm in TREATED_ARMS
            },
        },
        {
            "decision_match_rate": 0.6,
            "relative_regret": 0.20,
            "arms": {
                arm: {"mean_model_revenue": 300.0} for arm in TREATED_ARMS
            },
        },
    ]


def test_seed_variance_computed_from_caller_supplied_stability_runs():
    runs = stability_runs()
    block = _seed_variance_block(runs)

    assert block["status"] == "computed_from_caller_supplied_stability_runs"
    assert block["n_runs"] == 2
    assert block["decision_match_rate_sd"] == pytest.approx(0.1)
    assert block["relative_regret_sd"] == pytest.approx(0.05)
    assert all(
        block["mean_model_revenue_sd_by_arm"][arm] == pytest.approx(100.0)
        for arm in TREATED_ARMS
    )


def test_seed_variance_defaults_to_not_computed_filled_by_caller(day6_world):
    block = day6_world["report"]["uncertainty_inventory"]["seed_variance"]

    assert block["status"] == "not_computed"
    assert "caller" in block["note"].lower()


def test_seed_variance_rejects_malformed_or_empty_stability_runs():
    with pytest.raises(ValueError, match="stability"):
        _seed_variance_block([])
    malformed = [
        {
            "decision_match_rate": 0.5,
            "relative_regret": 0.1,
            "arms": {arm: {"mean_model_revenue": 1.0} for arm in TREATED_ARMS},
        },
        {"decision_match_rate": 0.5, "arms": {arm: {"mean_model_revenue": 2.0} for arm in TREATED_ARMS}},
    ]
    with pytest.raises(ValueError, match="relative_regret"):
        _seed_variance_block(malformed)


def test_seed_variance_rejects_arm_entry_missing_mean_model_revenue():
    """F5: a treated-arm entry lacking 'mean_model_revenue' must fail
    loudly naming BOTH the offending arm and the missing key."""
    runs = [
        {
            "decision_match_rate": 0.7,
            "relative_regret": 0.1,
            "arms": {
                arm: {"mean_model_revenue": 100.0} for arm in TREATED_ARMS
            },
        },
        {
            "decision_match_rate": 0.9,
            "relative_regret": 0.2,
            "arms": {
                **{
                    arm: {"mean_model_revenue": 200.0}
                    for arm in TREATED_ARMS
                    if arm != "HUMAN_REVIEW"
                },
                "HUMAN_REVIEW": {"n": 42},
            },
        },
    ]

    with pytest.raises(ValueError) as excinfo:
        _seed_variance_block(runs)

    message = str(excinfo.value)
    assert "HUMAN_REVIEW" in message
    assert "mean_model_revenue" in message


# ---------------------------------------------------------------------------
# 4. Denominator guard end-to-end, determinism, digest
# ---------------------------------------------------------------------------


def test_relative_regret_flows_through_wrapper_and_ci_contains_it(day6_world):
    report = day6_world["report"]

    assert report["relative_regret"] is not None
    assert report["absolute_regret_inr"] >= 0.0
    assert report["relative_regret_reason"] is None


def test_determinism_two_identical_calls_byte_identical(day6_world):
    bundle = day6_world["calibrated_bundle"]
    frame = day6_world["test_obs"]

    first = decision_evidence(bundle, frame, POLICY, seed=SEED)
    second = decision_evidence(bundle, frame, POLICY, seed=SEED)

    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )


def test_report_walk_contains_no_nan_or_infinity_anywhere(day6_world):
    """F2 guard: the digest serializes with allow_nan=False, so any
    non-finite float anywhere in the emitted structure would crash the
    provenance hash; walk every dict/list/float and assert finiteness."""

    def walk(value) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, float):
            assert math.isfinite(value), f"non-finite float {value!r} emitted"

    walk(day6_world["report"])


def test_provenance_digest_self_consistent_and_perturbation_sensitive(day6_world):
    bundle = day6_world["calibrated_bundle"]
    frame = day6_world["test_obs"]
    report = decision_evidence(bundle, frame, POLICY, seed=SEED)

    stored = report["provenance_digest"]
    digest_source = {key: value for key, value in report.items() if key != "provenance_digest"}
    assert _provenance_digest(digest_source) == stored

    # Perturbing ONE amount changes revenues/regret -> different digest.
    perturbed = frame.copy()
    perturbed.iloc[0, perturbed.columns.get_loc("amount_inr")] *= 1.05
    perturbed_report = decision_evidence(bundle, perturbed, POLICY, seed=SEED)
    assert perturbed_report["provenance_digest"] != stored

    # A different bootstrap seed reshuffles the resamples -> different CIs
    # (and therefore a different metrics payload) with probability ~1.
    other_seed_report = decision_evidence(bundle, frame, POLICY, seed=SEED + 1)
    assert other_seed_report["provenance_digest"] != stored


def test_input_frame_never_mutated_by_decision_evidence(day6_world):
    bundle = day6_world["calibrated_bundle"]
    frame = day6_world["snapshot"]

    decision_evidence(bundle, frame, POLICY, seed=SEED)

    pd.testing.assert_frame_equal(frame, day6_world["snapshot"])


# ---------------------------------------------------------------------------
# 5. Loud input validation
# ---------------------------------------------------------------------------


def test_zero_randomized_rows_raise_value_error(day6_world):
    bundle = day6_world["calibrated_bundle"]
    empty = day6_world["test_obs"].iloc[:0]

    with pytest.raises(ValueError, match="randomized"):
        decision_evidence(bundle, empty, POLICY, seed=SEED)


def test_non_action_model_bundle_raises_value_error(day6_world):
    with pytest.raises(ValueError, match="ActionModelBundle"):
        decision_evidence({"models": {}}, day6_world["test_obs"], POLICY, seed=SEED)


def test_non_treatment_policy_argument_raises_value_error(day6_world):
    """F6: the evidence must be computed against the declared synthetic
    world only; a non-TreatmentPolicy second argument fails loudly."""
    with pytest.raises(ValueError, match="TreatmentPolicy"):
        decision_evidence(
            day6_world["calibrated_bundle"],
            day6_world["test_obs"],
            {"main_effects_logit": {}},
            seed=SEED,
        )


def test_missing_feature_columns_raise_module_error_naming_offender(day6_world):
    bundle = day6_world["calibrated_bundle"]
    thinned = day6_world["test_obs"].drop(columns=["country"])

    with pytest.raises(ValueError, match="country"):
        decision_evidence(bundle, thinned, POLICY, seed=SEED)


def test_nonpositive_bootstrap_replications_raise_value_error(day6_world):
    with pytest.raises(ValueError, match="bootstrap_replications"):
        decision_evidence(
            day6_world["calibrated_bundle"],
            day6_world["test_obs"],
            POLICY,
            seed=SEED,
            bootstrap_replications=0,
        )


# ---------------------------------------------------------------------------
# 5b. Native policy-safety probe (review fix F1) + Task-5 contract closure
# ---------------------------------------------------------------------------

PROBE_CONTEXT_NAMES = ("customer_opted_out", "fraud_risk", "hard_decline")


class _FixedProbabilityPipeline:
    """Deterministic stand-in arm pipeline for adversarial/degenerate probes."""

    def __init__(self, probability: float):
        self._probability = float(probability)

    def predict_proba(self, X) -> np.ndarray:
        rows = len(X)
        return np.column_stack(
            [
                np.full(rows, 1.0 - self._probability),
                np.full(rows, self._probability),
            ]
        )


def _stub_action_bundle(probabilities: dict[str, float]) -> ActionModelBundle:
    return ActionModelBundle(
        models={
            arm: _FixedProbabilityPipeline(probabilities[arm]) for arm in ARM_ORDER
        },
        arms=ARM_ORDER,
        metadata={"stub": "fixed_probability_pipeline"},
    )


def test_report_embeds_native_passing_probe(day6_world):
    report = day6_world["report"]

    assert report["policy_safety_probe_passed"] is True
    details = report["policy_safety_probe_details"]
    assert [entry["context"] for entry in details] == list(PROBE_CONTEXT_NAMES)
    assert all(entry["authorized"] == "STOP" for entry in details)
    assert all(entry["candidate"] in TREATED_ARMS for entry in details)
    assert all(entry["overrode"] is True for entry in details)


def test_policy_safety_probe_direct_call_contract_and_determinism(day6_world):
    bundle = day6_world["calibrated_bundle"]

    probe = policy_safety_probe(bundle)
    again = policy_safety_probe(bundle)

    assert set(probe) == {
        "policy_safety_probe_passed",
        "probe_details",
        "label",
    }
    assert probe["label"] == "OBSERVED SIMULATED OUTCOME"
    assert probe["policy_safety_probe_passed"] is True
    assert len(probe["probe_details"]) == 3
    assert json.dumps(probe, sort_keys=True, allow_nan=False) == json.dumps(
        again, sort_keys=True, allow_nan=False
    )


def test_probe_rejects_non_bundle_and_bad_policy_config(day6_world):
    with pytest.raises(ValueError, match="ActionModelBundle"):
        policy_safety_probe({"models": {}})

    with pytest.raises(ValueError, match="PolicyConfig"):
        policy_safety_probe(
            day6_world["calibrated_bundle"], policy_config={"rules": []}
        )


def test_adversarial_stub_favoring_treatment_cannot_authorize_non_stop():
    """Dominance pin: even a bundle engineered so every treated arm looks
    wildly better than CONTROL (positive candidate revenue everywhere, high
    injected recovery_probability) cannot move any authorized action off
    STOP -- the recommendation layer NEVER overrides policy."""
    adversarial = _stub_action_bundle(
        {
            "CONTROL": 0.01,
            "RETRY_NOW": 0.99,
            "RETRY_LATER": 0.98,
            "REQUEST_UPDATE": 0.97,
            "HUMAN_REVIEW": 0.96,
        }
    )

    probe = policy_safety_probe(adversarial)

    assert probe["policy_safety_probe_passed"] is True
    for entry in probe["probe_details"]:
        assert entry["authorized"] == "STOP"
        assert entry["candidate"] in TREATED_ARMS
        assert entry["overrode"] is True


def test_degenerate_tie_bundle_still_stops_and_perturbs_digest(day6_world):
    """All-equal probabilities force the ARM_ORDER tie-break candidate
    (RETRY_NOW); every context still authorizes STOP so the gate passes.
    The probe verdict lives INSIDE the hashed content: recomputing the
    sorted-json digest over a tampered probe block must NOT reproduce the
    stored digest, and the degenerate world's digest differs from the
    honest canonical run's digest."""
    degenerate = _stub_action_bundle({arm: 0.50 for arm in ARM_ORDER})

    degenerate_report = decision_evidence(
        degenerate, day6_world["test_obs"], POLICY, seed=SEED
    )
    honest_report = day6_world["report"]

    assert degenerate_report["policy_safety_probe_passed"] is True
    assert honest_report["policy_safety_probe_passed"] is True
    tie_candidates = {
        entry["candidate"]
        for entry in degenerate_report["policy_safety_probe_details"]
    }
    assert tie_candidates == {"RETRY_NOW"}

    digest_source = {
        key: value
        for key, value in degenerate_report.items()
        if key != "provenance_digest"
    }
    assert _provenance_digest(digest_source) == (
        degenerate_report["provenance_digest"]
    )
    tampered_probe = [
        {**entry, "authorized": "RETRY_LATER"}
        for entry in degenerate_report["policy_safety_probe_details"]
    ]
    assert _provenance_digest(
        {**digest_source, "policy_safety_probe_details": tampered_probe}
    ) != degenerate_report["provenance_digest"]

    assert (
        degenerate_report["provenance_digest"]
        != honest_report["provenance_digest"]
    )


def test_genuine_evidence_satisfies_the_task5_classifier_contract(day6_world):
    """Contract-defect regression pin (review F1): the Task 5 classifier
    REQUIRES policy_safety_probe_passed; before this fix the GENUINE
    decision_evidence output raised ValueError naming that missing key, so
    no canonical bundle could ever pass the gate. The genuine report must
    validate structurally and its probe criterion must PASS."""
    verdict = classify_optimizer_justification(day6_world["report"])

    probe_criteria = [
        entry
        for entry in verdict["criteria"]
        if entry["criterion"] == "policy_safety_probe_passed"
    ]
    assert len(probe_criteria) == 1
    assert probe_criteria[0]["passed"] is True


# ---------------------------------------------------------------------------
# 6. Purity: imports, single seeded rng, honest language
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "hashlib",
        "json",
        "numpy",
        "pandas",
        "ml",
        "recovery",
        "simulation",
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


def test_decision_evidence_import_roots_whitelisted():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )
    # Documented necessities: hashlib/json power the D-E5 provenance digest;
    # the Day-2 cost basis is IMPORTED from recovery.scoring, never restated.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "from recovery.scoring import" in source
    assert "RETRY_INTERVENTION_COST_INR" in source
    assert "UNKNOWN_CATEGORY_RISK_FRACTION" in source
    assert re.search(r"^RETRY_INTERVENTION_COST_INR\s*[:=]", source, re.M) is None
    assert re.search(r"^UNKNOWN_CATEGORY_RISK_FRACTION\s*[:=]", source, re.M) is None


def test_exactly_one_seeded_default_rng_named_seed_pattern():
    code = _source_without_docstring()

    occurrences = [match.start() for match in re.finditer(r"default_rng", code)]
    assert len(occurrences) == 1, (
        "the stratified bootstrap is the ONLY sanctioned randomness consumer; "
        "its generator must be derived once from the named seed parameter"
    )
    pattern = r"np\.random\.default_rng\(\s*seed\s*\)"
    assert re.search(pattern, code) is not None


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_no_wall_clock_or_stdlib_randomness_tokens(pattern):
    code = _source_without_docstring()

    assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"


def test_docstring_embeds_labels_scope_classifier_no_causal_language():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))

    assert docstring is not None
    assert "OBSERVED SIMULATED OUTCOME" in docstring
    assert "SIMULATED GROUND TRUTH" in docstring
    assert "SYNTHETIC" in docstring
    assert "nothing causal" in docstring.lower()
    assert "classifier" in docstring.lower()
    assert "causal estimate" not in source.lower()
