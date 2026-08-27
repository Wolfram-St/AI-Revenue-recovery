"""Tests for the Day 6 pooled-vs-per-arm comparison harness (plan Task 3, D-E1/D-E2/D-E6).

Layers pinned here:

1. Report contract: ``compare_models`` emits the five pre-registered sections
   (predictive, ground_truth_agreement, effect_contrasts, rule_application,
   complexity) with synthetic-world labels embedded, calibrated-bundle
   enforcement, and a D-E2 verdict that is "pooled" ONLY when all four
   pre-registered criteria pass -- per-arm stays the reference otherwise.
2. Rule engine isolation: ``_apply_rule`` is a PURE function over plain
   metric dicts; synthetic dicts force every criterion outcome (pooled
   better, per-arm better, agreement fail, smallest-arm fail, interaction
   band fail, NaN-fails-closed, attenuated-cell-not-gated) without any
   refitting.
3. Fixture-scale reality: both families evaluated on the identical randomized
   test segment of the full frozen chain at n=1500; numbers finite; seeded
   stratified-bootstrap CIs bracket their point estimates; determinism holds
   across two calls once the single documented wall-clock field is excluded.
4. Separable protocols: ``learning_curves`` (fractions apply to the TRAIN
   segment only; calibration on the FULL validation segment; per-arm refits
   all five arms) and ``stability_check`` (multi-seed micro-Brier + arm-mean
   incremental spread) are standalone functions Task 6 calls explicitly;
   exercised here only on tiny fraction/seed sets.
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
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    TARGET_COLUMN,
    calibrate_action_models,
    train_action_models,
)
from ml.model_comparison import (
    _apply_rule,
    compare_models,
    learning_curves,
    stability_check,
)
from ml.pooled_model import (
    calibrate_pooled_model,
    train_pooled_model,
)
from ml.train import predict_recovery_probability, train_baseline
from simulation.config import load_treatment_policy
from simulation.observations import assemble_observations, split_observations

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "ml" / "model_comparison.py"

ATTEMPT_ROWS = 1500
SEED = 20260826

TREATED_ARMS = tuple(arm for arm in ARM_ORDER if arm != "CONTROL")

CRITERION_NAMES = (
    "strict_micro_brier_ci_non_overlap",
    "ground_truth_agreement_no_worse",
    "smallest_arm_brier_no_worse",
    "interaction_recovery_within_band",
)


@pytest.fixture(scope="module")
def day6_world():
    """Full frozen chain artifact plus both calibrated families and the report.

    Builds the canonical 1500-attempt world exactly like the prior task
    fixtures, trains AND calibrates both model families on the identical
    randomized train/validation segments, snapshots the test frame, runs one
    full ``compare_models`` (with the optional timed-refit frames supplied),
    and keeps the raw bundles for rejection-path tests.
    """
    attempts = generate_dataset(ATTEMPT_ROWS, seed=42).reset_index(drop=True)
    train_df, validation_df, _ = chronological_split(attempts, 0.70, 0.15)
    baseline, _metadata = train_baseline(train_df, validation_df, seed=42)
    probabilities = [
        float(value) for value in predict_recovery_probability(baseline, attempts)
    ]
    assembled = assemble_observations(attempts, probabilities, POLICY)
    train_obs, validation_obs, test_obs = split_observations(assembled)

    per_arm_raw, _ = train_action_models(train_obs, validation_obs, seed=SEED)
    pooled_raw, _ = train_pooled_model(train_obs, validation_obs, seed=SEED)
    per_arm_cal = calibrate_action_models(per_arm_raw, validation_obs)
    pooled_cal = calibrate_pooled_model(pooled_raw, validation_obs)

    test_snapshot = test_obs.copy(deep=True)
    report = compare_models(
        per_arm_cal,
        pooled_cal,
        baseline,
        test_obs,
        POLICY,
        seed=SEED,
        train_frame=train_obs,
        validation_frame=validation_obs,
    )
    return {
        "report": report,
        "per_arm_raw": per_arm_raw,
        "pooled_raw": pooled_raw,
        "per_arm_cal": per_arm_cal,
        "pooled_cal": pooled_cal,
        "baseline": baseline,
        "train_obs": train_obs,
        "validation_obs": validation_obs,
        "test_obs": test_obs,
        "test_snapshot": test_snapshot,
    }


# ---------------------------------------------------------------------------
# Synthetic metric dicts for pure rule-engine unit tests (no refitting)
# ---------------------------------------------------------------------------


def _synth_predictive(
    pooled_micro_ci=(0.10, 0.12),
    per_arm_micro_ci=(0.13, 0.15),
    pooled_small_brier=0.11,
    per_arm_small_brier=0.14,
):
    def family(micro_ci, small_brier):
        return {
            "micro": {
                "auc": 0.80,
                "auc_ci95": [0.70, 0.90],
                "brier": sum(micro_ci) / 2.0,
                "brier_ci95": list(micro_ci),
                "n": 900,
                "pr_auc": 0.60,
            },
            "arms": {
                arm: {
                    "auc": 0.75,
                    "auc_ci95": [0.65, 0.85],
                    "brier": small_brier if arm == "HUMAN_REVIEW" else small_brier + 0.01,
                    "brier_ci95": [small_brier - 0.02, small_brier + 0.02],
                    "n": 180,
                    "pr_auc": 0.55,
                    "small_segment": False,
                }
                for arm in ARM_ORDER
            },
        }

    return {
        "per_arm": family(per_arm_micro_ci, per_arm_small_brier),
        "pooled": family(pooled_micro_ci, pooled_small_brier),
    }


def _synth_agreement(pooled=0.08, per_arm=0.10):
    return {
        "per_arm": {"arm_mean_abs_error": per_arm, "arms": {}, "label": "x"},
        "pooled": {"arm_mean_abs_error": pooled, "arms": {}, "label": "x"},
    }


def _synth_effect_contrasts(gated_gap=0.05, fatigue_gap=-5.0, band=0.40):
    gated_cell = {
        "attenuation_expected": False,
        "configured_effect_logit": 0.40,
        "estimated_cell_contrast_logit": 0.40 + gated_gap,
        "n_cell": 60,
        "recovery_gap_logit": gated_gap,
    }
    fatigue_cell = {
        "attenuation_expected": True,
        "configured_effect_logit": -0.25,
        "estimated_cell_contrast_logit": -0.25 + fatigue_gap,
        "n_cell": 90,
        "recovery_gap_logit": fatigue_gap,
    }
    return {
        "gate_band_logit": band,
        "note": "synthetic",
        "per_arm": {
            "interaction_cells": {
                "RETRY_LATER|attempt_number>=3": dict(fatigue_cell)
            },
            "main_effects": {},
        },
        "pooled": {
            "interaction_cells": {
                "RETRY_LATER|attempt_number>=3": dict(fatigue_cell),
                "RETRY_NOW|failure_category==temporary_decline": gated_cell,
            },
            "main_effects": {},
        },
    }


def _verdict_of(result):
    return result["preferred_model"]


# ---------------------------------------------------------------------------
# 1. Report structure contract
# ---------------------------------------------------------------------------


def test_top_level_sections_keys_labels_and_calibrated_kind(day6_world):
    report = day6_world["report"]

    assert set(report) == {
        "bootstrap",
        "bundle_kind",
        "complexity",
        "effect_contrasts",
        "ground_truth_agreement",
        "label",
        "n_randomized_test_rows",
        "predictive",
        "rule_application",
        "scope_note",
        "seed",
        "smallest_test_arm",
        "truth_label",
    }
    assert report["label"] == "OBSERVED SIMULATED OUTCOME"
    assert report["truth_label"] == "SIMULATED GROUND TRUTH"
    assert report["bundle_kind"] == "calibrated"
    assert report["seed"] == SEED
    # Independent recount: the report counts RANDOMIZED test rows only.
    assert report["n_randomized_test_rows"] == int(
        (
            (day6_world["test_obs"][STRATUM_COLUMN] == STRATUM_RANDOMIZED)
        ).sum()
    )
    assert report["n_randomized_test_rows"] < len(day6_world["test_obs"])
    assert "nothing causal" in report["scope_note"]
    assert report["rule_application"]["preferred_model"] in {"pooled", "per_arm"}


def test_predictive_section_shape_for_both_families(day6_world):
    predictive = day6_world["report"]["predictive"]

    assert set(predictive) == {"per_arm", "pooled"}
    for family in ("per_arm", "pooled"):
        assert set(predictive[family]) == {"micro", "arms"}
        assert set(predictive[family]["micro"]) == {
            "auc",
            "auc_ci95",
            "brier",
            "brier_ci95",
            "n",
            "pr_auc",
        }
        assert set(predictive[family]["arms"]) == set(ARM_ORDER)
        assert sum(
            predictive[family]["arms"][arm]["n"] for arm in ARM_ORDER
        ) == predictive[family]["micro"]["n"]
        for arm in ARM_ORDER:
            assert set(predictive[family]["arms"][arm]) == {
                "auc",
                "auc_ci95",
                "brier",
                "brier_ci95",
                "n",
                "pr_auc",
                "small_segment",
            }
    # Identical randomized test segment for both families (D-E1).
    assert predictive["per_arm"]["micro"]["n"] == predictive["pooled"]["micro"]["n"]
    assert predictive["per_arm"]["micro"]["n"] > 0


def test_rule_section_structure_criterion_order_and_invariant(day6_world):
    rule = day6_world["report"]["rule_application"]

    assert set(rule) == {"applies_to", "criteria", "preferred_model", "rule_id"}
    assert rule["rule_id"] == "D-E2"
    assert rule["applies_to"] == "calibrated_bundles_only"
    assert [entry["criterion"] for entry in rule["criteria"]] == list(CRITERION_NAMES)
    for entry in rule["criteria"]:
        assert set(entry) == {"criterion", "evidence", "passed"}
        assert isinstance(entry["passed"], bool)
        assert isinstance(entry["evidence"], dict)
    all_passed = all(entry["passed"] for entry in rule["criteria"])
    assert (rule["preferred_model"] == "pooled") == all_passed
    # Smallest-arm anchor: at fixture scale HUMAN_REVIEW is the thinnest
    # randomized test arm, matching D-E2 criterion 3's named example.
    assert rule is not None
    assert day6_world["report"]["smallest_test_arm"] == "HUMAN_REVIEW"


# ---------------------------------------------------------------------------
# 2. Pure rule-engine unit tests on synthetic metric dicts
# ---------------------------------------------------------------------------


def test_rule_all_criteria_pass_prefers_pooled():
    result = _apply_rule(
        _synth_predictive(),
        _synth_agreement(pooled=0.08, per_arm=0.10),
        _synth_effect_contrasts(gated_gap=0.05),
        "HUMAN_REVIEW",
    )

    assert _verdict_of(result) == "pooled"
    assert all(entry["passed"] for entry in result["criteria"])
    c1 = result["criteria"][0]
    assert c1["evidence"]["pooled_upper"] < c1["evidence"]["per_arm_lower"]


def test_rule_criterion1_fail_keeps_per_arm_reference():
    """Overlapping micro-Brier CIs: point ordering alone is insufficient."""
    result = _apply_rule(
        _synth_predictive(
            pooled_micro_ci=(0.10, 0.16), per_arm_micro_ci=(0.13, 0.15)
        ),
        _synth_agreement(pooled=0.08, per_arm=0.10),
        _synth_effect_contrasts(gated_gap=0.05),
        "HUMAN_REVIEW",
    )

    assert _verdict_of(result) == "per_arm"
    assert result["criteria"][0]["passed"] is False
    assert all(entry["passed"] for entry in result["criteria"][1:])


def test_rule_agreement_regression_blocks_pooled():
    result = _apply_rule(
        _synth_predictive(),
        _synth_agreement(pooled=0.20, per_arm=0.10),
        _synth_effect_contrasts(gated_gap=0.05),
        "HUMAN_REVIEW",
    )

    assert _verdict_of(result) == "per_arm"
    assert result["criteria"][1]["passed"] is False
    assert result["criteria"][0]["passed"] is True


def test_rule_smallest_arm_regression_blocks_pooled():
    result = _apply_rule(
        _synth_predictive(pooled_small_brier=0.30, per_arm_small_brier=0.14),
        _synth_agreement(pooled=0.08, per_arm=0.10),
        _synth_effect_contrasts(gated_gap=0.05),
        "HUMAN_REVIEW",
    )

    assert _verdict_of(result) == "per_arm"
    assert result["criteria"][2]["passed"] is False
    assert result["criteria"][2]["evidence"]["smallest_test_arm"] == "HUMAN_REVIEW"


def test_rule_interaction_band_failure_blocks_pooled():
    result = _apply_rule(
        _synth_predictive(),
        _synth_agreement(pooled=0.08, per_arm=0.10),
        _synth_effect_contrasts(gated_gap=0.90),
        "HUMAN_REVIEW",
    )

    assert _verdict_of(result) == "per_arm"
    assert result["criteria"][3]["passed"] is False


def test_rule_attenuated_fatigue_cell_is_never_gated():
    """The weak-negative late-stage cell carries attenuation_expected and is
    REPORTED, not gated (Day 5 semantics carried into D-E2 criterion 4);
    an enormous fatigue gap cannot flip the verdict by itself."""
    result = _apply_rule(
        _synth_predictive(),
        _synth_agreement(pooled=0.08, per_arm=0.10),
        _synth_effect_contrasts(gated_gap=0.05, fatigue_gap=-5.0),
        "HUMAN_REVIEW",
    )

    assert _verdict_of(result) == "pooled"
    assert result["criteria"][3]["passed"] is True
    assert result["criteria"][3]["evidence"]["annotated_not_gated"] == [
        "RETRY_LATER|attempt_number>=3"
    ]


def test_rule_nan_inputs_fail_closed_toward_per_arm():
    """Every criterion whose OWN inputs are non-finite fails closed toward
    the per-arm reference; criteria whose inputs stay finite still evaluate
    on their numbers (here criterion 3 sees favorable finite Briers)."""
    result = _apply_rule(
        _synth_predictive(
            pooled_micro_ci=(float("nan"), float("nan")),
        ),
        _synth_agreement(pooled=float("nan"), per_arm=0.10),
        _synth_effect_contrasts(gated_gap=float("nan")),
        "HUMAN_REVIEW",
    )

    assert _verdict_of(result) == "per_arm"
    assert [entry["passed"] for entry in result["criteria"]] == [
        False,
        False,
        True,
        False,
    ]


# ---------------------------------------------------------------------------
# 3. Fixture-scale real run: finiteness, CI bracketing, section contents
# ---------------------------------------------------------------------------


def test_real_run_metrics_finite_and_cis_bracket_point_estimates(day6_world):
    predictive = day6_world["report"]["predictive"]

    for family in ("per_arm", "pooled"):
        micro = predictive[family]["micro"]
        assert math.isfinite(micro["auc"])
        assert math.isfinite(micro["brier"])
        assert micro["brier_ci95"][0] <= micro["brier"] <= micro["brier_ci95"][1]
        assert micro["auc_ci95"][0] <= micro["auc"] <= micro["auc_ci95"][1]
        for arm in ARM_ORDER:
            entry = predictive[family]["arms"][arm]
            assert entry["n"] > 0
            assert math.isfinite(entry["brier"])
            assert entry["brier_ci95"][0] <= entry["brier"] <= entry["brier_ci95"][1]


def test_real_run_agreement_and_effect_contrasts_contents(day6_world):
    report = day6_world["report"]
    agreement = report["ground_truth_agreement"]
    contrasts = report["effect_contrasts"]

    assert agreement["truth_label"] == "SIMULATED GROUND TRUTH"
    for family in ("per_arm", "pooled"):
        assert math.isfinite(agreement[family]["arm_mean_abs_error"])
        assert agreement[family]["arm_mean_abs_error"] > 0.0
        for arm in ARM_ORDER:
            entry = agreement[family]["arms"][arm]
            assert math.isfinite(entry["mean_abs_error_vs_integrated_true"])
            assert math.isfinite(entry["pearson_r"])
            assert math.isfinite(entry["spearman_rho"])

    assert contrasts["gate_band_logit"] == pytest.approx(0.40)
    for family in ("per_arm", "pooled"):
        cells = contrasts[family]["interaction_cells"]
        assert set(cells) == {
            "RETRY_NOW|failure_category==temporary_decline",
            "RETRY_LATER|attempt_number>=3",
        }
        assert cells["RETRY_LATER|attempt_number>=3"]["attenuation_expected"] is True
        assert cells["RETRY_NOW|failure_category==temporary_decline"][
            "attenuation_expected"
        ] is False
        assert cells["RETRY_LATER|attempt_number>=3"]["n_cell"] > 0
        assert cells["RETRY_NOW|failure_category==temporary_decline"]["n_cell"] > 0
        for arm in ARM_ORDER:
            main = contrasts[family]["main_effects"][arm]
            assert math.isfinite(main["estimated_contrast_logit"])
            assert math.isfinite(main["gap_logit"])


def test_complexity_records_timed_refit_and_parameter_proxies(day6_world):
    complexity = day6_world["report"]["complexity"]

    assert complexity["families"]["per_arm"]["fits"] == 5
    assert complexity["families"]["pooled"]["fits"] == 1
    for family in ("per_arm", "pooled"):
        seconds = complexity["families"][family]["fit_seconds"]
        assert isinstance(seconds, float)
        assert seconds > 0.0
    per_pipeline = complexity["families"]["per_arm"]["parameter_count_proxy"][
        "per_pipeline"
    ]
    pooled_proxy = complexity["families"]["pooled"]["parameter_count_proxy"]
    assert per_pipeline["n_estimators"] == 300
    assert per_pipeline["max_depth"] == 4
    assert complexity["families"]["per_arm"]["parameter_count_proxy"]["total"] == (
        5 * 300 * 4
    )
    assert pooled_proxy["per_pipeline"] == per_pipeline
    assert pooled_proxy["total"] == 300 * 4
    # Documented capacity PROXY language, never an exact node-count claim.
    assert "proxy" in complexity["parameter_count_note"]


# ---------------------------------------------------------------------------
# 4. Determinism, purity of the caller's frame, calibrated-bundle enforcement
# ---------------------------------------------------------------------------


def _strip_documented_wall_clock(report):
    stripped = json.loads(json.dumps(report, allow_nan=True))
    for family in ("per_arm", "pooled"):
        stripped["complexity"]["families"][family].pop("fit_seconds")
    return stripped


def test_two_compare_calls_identical_after_dropping_fit_seconds(day6_world):
    world = day6_world

    second = compare_models(
        world["per_arm_cal"],
        world["pooled_cal"],
        world["baseline"],
        world["test_obs"],
        POLICY,
        seed=SEED,
        train_frame=world["train_obs"],
        validation_frame=world["validation_obs"],
    )

    left = json.dumps(
        _strip_documented_wall_clock(world["report"]), sort_keys=True, allow_nan=True
    )
    right = json.dumps(_strip_documented_wall_clock(second), sort_keys=True, allow_nan=True)
    assert left == right


def test_compare_models_rejects_uncalibrated_bundles_naming_the_family(day6_world):
    world = day6_world

    with pytest.raises(ValueError) as per_arm_error:
        compare_models(
            world["per_arm_raw"],
            world["pooled_cal"],
            world["baseline"],
            world["test_obs"],
            POLICY,
            seed=SEED,
        )
    assert "per_arm" in str(per_arm_error.value)
    assert "calibrated" in str(per_arm_error.value)

    with pytest.raises(ValueError) as pooled_error:
        compare_models(
            world["per_arm_cal"],
            world["pooled_raw"],
            world["baseline"],
            world["test_obs"],
            POLICY,
            seed=SEED,
        )
    assert "pooled" in str(pooled_error.value)
    assert "calibrated" in str(pooled_error.value)


def test_compare_models_never_mutates_the_test_frame(day6_world):
    pd.testing.assert_frame_equal(day6_world["test_obs"], day6_world["test_snapshot"])
    assert list(day6_world["test_obs"].dtypes) == list(
        day6_world["test_snapshot"].dtypes
    )


# ---------------------------------------------------------------------------
# 5. Purity: import whitelist, one seeded rng derivation, token bans, labels
# ---------------------------------------------------------------------------

# DEVIATION NOTE (documented in the module docstring too): the Task 3 contract
# requires MEASURED fit wall-clock seconds in the complexity section, which
# needs stdlib ``time`` (via ``from time import perf_counter``). Every other
# root matches the pinned whitelist; dataclasses is unused here.
ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "numpy", "pandas", "time", "ml", "simulation"}
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


def test_model_comparison_import_roots_match_whitelist_exactly():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )


def test_exactly_one_seeded_default_rng_derivation():
    code = _source_without_docstring()

    occurrences = [match.start() for match in re.finditer(r"default_rng", code)]
    assert len(occurrences) == 1, (
        "the pooled-family bootstrap is the ONLY sanctioned randomness "
        "consumer; it must derive its generator once from the named seed"
    )
    assert re.search(r"np\.random\.default_rng\(\s*seed\s*\)", code) is not None


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_no_wall_clock_or_stdlib_randomness_token(pattern):
    code = _source_without_docstring()

    assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"


def test_docstring_embeds_labels_gate_and_nothing_causal():
    docstring = ast.get_docstring(ast.parse(SOURCE_PATH.read_text(encoding="utf-8")))
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert docstring is not None
    assert "OBSERVED SIMULATED OUTCOME" in docstring
    assert "SIMULATED GROUND TRUTH" in docstring
    assert "D-E2" in docstring
    assert "per-arm" in docstring
    assert "reference" in docstring
    assert "nothing causal" in docstring
    assert "causal estimate" not in source.lower()


# ---------------------------------------------------------------------------
# 6. Standalone protocols: learning curves (tiny fractions) + stability (2 seeds)
# ---------------------------------------------------------------------------


def test_learning_curves_structure_on_tiny_fractions(day6_world):
    world = day6_world

    curves = learning_curves(
        world["train_obs"],
        world["validation_obs"],
        world["test_obs"],
        fractions=(0.5, 1.0),
        seed=SEED,
    )

    assert set(curves) == {
        "curves",
        "fractions",
        "label",
        "protocol_note",
        "smallest_test_arm",
    }
    assert curves["label"] == "OBSERVED SIMULATED OUTCOME"
    assert curves["fractions"] == [0.5, 1.0]
    assert curves["smallest_test_arm"] == "HUMAN_REVIEW"
    assert "TRAIN segment" in curves["protocol_note"]
    assert "FULL validation" in curves["protocol_note"]
    assert set(curves["curves"]) == {"0.50", "1.00"}
    previous_rows = -1
    for key in ("0.50", "1.00"):
        point_block = curves["curves"][key]
        assert point_block["n_train_rows_used"] > previous_rows
        previous_rows = point_block["n_train_rows_used"]
        for family in ("per_arm", "pooled"):
            point = point_block[family]
            assert set(point) == {
                "micro_brier",
                "smallest_arm",
                "smallest_arm_brier",
                "smallest_arm_n",
            }
            assert 0.0 < point["micro_brier"] < 1.0
            assert 0.0 < point["smallest_arm_brier"] < 1.0
            assert point["smallest_arm"] == "HUMAN_REVIEW"
            assert point["smallest_arm_n"] > 0


def test_stability_check_structure_and_manual_sd_crosscheck(day6_world):
    world = day6_world

    stability = stability_check(
        world["train_obs"],
        world["validation_obs"],
        world["test_obs"],
        seeds=(SEED, 1),
    )

    assert set(stability) == {
        "incremental_note",
        "label",
        "metric_note",
        "per_seed",
        "seeds",
        "stability",
    }
    assert stability["label"] == "OBSERVED SIMULATED OUTCOME"
    assert stability["seeds"] == [SEED, 1]
    assert set(stability["per_seed"]) == {str(SEED), "1"}
    for seed_key in (str(SEED), "1"):
        for family in ("per_arm", "pooled"):
            entry = stability["per_seed"][seed_key][family]
            assert math.isfinite(entry["micro_brier"])
            assert set(entry["mean_incremental_by_arm"]) == set(TREATED_ARMS)
            for arm in TREATED_ARMS:
                assert math.isfinite(entry["mean_incremental_by_arm"][arm])

    for family in ("per_arm", "pooled"):
        values = [
            stability["per_seed"][key][family]["micro_brier"]
            for key in (str(SEED), "1")
        ]
        expected = float(np.std(np.asarray(values, dtype=float), ddof=0))
        reported = stability["stability"]["micro_brier_sd"][family]
        assert reported == pytest.approx(expected, abs=0.0)
        for arm in TREATED_ARMS:
            increments = [
                stability["per_seed"][key][family]["mean_incremental_by_arm"][arm]
                for key in (str(SEED), "1")
            ]
            expected_arm = float(
                np.std(np.asarray(increments, dtype=float), ddof=0)
            )
            assert stability["stability"]["mean_incremental_sd_by_arm"][family][
                arm
            ] == pytest.approx(expected_arm, abs=0.0)
