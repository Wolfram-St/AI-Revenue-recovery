"""Tests for Day 5 per-arm action-aware recovery model training + prediction."""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.action_model import (
    ACTION_COLUMN,
    ARM_ORDER,
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    TARGET_COLUMN,
    ActionModelBundle,
    calibrate_action_models,
    predict_action_probability,
    predict_all_actions,
    train_action_models,
)
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml.train import predict_recovery_probability, train_baseline
from simulation.config import load_treatment_policy
from simulation.observations import assemble_observations, split_observations

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "ml" / "action_model.py"

# F1 (review): raised from 1500 so the stricter MIN_BRIER_HALF_ROWS=30 Brier
# floor stays achievable. At 1500 attempts the largest randomized validation
# arm held only 58 rows (29 per parity half), so a 30-row floor would have
# skipped EVERY arm and left the calibration sanity gate with zero coverage.
ATTEMPT_ROWS = 3000
SEED = 20260826

# Honest Monte-Carlo sizing for the remainder-subslice Brier difference
# (review F1). The gate statistic is the PAIRED difference
# delta = Brier_calibrated - Brier_raw over the SAME n remainder rows, whose
# standard error is sd(d)/sqrt(n) with d_i = (y-p_cal)^2 - (y-p_raw)^2.
# Measured across this fixture's evaluable arms: sd(d) ~= 0.17-0.30 depending
# on base rate, i.e. sd(delta) ~= 0.02-0.03 generically (0.023-0.048 as run).
# Scaling the worst measured spread to the enforced half-size floor gives
# sd(delta) <= 0.30/sqrt(30) ~= 0.055, so the plan's >=3x-sd rule demands a
# tolerance >= 3 * 0.055 ~= 0.164. The old flat 0.04 sat at ~1.25 sigma by the
# review's own arithmetic (~0.8-1.3 sigma measured here) -- below the
# requirement -- and was raised (never tightened) to the computed value.
BRIER_SIGMA_BUDGET = 3
MAX_MEASURED_PAIRED_ROW_SD = 0.30
MIN_BRIER_HALF_ROWS = 30
CALIBRATION_BRIER_TOLERANCE = float(
    np.ceil(
        BRIER_SIGMA_BUDGET
        * MAX_MEASURED_PAIRED_ROW_SD
        / np.sqrt(MIN_BRIER_HALF_ROWS)
        * 100
    )
    / 100
)  # == 0.17: the 3-sigma budget rounded UP to whole hundredths, never down


@pytest.fixture(scope="module")
def observation_bundle():
    """Full chain artifact: ATTEMPT_ROWS attempts -> baseline -> observations -> splits."""
    attempts = generate_dataset(ATTEMPT_ROWS, seed=42).reset_index(drop=True)
    train_df, validation_df, _ = chronological_split(attempts, 0.70, 0.15)
    baseline, _metadata = train_baseline(train_df, validation_df, seed=42)
    probabilities = [
        float(value) for value in predict_recovery_probability(baseline, attempts)
    ]
    assembled = assemble_observations(attempts, probabilities, POLICY)
    train_obs, validation_obs, test_obs = split_observations(assembled)
    return assembled, train_obs, validation_obs, test_obs


@pytest.fixture(scope="module")
def trained_bundle(observation_bundle):
    _, train_obs, validation_obs, _ = observation_bundle
    bundle, metadata = train_action_models(train_obs, validation_obs, seed=SEED)
    return bundle, metadata


@pytest.fixture(scope="module")
def calibrated_bundle(observation_bundle, trained_bundle):
    """Raw bundle after one full sigmoid calibration pass on validation."""
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle
    return calibrate_action_models(bundle, validation_obs)


def _randomized_arm_counts(frame: pd.DataFrame) -> dict[str, int]:
    randomized = frame.loc[frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]
    return {arm: int((randomized[ACTION_COLUMN] == arm).sum()) for arm in ARM_ORDER}


# ---------------------------------------------------------------------------
# 1. Bundle shape: five canonical arms, frozen dataclass, distinct pipelines
# ---------------------------------------------------------------------------


def test_bundle_trains_exactly_five_distinct_arm_pipelines(trained_bundle):
    bundle, _ = trained_bundle

    assert isinstance(bundle, ActionModelBundle)
    assert tuple(bundle.models) == ARM_ORDER
    assert bundle.arms == ARM_ORDER
    assert len({id(model) for model in bundle.models.values()}) == len(ARM_ORDER)


def test_bundle_is_a_frozen_dataclass(trained_bundle):
    bundle, _ = trained_bundle

    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.ars = ARM_ORDER  # typo-proof: any attribute rebinding must fail


# ---------------------------------------------------------------------------
# 2. Metadata contract: keys, values, independent row-count recomputation
# ---------------------------------------------------------------------------


def test_metadata_contract_keys_and_scalar_values(trained_bundle):
    _, metadata = trained_bundle

    assert set(metadata) == {
        "model_family",
        "seed",
        "train_rows",
        "validation_rows",
        "feature_names",
        "small_segments",
    }
    assert metadata["model_family"] == "per_arm_xgboost"
    assert metadata["seed"] == SEED
    assert metadata["feature_names"] == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_metadata_row_counts_match_independent_randomized_arm_counts(
    observation_bundle, trained_bundle
):
    _, train_obs, validation_obs, _ = observation_bundle
    _, metadata = trained_bundle

    assert metadata["train_rows"] == _randomized_arm_counts(train_obs)
    assert metadata["validation_rows"] == _randomized_arm_counts(validation_obs)


def test_small_segments_flag_every_randomized_arm_segment_below_100(
    observation_bundle, trained_bundle
):
    _, train_obs, validation_obs, _ = observation_bundle
    _, metadata = trained_bundle

    expected = sorted(
        (segment, arm)
        for segment, frame in (("train", train_obs), ("validation", validation_obs))
        for arm, count in _randomized_arm_counts(frame).items()
        if count < 100
    )

    assert sorted(metadata["small_segments"]) == expected
    # Fixture anchor (F1 note): ATTEMPT_ROWS grew 1500 -> 3000 to keep the
    # 30-row-per-half calibration gate evaluable, which lifts every TRAIN arm
    # segment past the 100-row threshold; the genuinely small segments are now
    # the thin validation slices, led by HUMAN_REVIEW (the very arm whose
    # validation slice cannot fill a Brier half at the new floor).
    assert ("validation", "HUMAN_REVIEW") in metadata["small_segments"]


# ---------------------------------------------------------------------------
# 3. Reproducibility: identical predictions and metadata under the same seed
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_identical_predictions_for_every_arm(
    observation_bundle, trained_bundle
):
    _, train_obs, validation_obs, probe = observation_bundle
    bundle, _ = trained_bundle
    rebundle, _ = train_action_models(train_obs, validation_obs, seed=SEED)

    for arm in ARM_ORDER:
        np.testing.assert_array_equal(
            predict_action_probability(bundle, probe, arm),
            predict_action_probability(rebundle, probe, arm),
        )


def test_metadata_dicts_are_deterministic_across_runs(
    observation_bundle, trained_bundle
):
    _, train_obs, validation_obs, _ = observation_bundle
    _, metadata = trained_bundle

    _, remetadata = train_action_models(train_obs, validation_obs, seed=SEED)

    assert remetadata == metadata


# ---------------------------------------------------------------------------
# 4. Prediction surface: bounds, lengths, column order, consistency
# ---------------------------------------------------------------------------


def test_predictions_are_finite_probabilities_in_unit_bounds(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    for arm in ARM_ORDER:
        predictions = predict_action_probability(bundle, probe, arm)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(probe)
        assert np.isfinite(predictions).all()
        assert (predictions >= 0.0).all()
        assert (predictions <= 1.0).all()


def test_predict_all_actions_columns_follow_arm_order_exactly(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    table = predict_all_actions(bundle, probe)

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == list(ARM_ORDER)
    assert len(table) == len(probe)


def test_predict_all_actions_matches_single_action_predictions_per_row(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    table = predict_all_actions(bundle, probe)

    for arm in ARM_ORDER:
        np.testing.assert_array_equal(
            table[arm].to_numpy(),
            predict_action_probability(bundle, probe, arm),
        )


def test_predict_all_actions_is_deterministic(observation_bundle, trained_bundle):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    pd.testing.assert_frame_equal(
        predict_all_actions(bundle, probe), predict_all_actions(bundle, probe)
    )


# ---------------------------------------------------------------------------
# 4b. Per-arm independence: behavioral divergence, not just object identity
# ---------------------------------------------------------------------------


def _pairwise_arm_pairs() -> list[tuple[str, str]]:
    return [
        (left, right)
        for index, left in enumerate(ARM_ORDER)
        for right in ARM_ORDER[index + 1 :]
    ]


def test_per_arm_prediction_columns_diverge_on_every_pairwise_comparison(
    observation_bundle, trained_bundle
):
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle
    probe = validation_obs.loc[
        validation_obs[STRATUM_COLUMN] == STRATUM_RANDOMIZED
    ].reset_index(drop=True)
    assert len(probe) > 0

    table = predict_all_actions(bundle, probe)

    assert len(_pairwise_arm_pairs()) == 10
    for left, right in _pairwise_arm_pairs():
        assert not np.array_equal(table[left].to_numpy(), table[right].to_numpy()), (
            f"arms {left} and {right} produced identical prediction vectors"
        )


def test_per_arm_classifiers_learned_pairwise_distinct_feature_importances(
    trained_bundle,
):
    bundle, _ = trained_bundle

    importances = {
        arm: bundle.models[arm].named_steps["classifier"].feature_importances_
        for arm in ARM_ORDER
    }

    assert all(len(vector) > 0 for vector in importances.values())
    for left, right in _pairwise_arm_pairs():
        assert not np.array_equal(importances[left], importances[right]), (
            f"arms {left} and {right} learned identical feature importances"
        )


# ---------------------------------------------------------------------------
# 5. Rejection behavior: unknown action, empty arm, missing columns
# ---------------------------------------------------------------------------


def test_unknown_action_raises_naming_it_and_available_arms(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    with pytest.raises(ValueError) as error:
        predict_action_probability(bundle, probe, "RETRY_FOREVER")

    message = str(error.value)
    assert "RETRY_FOREVER" in message
    for arm in ARM_ORDER:
        assert arm in message


def test_zero_randomized_row_arm_raises_value_error_naming_the_arm(
    observation_bundle,
):
    assembled, _, _, _ = observation_bundle
    reduced = assembled.loc[assembled[ACTION_COLUMN] != "RETRY_NOW"].reset_index(
        drop=True
    )
    train_obs, validation_obs, _ = split_observations(reduced)

    with pytest.raises(ValueError, match="RETRY_NOW"):
        train_action_models(train_obs, validation_obs, seed=SEED)


def test_missing_required_columns_raise_value_error_naming_them(
    observation_bundle,
):
    assembled, _, validation_obs, _ = observation_bundle

    with pytest.raises(ValueError, match="simulated_recovered"):
        train_action_models(
            assembled.drop(columns=[TARGET_COLUMN]), validation_obs, seed=SEED
        )
    with pytest.raises(ValueError, match="assigned_action"):
        train_action_models(
            assembled, validation_obs.drop(columns=[ACTION_COLUMN]), seed=SEED
        )
    with pytest.raises(ValueError, match="stratum"):
        train_action_models(
            assembled.drop(columns=[STRATUM_COLUMN]), validation_obs, seed=SEED
        )


# ---------------------------------------------------------------------------
# 6. Wrong-label purity: fits track simulated_recovered, never recovered
# ---------------------------------------------------------------------------


def test_models_track_simulated_recovered_and_never_the_recovered_column(
    observation_bundle,
):
    assembled, _, _, _ = observation_bundle

    wrong_label_frame = assembled.copy()
    wrong_label_frame[TARGET_COLUMN] = 1 - wrong_label_frame["recovered"].astype(int)
    flipped_frame = wrong_label_frame.copy()
    flipped_frame[TARGET_COLUMN] = 1 - flipped_frame[TARGET_COLUMN]

    assert (wrong_label_frame["recovered"].astype(int) != wrong_label_frame[TARGET_COLUMN]).all()
    assert (flipped_frame["recovered"].astype(int) == flipped_frame[TARGET_COLUMN]).all()

    wrong_bundle, wrong_metadata = train_action_models(
        wrong_label_frame, wrong_label_frame, seed=SEED
    )
    flipped_bundle, flipped_metadata = train_action_models(
        flipped_frame, flipped_frame, seed=SEED
    )
    assert wrong_metadata["train_rows"] == flipped_metadata["train_rows"]

    contexts = wrong_label_frame
    recovered = wrong_label_frame["recovered"].astype(int).to_numpy()

    for arm in ARM_ORDER:
        wrong_predictions = predict_action_probability(wrong_bundle, contexts, arm)
        flipped_predictions = predict_action_probability(flipped_bundle, contexts, arm)

        materially_different = float(
            np.mean(np.abs(wrong_predictions - flipped_predictions))
        )
        assert materially_different > 0.02

        assert (
            wrong_predictions[recovered == 1].mean()
            < wrong_predictions[recovered == 0].mean()
        ), f"{arm} fit drifted toward the forbidden 'recovered' label"
        assert (
            flipped_predictions[recovered == 1].mean()
            > flipped_predictions[recovered == 0].mean()
        ), f"{arm} fit failed to track its own simulated_recovered target"


# ---------------------------------------------------------------------------
# 7. Non-randomized exclusion: safety_censored rows never enter any fit
# ---------------------------------------------------------------------------


def test_train_rows_sum_equals_randomized_intersection_not_frame_size(
    observation_bundle, trained_bundle
):
    _, train_obs, _, _ = observation_bundle
    _, metadata = trained_bundle

    randomized_total = int((train_obs[STRATUM_COLUMN] == STRATUM_RANDOMIZED).sum())

    assert sum(metadata["train_rows"].values()) == randomized_total
    assert randomized_total < len(train_obs)


def test_duplicating_safety_censored_rows_leaves_fits_byte_identical(
    observation_bundle, trained_bundle
):
    _, train_obs, validation_obs, probe = observation_bundle
    bundle, metadata = trained_bundle

    censored = train_obs.loc[train_obs[STRATUM_COLUMN] == "safety_censored"]
    assert len(censored) > 0
    doubled_train = pd.concat([train_obs, censored], ignore_index=True)
    assert int((doubled_train[STRATUM_COLUMN] == "safety_censored").sum()) == 2 * len(
        censored
    )

    doubled_bundle, doubled_metadata = train_action_models(
        doubled_train, validation_obs, seed=SEED
    )

    assert doubled_metadata["train_rows"] == metadata["train_rows"]
    for arm in ARM_ORDER:
        np.testing.assert_array_equal(
            predict_action_probability(doubled_bundle, probe, arm),
            predict_action_probability(bundle, probe, arm),
        )


# ---------------------------------------------------------------------------
# 8. Purity: import whitelist, no local randomness, no wall clock, honest claims
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset({"__future__", "numpy", "pandas", "dataclasses", "ml"})

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


def test_action_model_import_roots_whitelisted_without_simulator_or_estimators():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )
    assert "simulation" not in roots, (
        "training must consume the assembled frame, never compose the simulator"
    )
    assert "sklearn" not in roots and "xgboost" not in roots, (
        "pipeline shape must be reused from the Day 2 builder, never redefined"
    )
    assert "recovery" not in roots, "action models must not depend on recovery/*"


def test_no_rng_derivation_exists_in_source():
    code = _source_without_docstring()

    assert code.count("default_rng") == 0, (
        "fitting must draw no randomness beyond the seeded estimator itself"
    )


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_no_wall_clock_or_stdlib_randomness_token(pattern):
    code = _source_without_docstring()

    assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"


def test_docstring_declares_synthetic_world_only_scope():
    docstring = ast.get_docstring(ast.parse(SOURCE_PATH.read_text(encoding="utf-8")))

    assert docstring is not None
    assert "synthetic-world-only" in docstring


def test_no_causal_estimate_claim_language_anywhere():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "causal estimate" not in source.lower()


# ---------------------------------------------------------------------------
# 9. Input immutability
# ---------------------------------------------------------------------------


def test_training_frames_are_never_mutated(observation_bundle):
    _, train_obs, validation_obs, _ = observation_bundle
    train_snapshot = train_obs.copy(deep=True)
    validation_snapshot = validation_obs.copy(deep=True)

    train_action_models(train_obs, validation_obs, seed=SEED)

    pd.testing.assert_frame_equal(train_obs, train_snapshot)
    pd.testing.assert_frame_equal(validation_obs, validation_snapshot)


# ---------------------------------------------------------------------------
# 10. Task 3: per-arm sigmoid calibration on validation randomized rows
# ---------------------------------------------------------------------------


def test_calibration_leaves_raw_bundle_predictions_byte_identical(
    observation_bundle, trained_bundle
):
    """FrozenEstimator seals each raw pipeline (never refits it), so the
    input bundle's predictions and metadata must survive a calibration call
    byte-for-byte."""
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle
    probe = validation_obs
    before = {
        arm: predict_action_probability(bundle, probe, arm).copy()
        for arm in ARM_ORDER
    }

    calibrate_action_models(bundle, validation_obs)

    for arm in ARM_ORDER:
        np.testing.assert_array_equal(
            before[arm], predict_action_probability(bundle, probe, arm)
        )
    assert "calibration" not in bundle.metadata


def test_calibrated_probabilities_are_finite_unit_interval_with_correct_lengths(
    observation_bundle, calibrated_bundle
):
    _, _, _, probe = observation_bundle

    assert calibrated_bundle.arms == ARM_ORDER
    assert tuple(calibrated_bundle.models) == ARM_ORDER
    for arm in ARM_ORDER:
        predictions = predict_action_probability(calibrated_bundle, probe, arm)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(probe)
        assert np.isfinite(predictions).all()
        assert (predictions >= 0.0).all()
        assert (predictions <= 1.0).all()


def test_calibration_metadata_records_rows_per_arm_with_exact_labels(
    observation_bundle, trained_bundle
):
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle

    calibrated = calibrate_action_models(bundle, validation_obs)

    assert calibrated is not bundle
    assert "calibration" not in bundle.metadata
    assert set(calibrated.metadata) == set(bundle.metadata) | {"calibration"}
    # Independent recount of randomized ∩ arm ∩ validation_frame rows.
    assert calibrated.metadata["calibration"] == {
        "method": "sigmoid",
        "rows": _randomized_arm_counts(validation_obs),
        "fit_on": "validation_randomized_only",
    }


def test_calibrated_brier_beats_or_matches_raw_on_validation_remainder_subslice(
    observation_bundle, trained_bundle
):
    """D-M4 remainder sanity: sigmoid calibration fitted on HALF of a
    validation arm slice must not degrade Brier on the OTHER half beyond
    finite-sample noise; the test segment is never touched.

    Tolerance derivation (restated honestly per review F1): both Brier scores
    are computed over the SAME remainder rows, so the sampled statistic is the
    PAIRED difference delta = Brier_calibrated - Brier_raw and its Monte-Carlo
    standard error is sd(d)/sqrt(n), d_i = (y - p_calibrated)^2 -
    (y - p_raw)^2. Measured across this fixture's evaluable arms:
    sd(d) ~= 0.17-0.30 depending on base rate, giving sd(delta) ~= 0.02-0.03
    generically (0.023-0.048 as run). Scaling the worst measured spread to the
    enforced half-size floor gives sd(delta) <= 0.30/sqrt(30) ~= 0.055 at
    n = MIN_BRIER_HALF_ROWS, so the plan's >=3x-sd rule requires a tolerance
    of at least 3 * 0.055 ~= 0.164: CALIBRATION_BRIER_TOLERANCE is exactly that
    budget rounded UP to 0.17. The previous flat 0.04 was ~1.25 sigma by the
    review's arithmetic (~0.8-1.3 sigma measured here) -- below requirement.

    Halves are cut by round-robin parity (even/odd positions) along the
    deterministic chronological row order -- no randomness involved. A plain
    first-half/second-half cut is deliberately avoided: the observation frame
    drifts mildly along event_timestamp (plan Task 6 documents exactly this
    eligibility drift), so a contiguous split would score the calibrator on
    temporal drift rather than on calibration quality.

    Skip profile on this fixture (graceful, by design): REQUEST_UPDATE (44
    randomized validation rows -> 22 per half) and HUMAN_REVIEW (31 -> 16/15)
    fall below the 30-row floor -- binomial noise at that size swamps any
    calibration signal -- while CONTROL (38/38), RETRY_NOW (54/54) and
    RETRY_LATER (40/40) are evaluated. As-run margins: deltas -0.0577 /
    -0.0146 / -0.1670 (strongly negative), sitting 7.6 / 8.2 / 7.0 measured
    sigma below the +0.17 ceiling, which is itself 5.7 / 7.5 / 3.6 sigma wide.
    """
    _, _, validation_obs, _ = observation_bundle
    raw_bundle, _ = trained_bundle
    randomized = validation_obs.loc[
        validation_obs[STRATUM_COLUMN] == STRATUM_RANDOMIZED
    ]

    evaluated = []
    skipped = []
    for arm in ARM_ORDER:
        arm_rows = (
            randomized.loc[randomized[ACTION_COLUMN] == arm]
            .reset_index(drop=True)
        )
        calibrate_half = arm_rows.iloc[np.arange(len(arm_rows)) % 2 == 0]
        remainder = arm_rows.iloc[np.arange(len(arm_rows)) % 2 == 1]
        if (
            len(calibrate_half) < MIN_BRIER_HALF_ROWS
            or len(remainder) < MIN_BRIER_HALF_ROWS
        ):
            skipped.append(arm)
            continue

        mini_raw = ActionModelBundle(
            models={arm: raw_bundle.models[arm]}, arms=(arm,), metadata={}
        )
        mini_calibrated = calibrate_action_models(mini_raw, calibrate_half)

        y_remainder = remainder[TARGET_COLUMN].astype(int).to_numpy()
        raw_brier = brier_score_loss(
            y_remainder,
            predict_action_probability(raw_bundle, remainder, arm),
        )
        calibrated_brier = brier_score_loss(
            y_remainder,
            predict_action_probability(mini_calibrated, remainder, arm),
        )
        assert calibrated_brier <= raw_brier + CALIBRATION_BRIER_TOLERANCE, (
            f"{arm}: remainder-subslice Brier degraded beyond the "
            f"{CALIBRATION_BRIER_TOLERANCE:.2f} tolerance (= {BRIER_SIGMA_BUDGET}"
            f"x-sd at the {MIN_BRIER_HALF_ROWS}-row half floor; review F1) "
            f"({calibrated_brier:.4f} vs raw {raw_brier:.4f})"
        )
        evaluated.append(arm)

    for arm in skipped:
        arm_rows = randomized.loc[randomized[ACTION_COLUMN] == arm]
        assert min(
            int(np.ceil(len(arm_rows) / 2)), len(arm_rows) // 2
        ) < MIN_BRIER_HALF_ROWS, (
            f"{arm} was skipped despite having enough rows in both halves"
        )
    assert len(evaluated) >= 2, (
        f"Brier sanity lost coverage; evaluated={evaluated}, skipped={skipped}"
    )


def test_calibration_tracks_simulated_recovered_never_the_recovered_column(
    observation_bundle, trained_bundle
):
    """Task 2 purity pattern lifted to calibration: with a fixture where
    recovered != simulated_recovered on EVERY row, a calibrator that consumed
    the forbidden Day-1 ``recovered`` label would produce predictions identical
    to one fitted on the true frame; material divergence proves the sigmoid
    fits track the simulator column explicitly (same frozen pipelines feed
    both wrappers, so only the calibration target can differ)."""
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle

    flipped = validation_obs.copy()
    flipped[TARGET_COLUMN] = 1 - flipped["recovered"].astype(int)
    assert (flipped["recovered"].astype(int) != flipped[TARGET_COLUMN]).all()

    true_calibrated = calibrate_action_models(bundle, validation_obs)
    flipped_calibrated = calibrate_action_models(bundle, flipped)

    probe = validation_obs
    for arm in ARM_ORDER:
        true_predictions = predict_action_probability(true_calibrated, probe, arm)
        flipped_predictions = predict_action_probability(
            flipped_calibrated, probe, arm
        )
        divergence = float(np.mean(np.abs(true_predictions - flipped_predictions)))
        assert divergence > 0.01, (
            f"{arm}: calibration ignored simulated_recovered "
            f"(mean abs divergence {divergence:.4f} <= 0.01)"
        )


def test_calibration_with_zero_row_arm_raises_value_error_naming_it(
    observation_bundle, trained_bundle
):
    assembled, _, _, _ = observation_bundle
    bundle, _ = trained_bundle
    reduced = assembled.loc[assembled[ACTION_COLUMN] != "RETRY_NOW"].reset_index(
        drop=True
    )
    _, validation_obs, _ = split_observations(reduced)

    with pytest.raises(ValueError, match="RETRY_NOW"):
        calibrate_action_models(bundle, validation_obs)


def test_calibration_requires_observation_columns_naming_them_missing(
    observation_bundle, trained_bundle
):
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle

    with pytest.raises(ValueError, match="simulated_recovered"):
        calibrate_action_models(
            bundle, validation_obs.drop(columns=[TARGET_COLUMN])
        )
    with pytest.raises(ValueError, match="assigned_action"):
        calibrate_action_models(
            bundle, validation_obs.drop(columns=[ACTION_COLUMN])
        )
    with pytest.raises(ValueError, match="stratum"):
        calibrate_action_models(
            bundle, validation_obs.drop(columns=[STRATUM_COLUMN])
        )


def test_calibration_is_deterministic_across_repeat_calls(
    observation_bundle, trained_bundle
):
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle
    probe = validation_obs

    first = calibrate_action_models(bundle, validation_obs)
    second = calibrate_action_models(bundle, validation_obs)

    assert first.metadata["calibration"] == second.metadata["calibration"]
    for arm in ARM_ORDER:
        np.testing.assert_array_equal(
            predict_action_probability(first, probe, arm),
            predict_action_probability(second, probe, arm),
        )
