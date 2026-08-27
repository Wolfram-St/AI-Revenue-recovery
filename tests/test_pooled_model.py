"""Tests for the Day 6 pooled comparison-candidate model (plan Task 2, D-E1/D-E2)."""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.action_model import ARM_ORDER as ACTION_MODEL_ARM_ORDER
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml.pooled_model import (
    ACTION_COLUMN,
    ARM_ORDER,
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    TARGET_COLUMN,
    PooledModelBundle,
    calibrate_pooled_model,
    predict_pooled_all_actions,
    predict_pooled_probability,
    train_pooled_model,
)
from ml.train import predict_recovery_probability, train_baseline
from simulation.config import load_treatment_policy
from simulation.observations import assemble_observations, split_observations

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "ml" / "pooled_model.py"

# Plan Task 2 fixture sizing: the canonical 1500-attempt world with RAW
# baseline probabilities, mirroring the Day 5 fixture pattern. The canonical
# CALIBRATED pooled-vs-per-arm comparison lands in Task 3 (model_comparison).
ATTEMPT_ROWS = 1500
SEED = 20260826


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
    bundle, metadata = train_pooled_model(train_obs, validation_obs, seed=SEED)
    return bundle, metadata


@pytest.fixture(scope="module")
def calibrated_bundle(observation_bundle, trained_bundle):
    """Raw pooled bundle after one full sigmoid calibration pass on validation."""
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle
    return calibrate_pooled_model(bundle, validation_obs)


def _randomized_count(frame: pd.DataFrame) -> int:
    return int((frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED).sum())


# ---------------------------------------------------------------------------
# 1. Bundle shape: one shared pipeline, canonical arms, frozen dataclass
# ---------------------------------------------------------------------------


def test_bundle_is_a_frozen_dataclass_holding_one_shared_model(trained_bundle):
    bundle, _ = trained_bundle

    assert isinstance(bundle, PooledModelBundle)
    assert bundle.arms == ARM_ORDER
    assert bundle.model is not None

    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.arms = ARM_ORDER  # any attribute rebinding must fail


def test_arm_order_is_the_day5_reference_constant_not_a_drifted_copy():
    """Drift tripwire: ARM_ORDER is duplicated across the two model modules
    on purpose (the purity whitelists forbid pooled_model importing
    action_model), so this cross-module comparison -- allowed only here in a
    test -- must keep the two constants byte-identical."""
    assert ARM_ORDER == ACTION_MODEL_ARM_ORDER


# ---------------------------------------------------------------------------
# 2. Metadata contract: keys, values, independent randomized recount
# ---------------------------------------------------------------------------


def test_metadata_contract_keys_and_scalar_values(trained_bundle):
    _, metadata = trained_bundle

    assert set(metadata) == {
        "model_family",
        "seed",
        "train_rows",
        "validation_rows",
        "feature_names",
        "arm_rows",
        "pooled",
    }
    assert metadata["model_family"] == "pooled_xgboost"
    assert metadata["seed"] == SEED
    assert metadata["feature_names"] == (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + [ACTION_COLUMN]
    )
    # Per-arm small_segments flags are NOT applicable to a single pooled fit;
    # the metadata records the pooled shape explicitly instead.
    assert metadata["pooled"] is True
    assert "small_segments" not in metadata


def test_arm_rows_metadata_matches_independent_randomized_arm_recounts(
    observation_bundle, trained_bundle
):
    """Per-arm randomized counts feed D-E2 criterion 3 (smallest-arm
    comparison), so they must equal a fully independent recount and sum to
    the pooled totals."""
    _, train_obs, validation_obs, _ = observation_bundle
    _, metadata = trained_bundle

    expected = {}
    for arm in ARM_ORDER:
        expected[arm] = {
            "train": int(
                (
                    (train_obs[STRATUM_COLUMN] == STRATUM_RANDOMIZED)
                    & (train_obs[ACTION_COLUMN] == arm)
                ).sum()
            ),
            "validation": int(
                (
                    (validation_obs[STRATUM_COLUMN] == STRATUM_RANDOMIZED)
                    & (validation_obs[ACTION_COLUMN] == arm)
                ).sum()
            ),
        }

    assert metadata["arm_rows"] == expected
    assert sum(cell["train"] for cell in metadata["arm_rows"].values()) == (
        metadata["train_rows"]
    )
    assert sum(
        cell["validation"] for cell in metadata["arm_rows"].values()
    ) == metadata["validation_rows"]


def test_metadata_row_counts_match_independent_randomized_recounts(
    observation_bundle, trained_bundle
):
    _, train_obs, validation_obs, _ = observation_bundle
    _, metadata = trained_bundle

    assert metadata["train_rows"] == _randomized_count(train_obs)
    assert metadata["validation_rows"] == _randomized_count(validation_obs)


def test_train_rows_equal_randomized_intersection_not_frame_size(
    observation_bundle, trained_bundle
):
    _, train_obs, _, _ = observation_bundle
    _, metadata = trained_bundle

    randomized_total = _randomized_count(train_obs)

    assert metadata["train_rows"] == randomized_total
    assert randomized_total < len(train_obs)


def test_metadata_is_deterministic_across_runs(observation_bundle, trained_bundle):
    _, train_obs, validation_obs, _ = observation_bundle
    _, metadata = trained_bundle

    _, remetadata = train_pooled_model(train_obs, validation_obs, seed=SEED)

    assert remetadata == metadata


# ---------------------------------------------------------------------------
# 3. Reproducibility, bounds, lengths
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_identical_predictions(observation_bundle, trained_bundle):
    _, train_obs, validation_obs, probe = observation_bundle
    bundle, _ = trained_bundle
    rebundle, _ = train_pooled_model(train_obs, validation_obs, seed=SEED)

    for arm in ARM_ORDER:
        np.testing.assert_array_equal(
            predict_pooled_probability(bundle, probe, arm),
            predict_pooled_probability(rebundle, probe, arm),
        )


def test_predictions_are_finite_unit_interval_with_correct_lengths(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    for arm in ARM_ORDER:
        predictions = predict_pooled_probability(bundle, probe, arm)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(probe)
        assert np.isfinite(predictions).all()
        assert (predictions >= 0.0).all()
        assert (predictions <= 1.0).all()


def test_calibrated_probabilities_are_finite_unit_interval_with_correct_lengths(
    observation_bundle, calibrated_bundle
):
    _, _, _, probe = observation_bundle

    assert calibrated_bundle.arms == ARM_ORDER
    for arm in ARM_ORDER:
        predictions = predict_pooled_probability(calibrated_bundle, probe, arm)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(probe)
        assert np.isfinite(predictions).all()
        assert (predictions >= 0.0).all()
        assert (predictions <= 1.0).all()


# ---------------------------------------------------------------------------
# 4. All-actions surface: deterministic ARM_ORDER columns matching queries
# ---------------------------------------------------------------------------


def test_predict_all_actions_columns_follow_arm_order_exactly(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    table = predict_pooled_all_actions(bundle, probe)

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == list(ARM_ORDER)
    assert len(table) == len(probe)


def test_predict_all_actions_matches_single_action_predictions_per_row(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    table = predict_pooled_all_actions(bundle, probe)

    for arm in ARM_ORDER:
        np.testing.assert_array_equal(
            table[arm].to_numpy(),
            predict_pooled_probability(bundle, probe, arm),
        )


def test_predict_all_actions_is_deterministic(observation_bundle, trained_bundle):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    pd.testing.assert_frame_equal(
        predict_pooled_all_actions(bundle, probe),
        predict_pooled_all_actions(bundle, probe),
    )


# ---------------------------------------------------------------------------
# 5. Counterfactual semantics: arm overwrite drives the prediction
# ---------------------------------------------------------------------------


def test_counterfactual_queries_under_different_actions_diverge_materially(
    observation_bundle, trained_bundle
):
    """Same contexts scored under two different requested arms must yield
    genuinely different probability vectors -- otherwise the appended arm
    feature is decorative and the pooled model cannot rank actions."""
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    left = predict_pooled_probability(bundle, probe, "RETRY_NOW")
    right = predict_pooled_probability(bundle, probe, "HUMAN_REVIEW")

    divergence = float(np.mean(np.abs(left - right)))
    assert divergence > 0.01, (
        f"counterfactual arm overwrite changed predictions by only "
        f"{divergence:.4f} on average"
    )


def test_same_action_counterfactual_query_is_byte_identical_on_repeat(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    np.testing.assert_array_equal(
        predict_pooled_probability(bundle, probe, "RETRY_NOW"),
        predict_pooled_probability(bundle, probe, "RETRY_NOW"),
    )


def test_predict_calls_never_mutate_the_context_frame(
    observation_bundle, trained_bundle
):
    """Values AND dtypes of the caller's frame survive every predict path."""
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle
    snapshot = probe.copy(deep=True)

    predict_pooled_probability(bundle, probe, "RETRY_NOW")
    predict_pooled_probability(bundle, probe, "HUMAN_REVIEW")
    predict_pooled_all_actions(bundle, probe)

    pd.testing.assert_frame_equal(probe, snapshot)
    assert list(probe.dtypes) == list(snapshot.dtypes)


# ---------------------------------------------------------------------------
# 6. Calibration: raw untouched, exact metadata block, bounded outputs
# ---------------------------------------------------------------------------


def test_calibration_leaves_raw_bundle_predictions_byte_identical(
    observation_bundle, trained_bundle
):
    """FrozenEstimator seals the raw pipeline (never refits it), so the input
    bundle's predictions and metadata must survive a calibration call
    byte-for-byte."""
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle
    probe = validation_obs
    before = predict_pooled_probability(bundle, probe, "RETRY_NOW").copy()

    calibrated = calibrate_pooled_model(bundle, validation_obs)

    np.testing.assert_array_equal(
        before, predict_pooled_probability(bundle, probe, "RETRY_NOW")
    )
    assert calibrated is not bundle
    assert "calibration" not in bundle.metadata


def test_calibration_metadata_records_exact_block_over_an_independent_recount(
    observation_bundle, trained_bundle
):
    _, _, validation_obs, _ = observation_bundle
    bundle, _ = trained_bundle

    calibrated = calibrate_pooled_model(bundle, validation_obs)

    assert set(calibrated.metadata) == set(bundle.metadata) | {"calibration"}
    # Independent recount of the randomized validation rows feeding the
    # single pooled sigmoid calibrator.
    assert calibrated.metadata["calibration"] == {
        "method": "sigmoid",
        "rows": _randomized_count(validation_obs),
        "fit_on": "validation_randomized_only",
    }


# ---------------------------------------------------------------------------
# 7. Rejection behavior: unknown action, zero randomized rows, missing columns
# ---------------------------------------------------------------------------


def test_unknown_action_raises_naming_it_and_available_arms(
    observation_bundle, trained_bundle
):
    _, _, _, probe = observation_bundle
    bundle, _ = trained_bundle

    with pytest.raises(ValueError) as error:
        predict_pooled_probability(bundle, probe, "RETRY_FOREVER")

    message = str(error.value)
    assert "RETRY_FOREVER" in message
    for arm in ARM_ORDER:
        assert arm in message


def test_zero_randomized_training_rows_raise_value_error(observation_bundle):
    assembled, _, _, _ = observation_bundle
    censored_only = assembled.loc[
        assembled[STRATUM_COLUMN] == "safety_censored"
    ].reset_index(drop=True)
    assert len(censored_only) > 0
    train_obs, validation_obs, _ = split_observations(censored_only)
    assert _randomized_count(train_obs) == 0

    with pytest.raises(ValueError, match="randomized"):
        train_pooled_model(train_obs, validation_obs, seed=SEED)


def test_randomized_rows_missing_for_one_arm_raise_value_error_naming_it(
    observation_bundle,
):
    """A pooled fit that silently skipped an arm would hand downstream
    consumers an all-zero one-hot encoding for counterfactual queries of
    that arm -- the guard must name every arm absent from the randomized
    training pool, and ONLY those arms. Safety-censored rows of the removed
    arm deliberately stay behind: they are excluded from fits anyway and
    must not satisfy the presence guard."""
    assembled, _, _, _ = observation_bundle
    keep = (assembled[ACTION_COLUMN] != "HUMAN_REVIEW") | (
        assembled[STRATUM_COLUMN] != STRATUM_RANDOMIZED
    )
    reduced = assembled.loc[keep].reset_index(drop=True)
    assert (
        int(
            (
                (reduced[STRATUM_COLUMN] == STRATUM_RANDOMIZED)
                & (reduced[ACTION_COLUMN] == "HUMAN_REVIEW")
            ).sum()
        )
        == 0
    )
    train_obs, validation_obs, _ = split_observations(reduced)

    with pytest.raises(ValueError) as error:
        train_pooled_model(train_obs, validation_obs, seed=SEED)

    message = str(error.value)
    assert "missing randomized rows for arm(s)" in message
    assert "HUMAN_REVIEW" in message
    for arm in ARM_ORDER:
        if arm != "HUMAN_REVIEW":
            assert arm not in message, "only genuinely absent arms may be named"


def test_missing_required_columns_raise_value_error_naming_them(
    observation_bundle,
):
    assembled, _, validation_obs, _ = observation_bundle

    with pytest.raises(ValueError, match="simulated_recovered"):
        train_pooled_model(
            assembled.drop(columns=[TARGET_COLUMN]), validation_obs, seed=SEED
        )
    with pytest.raises(ValueError, match="assigned_action"):
        train_pooled_model(
            assembled, validation_obs.drop(columns=[ACTION_COLUMN]), seed=SEED
        )
    with pytest.raises(ValueError, match="stratum"):
        train_pooled_model(
            assembled.drop(columns=[STRATUM_COLUMN]), validation_obs, seed=SEED
        )


# ---------------------------------------------------------------------------
# 8. Wrong-label purity: the fit tracks simulated_recovered, never recovered
# ---------------------------------------------------------------------------


def test_pooled_fit_tracks_simulated_recovered_never_the_recovered_column(
    observation_bundle,
):
    """With simulated_recovered := 1 - recovered on EVERY row, a fit that
    secretly consumed the forbidden Day-1 ``recovered`` label would produce
    predictions pointing the opposite way from an honest fit of the declared
    target; material divergence plus opposing orderings expose any drift."""
    assembled, _, _, _ = observation_bundle

    wrong_label_frame = assembled.copy()
    wrong_label_frame[TARGET_COLUMN] = 1 - wrong_label_frame["recovered"].astype(int)
    flipped_frame = wrong_label_frame.copy()
    flipped_frame[TARGET_COLUMN] = 1 - flipped_frame[TARGET_COLUMN]

    assert (
        wrong_label_frame["recovered"].astype(int) != wrong_label_frame[TARGET_COLUMN]
    ).all()
    assert (
        flipped_frame["recovered"].astype(int) == flipped_frame[TARGET_COLUMN]
    ).all()

    wrong_bundle, wrong_metadata = train_pooled_model(
        wrong_label_frame, wrong_label_frame, seed=SEED
    )
    flipped_bundle, flipped_metadata = train_pooled_model(
        flipped_frame, flipped_frame, seed=SEED
    )
    assert wrong_metadata["train_rows"] == flipped_metadata["train_rows"]

    contexts = wrong_label_frame
    recovered = wrong_label_frame["recovered"].astype(int).to_numpy()
    wrong_predictions = predict_pooled_probability(wrong_bundle, contexts, "RETRY_NOW")
    flipped_predictions = predict_pooled_probability(
        flipped_bundle, contexts, "RETRY_NOW"
    )

    materially_different = float(
        np.mean(np.abs(wrong_predictions - flipped_predictions))
    )
    assert materially_different > 0.02

    assert (
        wrong_predictions[recovered == 1].mean()
        < wrong_predictions[recovered == 0].mean()
    ), "pooled fit drifted toward the forbidden 'recovered' label"
    assert (
        flipped_predictions[recovered == 1].mean()
        > flipped_predictions[recovered == 0].mean()
    ), "pooled fit failed to track its own simulated_recovered target"


# ---------------------------------------------------------------------------
# 9. Stratum exclusion: safety_censored rows never enter the pooled fit
# ---------------------------------------------------------------------------


def test_duplicating_safety_censored_rows_leaves_fit_byte_identical(
    observation_bundle, trained_bundle
):
    _, train_obs, validation_obs, probe = observation_bundle
    bundle, metadata = trained_bundle

    censored = train_obs.loc[train_obs[STRATUM_COLUMN] == "safety_censored"]
    assert len(censored) > 0
    doubled_train = pd.concat([train_obs, censored], ignore_index=True)
    assert int((doubled_train[STRATUM_COLUMN] == "safety_censored").sum()) == (
        2 * len(censored)
    )

    doubled_bundle, doubled_metadata = train_pooled_model(
        doubled_train, validation_obs, seed=SEED
    )

    assert doubled_metadata["train_rows"] == metadata["train_rows"]
    np.testing.assert_array_equal(
        predict_pooled_probability(doubled_bundle, probe, "RETRY_NOW"),
        predict_pooled_probability(bundle, probe, "RETRY_NOW"),
    )
    np.testing.assert_array_equal(
        predict_pooled_all_actions(doubled_bundle, probe).to_numpy(),
        predict_pooled_all_actions(bundle, probe).to_numpy(),
    )


# ---------------------------------------------------------------------------
# 10. Training-input immutability
# ---------------------------------------------------------------------------


def test_training_frames_are_never_mutated(observation_bundle):
    _, train_obs, validation_obs, _ = observation_bundle
    train_snapshot = train_obs.copy(deep=True)
    validation_snapshot = validation_obs.copy(deep=True)

    train_pooled_model(train_obs, validation_obs, seed=SEED)

    pd.testing.assert_frame_equal(train_obs, train_snapshot)
    pd.testing.assert_frame_equal(validation_obs, validation_snapshot)


# ---------------------------------------------------------------------------
# 11. Purity: exact import roots, no local randomness, no wall clock
# ---------------------------------------------------------------------------

# The pooled pipeline cannot reuse ml.train._build_pipeline (its categorical
# block needs the extra assigned_action column), so local sklearn/xgboost
# construction is required and BOTH roots are whitelisted explicitly.
# Everything else stays as strict as the Day 5 whitelist: no recovery/*, no
# simulator composition, no stdlib randomness, no wall clock.
ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "numpy", "pandas", "dataclasses", "ml", "sklearn", "xgboost"}
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


def test_pooled_model_import_roots_match_whitelist_exactly():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )
    assert "recovery" not in roots, "pooled model must not depend on recovery/*"
    assert "simulation" not in roots, (
        "training must consume the assembled frame, never compose the simulator"
    )


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
