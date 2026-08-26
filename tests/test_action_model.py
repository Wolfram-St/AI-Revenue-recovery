"""Tests for Day 5 per-arm action-aware recovery model training + prediction."""

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
from ml.action_model import (
    ACTION_COLUMN,
    ARM_ORDER,
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    TARGET_COLUMN,
    ActionModelBundle,
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

ATTEMPT_ROWS = 1500
SEED = 20260826


@pytest.fixture(scope="module")
def observation_bundle():
    """Full chain artifact: 1500 attempts -> baseline -> observations -> splits."""
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
    assert ("train", "HUMAN_REVIEW") in metadata["small_segments"]


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
