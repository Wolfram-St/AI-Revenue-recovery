"""Per-arm action-aware recovery models for Day 5 (plan Tasks D-M1..D-M4).

Fits exactly one calibrated-ready binary pipeline PER ARM estimating
``P(recovered | decision-time context, action=a)`` for each canonical arm in
``ARM_ORDER``, using only the ``randomized`` stratum of the assembled Day 5
observation frame (plan decisions D-M1/D-M2): safety-censored rows are all
control by rule and are excluded from every fit so eligibility can never be
confused with treatment. The regression target is EXPLICITLY the
``simulated_recovered`` column produced by the Day 4 simulator -- the
Day-1-style ``recovered`` label returned by the shared feature builder is
discarded on purpose -- which makes these models synthetic-world-only
quantities that support never a production claim; the Day 2 baseline
``P(recovered | context)`` remains untouched elsewhere as the control
reference.

Pipeline shape mirrors Day 2 byte-for-byte by reusing its private builder:
``ColumnTransformer`` numeric passthrough + ``OneHotEncoder(handle_unknown=
"ignore", sparse_output=False)`` followed by a seeded ``XGBClassifier``.
The module draws no randomness of its own (the estimator's ``random_state``
is the only stochastic input), consults no wall clock, and never mutates its
inputs, so identical frames and seed reproduce byte-identical bundles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.evaluate import CalibratedClassifierCV, FrozenEstimator
from ml.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_feature_matrix
from ml.train import _build_pipeline

ARM_ORDER = ("CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")

STRATUM_COLUMN = "stratum"
STRATUM_RANDOMIZED = "randomized"
ACTION_COLUMN = "assigned_action"
TARGET_COLUMN = "simulated_recovered"

REQUIRED_COLUMNS = (STRATUM_COLUMN, ACTION_COLUMN, TARGET_COLUMN)

SMALL_SEGMENT_THRESHOLD = 100


@dataclass(frozen=True)
class ActionModelBundle:
    """Immutable container of fitted per-arm pipelines plus fit metadata."""

    models: dict
    arms: tuple
    metadata: dict


def _require_frame(value: object, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(value).__name__}")


def _require_observation_columns(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required observation columns {missing}; pass "
            "frames produced by assemble_observations (stratum + "
            f"{ACTION_COLUMN} + {TARGET_COLUMN} required)"
        )


def _randomized_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]


def train_action_models(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    seed: int = 20260826,
) -> tuple[ActionModelBundle, dict]:
    """Fit one Day-2-shaped pipeline per arm on ``randomized ∩ arm`` rows.

    For each arm in ``ARM_ORDER`` the training rows are exactly
    ``train_frame[(stratum == "randomized") & (assigned_action == arm)]``;
    features come from ``build_feature_matrix`` while the target is taken
    EXPLICITLY from the ``simulated_recovered`` simulator column (never the
    builder's returned ``recovered`` label). An arm with zero such rows
    raises ``ValueError`` naming the arm. Validation rows are counted for
    metadata only -- calibration arrives in Task 3. Segments holding fewer
    than 100 randomized rows of an arm are flagged in ``small_segments``.
    Neither input frame is mutated.
    """
    _require_frame(train_frame, "train_frame")
    _require_frame(validation_frame, "validation_frame")
    _require_observation_columns(train_frame, "train_frame")
    _require_observation_columns(validation_frame, "validation_frame")

    randomized_train = _randomized_rows(train_frame)
    randomized_validation = _randomized_rows(validation_frame)

    models = {}
    train_rows = {}
    validation_rows = {}
    small_segments = []
    for arm in ARM_ORDER:
        arm_train = randomized_train.loc[randomized_train[ACTION_COLUMN] == arm]
        if len(arm_train) == 0:
            raise ValueError(
                f"arm '{arm}' has zero randomized training rows in train_frame; "
                "cannot fit an action-conditional model without observations"
            )
        X_train, _baseline_label = build_feature_matrix(arm_train)
        y_train = arm_train[TARGET_COLUMN].astype(int)
        pipeline = _build_pipeline(seed)
        pipeline.fit(X_train, y_train)
        models[arm] = pipeline

        arm_validation = randomized_validation.loc[
            randomized_validation[ACTION_COLUMN] == arm
        ]
        train_rows[arm] = int(len(arm_train))
        validation_rows[arm] = int(len(arm_validation))
        if train_rows[arm] < SMALL_SEGMENT_THRESHOLD:
            small_segments.append(("train", arm))
        if validation_rows[arm] < SMALL_SEGMENT_THRESHOLD:
            small_segments.append(("validation", arm))

    metadata = {
        "model_family": "per_arm_xgboost",
        "seed": seed,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "feature_names": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "small_segments": small_segments,
    }
    return ActionModelBundle(models=models, arms=ARM_ORDER, metadata=metadata), metadata


def calibrate_action_models(
    bundle: ActionModelBundle,
    validation_frame: pd.DataFrame,
) -> ActionModelBundle:
    """Wrap every arm pipeline in per-arm sigmoid probability calibration.

    Mirrors the Day 2 ``calibrate_model`` discipline one arm at a time: each
    fitted raw pipeline is sealed inside ``FrozenEstimator`` (so its learned
    state can never refit and its own predictions stay byte-identical) and a
    fresh ``CalibratedClassifierCV(method="sigmoid")`` wrapper is fitted on
    exactly ``validation_frame[(stratum == "randomized") & (assigned_action ==
    arm)]`` rows. The wrapper classes are imported through ``ml.evaluate`` so
    this module's import-root whitelist stays untouched. The calibration
    target is EXPLICITLY the ``simulated_recovered`` simulator column -- the
    identical synthetic-world-only discipline as training -- never the
    builder's ``recovered`` label. An arm with zero such validation rows
    raises ``ValueError`` naming the arm. Returns a NEW frozen bundle whose
    metadata extends the input copy-for-copy with a ``calibration`` record;
    the input bundle, its pipelines, and its metadata are left unmutated.
    """
    _require_frame(validation_frame, "validation_frame")
    _require_observation_columns(validation_frame, "validation_frame")

    randomized_validation = _randomized_rows(validation_frame)

    calibrated_models = {}
    calibration_rows = {}
    for arm in bundle.arms:
        arm_validation = randomized_validation.loc[
            randomized_validation[ACTION_COLUMN] == arm
        ]
        if len(arm_validation) == 0:
            raise ValueError(
                f"arm '{arm}' has zero randomized validation rows in "
                "validation_frame; cannot fit a calibrator without observations"
            )
        X_validation, _baseline_label = build_feature_matrix(arm_validation)
        y_validation = arm_validation[TARGET_COLUMN].astype(int)
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(bundle.models[arm]), method="sigmoid"
        )
        calibrated.fit(X_validation, y_validation)
        calibrated_models[arm] = calibrated
        calibration_rows[arm] = int(len(arm_validation))

    metadata = {
        **bundle.metadata,
        "calibration": {
            "method": "sigmoid",
            "rows": calibration_rows,
            "fit_on": "validation_randomized_only",
        },
    }
    return ActionModelBundle(
        models=calibrated_models, arms=bundle.arms, metadata=metadata
    )


def predict_action_probability(
    bundle: ActionModelBundle,
    context_frame: pd.DataFrame,
    action: str,
) -> np.ndarray:
    """Return ``P(action=a recovers | context)`` for each row of the frame."""
    _require_frame(context_frame, "context_frame")
    if action not in bundle.models:
        raise ValueError(
            f"unknown action {action!r}; this bundle holds fitted models for "
            f"the arms {sorted(bundle.models)}"
        )
    X, _ = build_feature_matrix(context_frame)
    return bundle.models[action].predict_proba(X)[:, 1]


def predict_all_actions(
    bundle: ActionModelBundle,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-row probabilities with one column per arm in ARM_ORDER."""
    columns = {
        arm: predict_action_probability(bundle, context_frame, arm)
        for arm in ARM_ORDER
    }
    return pd.DataFrame(data=columns, columns=list(ARM_ORDER))
