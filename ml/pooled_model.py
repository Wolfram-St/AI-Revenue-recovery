"""Pooled comparison-candidate model for the Day 6 investigation (plan D-E1).

Fits exactly ONE shared XGBoost pipeline estimating ``P(recovered |
decision-time context, assigned arm)``: the Day 2 decision-time feature
whitelist enters unchanged and the assigned arm joins the categorical block
as one more one-hot feature, so a single fit serves every canonical arm.
This is the sample-efficiency comparison candidate for plan Gate A -- the
Day 5 per-arm models remain the REFERENCE production form, and this pooled
family exists only to be measured against them under the pre-registered
D-E2 rule before any preference is claimed.

Training discipline mirrors Day 5 exactly: only ``randomized``-stratum rows
of the assembled observation frame enter the fit (safety-censored rows are
excluded so eligibility can never be confused with treatment), and the
regression target is EXPLICITLY the simulator's ``simulated_recovered``
column -- the Day-1-style ``recovered`` label returned by the shared feature
builder is discarded on purpose. The pipeline cannot reuse the Day 2 private
builder (its categorical block needs one extra column), so it is constructed
here with byte-identical estimator hyperparameters. The module draws no
randomness of its own (the estimator's ``random_state`` is the only
stochastic input), consults no wall clock, and never mutates its inputs.

Prediction is counterfactual by construction: a query copies the caller's
context frame, OVERWRITES the ``assigned_action`` column with the requested
arm on that copy only, and scores the result, so identical contexts queried
under different arms isolate exactly the arm effect this model estimates.

Like every action-aware quantity in this repository these probabilities are
synthetic-world-only MODEL ESTIMATES of the simulated world; they support
never a production or real-world claim, and the Day 2 baseline
``P(recovered | context)`` remains untouched elsewhere as the control
reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from ml.evaluate import CalibratedClassifierCV, FrozenEstimator
from ml.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_feature_matrix

ARM_ORDER = ("CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")

STRATUM_COLUMN = "stratum"
STRATUM_RANDOMIZED = "randomized"
ACTION_COLUMN = "assigned_action"
TARGET_COLUMN = "simulated_recovered"

REQUIRED_COLUMNS = (STRATUM_COLUMN, ACTION_COLUMN, TARGET_COLUMN)


@dataclass(frozen=True)
class PooledModelBundle:
    """Immutable container of the fitted shared pipeline plus fit metadata."""

    model: object
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


def _build_pooled_pipeline(seed: int) -> Pipeline:
    """Day-2-shaped pipeline whose categorical block also one-hots the arm."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES + [ACTION_COLUMN],
            ),
        ],
        remainder="drop",
    )
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def _pooled_design_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Decision-time features PLUS the assigned-arm column (never mutates)."""
    features = build_feature_matrix(rows)[0]
    features[ACTION_COLUMN] = rows[ACTION_COLUMN].to_numpy()
    return features


def train_pooled_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    seed: int = 20260826,
) -> tuple[PooledModelBundle, dict]:
    """Fit ONE shared pipeline on all ``randomized`` train rows at once.

    The training rows are exactly ``train_frame[stratum == "randomized"]``
    regardless of arm; features come from ``build_feature_matrix`` extended
    with the row's ``assigned_action`` while the target is taken EXPLICITLY
    from the ``simulated_recovered`` simulator column (never the builder's
    ``recovered`` label). Zero randomized training rows raises ``ValueError``
    even though a pooled fit could technically run empty-free elsewhere --
    the guard keeps every failure mode loud -- and so does an arm with no
    randomized training row at all: such an arm would one-hot encode to a
    silent all-zero column at counterfactual query time, so its absence
    raises ``ValueError`` naming it. Validation rows are counted for
    metadata only; calibration arrives via :func:`calibrate_pooled_model`.
    Per-arm small-segment flags are NOT applicable to a single pooled fit,
    so the metadata records its pooled shape plus per-arm randomized counts
    in ``arm_rows``, letting downstream smallest-arm comparisons audit
    coverage without recounting or refitting. Neither input frame is
    mutated.
    """
    _require_frame(train_frame, "train_frame")
    _require_frame(validation_frame, "validation_frame")
    _require_observation_columns(train_frame, "train_frame")
    _require_observation_columns(validation_frame, "validation_frame")

    randomized_train = _randomized_rows(train_frame)
    randomized_validation = _randomized_rows(validation_frame)
    if len(randomized_train) == 0:
        raise ValueError(
            "train_frame has zero 'randomized' stratum rows; the pooled "
            "comparison candidate can only be fitted on randomized-stratum "
            "observations of the assembled frame"
        )
    absent_arms = [
        arm for arm in ARM_ORDER if not bool((randomized_train[ACTION_COLUMN] == arm).any())
    ]
    if absent_arms:
        raise ValueError(
            "pooled training data is missing randomized rows for arm(s): "
            f"{absent_arms} — counterfactual queries for them would be "
            "silent all-zero encodings"
        )

    X_train = _pooled_design_matrix(randomized_train)
    y_train = randomized_train[TARGET_COLUMN].astype(int)
    pipeline = _build_pooled_pipeline(seed)
    pipeline.fit(X_train, y_train)

    arm_rows = {
        arm: {
            "train": int((randomized_train[ACTION_COLUMN] == arm).sum()),
            "validation": int((randomized_validation[ACTION_COLUMN] == arm).sum()),
        }
        for arm in ARM_ORDER
    }
    metadata = {
        "model_family": "pooled_xgboost",
        "seed": seed,
        "train_rows": int(len(randomized_train)),
        "validation_rows": int(len(randomized_validation)),
        "feature_names": NUMERIC_FEATURES + CATEGORICAL_FEATURES + [ACTION_COLUMN],
        "arm_rows": arm_rows,
        "pooled": True,
    }
    return PooledModelBundle(model=pipeline, arms=ARM_ORDER, metadata=metadata), metadata


def calibrate_pooled_model(
    bundle: PooledModelBundle,
    validation_frame: pd.DataFrame,
) -> PooledModelBundle:
    """Wrap the shared pipeline in sigmoid probability calibration.

    Mirrors the Day 2/Day 5 discipline for the single pooled pipeline: the
    fitted raw pipeline is sealed inside ``FrozenEstimator`` (so its learned
    state can never refit and its own predictions stay byte-identical) and a
    fresh ``CalibratedClassifierCV(method="sigmoid")`` wrapper is fitted on
    exactly ``validation_frame[stratum == "randomized"]`` rows. The wrapper
    classes are imported through ``ml.evaluate`` like the per-arm module.
    The calibration target is EXPLICITLY the ``simulated_recovered``
    simulator column -- the identical synthetic-world-only discipline as
    training -- never the builder's ``recovered`` label. Zero randomized
    validation rows raises ``ValueError``. Returns a NEW frozen bundle whose
    metadata extends the input copy-for-copy with a ``calibration`` record;
    the input bundle, its pipeline, and its metadata are left unmutated.
    """
    _require_frame(validation_frame, "validation_frame")
    _require_observation_columns(validation_frame, "validation_frame")

    randomized_validation = _randomized_rows(validation_frame)
    if len(randomized_validation) == 0:
        raise ValueError(
            "validation_frame has zero 'randomized' stratum rows; cannot "
            "fit a calibrator without observations"
        )
    X_validation = _pooled_design_matrix(randomized_validation)
    y_validation = randomized_validation[TARGET_COLUMN].astype(int)
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(bundle.model), method="sigmoid"
    )
    calibrated.fit(X_validation, y_validation)

    metadata = {
        **bundle.metadata,
        "calibration": {
            "method": "sigmoid",
            "rows": int(len(randomized_validation)),
            "fit_on": "validation_randomized_only",
        },
    }
    return PooledModelBundle(model=calibrated, arms=bundle.arms, metadata=metadata)


def predict_pooled_probability(
    bundle: PooledModelBundle,
    context_frame: pd.DataFrame,
    action: str,
) -> np.ndarray:
    """Return ``P(action=a recovers | context)`` under the shared pipeline.

    Counterfactual query semantics: the context frame is copied, the copy's
    ``assigned_action`` column is OVERWRITTEN with ``action``, and the
    overwritten copy feeds the feature builder plus the appended arm column.
    The caller's frame is never mutated. An ``action`` outside ``ARM_ORDER``
    raises ``ValueError`` naming it and the available arms.
    """
    _require_frame(context_frame, "context_frame")
    if action not in ARM_ORDER:
        raise ValueError(
            f"unknown action {action!r}; the pooled model supports "
            f"counterfactual queries for the arms {list(ARM_ORDER)} only"
        )
    counterfactual = context_frame.copy()
    counterfactual[ACTION_COLUMN] = action
    X = _pooled_design_matrix(counterfactual)
    return bundle.model.predict_proba(X)[:, 1]


def predict_pooled_all_actions(
    bundle: PooledModelBundle,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-row probabilities with one column per arm in ARM_ORDER."""
    columns = {
        arm: predict_pooled_probability(bundle, context_frame, arm)
        for arm in ARM_ORDER
    }
    return pd.DataFrame(data=columns, columns=list(ARM_ORDER))
