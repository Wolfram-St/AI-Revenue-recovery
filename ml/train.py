"""Baseline recoverability model.

Trains an XGBoost classifier on decision-time features to estimate
``P(recovered | decision-time context)``. This is a general recoverability
estimate only: it is not a per-action probability and supports no causal
claims about any specific intervention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from data.generate_dataset import LABEL_COLUMNS
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix

MODEL_NAME = "xgboost"


def _build_pipeline(seed: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
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


def train_baseline(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, object]]:
    """Fit the baseline on training data only and return it with metadata."""
    if train_df[LABEL_COLUMNS[0]].nunique() < 2:
        raise ValueError("training data must contain both classes of 'recovered'")
    X_train, y_train = build_feature_matrix(train_df)
    model = _build_pipeline(seed)
    model.fit(X_train, y_train)

    metadata = {
        "model": MODEL_NAME,
        "seed": seed,
        "feature_names": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "positive_class_rate": float(train_df[LABEL_COLUMNS[0]].mean()),
        "classifier_params": {
            key: value
            for key, value in model.named_steps["classifier"].get_params().items()
            if key in {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"}
        },
    }
    return model, metadata


def predict_recovery_probability(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Return calibrated-in-shape recovery probabilities for each row."""
    X, _ = build_feature_matrix(df)
    return model.predict_proba(X)[:, 1]
