"""Model diagnostics and thresholded revenue metrics.

The business numbers here come from a clearly labeled *thresholded
simulation* on observed outcomes. They are not causal estimates of
incremental recovery per intervention: the baseline dataset records whether
a payment eventually recovered, not which action caused recovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml.features import build_feature_matrix


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
) -> dict[str, float]:
    """Return ROC-AUC, PR-AUC, and Brier score for predicted probabilities.

    ``amounts`` is accepted to keep the frozen Day 2 interface stable for
    amount-aware reporting elsewhere; AUC and Brier math is amount-independent.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if np.unique(y_true).size < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, probabilities))
        pr_auc = float(average_precision_score(y_true, probabilities))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def calculate_revenue_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return thresholded-simulation revenue metrics in INR."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    amounts = np.asarray(amounts, dtype=float)

    selected = probabilities >= threshold
    revenue_at_risk = float(amounts.sum())
    intervention_count = int(selected.sum())

    actual_recovered_revenue = float(amounts[selected & (y_true == 1)].sum())
    predicted_recoverable_revenue = float((amounts[selected] * probabilities[selected]).sum())

    return {
        "revenue_at_risk_inr": revenue_at_risk,
        "intervention_count": float(intervention_count),
        "actual_recovered_revenue_inr": actual_recovered_revenue,
        "predicted_recoverable_revenue_inr": predicted_recoverable_revenue,
        "recovery_rate": float(y_true[selected].mean()) if intervention_count else 0.0,
        "recovered_share_of_revenue_at_risk": (
            actual_recovered_revenue / revenue_at_risk if revenue_at_risk else 0.0
        ),
        "false_positive_interventions": float(int((selected & (y_true == 0)).sum())),
        "missed_recoverable_cases": float(int((~selected & (y_true == 1)).sum())),
    }


def calibrate_model(model: object, validation_df: pd.DataFrame) -> CalibratedClassifierCV:
    """Fit a probability calibrator on validation data without refitting the model."""
    X_validation, y_validation = build_feature_matrix(validation_df)
    try:
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method="sigmoid"
        )
        calibrated.fit(X_validation, y_validation)
    except (TypeError, NotFittedError) as error:
        raise ValueError("calibration requires a fitted pipeline") from error
    return calibrated
