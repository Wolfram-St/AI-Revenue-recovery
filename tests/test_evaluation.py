import numpy as np

from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.evaluate import (
    calculate_revenue_metrics,
    calibrate_model,
    evaluate_predictions,
)
from ml.features import build_feature_matrix
from ml.train import predict_recovery_probability, train_baseline


def test_evaluation_returns_required_metrics():
    result = evaluate_predictions(
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.8, 0.7, 0.2]),
        np.array([100.0, 200.0, 300.0, 400.0]),
    )
    assert {"roc_auc", "pr_auc", "brier_score"}.issubset(result)
    assert 0.5 < result["roc_auc"] <= 1.0


def test_evaluation_handles_single_class_labels_without_crashing():
    result = evaluate_predictions(
        np.array([1, 1]),
        np.array([0.4, 0.6]),
        np.array([100.0, 200.0]),
    )
    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["pr_auc"])


def test_revenue_metrics_are_nonnegative():
    result = calculate_revenue_metrics(
        np.array([0, 1]), np.array([0.2, 0.9]), np.array([100.0, 1000.0]), threshold=0.5
    )
    assert result["revenue_at_risk_inr"] >= 0
    assert result["predicted_recoverable_revenue_inr"] >= 0


def test_revenue_metrics_separate_predicted_from_actual_money():
    result = calculate_revenue_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([100.0, 300.0, 500.0, 700.0]),
        threshold=0.5,
    )
    assert result["intervention_count"] == 2
    assert result["actual_recovered_revenue_inr"] == 300.0
    assert result["false_positive_interventions"] == 1
    assert result["missed_recoverable_cases"] == 1
    assert result["recovery_rate"] == 0.5
    assert result["revenue_at_risk_inr"] == 1600.0
    assert 0 < result["predicted_recoverable_revenue_inr"] < result["revenue_at_risk_inr"]


def test_revenue_metrics_with_no_interventions_is_zero_divided():
    result = calculate_revenue_metrics(
        np.array([1, 1]), np.array([0.1, 0.2]), np.array([100.0, 200.0]), threshold=0.5
    )
    assert result["intervention_count"] == 0
    assert result["actual_recovered_revenue_inr"] == 0
    assert result["recovery_rate"] == 0


def test_calibration_produces_valid_probabilities_on_test_data():
    train, validation, test = chronological_split(generate_dataset(400, seed=42))
    model, _ = train_baseline(train, validation, seed=42)
    calibrated = calibrate_model(model, validation)
    X_test, _ = build_feature_matrix(test)
    probabilities = calibrated.predict_proba(X_test)[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert len(probabilities) == len(test)


def test_calibration_comparison_beats_or_matches_uncalibrated_brier_on_average():
    train, validation, test = chronological_split(generate_dataset(1500, seed=42))
    model, _ = train_baseline(train, validation, seed=42)
    calibrated = calibrate_model(model, validation)
    X_test, y_test = build_feature_matrix(test)
    raw_brier = evaluate_predictions(
        y_test.to_numpy(), predict_recovery_probability(model, test), test["amount_inr"].to_numpy()
    )["brier_score"]
    calibrated_brier = evaluate_predictions(
        y_test.to_numpy(), calibrated.predict_proba(X_test)[:, 1], test["amount_inr"].to_numpy()
    )["brier_score"]
    assert calibrated_brier <= raw_brier + 0.01

def test_calibration_does_not_modify_base_pipeline():
    train, validation, test = chronological_split(generate_dataset(400, seed=42))
    model, _ = train_baseline(train, validation, seed=42)
    probe_before = predict_recovery_probability(model, test)
    importances_before = model.named_steps["classifier"].feature_importances_.copy()
    calibrate_model(model, validation)
    np.testing.assert_array_equal(probe_before, predict_recovery_probability(model, test))
    np.testing.assert_array_equal(
        importances_before, model.named_steps["classifier"].feature_importances_
    )
