from data.generate_dataset import generate_dataset
from ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_matrix,
)

FORBIDDEN_COLUMNS = {
    "attempt_id", "payment_id", "customer_id",
    "event_timestamp", "recovered", "recovery_time_hours",
    "recovery_action", "action_outcome", "recovered_amount_inr",
}


def test_feature_matrix_excludes_outcome_and_metadata_columns():
    X, y = build_feature_matrix(generate_dataset(100, seed=1))
    assert FORBIDDEN_COLUMNS.isdisjoint(X.columns)
    assert len(X) == len(y) == 100


def test_feature_matrix_uses_exactly_the_declared_decision_time_features():
    X, _ = build_feature_matrix(generate_dataset(100, seed=1))
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_labels_align_with_rows():
    df = generate_dataset(80, seed=9)
    _, y = build_feature_matrix(df)
    assert y.tolist() == df["recovered"].astype(int).tolist()


def test_declared_features_are_all_present_in_source_frame():
    df = generate_dataset(10, seed=13)
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert column in df.columns
