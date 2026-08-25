import numpy as np

from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.train import predict_recovery_probability, train_baseline


def test_baseline_returns_probabilities_in_range():
    train, validation, _ = chronological_split(generate_dataset(300, seed=42))
    model, metadata = train_baseline(train, validation, seed=42)
    probabilities = predict_recovery_probability(model, validation)
    assert len(probabilities) == len(validation)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert metadata["model"] == "xgboost"


def test_training_is_reproducible_for_a_fixed_seed():
    train, validation, _ = chronological_split(generate_dataset(300, seed=17))
    first_model, _ = train_baseline(train, validation, seed=99)
    second_model, _ = train_baseline(train, validation, seed=99)
    np.testing.assert_allclose(
        predict_recovery_probability(first_model, validation),
        predict_recovery_probability(second_model, validation),
    )


def test_metadata_records_training_context():
    train, validation, _ = chronological_split(generate_dataset(300, seed=5))
    _, metadata = train_baseline(train, validation, seed=21)
    assert metadata["seed"] == 21
    assert metadata["train_rows"] == len(train)
    assert metadata["validation_rows"] == len(validation)
    assert metadata["positive_class_rate"] == train["recovered"].mean()
    assert len(metadata["feature_names"]) == 14
