from data.generate_dataset import generate_dataset, dataset_contract


def test_generator_returns_exact_day1_shape():
    df = generate_dataset(n_rows=100, seed=42)
    assert df.shape == (100, 19)
    assert list(df.columns) == [
        "attempt_id", "payment_id", "customer_id", "event_timestamp", "amount_inr",
        "payment_method", "attempt_number", "customer_tenure_days",
        "successful_payment_count", "failed_payment_count",
        "historical_recovery_count", "customer_opted_out", "failure_code",
        "failure_category", "issuer_response", "device_type", "country",
        "fraud_risk", "recovered",
    ]


def test_generator_is_deterministic_for_same_seed():
    first = generate_dataset(n_rows=50, seed=7)
    second = generate_dataset(n_rows=50, seed=7)
    assert first.equals(second)


def test_event_time_is_strictly_ordered_for_temporal_split():
    df = generate_dataset(n_rows=100, seed=42)
    assert df["event_timestamp"].is_monotonic_increasing


def test_outcome_is_not_identical_to_policy_rule():
    df = generate_dataset(n_rows=1000, seed=123)
    temporary = df["failure_category"].eq("temporary_decline")
    assert temporary.any()
    assert df.loc[temporary, "recovered"].nunique() == 2


def test_dataset_contract_exposes_feature_label_metadata_and_outcome_groups():
    contract = dataset_contract()
    assert contract["rows"] == 5000
    assert contract["columns"] == 19
    assert "event_timestamp" in contract["metadata"]
    assert "recovered" in contract["labels"]
    assert "recovery_time_hours" in contract["outcomes"]
    assert "recovered" not in contract["features"]
