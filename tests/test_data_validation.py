from data.generate_dataset import generate_dataset
from data.validate_dataset import validate_dataset


def test_validation_accepts_canonical_dataset():
    report = validate_dataset(generate_dataset(500, seed=42))
    assert report["valid"] is True
    assert report["column_count"] == 19
    assert report["missing_cells"] == 0
    assert report["violations"] == []


def test_validation_rejects_duplicate_attempt_ids():
    df = generate_dataset(50, seed=42)
    df.loc[1, "attempt_id"] = df.loc[0, "attempt_id"]
    report = validate_dataset(df)
    assert report["valid"] is False
    assert report["duplicate_attempt_ids"] > 0


def test_validation_rejects_duplicate_payment_ids():
    df = generate_dataset(50, seed=42)
    df.loc[2, "payment_id"] = df.loc[0, "payment_id"]
    report = validate_dataset(df)
    assert report["valid"] is False
    assert report["duplicate_payment_ids"] > 0


def test_validation_rejects_missing_cells():
    df = generate_dataset(50, seed=42)
    df.loc[3, "amount_inr"] = None
    report = validate_dataset(df)
    assert report["valid"] is False
    assert report["missing_cells"] > 0


def test_validation_rejects_non_binary_label():
    df = generate_dataset(50, seed=42)
    df["recovered"] = df["recovered"].astype("int64")
    df.loc[4, "recovered"] = 2
    report = validate_dataset(df)
    assert report["valid"] is False


def test_validation_rejects_unknown_failure_category():
    df = generate_dataset(50, seed=42)
    df.loc[5, "failure_category"] = "quantum_decline"
    report = validate_dataset(df)
    assert report["valid"] is False


def test_validation_rejects_non_monotonic_event_time():
    df = generate_dataset(50, seed=42)
    shuffled = df.iloc[[1, 0] + list(range(2, len(df)))]
    report = validate_dataset(shuffled)
    assert report["valid"] is False
    assert report["timestamp_monotonic_increasing"] is False


def test_validation_rejects_column_contract_violation():
    df = generate_dataset(50, seed=42)
    df = df.drop(columns=["fraud_risk"])
    report = validate_dataset(df)
    assert report["valid"] is False
    assert report["column_count"] == 18


def test_validation_reports_class_balance():
    report = validate_dataset(generate_dataset(500, seed=42))
    balance = report["class_balance"]
    assert balance[0] > 0
    assert balance[1] > 0
    assert balance[0] + balance[1] == 500

def test_validation_rejects_empty_dataset():
    df = generate_dataset(50, seed=42).iloc[0:0]
    report = validate_dataset(df)
    assert report["valid"] is False
    assert any("empty" in violation for violation in report["violations"])


def test_validation_reports_timestamp_dtype_violation_separately():
    df = generate_dataset(50, seed=42)
    df["event_timestamp"] = df["event_timestamp"].astype(str)
    report = validate_dataset(df)
    assert report["valid"] is False
    assert any("datetime" in violation for violation in report["violations"])
