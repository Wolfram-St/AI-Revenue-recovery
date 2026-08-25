import pandas as pd

from data.generate_dataset import generate_dataset
from data.splits import chronological_split


def test_chronological_split_preserves_time_order():
    train, validation, test = chronological_split(generate_dataset(100, seed=1))
    assert train["event_timestamp"].max() < validation["event_timestamp"].min()
    assert validation["event_timestamp"].max() < test["event_timestamp"].min()


def test_split_is_lossless_and_disjoint():
    df = generate_dataset(500, seed=7)
    train, validation, test = chronological_split(df)
    assert len(train) + len(validation) + len(test) == len(df)
    combined = pd.concat([train["attempt_id"], validation["attempt_id"], test["attempt_id"]])
    assert combined.nunique() == len(df)
    assert not set(train["attempt_id"]) & set(validation["attempt_id"])
    assert not set(validation["attempt_id"]) & set(test["attempt_id"])


def test_split_respects_configured_fractions():
    df = generate_dataset(200, seed=3)
    train, validation, test = chronological_split(df)
    assert (len(train), len(validation), len(test)) == (140, 30, 30)


def test_split_does_not_shuffle_rows():
    df = generate_dataset(60, seed=11).reset_index(drop=True)
    train, _, _ = chronological_split(df)
    expected = df.sort_values("event_timestamp").head(len(train))
    assert train["attempt_id"].tolist() == expected["attempt_id"].tolist()


def test_split_rejects_invalid_fractions():
    df = generate_dataset(20, seed=5)
    try:
        chronological_split(df, train_fraction=0.9, validation_fraction=0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for fractions summing above 1.0")
