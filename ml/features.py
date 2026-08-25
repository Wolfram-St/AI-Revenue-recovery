"""Decision-time feature construction.

Every column here is known immediately after the failed attempt and before
any recovery intervention. Identifiers, the event timestamp, the label, and
all post-intervention outcome fields are excluded by construction.
"""

from __future__ import annotations

import pandas as pd

from data.generate_dataset import LABEL_COLUMNS, TIME_COLUMN

IDENTIFIER_COLUMNS = ["attempt_id", "payment_id", "customer_id"]

NUMERIC_FEATURES = [
    "amount_inr",
    "attempt_number",
    "customer_tenure_days",
    "successful_payment_count",
    "failed_payment_count",
    "historical_recovery_count",
    "customer_opted_out",
    "fraud_risk",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "failure_code",
    "failure_category",
    "issuer_response",
    "device_type",
    "country",
]

FORBIDDEN_FEATURES = frozenset(
    IDENTIFIER_COLUMNS
    + [TIME_COLUMN]
    + LABEL_COLUMNS
    + [
        "recovery_time_hours",
        "recovery_action",
        "action_outcome",
        "recovered_amount_inr",
    ]
)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the decision-time feature frame and the ``recovered`` label."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if FORBIDDEN_FEATURES & set(feature_columns):
        raise AssertionError("decision-time features must never include forbidden columns")

    X = df.loc[:, feature_columns].copy()
    for boolean_column in ("customer_opted_out", "fraud_risk"):
        X[boolean_column] = X[boolean_column].astype("int8")
    y = df["recovered"].astype(int)
    return X, y
