"""Validate a payment-attempt frame against the canonical Day 1 contract.

The validator is a pure function: it returns a structured report and never
prints. A report is ``valid`` only when every contract check passes.
"""

from __future__ import annotations

import pandas as pd

from data.generate_dataset import FAILURE_MAP, FEATURE_COLUMNS, LABEL_COLUMNS, TIME_COLUMN

EXPECTED_COLUMNS = (
    FEATURE_COLUMNS[:3] + [TIME_COLUMN] + FEATURE_COLUMNS[3:] + LABEL_COLUMNS
)
ALLOWED_FAILURE_CATEGORIES = sorted({category for category, _ in FAILURE_MAP.values()})

DUPLICATE_IDENTIFIER_COLUMNS = ("attempt_id", "payment_id")


def validate_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Return a structured contract report for a payment-attempt frame."""
    violations: list[str] = []
    column_count = int(df.shape[1])

    if list(df.columns) != list(EXPECTED_COLUMNS):
        violations.append(
            "column contract violated: expected "
            f"{list(EXPECTED_COLUMNS)}, found {list(df.columns)}"
        )

    missing_cells = 0
    if not df.empty:
        missing_cells = int(df.isna().to_numpy().sum())
    if missing_cells > 0:
        violations.append(f"missing cells present: {missing_cells}")

    duplicate_counts = {
        column: (int(df[column].duplicated().sum()) if column in df.columns else -1)
        for column in DUPLICATE_IDENTIFIER_COLUMNS
    }
    duplicate_attempt_ids = max(duplicate_counts["attempt_id"], 0)
    duplicate_payment_ids = max(duplicate_counts["payment_id"], 0)
    if "attempt_id" in df.columns and duplicate_attempt_ids > 0:
        violations.append(f"duplicate attempt_id values: {duplicate_attempt_ids}")
    if "payment_id" in df.columns and duplicate_payment_ids > 0:
        violations.append(f"duplicate payment_id values: {duplicate_payment_ids}")

    class_balance = {0: 0, 1: 0}
    label_binary = True
    if "recovered" in df.columns:
        label_values = set(pd.unique(df["recovered"]))
        label_binary = label_values.issubset({0, 1, False, True})
        if not label_binary:
            violations.append(f"recovered must be binary, found values: {sorted(label_values)}")
        else:
            counts = df["recovered"].astype(int).value_counts()
            class_balance = {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}
    else:
        violations.append("label column missing: recovered")
        label_binary = False

    timestamp_monotonic_increasing = bool(
        "event_timestamp" in df.columns
        and pd.api.types.is_datetime64_any_dtype(df["event_timestamp"])
        and df["event_timestamp"].is_monotonic_increasing
    )
    if "event_timestamp" in df.columns and not timestamp_monotonic_increasing:
        violations.append("event_timestamp must be nondecreasing")

    if "failure_category" in df.columns:
        unknown_categories = set(pd.unique(df["failure_category"])) - set(ALLOWED_FAILURE_CATEGORIES)
        if unknown_categories:
            violations.append(f"unknown failure categories: {sorted(unknown_categories)}")

    return {
        "valid": len(violations) == 0,
        "row_count": int(df.shape[0]),
        "column_count": column_count,
        "missing_cells": missing_cells,
        "duplicate_attempt_ids": duplicate_attempt_ids,
        "duplicate_payment_ids": duplicate_payment_ids,
        "class_balance": class_balance,
        "timestamp_monotonic_increasing": timestamp_monotonic_increasing,
        "allowed_failure_categories": ALLOWED_FAILURE_CATEGORIES,
        "violations": violations,
    }
