"""Chronological splitting of the payment-attempt dataset.

Temporal order is the backbone of leakage-safe evaluation: the split never
shuffles and never lets a later event into an earlier segment.
"""

from __future__ import annotations

import pandas as pd

from data.generate_dataset import TIME_COLUMN


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into earliest train, next validation, latest test."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be strictly between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave room for test")
    if TIME_COLUMN not in df.columns:
        raise ValueError(f"chronological_split requires the {TIME_COLUMN} column")

    ordered = df.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = train_end + int(len(ordered) * validation_fraction)

    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )
