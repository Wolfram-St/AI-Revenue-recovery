"""Pure opportunity scoring for ranked recovery interventions.

Expected Recovery Value is a prioritization score, not an action-conditional causal estimate and not realized revenue.

The score combines the existing baseline recoverability estimate
``P(recovered | context)`` with deterministic cost and risk constants:

    expected_recovery_value_inr =
        P(recovered | context) * amount_inr
        - intervention_cost_inr
        - risk_penalty_inr

where ``risk_penalty_inr`` equals ``UNKNOWN_CATEGORY_RISK_FRACTION *
amount_inr`` iff ``failure_category == "unknown"``, else ``0.0``. The cost
basis is the automated-retry cost because both automated actions are retries;
REQUEST_UPDATE / HUMAN_REVIEW economics belong to policy, not scoring, and
nothing here estimates per-action recovery probabilities.

This module never calls the ML model, never invokes the policy engine,
performs no I/O, reads no wall clock, and draws no randomness: identical
inputs always produce identical outputs.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

RETRY_INTERVENTION_COST_INR: float = 10.0
"""Illustrative simulation parameter pending product calibration."""

UNKNOWN_CATEGORY_RISK_FRACTION: float = 0.05
"""Illustrative simulation parameter pending product calibration."""

INTERVENE = "INTERVENE"
NO_INTERVENTION = "NO_INTERVENTION"

_UNKNOWN_CATEGORY = "unknown"
_OUTPUT_COLUMNS = (
    "scoring_recommendation",
    "expected_recovery_value_inr",
    "recovery_probability",
    "payment_amount_inr",
    "risk_penalty_inr",
    "intervention_cost_inr",
    "worth_intervening",
    "opportunity_rank",
)


@dataclass(frozen=True)
class OpportunityScore:
    scoring_recommendation: str
    expected_recovery_value_inr: float
    recovery_probability: float
    payment_amount_inr: float
    risk_penalty_inr: float
    intervention_cost_inr: float
    worth_intervening: bool


def _validated_amount(name: str, value: Any) -> float:
    number = _validated_finite_real(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return number


def _validated_finite_real(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{name} must be a real finite number, got {value!r} of type "
            f"{type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def score_opportunity(
    recovery_probability: float,
    amount_inr: float,
    failure_category: str,
    intervention_cost_inr: float = RETRY_INTERVENTION_COST_INR,
) -> OpportunityScore:
    """Score one opportunity from ``P(recovered | context)`` and its amount.

    Expected Recovery Value is a prioritization score, not an action-conditional causal estimate and not realized revenue. The strict rule is:
    ``worth_intervening`` compares the UNROUNDED expected recovery value
    against zero, so zero or negative ERV is never worth intervening while
    displayed values still carry paise rounding. Probabilities 0.0 and 1.0
    are valid boundaries and yield deterministic results.

    Raises ValueError naming the offending value when any input violates its
    domain.
    """
    probability = _validated_probability(recovery_probability)
    amount = _validated_amount("amount_inr", amount_inr)
    cost = _validated_amount("intervention_cost_inr", intervention_cost_inr)
    if not isinstance(failure_category, str) or not failure_category:
        raise ValueError(
            f"failure_category must be a non-empty string, got {failure_category!r}"
        )

    if failure_category == _UNKNOWN_CATEGORY:
        risk_penalty = UNKNOWN_CATEGORY_RISK_FRACTION * amount
    else:
        risk_penalty = 0.0
    unrounded_erv = probability * amount - cost - risk_penalty
    worth = unrounded_erv > 0.0

    return OpportunityScore(
        scoring_recommendation=INTERVENE if worth else NO_INTERVENTION,
        expected_recovery_value_inr=round(unrounded_erv, 2),
        recovery_probability=probability,
        payment_amount_inr=round(amount, 2),
        risk_penalty_inr=round(risk_penalty, 2),
        intervention_cost_inr=round(cost, 2),
        worth_intervening=worth,
    )


def _validated_probability(value: Any) -> float:
    number = _validated_finite_real("recovery_probability", value)
    if number < 0.0 or number > 1.0:
        raise ValueError(
            f"recovery_probability must be within [0.0, 1.0], got {value!r}"
        )
    return number


def _label_precedence(index: pd.Index) -> list[int]:
    try:
        order_by_label = sorted(range(len(index)), key=lambda position: index[position])
    except TypeError as exc:
        raise ValueError(
            "opportunity_rank requires uniformly typed index labels, "
            "found mixed types"
        ) from exc
    precedence = [0] * len(index)
    for order, position in enumerate(order_by_label):
        precedence[position] = order
    return precedence


def score_opportunities(df: pd.DataFrame, probabilities: Sequence[float]) -> pd.DataFrame:
    """Score every row of ``df`` against aligned probabilities.

    Expected Recovery Value is a prioritization score, not an action-conditional causal estimate and not realized revenue. Returns a NEW
    frame indexed like ``df`` (the input frame and probability sequence are
    never mutated) with exactly ``_OUTPUT_COLUMNS``. ``opportunity_rank``
    densely numbers only rows whose ``worth_intervening`` is true, ordered
    by the paise-rounded displayed expected recovery value descending; ties
    break by index label ascending (lexicographic), never by positional row
    order, so assignment is stable under row shuffles and equals attempt_id
    ascending whenever ``df`` is labeled by attempt_id. Because ordering
    uses the displayed value, raw sub-paise ERV differences that round to
    equal values fall back to label precedence; monotone rounding bounds any
    resulting distortion at half a paise. Ranks start at 1 and non-worth rows
    carry ``pd.NA`` under the nullable ``Int64`` dtype. Deterministic for
    identical inputs; mixed-type index labels raise ValueError at rank time.

    Raises ValueError when the probability count mismatches the row count or
    required columns are absent.
    """
    probability_values = list(probabilities)
    if len(probability_values) != len(df):
        raise ValueError(
            f"probabilities length {len(probability_values)} does not match "
            f"df row count {len(df)}"
        )
    missing_columns = [
        column for column in ("amount_inr", "failure_category")
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"df is missing required columns: {missing_columns}")

    scores = [
        score_opportunity(probability, amount, category)
        for probability, amount, category in zip(
            probability_values, df["amount_inr"], df["failure_category"]
        )
    ]

    scored = pd.DataFrame(
        {
            "scoring_recommendation": [score.scoring_recommendation for score in scores],
            "expected_recovery_value_inr": [
                score.expected_recovery_value_inr for score in scores
            ],
            "recovery_probability": [score.recovery_probability for score in scores],
            "payment_amount_inr": [score.payment_amount_inr for score in scores],
            "risk_penalty_inr": [score.risk_penalty_inr for score in scores],
            "intervention_cost_inr": [score.intervention_cost_inr for score in scores],
            "worth_intervening": [score.worth_intervening for score in scores],
        },
        index=df.index,
    )
    scored["opportunity_rank"] = _dense_ranks(scored)
    return scored[list(_OUTPUT_COLUMNS)]


def _dense_ranks(scored: pd.DataFrame) -> pd.Series:
    row_count = len(scored)
    rank_values = pd.array([pd.NA] * row_count, dtype="Int64")
    worth_mask = scored["worth_intervening"].to_numpy(dtype=bool)
    eligible_positions = [position for position in range(row_count) if worth_mask[position]]
    if eligible_positions:
        displayed_erv = scored["expected_recovery_value_inr"].to_numpy()
        precedence = _label_precedence(scored.index)
        ordered = sorted(
            eligible_positions,
            key=lambda position: (-displayed_erv[position], precedence[position]),
        )
        for rank_number, position in enumerate(ordered, start=1):
            rank_values[position] = rank_number
    return pd.Series(rank_values, index=scored.index, dtype="Int64")
