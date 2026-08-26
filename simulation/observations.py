"""Assemble the action-aware observation dataset for Day 5 (plan Task 1).

This module composes the frozen Day 4 simulator chain -- treatment
assignment, simulated outcomes, timeline stamping -- over the canonical
attempts frame, producing exactly one joined observation row per failed
payment attempt. Every row is stratified into ``randomized`` (the
experimental sample whose action variation is unconfounded within the
eligible pool) or ``safety_censored`` (rows the policy gate forces to
CONTROL, which have no modeled counterfactual and are excluded from
action-model fitting per plan decision D-M1). Chronological splitting
reuses ``data.splits.chronological_split`` so temporal leakage discipline
stays identical to every earlier evaluation in this repository.

The module is a pure composition layer: it draws no randomness of its own
(all stochasticity originates inside the policy-seeded simulator streams),
consults no wall clock, and never mutates its inputs. Identical inputs and
policy therefore reproduce byte-identical observations, timestamps included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.splits import chronological_split
from simulation.config import TreatmentPolicy
from simulation.dataset import (
    ARM_SOURCE_SAFETY_CENSORED,
    JOIN_KEY,
)
from simulation.outcomes import simulate_outcomes
from simulation.temporal import stamp_treatment_timeline
from simulation.treatment import assign_treatments

STRATUM_COLUMN = "stratum"
STRATUM_RANDOMIZED = "randomized"
STRATUM_SAFETY_CENSORED = "safety_censored"

ASSIGNMENT_COLUMNS = ("assigned_action", "arm_source", "assignment_probability")
OUTCOME_COLUMNS = (
    "simulated_recovered",
    "simulated_recovered_amount_inr",
)
GROUND_TRUTH_COLUMNS = (
    "base_recovery_propensity",
    "action_effect_logit",
    "propensity_under_assignment",
)
TIMELINE_COLUMNS = ("treatment_timestamp", "outcome_timestamp")

APPENDIX_COLUMNS = (
    ASSIGNMENT_COLUMNS
    + OUTCOME_COLUMNS
    + GROUND_TRUTH_COLUMNS
    + TIMELINE_COLUMNS
    + (STRATUM_COLUMN,)
)

_APPENDIX_OWNED = frozenset(APPENDIX_COLUMNS)


def _require_frame(value: object, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(value).__name__}")


def _require_stratum_column(frame: pd.DataFrame) -> None:
    if STRATUM_COLUMN not in frame.columns:
        raise ValueError(
            f"frame is missing the required '{STRATUM_COLUMN}' column produced by "
            "assemble_observations; pass the assembled observation frame, not a "
            "raw attempts frame"
        )


def _joined_context(attempts_df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if JOIN_KEY not in attempts_df.columns:
        raise ValueError(
            f"attempts_df is missing the join key '{JOIN_KEY}' required to align "
            "assignments one-to-one with attempt rows"
        )
    collisions = sorted(set(attempts_df.columns) & _APPENDIX_OWNED)
    if collisions:
        raise ValueError(
            "attempts_df already carries observation columns owned by the "
            f"simulator chain: {collisions}; pass the raw 19-column attempts "
            "context instead"
        )
    keyed = assignments.copy()
    keyed[JOIN_KEY] = attempts_df[JOIN_KEY].to_numpy()
    # Alignment invariant: validate="one_to_one" re-emits a fresh RangeIndex preserving left row order, so the downstream index-based joins stay safe even for hostile input indexes.
    return attempts_df.merge(
        keyed, on=JOIN_KEY, how="left", sort=False, validate="one_to_one"
    )


def assemble_observations(
    attempts_df: pd.DataFrame,
    probabilities: object,
    policy: TreatmentPolicy,
    policy_config: object | None = None,
) -> pd.DataFrame:
    """Run the full Day 4 chain in-process and return the observation frame.

    The chain is composition only: ``assign_treatments`` gates and randomizes
    over the attempts context (with the caller-supplied recovery probabilities
    injected per row), the assignments are merged onto the context one-to-one
    on ``attempt_id`` (row-aligned, mirroring the ``simulation/dataset.py``
    discipline), ``simulate_outcomes`` produces the simulated label, revenue,
    and stored synthetic ground truth, and ``stamp_treatment_timeline`` adds
    the post-failure treatment/outcome timestamps.

    Returns a NEW frame with columns EXACTLY: all original ``attempts_df``
    columns first (input order preserved), then ``assigned_action``,
    ``arm_source``, ``assignment_probability``, then ``simulated_recovered``,
    ``simulated_recovered_amount_inr``, then ``base_recovery_propensity``,
    ``action_effect_logit``, ``propensity_under_assignment``, then
    ``treatment_timestamp``, ``outcome_timestamp``, and finally ``stratum``
    -- ``safety_censored`` where ``arm_source == "safety_censored"``, else
    ``randomized``.

    Neither ``attempts_df`` nor ``probabilities`` is mutated; identical inputs
    and policy yield byte-identical output because every draw comes from the
    policy-seeded seed-stream children inside the composed modules. An empty
    attempts frame yields an empty result carrying the exact same column list.
    Probability problems (wrong length, non-numeric, out of range) surface
    cleanly as the ``ValueError`` raised by ``assign_treatments``.
    """
    _require_frame(attempts_df, "attempts_df")
    assignments = assign_treatments(attempts_df, probabilities, policy, policy_config)
    context = _joined_context(attempts_df, assignments)
    outcomes = simulate_outcomes(context, policy)
    stamped = stamp_treatment_timeline(context.join(outcomes), policy)

    stratum = np.where(
        stamped["arm_source"].to_numpy() == ARM_SOURCE_SAFETY_CENSORED,
        STRATUM_SAFETY_CENSORED,
        STRATUM_RANDOMIZED,
    )
    stamped[STRATUM_COLUMN] = stratum
    return stamped.reset_index(drop=True)


def split_observations(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split observations chronologically into train/validation/test thirds.

    Thin documented wrapper over ``data.splits.chronological_split``: earliest
    rows train, next validation, latest test, never shuffled. Requires the
    ``stratum`` column (so every segment can be audited per stratum) and the
    ``event_timestamp`` column, whose presence ``chronological_split`` itself
    guards. Fraction validation is delegated likewise.
    """
    _require_frame(frame, "frame")
    _require_stratum_column(frame)
    return chronological_split(frame, train_fraction, validation_fraction)


def randomized_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the rows whose stratum is ``randomized``."""
    _require_frame(frame, "frame")
    _require_stratum_column(frame)
    return frame.loc[frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED].copy()


def safety_censored_count(frame: pd.DataFrame) -> int:
    """Return how many rows carry the ``safety_censored`` stratum."""
    _require_frame(frame, "frame")
    _require_stratum_column(frame)
    return int((frame[STRATUM_COLUMN] == STRATUM_SAFETY_CENSORED).sum())
