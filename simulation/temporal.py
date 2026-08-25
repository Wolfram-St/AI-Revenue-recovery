"""Temporal timeline stamping AFTER payment failure (plan decision D4).

This module stamps the treatment timeline strictly after each recorded
payment failure. Treated rows satisfy ``failure_ts < treatment_ts <
outcome_ts`` where ``treatment_ts = failure_ts + arm delay`` (per-arm
configured latency) and ``outcome_ts = treatment_ts + uniform resolution
window. CONTROL rows mark NO intervention at all: their
``treatment_timestamp`` is the documented null semantic ``pd.NaT``, and
their observation horizon starts at the failure itself --
``outcome_ts = failure_ts + uniform resolution window``.

Seed discipline (plan decision D1b): every stochastic draw comes exclusively
from seed-stream child ``SEED_STREAM_TEMPORAL``, obtained as element
``SEED_STREAM_TEMPORAL`` of ``default_rng(policy.master_seed).spawn(SEED_STREAM_TEMPORAL + 1)``
-- never from a freshly re-derived stream -- so assignment, outcome, and
temporal stages can never share or reorder streams. No wall clock is read
anywhere in this module; identical inputs and policy therefore reproduce
byte-identical timelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.config import CANONICAL_ARMS, SEED_STREAM_TEMPORAL, TreatmentPolicy

RESULT_COLUMNS = ("treatment_timestamp", "outcome_timestamp")

_REQUIRED_COLUMNS = ("attempt_id", "event_timestamp", "assigned_action", "arm_source")

_CONTROL_ARM = "CONTROL"
_UTC_DTYPE = "datetime64[ns, UTC]"


def _reject_missing_columns(df: pd.DataFrame) -> None:
    missing = [name for name in _REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(
            "stamp_treatment_timeline requires these identity/assignment "
            f"columns but they are missing: {missing}"
        )


def _normalized_event_timestamps(df: pd.DataFrame) -> pd.Series:
    raw = df["event_timestamp"]
    normalized = pd.to_datetime(raw, utc=True, errors="coerce", format="mixed")
    bad_positions = np.flatnonzero(normalized.isna().to_numpy(dtype=bool))
    if bad_positions.size:
        offenders = df.index[bad_positions].tolist()
        raise ValueError(
            "event_timestamp holds unparsable or NaT values that could not be "
            f"normalized to UTC; offending row indices: {offenders}"
        )
    return normalized.astype(_UTC_DTYPE)


def _reject_unknown_arms(assigned_values: np.ndarray) -> None:
    # Order-stable, type-safe dedupe: sorting a mixed str/NaN offender set
    # would itself raise TypeError before the loud ValueError is emitted.
    uniques = dict.fromkeys(assigned_values.tolist())
    offenders = [value for value in uniques if value not in CANONICAL_ARMS]
    if offenders:
        raise ValueError(
            "assigned_action values outside the canonical arm set: "
            f"{offenders}; expected a subset of {sorted(CANONICAL_ARMS)}"
        )


def stamp_treatment_timeline(
    df: pd.DataFrame, policy: TreatmentPolicy
) -> pd.DataFrame:
    """Stamp treatment/outcome timestamps strictly after payment failure.

    Plan decision D4 semantics, applied per row of ``df``:

    * treated arms (``assigned_action != "CONTROL"``):
      ``treatment_timestamp = event_timestamp +
      policy.treatment_delay_hours[arm]`` and ``outcome_timestamp =
      treatment_timestamp + u_i`` with ``u_i`` drawn uniformly over
      ``policy.resolution_window_hours``;
    * CONTROL rows receive NO intervention: ``treatment_timestamp`` is the
      documented null ``pd.NaT`` and ``outcome_timestamp =
      event_timestamp + u_i`` (observation horizon anchored at failure).

    Returns a NEW frame indexed like ``df`` holding every original column
    unchanged plus ``RESULT_COLUMNS``, both typed ``datetime64[ns, UTC]``;
    the input frame is never mutated. Identical inputs and policy yield
    byte-identical output because exactly ONE vectorized uniform batch over
    ALL rows in row order is consumed from seed-stream child
    SEED_STREAM_TEMPORAL (D1b), after every validation has passed, and no
    wall clock is ever consulted.

    Raises ``ValueError`` naming the offending items when required columns
    are missing, any ``event_timestamp`` value cannot be normalized to UTC
    (literal NaT inputs included), or ``assigned_action`` leaves the
    canonical arm set. Naive timestamp strings are interpreted as UTC by
    design; timezone-aware inputs normalize to their true UTC instants.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    _reject_missing_columns(df)
    event_timestamps = _normalized_event_timestamps(df)

    arm_values = df["assigned_action"].to_numpy()
    _reject_unknown_arms(arm_values)

    n_rows = len(df)
    low, high = (float(bound) for bound in policy.resolution_window_hours)
    treated_mask = pd.Series(arm_values != _CONTROL_ARM, index=df.index)

    # Single fixed-order draw from the mandated spawn child: one vectorized
    # uniform batch over ALL rows in row order, taken only after validation.
    temporal_rng = np.random.default_rng(policy.master_seed).spawn(
        SEED_STREAM_TEMPORAL + 1
    )[SEED_STREAM_TEMPORAL]
    window_hours = temporal_rng.uniform(low, high, size=n_rows)

    delay_lookup = dict(policy.treatment_delay_hours)
    delay_hours = np.array(
        [delay_lookup.get(str(arm), 0.0) for arm in arm_values], dtype=float
    )

    delay_deltas = pd.to_timedelta(delay_hours, unit="h").to_numpy()
    window_deltas = pd.to_timedelta(window_hours, unit="h").to_numpy()

    stamped_treatment = event_timestamps + delay_deltas
    treatment_series = stamped_treatment.where(treated_mask, other=pd.NaT).astype(
        _UTC_DTYPE
    )
    outcome_series = (
        (event_timestamps + window_deltas)
        .mask(treated_mask, other=stamped_treatment + window_deltas)
        .astype(_UTC_DTYPE)
    )

    result = df.copy(deep=True)
    result["treatment_timestamp"] = treatment_series
    result["outcome_timestamp"] = outcome_series
    return result
