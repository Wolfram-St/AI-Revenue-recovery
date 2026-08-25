"""Two-stage treatment assignment for the Day 4 simulated environment.

Assignment follows plan decision D1 as a two-stage hybrid. Stage 1 is a
deterministic safety-eligibility gate: the shipped ``decide_action`` policy is
called directly per row (the engine's ERV/no-op path is not part of
assignment) over the row's decision-time context, with the caller-supplied
recovery probability injected as ``recovery_probability``; any row whose
authorized action is ``STOP`` is forced to CONTROL as a "safety-censored
control". Stage 2 randomizes every remaining eligible row across the
configured arms.

Documented selection confounding (D1): eligibility depends on context AND on
the model-derived recovery probability (rules R006/R007/R008 consume it), so
the probability of being assignable varies with context. Within the eligible
pool assignment is randomized, therefore cross-arm comparisons are
adjusted-for-nothing naive differences and only within-eligible-pool contrasts
approach unconfoundedness.

Seed discipline (D1b): stage 2 uses seed-stream child SEED_STREAM_ASSIGNMENT
derived via Generator.spawn from master_seed (never a bare ``default_rng``
re-derivation of a fresh stream), so outcome and temporal draws can never
reuse this stream.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from recovery.policy import PolicyConfig, decide_action, load_policy_config
from simulation.config import SEED_STREAM_ASSIGNMENT, TreatmentPolicy

RESULT_COLUMNS = ("assigned_action", "arm_source", "assignment_probability")

_ARM_SOURCE_RANDOMIZED = "randomized"
_ARM_SOURCE_SAFETY_CENSORED = "safety_censored"

_ARM_DRAW_ORDER = (
    "CONTROL",
    "RETRY_NOW",
    "RETRY_LATER",
    "REQUEST_UPDATE",
    "HUMAN_REVIEW",
)

_GATE_CONTEXT_COLUMNS = (
    "customer_opted_out",
    "fraud_risk",
    "failure_category",
    "attempt_number",
    "amount_inr",
)


def _reject_nan_context(df: pd.DataFrame) -> None:
    offenders: list[str] = []
    for column in _GATE_CONTEXT_COLUMNS:
        if column not in df.columns:
            continue
        missing_count = int(df[column].isna().sum())
        if missing_count:
            offenders.append(f"{column}={missing_count} row(s)")
    if offenders:
        raise ValueError(
            "policy-referenced context columns contain NaN/None values, which "
            "would silently compare False inside STOP-rule conditions and "
            "corrupt safety censoring: " + ", ".join(offenders)
        )


def _validated_probabilities(probabilities: object, n_rows: int) -> list[float]:
    try:
        materialized = list(probabilities)
    except TypeError:
        raise ValueError(
            "probabilities must be a finite-length sequence aligned to df rows"
        ) from None
    if len(materialized) != n_rows:
        raise ValueError(
            f"probabilities length {len(materialized)} does not match df row "
            f"count {n_rows}"
        )
    validated: list[float] = []
    for position, raw in enumerate(materialized):
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"probabilities[{position}] must be a real number, got {raw!r}"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(
                f"probabilities[{position}] must be finite, got {raw!r}"
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"probabilities[{position}] must lie in [0, 1], got {value!r}"
            )
        validated.append(value)
    return validated


def _arm_draw_arrays(policy: TreatmentPolicy) -> tuple[np.ndarray, np.ndarray]:
    unexpected = sorted(set(policy.arm_probabilities) - set(_ARM_DRAW_ORDER))
    missing = sorted(set(_ARM_DRAW_ORDER) - set(policy.arm_probabilities))
    if unexpected or missing:
        raise ValueError(
            "policy arm_probabilities keys do not match the canonical arm set: "
            f"unexpected {unexpected}, missing {missing}"
        )
    names = np.asarray(_ARM_DRAW_ORDER, dtype=object)
    probabilities = np.asarray(
        [policy.arm_probabilities[arm] for arm in _ARM_DRAW_ORDER], dtype=float
    )
    return names, probabilities


def assign_treatments(
    df: pd.DataFrame,
    probabilities: object,
    policy: TreatmentPolicy,
    policy_config: PolicyConfig | None = None,
) -> pd.DataFrame:
    """Assign treatments via safety gate then randomized arms (plan D1/D1b).

    Returns a NEW frame indexed like ``df`` with columns exactly
    ``assigned_action``, ``arm_source``, ``assignment_probability``. The input
    frame and the probabilities sequence are never mutated; identical inputs
    and policy yield byte-identical output because stage 2 makes one single
    vectorized draw over eligible positions from seed-stream child 0. Missing
    values in any policy-referenced context column are rejected before the
    gate, because NaN/None compares False inside rule conditions and would
    silently bypass STOP rules, corrupting safety censoring.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    _reject_nan_context(df)
    validated = _validated_probabilities(probabilities, len(df))

    if policy_config is None:
        policy_config = load_policy_config()

    n_rows = len(df)
    assigned: list[str | None] = [None] * n_rows
    sources: list[str | None] = [None] * n_rows
    recorded: list[float] = [float("nan")] * n_rows
    eligible_positions: list[int] = []

    for position in range(n_rows):
        context = dict(df.iloc[position])
        context["recovery_probability"] = validated[position]
        decision = decide_action(context, policy_config)
        if decision.authorized_action == "STOP":
            assigned[position] = "CONTROL"
            sources[position] = _ARM_SOURCE_SAFETY_CENSORED
            recorded[position] = 0.0
        else:
            sources[position] = _ARM_SOURCE_RANDOMIZED
            eligible_positions.append(position)

    if eligible_positions:
        arm_names, arm_probabilities = _arm_draw_arrays(policy)
        assignment_rng = np.random.default_rng(policy.master_seed).spawn(
            SEED_STREAM_ASSIGNMENT + 1
        )[SEED_STREAM_ASSIGNMENT]
        draws = assignment_rng.choice(
            arm_names, size=len(eligible_positions), p=arm_probabilities
        )
        for offset, position in enumerate(eligible_positions):
            arm = str(draws[offset])
            assigned[position] = arm
            recorded[position] = float(policy.arm_probabilities[arm])

    return pd.DataFrame(
        {
            "assigned_action": pd.Series(assigned, index=df.index, dtype="object"),
            "arm_source": pd.Series(sources, index=df.index, dtype="object"),
            "assignment_probability": pd.Series(recorded, index=df.index, dtype="float64"),
        }
    )
