"""Simulated action-aware outcomes with stored SYNTHETIC GROUND TRUTH.

This module generates SIMULATED action-aware recovery outcomes for the Day 4
synthetic treatment environment together with stored SYNTHETIC GROUND TRUTH
columns. Nothing here estimates or claims real-world causal effects: the
simulator IS the world, every propensity below is a fixed synthetic
assumption loaded from the declarative policy, and every emitted field
describes the simulated environment only.

Only decision-time context columns plus the treatment-assignment metadata
column ``assigned_action`` are consumed; no post-decision, timing, or label
information is ever read, so the function remains safe to run at decision
time inside the simulation.

Seed discipline (plan decision D1b): every stochastic draw comes exclusively
from seed-stream child ``SEED_STREAM_OUTCOMES``, obtained as element
``SEED_STREAM_OUTCOMES`` of ``default_rng(policy.master_seed).spawn(SEED_STREAM_OUTCOMES + 1)[SEED_STREAM_OUTCOMES]``
-- never from a freshly re-derived stream -- so assignment, outcome, and
temporal stages can never share or reorder streams. Within the child
generator the draw order is FIXED: one vectorized uniform batch first, then
one Gaussian logit-noise batch, then the Bernoulli comparison. Identical
inputs and policy therefore reproduce byte-identical output.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from simulation.config import (
    CANONICAL_ARMS,
    CANONICAL_CATEGORIES,
    SEED_STREAM_OUTCOMES,
    TreatmentPolicy,
)

RESULT_COLUMNS = (
    "simulated_recovered",
    "base_recovery_propensity",
    "action_effect_logit",
    "propensity_under_assignment",
)

_REQUIRED_COLUMNS = (
    "amount_inr",
    "attempt_number",
    "assigned_action",
    "device_type",
    "failure_category",
    "fraud_risk",
    "historical_recovery_count",
    "payment_method",
    "successful_payment_count",
)

_HISTORICAL_RECOVERY_CAP = 5.0
_AMOUNT_SCALE_INR = 1000.0
_PROPENSITY_DECIMALS = 6
_LOGIT_REPRESENTABLE_BOUND = 30.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic transform: exp() only ever sees non-positive
    arguments, so no finite input can overflow (extreme magnitudes saturate)."""
    non_positive = np.where(values >= 0.0, -values, values)
    exponential = np.exp(non_positive)
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + exponential),
        exponential / (1.0 + exponential),
    )


def _reject_missing_columns(df: pd.DataFrame) -> None:
    missing = [name for name in _REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(
            "simulate_outcomes requires these decision-time/assignment "
            f"columns but they are missing: {missing}"
        )


def _reject_nan_columns(df: pd.DataFrame) -> None:
    offenders: list[str] = []
    for name in _REQUIRED_COLUMNS:
        nan_count = int(df[name].isna().sum())
        if nan_count:
            offenders.append(f"{name}={nan_count} row(s)")
    if offenders:
        raise ValueError(
            "consumed columns contain NaN/None values, which would silently "
            "propagate into NaN ground-truth propensities and force every "
            "noisy Bernoulli comparison to False (simulated_recovered=0) "
            "without any error, mirroring the treatment.py guard rationale: "
            + ", ".join(offenders)
        )


def _reject_unknown_arms(assigned_values: np.ndarray) -> None:
    offenders = sorted(set(assigned_values.tolist()) - set(CANONICAL_ARMS))
    if offenders:
        raise ValueError(
            "assigned_action values outside the canonical arm set: "
            f"{offenders}; expected a subset of {sorted(CANONICAL_ARMS)}"
        )


def _reject_unknown_categories(category_values: np.ndarray) -> None:
    offenders = sorted(set(category_values.tolist()) - set(CANONICAL_CATEGORIES))
    if offenders:
        raise ValueError(
            "failure_category values outside the canonical enum: "
            f"{offenders}; expected a subset of {sorted(CANONICAL_CATEGORIES)}"
        )


def simulate_outcomes(
    df_with_assignments: pd.DataFrame, policy: TreatmentPolicy
) -> pd.DataFrame:
    """Return SIMULATED outcomes plus SYNTHETIC GROUND TRUTH for each row.

    For row i with assigned arm ``a`` (plan decision D2):

    * ``base_logit(i)`` -- documented intercept/category/history/attempt/
      fraud/amount/method/device family mirroring the Day 1 generator;
    * ``effect_logit(i, a)`` -- ``main_effects_logit[a]`` plus every
      configured interaction whose rule matches arm ``a`` and the row's
      ``failure_category`` / ``attempt_number``;
    * ``propensity_under_assignment = sigmoid(base_logit + effect_logit)``
      is the PRE-noise probability actually drawn against.

    Returns a NEW frame indexed like ``df_with_assignments`` with columns
    exactly ``RESULT_COLUMNS``: an int8 ``simulated_recovered`` label and the
    three float ground-truth columns rounded to 6 decimals for stability.
    The input frame and policy are never mutated; identical inputs and policy
    yield byte-identical output because all randomness comes from the single
    fixed-order seed-stream child described in the module docstring.

    Raises ``ValueError`` naming the offending items when required columns
    are missing, any consumed column holds NaN/None, ``assigned_action``
    leaves the canonical arm set, ``failure_category`` leaves the canonical
    enum, or the computed base/assignment logits leave the representable
    synthetic range (|logit| > 30) instead of silently saturating every
    propensity to 0 or 1.
    """
    if not isinstance(df_with_assignments, pd.DataFrame):
        raise ValueError(
            f"df must be a pandas DataFrame, got {type(df_with_assignments).__name__}"
        )
    _reject_missing_columns(df_with_assignments)
    _reject_nan_columns(df_with_assignments)

    sigma = float(policy.noise_sigma_logit)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            f"policy.noise_sigma_logit must be finite and positive, got {sigma!r}"
        )

    assigned_values = df_with_assignments["assigned_action"].to_numpy()
    category_values = df_with_assignments["failure_category"].to_numpy()
    _reject_unknown_arms(assigned_values)
    _reject_unknown_categories(category_values)

    terms = policy.base_propensity_terms
    n_rows = len(df_with_assignments)

    successful = df_with_assignments["successful_payment_count"].to_numpy(dtype=float)
    historical = df_with_assignments["historical_recovery_count"].to_numpy(dtype=float)
    attempt_number = df_with_assignments["attempt_number"].to_numpy(dtype=float)
    fraud_risk = df_with_assignments["fraud_risk"].astype(float).to_numpy()
    amount_inr = df_with_assignments["amount_inr"].to_numpy(dtype=float)
    method_upi = (df_with_assignments["payment_method"] == "upi").to_numpy(dtype=float)
    device_android = (df_with_assignments["device_type"] == "android").to_numpy(dtype=float)

    category_term = np.array(
        [terms.category_effects[category] for category in category_values],
        dtype=float,
    )
    base_logits = (
        float(terms.intercept)
        + category_term
        + float(terms.successful_payment_count_log1p) * np.log1p(successful)
        + float(terms.historical_recovery_count_min5)
        * np.minimum(historical, _HISTORICAL_RECOVERY_CAP)
        + float(terms.attempt_number_prior_offset)
        * np.maximum(attempt_number - 1.0, 0.0)
        + float(terms.fraud_risk) * fraud_risk
        + float(terms.amount_log1p_per_k)
        * np.log1p(amount_inr / _AMOUNT_SCALE_INR)
        + float(terms.method_upi) * method_upi
        + float(terms.device_android) * device_android
    )

    effect_logits = np.array(
        [policy.main_effects_logit[arm] for arm in assigned_values], dtype=float
    )
    for rule in policy.interactions:
        fires = assigned_values == rule.action
        if rule.column == "failure_category":
            fires = fires & (category_values == rule.equals_value)
        else:
            fires = fires & (attempt_number >= float(rule.min_threshold))
        effect_logits = effect_logits + np.asarray(fires, dtype=float) * float(
            rule.effect_logit
        )

    logits = np.concatenate([base_logits, effect_logits])
    if n_rows > 0 and not np.all(np.isfinite(logits)):
        raise ValueError(
            "computed base/effect logits must stay finite; check the supplied "
            "policy coefficients and consumed column values"
        )

    assignment_logits = base_logits + effect_logits
    if n_rows > 0:
        checked_logits = np.concatenate([base_logits, assignment_logits])
        if not np.all(np.isfinite(checked_logits)):
            raise ValueError(
                "computed base/effect logits must stay finite; check the supplied "
                "policy coefficients and consumed column values"
            )
        if float(np.max(np.abs(checked_logits))) > _LOGIT_REPRESENTABLE_BOUND:
            raise ValueError(
                "logit out of representable synthetic range — check "
                "base_propensity_terms configuration"
            )

    base_recovery_propensity = _sigmoid(base_logits)
    propensity_under_assignment = _sigmoid(assignment_logits)

    # Single fixed-order draw pair from the mandated spawn child: uniforms
    # first, then Gaussian logit noise, then the Bernoulli comparison.
    outcome_rng = np.random.default_rng(policy.master_seed).spawn(
        SEED_STREAM_OUTCOMES + 1
    )[SEED_STREAM_OUTCOMES]
    uniforms = outcome_rng.random(n_rows)
    logit_noise = outcome_rng.normal(0.0, sigma, n_rows)
    recovered = (uniforms < _sigmoid(assignment_logits + logit_noise)).astype(np.int8)

    return pd.DataFrame(
        {
            "simulated_recovered": pd.Series(
                recovered, index=df_with_assignments.index, dtype="int8"
            ),
            "base_recovery_propensity": pd.Series(
                np.round(base_recovery_propensity, _PROPENSITY_DECIMALS),
                index=df_with_assignments.index,
                dtype="float64",
            ),
            "action_effect_logit": pd.Series(
                np.round(effect_logits, _PROPENSITY_DECIMALS),
                index=df_with_assignments.index,
                dtype="float64",
            ),
            "propensity_under_assignment": pd.Series(
                np.round(propensity_under_assignment, _PROPENSITY_DECIMALS),
                index=df_with_assignments.index,
                dtype="float64",
            ),
        },
        index=df_with_assignments.index,
    )
