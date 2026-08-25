"""Ground-truth and evaluation reporting for the Day 4 synthetic environment.

Label discipline (plan decision D6): every output of this module is labeled
either SIMULATED GROUND TRUTH -- quantities known by construction from the
declarative treatment-policy configuration -- or OBSERVED SIMULATED OUTCOME --
quantities measured from this synthetic run of the simulator.

Nothing here computes a causal estimate; cross-arm naive differences are
confounded by the documented eligibility-selection mechanism (the stage-1
safety gate routes context-dependent rows into safety-censored CONTROL, so
compared arms mix different strata) and must never be read as causal.

All functions are pure over the JOINED frame -- payment-attempt context
(``amount_inr`` at minimum) plus the treatment columns ``assigned_action`` /
``arm_source`` plus the outcome columns ``simulated_recovered`` /
``simulated_recovered_amount_inr``: no wall clock, no randomness, no I/O.
Identical inputs yield identical structures and byte-identical rendered
summaries. Missing required columns raise ``ValueError`` naming every missing
one; an empty frame yields valid zeroed results with labels intact.

Source discipline (plan decisions D1/D6): per-arm counts in
``summarize_arms`` and the assignment-probability ranges in
``overlap_diagnostics`` separate ``randomized`` versus ``safety_censored``
sources. CONTROL's aggregate ``recovery_rate`` and
``recovered_amount_inr_total`` intentionally include safety-censored rows
(descriptive); ``observed_differences`` uses randomized rows only, for both
the CONTROL baseline and every treated arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.config import InteractionRule, TreatmentPolicy

LABEL_GROUND_TRUTH = "SIMULATED GROUND TRUTH"
LABEL_OBSERVED = "OBSERVED SIMULATED OUTCOME"

BASELINE_ARM = "CONTROL"

ARM_ORDER: tuple[str, ...] = (
    "CONTROL",
    "RETRY_NOW",
    "RETRY_LATER",
    "REQUEST_UPDATE",
    "HUMAN_REVIEW",
)
TREATED_ARM_ORDER: tuple[str, ...] = tuple(
    arm for arm in ARM_ORDER if arm != BASELINE_ARM
)

ARM_SOURCE_RANDOMIZED = "randomized"
ARM_SOURCE_SAFETY_CENSORED = "safety_censored"

MIN_CASES_PER_STABLE_DIFFERENCE = 30

_REQUIRED_COLUMNS = (
    "amount_inr",
    "assigned_action",
    "arm_source",
    "simulated_recovered",
    "simulated_recovered_amount_inr",
)
_OVERLAP_EXTRA_COLUMNS = ("assignment_probability", "propensity_under_assignment")

_DIFFERENCE_NOTE = (
    "Naive cross-arm differences on OBSERVED SIMULATED OUTCOMES: they are "
    "confounded by eligibility selection (the stage-1 safety gate routes "
    "context-dependent rows to safety-censored CONTROL, so compared arms draw "
    "from different strata) and are therefore not causal."
)

_POSITIVITY_NOTE = (
    "Positivity holds within the eligible stratum by construction: every "
    "policy-eligible row kept positive stage-2 probability for every arm. "
    "Full-population estimands are not supported because eligibility was "
    "context-dependent."
)


def _require_columns(
    df: pd.DataFrame, function_name: str, extra: tuple[str, ...] = ()
) -> None:
    missing = [
        name for name in (*_REQUIRED_COLUMNS, *extra) if name not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{function_name} requires these joined-frame columns but they are "
            f"missing: {missing}"
        )


def _arm_mask(df: pd.DataFrame, arm: str) -> pd.Series:
    return df["assigned_action"] == arm


def _subset_metrics(frame: pd.DataFrame) -> tuple[int, float, float]:
    """Return (count, recovery rate, revenue per case) over a row subset."""
    count = int(len(frame))
    if count == 0:
        return 0, 0.0, 0.0
    rate = float(frame["simulated_recovered"].astype(float).mean())
    revenue_per_case = (
        float(frame["simulated_recovered_amount_inr"].astype(float).sum()) / count
    )
    return count, rate, revenue_per_case


def _value_range(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "max": 0.0}
    return {"min": float(values.min()), "max": float(values.max())}


def summarize_arms(df: pd.DataFrame) -> dict:
    """Per-arm counts split by source, recovery rates, recovered-INR totals.

    Label: OBSERVED SIMULATED OUTCOME -- every number is measured from this
    synthetic run. All five canonical arms appear in fixed order; an arm
    absent from the frame reports zero counts, ``recovery_rate`` 0.0, and a
    total of 0.0. CONTROL additionally carries ``randomized_recovery_rate``
    -- the rate over its randomized rows only -- so the gap between its
    aggregate rate (which includes safety-censored rows) and the randomized
    baseline used by ``observed_differences`` is locally explainable.
    """
    _require_columns(df, "summarize_arms")
    arms: dict[str, dict] = {}
    for arm in ARM_ORDER:
        rows = df.loc[_arm_mask(df, arm)]
        count = int(len(rows))
        if count == 0:
            entry: dict = {
                "count": 0,
                "randomized_count": 0,
                "safety_censored_count": 0,
                "recovery_rate": 0.0,
                "recovered_amount_inr_total": 0.0,
            }
        else:
            sources = rows["arm_source"]
            entry = {
                "count": count,
                "randomized_count": int((sources == ARM_SOURCE_RANDOMIZED).sum()),
                "safety_censored_count": int(
                    (sources == ARM_SOURCE_SAFETY_CENSORED).sum()
                ),
                "recovery_rate": float(
                    rows["simulated_recovered"].astype(float).mean()
                ),
                "recovered_amount_inr_total": round(
                    float(rows["simulated_recovered_amount_inr"].astype(float).sum()),
                    2,
                ),
            }
        if arm == BASELINE_ARM:
            randomized_rows = rows.loc[rows["arm_source"] == ARM_SOURCE_RANDOMIZED]
            entry["randomized_recovery_rate"] = (
                float(randomized_rows["simulated_recovered"].astype(float).mean())
                if int(len(randomized_rows))
                else 0.0
            )
        arms[arm] = entry
    return {"label": LABEL_OBSERVED, "case_count": int(len(df)), "arms": arms}


def observed_differences(df: pd.DataFrame) -> dict:
    """Naive treated-minus-control differences on OBSERVED SIMULATED OUTCOMES.

    The baseline arm is CONTROL restricted to randomized rows; treated arms
    likewise contribute randomized rows only. Differences are
    adjusted-for-nothing and carry the fixed disclaimer note.
    ``count_caveat`` is None when both sides have at least
    ``MIN_CASES_PER_STABLE_DIFFERENCE`` cases, otherwise a string naming both
    counts.
    """
    _require_columns(df, "observed_differences")
    randomized = (df["arm_source"] == ARM_SOURCE_RANDOMIZED).to_numpy()
    control_n, control_rate, control_revenue = _subset_metrics(
        df.loc[_arm_mask(df, BASELINE_ARM).to_numpy() & randomized]
    )
    treated: dict[str, dict] = {}
    for arm in TREATED_ARM_ORDER:
        arm_n, arm_rate, arm_revenue = _subset_metrics(
            df.loc[_arm_mask(df, arm).to_numpy() & randomized]
        )
        caveat: str | None = None
        if (
            arm_n < MIN_CASES_PER_STABLE_DIFFERENCE
            or control_n < MIN_CASES_PER_STABLE_DIFFERENCE
        ):
            caveat = (
                f"count caveat: {arm} n={arm_n} vs randomized CONTROL n={control_n}; "
                f"fewer than {MIN_CASES_PER_STABLE_DIFFERENCE} cases on either side "
                "makes this difference unstable"
            )
        treated[arm] = {
            "recovery_rate_difference": round(arm_rate - control_rate, 4),
            "revenue_per_case_difference_inr": round(
                arm_revenue - control_revenue, 2
            ),
            "count_caveat": caveat,
        }
    return {
        "label": LABEL_OBSERVED,
        "note": _DIFFERENCE_NOTE,
        "baseline_arm": BASELINE_ARM,
        "treated_arms": treated,
    }


def _interaction_condition(rule: InteractionRule) -> str:
    if rule.equals_value is not None:
        return f"{rule.column} == {rule.equals_value}"
    return f"{rule.column} >= {rule.min_threshold}"


def ground_truth_table(policy: TreatmentPolicy) -> dict:
    """The configured synthetic-world parameters as SIMULATED GROUND TRUTH.

    Every value is known by construction from the loaded policy; mappings are
    copied so callers can never mutate policy internals through the result.
    """
    terms = policy.base_propensity_terms
    return {
        "label": LABEL_GROUND_TRUTH,
        "master_seed": int(policy.master_seed),
        "noise_sigma_logit": float(policy.noise_sigma_logit),
        "arm_probabilities": {
            arm: float(value) for arm, value in policy.arm_probabilities.items()
        },
        "main_effects_logit": {
            arm: float(value) for arm, value in policy.main_effects_logit.items()
        },
        "interactions": [
            {
                "action": rule.action,
                "column": rule.column,
                "condition": _interaction_condition(rule),
                "effect_logit": float(rule.effect_logit),
            }
            for rule in policy.interactions
        ],
        "base_propensity_terms": {
            "intercept": float(terms.intercept),
            "category_effects": {
                name: float(value)
                for name, value in terms.category_effects.items()
            },
            "successful_payment_count_log1p": float(
                terms.successful_payment_count_log1p
            ),
            "historical_recovery_count_min5": float(
                terms.historical_recovery_count_min5
            ),
            "attempt_number_prior_offset": float(terms.attempt_number_prior_offset),
            "fraud_risk": float(terms.fraud_risk),
            "amount_log1p_per_k": float(terms.amount_log1p_per_k),
            "method_upi": float(terms.method_upi),
            "device_android": float(terms.device_android),
        },
    }


def overlap_diagnostics(df: pd.DataFrame) -> dict:
    """Assignment-probability and propensity ranges per arm, OBSERVED label.

    Assignment-probability ranges are split by ``arm_source``: treated arms
    report randomized rows only; CONTROL reports randomized versus
    safety-censored separately with distinct min/max (censored rows carry
    exactly 0.0 because no stage-2 draw occurred). Propensity ranges cover
    every row of the arm. Empty subsets report the zeroed range
    {"min": 0.0, "max": 0.0}.
    """
    # Deviation note (reviewer): per-arm randomized/censored split counts live in summarize_arms; this function reports global eligible/safety_censored counts plus probability ranges.
    _require_columns(df, "overlap_diagnostics", extra=_OVERLAP_EXTRA_COLUMNS)
    randomized = (df["arm_source"] == ARM_SOURCE_RANDOMIZED).to_numpy()
    censored = (df["arm_source"] == ARM_SOURCE_SAFETY_CENSORED).to_numpy()
    probabilities = df["assignment_probability"].astype(float).to_numpy()
    propensities = df["propensity_under_assignment"].astype(float).to_numpy()

    assignment_ranges: dict[str, dict] = {}
    propensity_ranges: dict[str, dict] = {}
    for arm in ARM_ORDER:
        arm_mask = _arm_mask(df, arm).to_numpy()
        ranges: dict[str, dict] = {
            ARM_SOURCE_RANDOMIZED: _value_range(probabilities[arm_mask & randomized])
        }
        if arm == BASELINE_ARM:
            ranges[ARM_SOURCE_SAFETY_CENSORED] = _value_range(
                probabilities[arm_mask & censored]
            )
        assignment_ranges[arm] = ranges
        propensity_ranges[arm] = _value_range(propensities[arm_mask])

    return {
        "label": LABEL_OBSERVED,
        "eligible_count": int(randomized.sum()),
        "safety_censored_count": int(censored.sum()),
        "assignment_probability_ranges": assignment_ranges,
        "propensity_under_assignment_ranges": propensity_ranges,
        "positivity_note": _POSITIVITY_NOTE,
    }


def render_summary(df: pd.DataFrame, policy: TreatmentPolicy) -> str:
    """Deterministic multi-line text combining sections 1-4 with their labels.

    Pure assembly over the same pure computations: byte-identical for
    identical inputs, no timestamps anywhere.
    """
    arms_summary = summarize_arms(df)
    differences = observed_differences(df)
    truth = ground_truth_table(policy)
    overlap = overlap_diagnostics(df)

    lines: list[str] = ["RecoverAI Day 4 simulated treatment/outcome summary", ""]
    lines.append(f"[{LABEL_OBSERVED}] arm summary")
    lines.append(f"cases: {arms_summary['case_count']}")
    for arm, stats in arms_summary["arms"].items():
        lines.append(
            f"  {arm}: count={stats['count']} "
            f"randomized={stats['randomized_count']} "
            f"safety_censored={stats['safety_censored_count']} "
            f"recovery_rate={stats['recovery_rate']:.4f} "
            f"recovered_amount_inr_total={stats['recovered_amount_inr_total']:.2f}"
        )
    lines.append("")
    lines.append(f"[{LABEL_OBSERVED}] differences vs {differences['baseline_arm']}")
    lines.append(f"note: {differences['note']}")
    for arm, entry in differences["treated_arms"].items():
        lines.append(
            f"  {arm}: recovery_rate_difference="
            f"{entry['recovery_rate_difference']:.4f} "
            f"revenue_per_case_difference_inr="
            f"{entry['revenue_per_case_difference_inr']:.2f} "
            f"count_caveat={entry['count_caveat']!r}"
        )
    lines.append("")
    lines.append(f"[{LABEL_GROUND_TRUTH}] policy table")
    lines.append(f"master_seed={truth['master_seed']}")
    lines.append(f"noise_sigma_logit={truth['noise_sigma_logit']!r}")
    lines.append(f"arm_probabilities={truth['arm_probabilities']!r}")
    lines.append(f"main_effects_logit={truth['main_effects_logit']!r}")
    for interaction in truth["interactions"]:
        lines.append(
            f"interaction: action={interaction['action']} "
            f"condition={interaction['condition']} "
            f"effect_logit={interaction['effect_logit']!r}"
        )
    lines.append(f"base_propensity_terms={truth['base_propensity_terms']!r}")
    lines.append("")
    lines.append(f"[{LABEL_OBSERVED}] overlap diagnostics")
    lines.append(
        f"eligible_count={overlap['eligible_count']} "
        f"safety_censored_count={overlap['safety_censored_count']}"
    )
    for arm, ranges in overlap["assignment_probability_ranges"].items():
        rendered = [
            f"{source_name}=[{bounds['min']!r}, {bounds['max']!r}]"
            for source_name, bounds in ranges.items()
        ]
        lines.append(f"  {arm}: assignment_probability {' '.join(rendered)}")
    for arm, bounds in overlap["propensity_under_assignment_ranges"].items():
        lines.append(
            f"  {arm}: propensity_under_assignment="
            f"[{bounds['min']!r}, {bounds['max']!r}]"
        )
    lines.append(f"positivity_note: {overlap['positivity_note']}")
    return "\n".join(lines) + "\n"
