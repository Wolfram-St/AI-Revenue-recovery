"""ADDITIVE action-aware treatment/outcome dataset contract (plan Task 5, D5).

This module defines the SIMULATED treatment/outcome dataset: an additive
schema joined to the existing payment attempts one-to-one on ``attempt_id``.
Every field carries an explicit D5 classification in ``FIELD_CLASSIFICATION``
(identifier / decision_time_feature / treatment_metadata / outcome_timing /
outcome / ground_truth / evaluation_metadata); today no field is classified
``decision_time_feature`` because all decision-time context remains in the
attempts frame by design.

The Day 2 baseline contract is untouched and structurally cannot consume any
of these fields: ``ml/features.py`` selects its features from an explicit
whitelist (NUMERIC_FEATURES + CATEGORICAL_FEATURES), so even a hostile frame
that concatenates treatment/outcome columns -- or deliberately poisoned decoy
columns -- onto the attempts context can never leak them into the baseline
feature matrix. The leakage-guard tests in ``tests/test_treatment_dataset.py``
prove this immunity end to end.

Nothing in this module touches a database -- it builds and validates pandas
frames only, no DB writes anywhere. The temporal columns are structural
placeholders until Task 6 stamps real timestamps; their ordering semantics
are enforced there, not here.
Unlike recovery.audit (which rejects NaT loudly), this validator
accepts NaT/None timestamps as legitimate pre-Task-6 placeholders and
CONTROL treatment-timestamp nulls; ordering enforcement arrives with the
timeline module.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from simulation.config import CANONICAL_ARMS

JOIN_KEY = "attempt_id"

ARM_SOURCE_RANDOMIZED = "randomized"
ARM_SOURCE_SAFETY_CENSORED = "safety_censored"
ARM_SOURCES = frozenset({ARM_SOURCE_RANDOMIZED, ARM_SOURCE_SAFETY_CENSORED})

FIELD_CLASSIFICATION: dict[str, str] = {
    "attempt_id": "identifier",
    "event_timestamp": "evaluation_metadata",
    "assigned_action": "treatment_metadata",
    "arm_source": "treatment_metadata",
    "assignment_probability": "treatment_metadata",
    "treatment_timestamp": "outcome_timing",
    "outcome_timestamp": "outcome_timing",
    "simulated_recovered": "outcome",
    "simulated_recovered_amount_inr": "outcome",
    "base_recovery_propensity": "ground_truth",
    "action_effect_logit": "ground_truth",
    "propensity_under_assignment": "ground_truth",
}

CANONICAL_COLUMNS: tuple[str, ...] = tuple(FIELD_CLASSIFICATION)

CLASSIFICATION_VOCABULARY = frozenset(
    {
        "identifier",
        "decision_time_feature",
        "treatment_metadata",
        "outcome_timing",
        "outcome",
        "ground_truth",
        "evaluation_metadata",
    }
)

_ATTEMPTS_CONTRIBUTION = ("attempt_id", "event_timestamp")
_ASSIGNMENTS_CONTRIBUTION = (
    "assigned_action",
    "arm_source",
    "assignment_probability",
)
_OUTCOMES_CONTRIBUTION = (
    "simulated_recovered",
    "simulated_recovered_amount_inr",
    "base_recovery_propensity",
    "action_effect_logit",
    "propensity_under_assignment",
)
_TIMELINE_CONTRIBUTION = ("attempt_id", "treatment_timestamp", "outcome_timestamp")

_TIMING_COLUMNS = ("treatment_timestamp", "outcome_timestamp")

_PROPENSITY_COLUMNS = ("base_recovery_propensity", "propensity_under_assignment")
_PROPENSITY_BOUNDS = (0.0, 1.0)
_EFFECT_LOGIT_BOUNDS = (-3.0, 3.0)

_TARGET_ROWS = 5000
_MAX_EXAMPLE_IDS = 3


def _require_frame(value: object, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(value).__name__}")


def _require_columns(frame: pd.DataFrame, needed: tuple[str, ...], name: str) -> None:
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _check_join_key(frame: pd.DataFrame, name: str) -> None:
    ids = frame["attempt_id"]
    null_count = int(ids.isna().sum())
    if null_count:
        raise ValueError(
            f"{name}.attempt_id contains {null_count} null value(s); the "
            "one-to-one join requires non-null keys"
        )
    if not ids.is_unique:
        raise ValueError(
            f"{name}.attempt_id must be unique for the one-to-one join; "
            f"{int(ids.duplicated().sum())} duplicate row(s) found"
        )


def _describe_id_difference(difference: set[str]) -> str:
    examples = sorted(difference)[:_MAX_EXAMPLE_IDS]
    rendered = ", ".join(repr(example) for example in examples)
    return f"{len(difference)} id(s) [examples: {rendered}]"


def _check_identical_id_sets(reference: pd.DataFrame, frame: pd.DataFrame, name: str) -> None:
    reference_set = set(reference["attempt_id"])
    frame_set = set(frame["attempt_id"])
    missing = reference_set - frame_set
    extra = frame_set - reference_set
    if missing or extra:
        parts = []
        if missing:
            parts.append(
                f"{name} is missing {_describe_id_difference(missing)} present in attempts_df"
            )
        if extra:
            parts.append(
                f"{name} carries {_describe_id_difference(extra)} absent from attempts_df"
            )
        raise ValueError(
            "attempt_id sets must be identical across frames: " + "; ".join(parts)
        )


def _check_column_collisions(
    contributions: list[tuple[str, tuple[str, ...]]],
    frames: list[pd.DataFrame],
) -> None:
    owners: dict[str, str] = {}
    for source_name, columns in contributions:
        for column in columns:
            if column != JOIN_KEY and column not in owners:
                owners[column] = source_name
    conflicts: list[str] = []
    for (source_name, _columns), frame in zip(contributions, frames):
        for column in frame.columns:
            if column == JOIN_KEY:
                continue
            owner = owners.get(column)
            if owner is not None and owner != source_name:
                conflicts.append(f"{column!r} owned by {owner} but also present in {source_name}")
    if conflicts:
        raise ValueError(
            "column collision across sources beyond each source's own canonical "
            "contribution: " + "; ".join(conflicts)
        )


def build_treatment_dataset(
    attempts_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    timeline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join attempts/assignments/outcomes(/timeline) into the D5 schema.

    Only ``attempt_id`` and ``event_timestamp`` are taken from the attempts
    frame -- the full decision-time context stays where it belongs. The four
    frames must carry non-null unique ``attempt_id`` values with IDENTICAL id
    sets; violations raise ``ValueError`` naming counts and up to three
    example ids. Any column collision beyond each source's own canonical
    contribution raises ``ValueError``.

    When ``timeline_df`` is omitted, ``treatment_timestamp`` and
    ``outcome_timestamp`` are created as object-dtype all-None placeholders
    until Task 6 stamps the real timeline. The result is a NEW frame in the
    canonical column order with a fresh RangeIndex; no input frame is mutated.
    """
    _require_frame(attempts_df, "attempts_df")
    _require_frame(assignments_df, "assignments_df")
    _require_frame(outcomes_df, "outcomes_df")
    contributions = [
        ("attempts_df", _ATTEMPTS_CONTRIBUTION),
        ("assignments_df", _ASSIGNMENTS_CONTRIBUTION),
        ("outcomes_df", _OUTCOMES_CONTRIBUTION),
    ]
    if timeline_df is not None:
        _require_frame(timeline_df, "timeline_df")
        contributions.append(("timeline_df", _TIMELINE_CONTRIBUTION))

    _require_columns(attempts_df, _ATTEMPTS_CONTRIBUTION, "attempts_df")
    _require_columns(assignments_df, (JOIN_KEY, *_ASSIGNMENTS_CONTRIBUTION), "assignments_df")
    _require_columns(outcomes_df, (JOIN_KEY, *_OUTCOMES_CONTRIBUTION), "outcomes_df")
    if timeline_df is not None:
        _require_columns(timeline_df, _TIMELINE_CONTRIBUTION, "timeline_df")
    _check_column_collisions(
        contributions,
        [attempts_df, assignments_df, outcomes_df]
        + ([timeline_df] if timeline_df is not None else []),
    )

    for frame, name in (
        (attempts_df, "attempts_df"),
        (assignments_df, "assignments_df"),
        (outcomes_df, "outcomes_df"),
    ):
        _check_join_key(frame, name)
    if timeline_df is not None:
        _check_join_key(timeline_df, "timeline_df")
        _check_identical_id_sets(attempts_df, timeline_df, "timeline_df")
    _check_identical_id_sets(attempts_df, assignments_df, "assignments_df")
    _check_identical_id_sets(attempts_df, outcomes_df, "outcomes_df")

    merged = attempts_df.loc[:, ["attempt_id", "event_timestamp"]].merge(
        assignments_df.loc[:, [JOIN_KEY, *_ASSIGNMENTS_CONTRIBUTION]],
        on=JOIN_KEY,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    merged = merged.merge(
        outcomes_df.loc[:, [JOIN_KEY, *_OUTCOMES_CONTRIBUTION]],
        on=JOIN_KEY,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if timeline_df is not None:
        merged = merged.merge(
            timeline_df.loc[:, list(_TIMELINE_CONTRIBUTION)],
            on=JOIN_KEY,
            how="left",
            sort=False,
            validate="one_to_one",
        )
    else:
        merged = pd.concat(
            [
                merged,
                pd.DataFrame(
                    {
                        column: pd.Series([None] * len(merged), dtype=object, index=merged.index)
                        for column in _TIMING_COLUMNS
                    }
                ),
            ],
            axis=1,
        )

    merged = merged.loc[:, list(CANONICAL_COLUMNS)]
    merged["assignment_probability"] = merged["assignment_probability"].astype("float64")
    for column in _OUTCOMES_CONTRIBUTION[1:]:
        merged[column] = merged[column].astype("float64")
    merged["simulated_recovered"] = merged["simulated_recovered"].astype("int8")
    return merged.reset_index(drop=True)


def dataset_contract() -> dict:
    """Machine-readable contract for the additive treatment/outcome dataset."""
    return {
        "rows": _TARGET_ROWS,
        "field_classification": dict(FIELD_CLASSIFICATION),
        "join_key": JOIN_KEY,
        "synthetic": True,
        "baseline_contract_untouched": True,
    }


def _is_structural_timestamp(raw: object) -> bool:
    if raw is None or raw is pd.NaT:
        return True
    if isinstance(raw, pd.Timestamp):
        return True
    if isinstance(raw, datetime):
        return True
    if isinstance(raw, str):
        try:
            pd.Timestamp(raw)
        except ValueError:
            return False
        return True
    return False


def _numeric_view(series: pd.Series) -> pd.Series | None:
    try:
        return series.astype(float)
    except (TypeError, ValueError):
        return None


def _real_number_mask(series: pd.Series) -> pd.Series:
    """Per-entry recognition of genuine int/float values (bools excluded).

    Unlike a column-wide ``astype(float)`` coercion -- which happily converts
    numeric-looking strings such as ``\"-300\"`` -- this mask treats every
    non-number entry (strings, Decimals, complex, bool, ...) as unrecognized,
    so tampered frames degrade to named violations instead of crashing the
    ``< 0`` comparison or silently passing coerced strings.
    """

    def _is_real_number(value: object) -> bool:
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, float, np.integer, np.floating))

    return series.map(_is_real_number)


def _check_bounded_numeric(series: pd.Series, column: str, bounds: tuple[float, float], violations: list[str]) -> None:
    numeric = _numeric_view(series)
    low, high = bounds
    if numeric is None:
        violations.append(f"{column} must hold numeric float values")
        return
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    non_finite_count = int((~finite).sum())
    if non_finite_count:
        violations.append(
            f"{column} must hold finite floats; {non_finite_count} row(s) are NaN/infinite"
        )
    within = (numeric >= low) & (numeric <= high)
    outside_count = int(((~within) & finite).sum())
    if outside_count:
        violations.append(
            f"{column} must lie within [{low}, {high}]; {outside_count} row(s) outside"
        )


def validate_treatment_dataset(df: pd.DataFrame) -> dict:
    """Return a structured report {valid, violations, row_count, column_count,
    classification_complete} checking schema, arm vocabulary, probability/arm-
    source rules, label/revenue bounds, ground-truth ranges, timestamp
    structure, and D5 classification completeness.

    The revenue check here is lower-bound only (>= 0): the upper bound versus
    each payment attempt's own settled payable lives where amounts coexist --
    joined views and reporting -- never inside this standalone frame. Temporal
    ordering of the timing columns is Task 6's concern; this validator applies
    structural dtype rules only.
    """
    _require_frame(df, "df")
    violations: list[str] = []

    unexpected = [column for column in df.columns if column not in set(CANONICAL_COLUMNS)]
    missing = [column for column in CANONICAL_COLUMNS if column not in set(df.columns)]
    if unexpected or missing or list(df.columns) != list(CANONICAL_COLUMNS):
        violations.append(
            "columns deviate from the canonical D5 set/order: "
            f"unexpected={unexpected}, missing={missing}, order={list(df.columns)}"
        )

    classification_complete = list(FIELD_CLASSIFICATION) == list(df.columns)
    if not classification_complete:
        violations.append(
            "classification_complete=False: FIELD_CLASSIFICATION keys do not "
            f"match the frame columns exactly; expected {list(FIELD_CLASSIFICATION)}, "
            f"got {list(df.columns)}"
        )

    if "assigned_action" in df.columns:
        assigned = df["assigned_action"]
        missing_count = int(assigned.isna().sum())
        if missing_count:
            violations.append(
                f"assigned_action has {missing_count} missing value(s)"
            )
        offenders = sorted(set(assigned.dropna().tolist()) - set(CANONICAL_ARMS))
        if offenders:
            violations.append(
                f"assigned_action leaves the canonical arm set: {offenders}; "
                f"expected a subset of {sorted(CANONICAL_ARMS)}"
            )

    if "arm_source" in df.columns:
        arm_source = df["arm_source"]
        missing_count = int(arm_source.isna().sum())
        if missing_count:
            violations.append(f"arm_source has {missing_count} missing value(s)")
        offenders = sorted(set(arm_source.dropna().tolist()) - set(ARM_SOURCES))
        if offenders:
            violations.append(
                f"arm_source must be one of {sorted(ARM_SOURCES)}; offenders: {offenders}"
            )

    if "assignment_probability" in df.columns:
        probabilities = df["assignment_probability"]
        nan_count = int(probabilities.isna().sum())
        if nan_count:
            violations.append(
                f"assignment_probability contains {nan_count} NaN value(s)"
            )
        if "arm_source" in df.columns:
            censored = df["arm_source"] == ARM_SOURCE_SAFETY_CENSORED
            bad_censored = censored & probabilities.notna() & (probabilities != 0.0)
            bad_censored_count = int(bad_censored.sum())
            if bad_censored_count:
                violations.append(
                    "safety_censored rows must carry assignment_probability == "
                    f"0.0 exactly (no stage-2 draw occurred); "
                    f"{bad_censored_count} row(s) violate"
                )
            randomized = df["arm_source"] == ARM_SOURCE_RANDOMIZED
            bad_randomized = randomized & probabilities.notna() & (probabilities <= 0.0)
            bad_randomized_count = int(bad_randomized.sum())
            if bad_randomized_count:
                violations.append(
                    "randomized rows must carry assignment_probability > 0 -- "
                    "the stage-2 draw probability must be positive within the "
                    f"eligible stratum; {bad_randomized_count} row(s) violate"
                )

    if "simulated_recovered" in df.columns:
        recovered = df["simulated_recovered"]
        missing_count = int(recovered.isna().sum())
        if missing_count:
            violations.append(
                f"simulated_recovered has {missing_count} missing value(s)"
            )
        offenders = sorted(set(pd.unique(recovered.dropna())) - {0, 1})
        if offenders:
            violations.append(
                f"simulated_recovered must be binary {{0, 1}}; offenders: {offenders}"
            )

    if "simulated_recovered_amount_inr" in df.columns:
        amounts = df["simulated_recovered_amount_inr"]
        nan_count = int(amounts.isna().sum())
        if nan_count:
            violations.append(
                f"simulated_recovered_amount_inr contains {nan_count} NaN value(s)"
            )
        recognized = _real_number_mask(amounts)
        non_numeric_count = int((~recognized & amounts.notna()).sum())
        if non_numeric_count:
            violations.append(
                "simulated_recovered_amount_inr must be numeric, found "
                f"{non_numeric_count} non-numeric row(s)"
            )
        coerced = pd.to_numeric(amounts.where(recognized), errors="coerce")
        negative_count = int((coerced < 0).sum())
        if negative_count:
            violations.append(
                f"simulated_recovered_amount_inr has {negative_count} negative "
                "row(s); simulated payouts are bounded below by zero. The upper "
                "bound versus each payment attempt's own settled payable is "
                "checked where amounts coexist (joined views/reporting), not "
                "inside this standalone frame."
            )

    for column in _PROPENSITY_COLUMNS:
        if column in df.columns:
            _check_bounded_numeric(df[column], column, _PROPENSITY_BOUNDS, violations)
    if "action_effect_logit" in df.columns:
        _check_bounded_numeric(
            df["action_effect_logit"], "action_effect_logit", _EFFECT_LOGIT_BOUNDS, violations
        )

    for column in _TIMING_COLUMNS:
        if column not in df.columns:
            continue
        offenders: list[str] = []
        for raw in df[column].tolist():
            if not _is_structural_timestamp(raw):
                offenders.append(repr(raw))
            if len(offenders) >= _MAX_EXAMPLE_IDS:
                break
        if offenders:
            violations.append(
                f"{column} values must be None, NaT, datetime, pd.Timestamp, or "
                "ISO-parseable strings; temporal ordering against failure/"
                "treatment times is enforced later (Task 6). Offending "
                f"examples: {offenders}"
            )

    return {
        "valid": not violations,
        "violations": violations,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "classification_complete": classification_complete,
    }
