"""Incremental recovery / incremental revenue reporting for Day 5 (D-M5/D-M6).

Every quantity in this module lives in the SYNTHETIC world defined by the
Day 4 simulator: incremental recovery contrasts modeled or replayed arm
propensities INSIDE the simulator's randomized stratum, and nothing causal
about any production system is supported anywhere in this file. Two label families
are embedded verbatim in every report -- ``MODEL ESTIMATE`` for quantities
derived from the fitted per-arm bundles and ``SIMULATED GROUND TRUTH`` for
the noise-integrated ``ground_truth_propensity`` replay (known by
construction) -- so a reader can never mistake a synthetic-world
bookkeeping number for a real-world claim.

The model-estimate contrast pairs each treated arm's estimated probability
against the SAME bundle's CONTROL probability evaluated on IDENTICAL rows
(the randomized rows of the supplied frame): naive within-stratum
comparison, unconfounded within stratum by construction, nothing more. The
truth twin replays ``simulation.outcomes.ground_truth_propensity`` per arm
on those identical rows. Incremental revenue is a pure accounting transform
of those labeled contrasts: ``IncRec x amount - intervention_cost -
risk_penalty``, reusing the Day 2 scoring constants IMPORTED from
``recovery.scoring`` (never restated here) and carrying D-M6's exact
uniform-retry-cost disclosure sentence. Nothing in this module is an
optimization target in Day 5: no optimizer, bandit, uplift estimator, or
threshold search consumes these numbers.

Discipline mirrors the rest of Day 5: inputs are never mutated, no wall
clock is read, and NO randomness is drawn anywhere (zero generator
derivations -- reporting is a deterministic projection of its inputs), so
identical inputs always reproduce byte-identical reports. Missing required
columns fail fast with a ValueError naming every offender before any
delegation, and empty (or fully safety-censored) frames yield zeroed but
valid report structures that still carry their labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.action_evaluation import TRUTH_LABEL_SIMULATED_GROUND_TRUTH
from ml.action_model import (
    ARM_ORDER,
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    predict_action_probability,
)
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from recovery.scoring import (
    RETRY_INTERVENTION_COST_INR,
    UNKNOWN_CATEGORY_RISK_FRACTION,
)
from simulation.outcomes import ground_truth_propensity

LABEL_MODEL_ESTIMATE = "MODEL ESTIMATE"
LABEL_SIMULATED_GROUND_TRUTH = TRUTH_LABEL_SIMULATED_GROUND_TRUTH

TREATED_ARMS = tuple(arm for arm in ARM_ORDER if arm != "CONTROL")

REVENUE_REQUIRED_COLUMNS = ("amount_inr", "failure_category")

NOTE_MODEL_TABLE = (
    "Naive within-randomized-stratum comparison in the synthetic world: "
    "each treated arm's estimated propensity minus the CONTROL model's "
    "propensity on IDENTICAL rows; unconfounded within stratum by "
    "construction; supports no production claim."
)

NOTE_TRUTH_TABLE = (
    "Noise-integrated SIMULATED GROUND TRUTH replayed per arm from the "
    "declarative policy; known by construction in the synthetic world; "
    "supports no production claim."
)

NOTE_REVENUE_TABLE = (
    "Incremental revenue is an accounting transform of labeled incremental "
    "recovery estimates in the synthetic world; nothing causal; NOT an "
    "optimization target in Day 5."
)

COST_SIMPLIFICATION_NOTE = (
    "A single retry-cost constant is applied uniformly to all treated arms "
    "including REQUEST_UPDATE and HUMAN_REVIEW, whose true economics differ."
)


def _require_frame(value: object, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(value).__name__}")


def _reject_missing_columns(
    frame: pd.DataFrame, name: str, columns: tuple[str, ...], hint: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns {missing}; {hint}")


def _reject_unknown_bundle(bundle: object) -> dict:
    models = getattr(bundle, "models", None)
    if not isinstance(models, dict) or not models:
        raise ValueError(
            "bundle must expose a non-empty dict of fitted per-arm pipelines "
            f"as its .models attribute (e.g. an ActionModelBundle), got "
            f"{type(bundle).__name__}"
        )
    missing_arms = [arm for arm in ARM_ORDER if arm not in models]
    if missing_arms:
        raise ValueError(
            f"bundle is missing fitted models for arms {missing_arms}; every "
            f"arm in {list(ARM_ORDER)} is required so each treated arm can be "
            "contrasted against CONTROL on identical rows"
        )
    return models


def _randomized_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]


def _row_identity(randomized: pd.DataFrame) -> dict | None:
    """Order-stable fingerprint of the EXACT row set behind every contrast.

    Uses ``pd.util.hash_pandas_object`` (fixed-key SipHash) summed over the
    ``attempt_id`` column: integer addition commutes, so the fingerprint
    identifies the ROW SET rather than the row order, letting callers verify
    that every arm was scored on the same rows. ``None`` when the column is
    absent or the slice is empty.
    """
    if len(randomized) == 0 or "attempt_id" not in randomized.columns:
        return None
    fingerprint = int(
        pd.util.hash_pandas_object(
            randomized["attempt_id"].astype(str), index=False
        ).sum()
    )
    return {
        "column": "attempt_id",
        "fingerprint": fingerprint,
        "n": int(len(randomized)),
    }


def _zeroed_contrast_entry() -> dict:
    return {
        "n": 0,
        "mean_probability_arm": 0.0,
        "mean_probability_control": 0.0,
        "mean_difference": 0.0,
        "paired_mean_difference": 0.0,
        "paired_median_difference": 0.0,
        "per_row_differences": [],
    }


def _contrast_entry(n_rows: int, arm_probabilities: np.ndarray, control: np.ndarray) -> dict:
    """Paired per-row contrast summary over ONE shared row block.

    ``mean_difference`` is difference-of-means; ``paired_mean_difference``
    is the mean of per-row differences -- algebraically equal on identical
    rows, both reported at 4-dp display precision alongside the median of
    the per-row differences. Full-precision per-row differences travel in
    ``per_row_differences`` so the revenue transform stays row-aligned.
    """
    differences = np.asarray(arm_probabilities, dtype=float) - np.asarray(
        control, dtype=float
    )
    return {
        "n": int(n_rows),
        "mean_probability_arm": round(float(np.mean(arm_probabilities)), 4),
        "mean_probability_control": round(float(np.mean(control)), 4),
        "mean_difference": round(
            float(np.mean(arm_probabilities)) - float(np.mean(control)), 4
        ),
        "paired_mean_difference": round(float(np.mean(differences)), 4),
        "paired_median_difference": round(float(np.median(differences)), 4),
        "per_row_differences": [float(value) for value in differences],
    }


def incremental_recovery_table(bundle: object, test_frame: pd.DataFrame) -> dict:
    """Per-treated-arm incremental recovery, labeled ``MODEL ESTIMATE``.

    Rows: EXACTLY ``test_frame[(stratum == "randomized")]`` -- one shared
    block. Each treated arm's contribution is ``mean P_hat_a - mean
    P_hat_CONTROL`` where BOTH columns come from this bundle's own pipelines
    run on the identical block (counterfactual scoring, not observed arm
    slices), reported at 4 dp together with the paired per-row mean/median
    difference, ``n``, and full-precision ``per_row_differences``. The
    embedded note declares the comparison naive-within-randomized-stratum,
    unconfounded within stratum by construction, and supports no production
    claim. A ``row_identity`` fingerprint (when ``attempt_id`` exists)
    proves every arm consumed the same row set.

    Raises ``ValueError`` naming the offenders when ``test_frame`` lacks the
    stratum column or the Day 2 feature whitelist, or when the bundle fails
    to expose a fitted pipeline for every canonical arm. An empty (or fully
    safety-censored) frame yields zeroed-but-valid per-arm structures under
    the unchanged labels. Inputs are never mutated; output is deterministic.
    """
    _require_frame(test_frame, "test_frame")
    _reject_missing_columns(
        test_frame,
        "test_frame",
        (STRATUM_COLUMN,),
        "pass frames produced by assemble_observations (a 'stratum' column "
        "is required to restrict scoring to the randomized stratum)",
    )
    _reject_missing_columns(
        test_frame,
        "test_frame",
        tuple(NUMERIC_FEATURES) + tuple(CATEGORICAL_FEATURES),
        "the bundle predicts through build_feature_matrix, so the full Day 2 "
        "decision-time feature whitelist must be present",
    )
    _reject_unknown_bundle(bundle)

    randomized = _randomized_rows(test_frame)
    n_rows = int(len(randomized))
    identity = _row_identity(randomized)
    if n_rows == 0:
        entries = {arm: _zeroed_contrast_entry() for arm in TREATED_ARMS}
    else:
        control = np.asarray(
            predict_action_probability(bundle, randomized, "CONTROL"), dtype=float
        )
        entries = {
            arm: _contrast_entry(
                n_rows,
                predict_action_probability(bundle, randomized, arm),
                control,
            )
            for arm in TREATED_ARMS
        }
    return {
        "label": LABEL_MODEL_ESTIMATE,
        "note": NOTE_MODEL_TABLE,
        "arms_covered": TREATED_ARMS,
        "n_randomized_test_rows": n_rows,
        "row_identity": identity,
        "arms": entries,
    }


def incremental_recovery_truth(policy: object, test_frame: pd.DataFrame) -> dict:
    """Same contrast quantities, labeled ``SIMULATED GROUND TRUTH``.

    Each treated arm's noise-integrated propensity comes from
    ``simulation.outcomes.ground_truth_propensity(df, policy, arm)``, and
    CONTROL from the same replay -- again on ONE shared randomized row
    block -- so the reported differences are known by construction in the
    synthetic world. Structure mirrors ``incremental_recovery_table`` key
    for key so the two tables can be placed side by side.

    Raises ``ValueError`` naming the offenders when the stratum column is
    missing or when the frame lacks any simulator-consumed decision-time
    column (the replay's own loud guard runs inside the loop). Empty or
    fully safety-censored frames yield zeroed-but-valid structures under
    the unchanged label; inputs are never mutated; output is deterministic.
    """
    _require_frame(test_frame, "test_frame")
    _reject_missing_columns(
        test_frame,
        "test_frame",
        (STRATUM_COLUMN,),
        "pass frames produced by assemble_observations (a 'stratum' column "
        "is required to restrict the replay to the randomized stratum)",
    )

    randomized = _randomized_rows(test_frame)
    n_rows = int(len(randomized))
    identity = _row_identity(randomized)
    if n_rows == 0:
        entries = {arm: _zeroed_contrast_entry() for arm in TREATED_ARMS}
    else:
        control = ground_truth_propensity(randomized, policy, "CONTROL")
        entries = {
            arm: _contrast_entry(
                n_rows, ground_truth_propensity(randomized, policy, arm), control
            )
            for arm in TREATED_ARMS
        }
    return {
        "label": LABEL_SIMULATED_GROUND_TRUTH,
        "note": NOTE_TRUTH_TABLE,
        "arms_covered": TREATED_ARMS,
        "n_randomized_test_rows": n_rows,
        "row_identity": identity,
        "arms": entries,
    }


def _zeroed_revenue_entry(mean_incremental_recovery: float) -> dict:
    return {
        "n": 0,
        "mean_incremental_recovery": round(float(mean_incremental_recovery), 4),
        "mean_incremental_revenue_per_case_inr": 0.0,
        "total_projected_incremental_revenue_inr": 0.0,
        "risk_penalty_applied_rows": 0,
    }


def incremental_revenue_table(
    recovery_table: dict,
    context_frame: pd.DataFrame,
    is_truth: bool = False,
) -> dict:
    """Project incremental revenue from a labeled recovery table (D-M6).

    For each treated arm and EACH row of the randomized block of
    ``context_frame`` (aligned position-for-position with the recovery
    table's ``per_row_differences``):

        IncrementalRevenue_i = IncRec_i * amount_i
                               - RETRY_INTERVENTION_COST_INR
                               - risk_penalty_i

    where ``risk_penalty_i = UNKNOWN_CATEGORY_RISK_FRACTION * amount_i``
    iff ``failure_category == "unknown"``, else 0. Both constants are
    IMPORTED from ``recovery.scoring`` -- the Day 2 cost basis is never
    restated here. Reported per arm: mean revenue-per-case difference and
    total projected revenue difference over the rows, rounded to 2 decimals,
    plus the count of rows to which the unknown-category penalty applied.
    The label is INHERITED verbatim from ``recovery_table``; ``is_truth``
    records the caller's declaration of which twin was passed in.

    ``cost_simplification_note`` embeds D-M6's exact disclosure sentence:
    the uniform retry-cost treatment of REQUEST_UPDATE / HUMAN_REVIEW is a
    simplification whose true economics differ. Nothing here is an
    optimization target in Day 5 and nothing causal is claimed.

    Raises ``ValueError`` naming the offenders when the recovery table is
    malformed (missing label / treated arms / per-row differences), when
    ``context_frame`` lacks the stratum / ``amount_inr`` /
    ``failure_category`` columns, when the randomized context row count
    disagrees with a stored difference vector, or -- when both sides carry
    a computable ``attempt_id`` fingerprint (``row_identity``) -- when the
    recovery table was computed on a different row set than the provided
    context frame. Empty contexts yield zeroed-but-valid structures under
    inherited labels.
    """
    if not isinstance(recovery_table, dict) or not isinstance(
        recovery_table.get("arms"), dict
    ):
        raise ValueError(
            "recovery_table must be the dict returned by "
            "incremental_recovery_table or incremental_recovery_truth "
            "(a non-empty 'arms' mapping is required)"
        )
    label = recovery_table.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError(
            "recovery_table must carry a non-string-empty 'label' so the "
            "revenue projection inherits MODEL ESTIMATE vs SIMULATED GROUND "
            "TRUTH verbatim"
        )
    _require_frame(context_frame, "context_frame")
    _reject_missing_columns(
        context_frame,
        "context_frame",
        (STRATUM_COLUMN,),
        "the revenue projection aligns amounts with the randomized rows the "
        "recovery table was computed on",
    )
    _reject_missing_columns(
        context_frame,
        "context_frame",
        REVENUE_REQUIRED_COLUMNS,
        "amount_inr and failure_category drive the revenue accounting "
        "(retry cost minus unknown-category risk penalty)",
    )
    arms_block = recovery_table["arms"]
    missing_arms = [arm for arm in TREATED_ARMS if arm not in arms_block]
    if missing_arms:
        raise ValueError(
            f"recovery_table is missing treated arms {missing_arms}; revenue "
            f"projection covers every treated arm in {list(TREATED_ARMS)}"
        )

    randomized = _randomized_rows(context_frame)
    n_rows = int(len(randomized))
    # Review F1 cross-check: when BOTH sides carry a computable row-set
    # fingerprint, count equality is not enough -- require the SAME rows.
    table_identity = recovery_table.get("row_identity")
    context_identity = _row_identity(randomized)
    if (
        isinstance(table_identity, dict)
        and isinstance(context_identity, dict)
        and table_identity.get("fingerprint") != context_identity.get("fingerprint")
    ):
        raise ValueError(
            "recovery table was computed on a different row set than the "
            "provided context frame: recovery_table row_identity fingerprint "
            f"{table_identity.get('fingerprint')!r} vs context_frame "
            f"fingerprint {context_identity.get('fingerprint')!r}; amounts "
            "and contrasts would be misaligned one-to-one"
        )
    amounts = randomized["amount_inr"].to_numpy(dtype=float)
    unknown_mask = (
        randomized["failure_category"].to_numpy() == "unknown"
    ) if n_rows > 0 else np.zeros(0, dtype=bool)

    entries = {}
    for arm in TREATED_ARMS:
        block = arms_block[arm]
        if not isinstance(block, dict) or "per_row_differences" not in block:
            raise ValueError(
                f"recovery_table['{arm}'] must carry 'per_row_differences' "
                "(the full-precision paired contrasts the revenue transform "
                "consumes row-aligned)"
            )
        source_mean = block.get("paired_mean_difference", 0.0)
        differences = np.asarray(block["per_row_differences"], dtype=float)
        if n_rows == 0:
            entries[arm] = _zeroed_revenue_entry(source_mean)
            continue
        if differences.shape != (n_rows,):
            raise ValueError(
                f"row alignment mismatch for arm '{arm}': recovery_table "
                f"carries {int(differences.size)} paired differences but "
                f"context_frame provides {n_rows} randomized rows; amounts "
                "and contrasts must be row-aligned one-to-one"
            )
        risk_penalty = np.where(
            unknown_mask, UNKNOWN_CATEGORY_RISK_FRACTION * amounts, 0.0
        )
        revenue_rows = (
            differences * amounts - RETRY_INTERVENTION_COST_INR - risk_penalty
        )
        entries[arm] = {
            "n": n_rows,
            "mean_incremental_recovery": round(float(source_mean), 4),
            "mean_incremental_revenue_per_case_inr": round(
                float(np.mean(revenue_rows)), 2
            ),
            "total_projected_incremental_revenue_inr": round(
                float(np.sum(revenue_rows)), 2
            ),
            "risk_penalty_applied_rows": int(unknown_mask.sum()),
        }
    return {
        "label": label,
        "is_truth_input": bool(is_truth),
        "note": NOTE_REVENUE_TABLE,
        "cost_simplification_note": COST_SIMPLIFICATION_NOTE,
        "intervention_cost_inr": float(RETRY_INTERVENTION_COST_INR),
        "unknown_category_risk_fraction": float(UNKNOWN_CATEGORY_RISK_FRACTION),
        "arms_covered": TREATED_ARMS,
        "n_rows": n_rows,
        "arms": entries,
    }
