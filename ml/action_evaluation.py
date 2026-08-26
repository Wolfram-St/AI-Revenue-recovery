"""Ground-truth replay evaluation for Day 5 action-aware models (D-M5/D-M7).

This module evaluates fitted per-arm action-aware recovery models against
the simulator's known SYNTHETIC GROUND TRUTH. Every report embeds the label
``OBSERVED SIMULATED OUTCOME`` for metrics computed on simulated labels and
``SIMULATED GROUND TRUTH`` for quantities replayed from the declarative
policy via ``simulation.outcomes.ground_truth_propensity``. The PRIMARY
agreement check is logit-scale effect recovery -- mean-logit arm contrasts
and interaction-cell contrasts against the configured effects -- because the
additive Gaussian logit noise leaves contrasts interpretable there. The
probability-scale mean |P_hat - integrated TRUE| is a SECONDARY comparison;
it carries the documented Jensen floor (a perfectly fitted model converges
to the noise-integrated propensity, never to the pre-noise sigmoid) and is
never used as a pass/fail gate. These comparisons validate whether the model
RECOVERS THE SYNTHETIC STRUCTURE and support nothing causal about any
production system.

The evaluator consumes ONLY ``stratum``, ``assigned_action``, and
``simulated_recovered`` plus decision-time features built through
``build_feature_matrix`` -- never the Day-1 ``recovered`` label, timestamps,
or assignment metadata beyond the assigned arm. The caller supplies the
fitted Day 2 baseline pipeline (D-M7 interface change versus the original
plan sketch): baseline Brier scores are reported per arm ON THE SAME rows so
the comparison shows what action conditioning adds, absorbing the
Day-1-to-Day-4 DGP transfer mismatch rather than isolating it.

Randomness discipline: the ONLY stochastic consumer is the seeded stratified
row-level bootstrap (B=500 default). Its generator is derived EXACTLY once,
from the named ``seed`` parameter via ``np.random.default_rng(seed)`` (the
documented named-seed pattern, default 20260826); resampling is performed
within arm slices (stratified), so identical inputs, bundle, baseline, and
seed reproduce byte-identical reports.

Bundle-kind discipline: reports record ``bundle_kind`` -- detected from the
bundle metadata ("calibration" key present means sigmoid-calibrated) -- and
the matching documented logit-scale effect-contrast gate band
(``gate_band_logit``): +-0.25 for raw pipelines, +-0.40 for calibrated ones
(calibrated pipelines shrink logits; the wider band is annotated in output).

Metric implementations live on numpy/pandas only (no scipy/sklearn imports):
ROC-AUC is the tie-corrected Mann-Whitney statistic via average ranks, PR-AUC
is the step-wise average-precision sum over DISTINCT score thresholds (ties
grouped per the sklearn definition), Spearman rho is Pearson r of average
ranks. The logit transform clips probabilities to [1e-6, 1 - 1e-6] before
ln(p/(1-p)) -- a documented numeric guard against infinite logits at
saturated predictions, not a statistical correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.action_model import (
    ACTION_COLUMN,
    REQUIRED_COLUMNS,
    SMALL_SEGMENT_THRESHOLD,
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    TARGET_COLUMN,
    ActionModelBundle,
    predict_action_probability,
)
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml.train import predict_recovery_probability
from simulation.config import CANONICAL_ARMS, TreatmentPolicy
from simulation.outcomes import ground_truth_propensity

LABEL_OBSERVED_SIMULATED_OUTCOME = "OBSERVED SIMULATED OUTCOME"
TRUTH_LABEL_SIMULATED_GROUND_TRUTH = "SIMULATED GROUND TRUTH"

PRIMARY_AGREEMENT_CHECK = "logit_scale_effect_contrast_recovery"
SECONDARY_COMPARISON_NOTE = (
    "secondary comparison; carries the documented Jensen floor (models "
    "converge to the noise-integrated propensity, not the pre-noise sigmoid)"
)

# Bundle-kind awareness (review F2): the effect-contrast gate band is a
# REPORTED constant that depends on whether the bundle's pipelines are
# sigmoid-calibrated. Calibration shrinks logits toward the mean, so
# calibrated bundles get the wider documented band; both kinds annotate it
# in the report via ``gate_band_logit`` next to ``bundle_kind``.
BUNDLE_KIND_RAW = "raw"
BUNDLE_KIND_CALIBRATED = "calibrated"
RAW_GATE_BAND_LOGIT = 0.25
CALIBRATED_GATE_BAND_LOGIT = 0.40

_LOGIT_CLIP_LOW = 1e-6
_LOGIT_CLIP_HIGH = 1.0 - 1e-6


def _require_frame(value: object, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(value).__name__}")


def _reject_missing_observation_columns(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required observation columns {missing}; pass "
            "frames produced by assemble_observations (stratum + "
            f"{ACTION_COLUMN} + {TARGET_COLUMN} required)"
        )


def _reject_missing_feature_columns(frame: pd.DataFrame, name: str) -> None:
    """Review F9 guard: fail HERE, naming every offender, before any
    delegation to build_feature_matrix (whose pandas layer would surface a
    bare KeyError instead of this module's loud ValueError)."""
    missing = [
        column
        for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{name} is missing required decision-time feature columns "
            f"{missing}; evaluation builds model inputs exclusively through "
            "build_feature_matrix, so the full Day 2 feature whitelist must "
            "be present"
        )


def _detect_bundle_kind(bundle: ActionModelBundle) -> str:
    """Calibration status from metadata alone (review F2): a ``calibration``
    record means the pipelines are sigmoid-calibrated; anything else is raw."""
    metadata = getattr(bundle, "metadata", None)
    if isinstance(metadata, dict) and "calibration" in metadata:
        return BUNDLE_KIND_CALIBRATED
    return BUNDLE_KIND_RAW


def _safe_logit(probabilities: np.ndarray) -> np.ndarray:
    """Overflow-guarded logit: clip to [1e-6, 1 - 1e-6] before ln(p/(1-p)).

    Documented numeric guard: saturated XGBoost/calibration outputs would
    otherwise produce +-inf logits and poison every mean contrast; the clip
    bounds |logit| near 13.8 while staying far outside the resolution any
    honest comparison needs.
    """
    clipped = np.clip(
        np.asarray(probabilities, dtype=float), _LOGIT_CLIP_LOW, _LOGIT_CLIP_HIGH
    )
    return np.log(clipped / (1.0 - clipped))


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney ROC-AUC; NaN on single-class/empty input."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    rank_sum_positives = float(ranks[labels == 1].sum())
    return (rank_sum_positives - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise average precision with ties GROUPED per the sklearn
    definition: one threshold step per DISTINCT score value, AP =
    sum_k (R_k - R_{k-1}) * P_k. The previous per-positive-index mean broke
    on tied scores (worst observed deviation vs sklearn 0.14 on integer-grid
    slices). Equivalence probe kept after review F1 fix: seeded 500-slice
    sweep over small-integer-grid scores plus an explicit duplicated-block
    case showed max deviation 1.11e-16 (all-tied extreme: exactly 0.0)
    against sklearn.metrics.average_precision_score -- far inside the 1e-9
    claim. NaN when no positive labels.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    total_positives = int((labels == 1).sum())
    if labels.size == 0 or total_positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positive = labels[order] == 1
    true_positives = np.cumsum(sorted_positive)
    # Last position of every tie block == one distinct-threshold step end.
    block_ends = np.flatnonzero(
        np.concatenate([sorted_scores[:-1] != sorted_scores[1:], [True]])
    )
    precision_at_block = true_positives[block_ends] / (block_ends + 1.0)
    recall_at_block = true_positives[block_ends] / total_positives
    recall_steps = np.diff(np.concatenate([[0.0], recall_at_block]))
    return float(np.sum(precision_at_block * recall_steps))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson r with explicit NaN guards for degenerate variance."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = (
        float((left_centered * left_centered).sum())
        * float((right_centered * right_centered).sum())
    ) ** 0.5
    if denominator == 0.0:
        return float("nan")
    return float((left_centered * right_centered).sum() / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rho WITHOUT scipy: Pearson r of pandas average ranks."""
    left_ranks = (
        pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy(dtype=float)
    )
    right_ranks = (
        pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy(dtype=float)
    )
    return _pearson(left_ranks, right_ranks)


def _brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    residual = np.asarray(labels, dtype=float) - np.asarray(probabilities, dtype=float)
    return float(np.mean(residual * residual))


def _percentile_bounds(statistics: np.ndarray) -> list[float]:
    finite = statistics[np.isfinite(statistics)]
    if finite.size == 0:
        return [float("nan"), float("nan")]
    low, high = np.percentile(finite, [2.5, 97.5])
    return [float(low), float(high)]


def _stratified_bootstrap_cis(
    labels: np.ndarray,
    model_probabilities: np.ndarray,
    strata: list[np.ndarray],
    rng: np.random.Generator,
    replications: int,
) -> tuple[list[float], list[float]]:
    """Seeded stratified bootstrap 95% CIs for ROC-AUC and Brier.

    Resampling happens WITHIN each stratum (per-arm row blocks), mirroring
    the D-M4 requirement; ``strata`` partitions the row positions of the
    supplied arrays. Replicates that collapse to a single class contribute
    NaN AUCs which the percentile step ignores.
    """
    auc_statistics = np.full(replications, np.nan, dtype=float)
    brier_statistics = np.full(replications, np.nan, dtype=float)
    for replication in range(replications):
        resampled = np.concatenate(
            [
                block[rng.integers(0, block.size, block.size)]
                for block in strata
            ]
        )
        y_resampled = labels[resampled]
        p_resampled = model_probabilities[resampled]
        auc_statistics[replication] = _roc_auc(y_resampled, p_resampled)
        brier_statistics[replication] = _brier(y_resampled, p_resampled)
    return _percentile_bounds(auc_statistics), _percentile_bounds(brier_statistics)


def _interaction_key(rule: object) -> str:
    if rule.column == "failure_category":
        condition = f"{rule.column}=={rule.equals_value}"
    else:
        condition = f"{rule.column}>={rule.min_threshold}"
    return f"{rule.action}|{condition}"


def _rule_row_mask(rows: pd.DataFrame, rule: object) -> np.ndarray:
    if rule.column == "failure_category":
        return (rows["failure_category"].to_numpy() == rule.equals_value) & (
            rows[ACTION_COLUMN].to_numpy() == rule.action
        )
    return (
        rows["attempt_number"].to_numpy(dtype=float) >= float(rule.min_threshold)
    ) & (rows[ACTION_COLUMN].to_numpy() == rule.action)


def _control_rule_mask(rows: pd.DataFrame, rule: object) -> np.ndarray:
    if rule.column == "failure_category":
        return rows["failure_category"].to_numpy() == rule.equals_value
    return rows["attempt_number"].to_numpy(dtype=float) >= float(rule.min_threshold)


def evaluate_action_models(
    bundle: ActionModelBundle,
    baseline_model: object,
    test_frame: pd.DataFrame,
    policy: TreatmentPolicy,
    *,
    seed: int = 20260826,
    bootstrap_replications: int = 500,
) -> dict:
    """Evaluate per-arm action-aware models against simulator ground truth.

    Rows evaluated per arm: ``test_frame[(stratum == "randomized") &
    (assigned_action == arm)]``. For each arm the report carries: row count
    and small-segment flag (<100), ROC-AUC (NaN-safe on single-class slices)
    with seeded stratified bootstrap 95% CI, PR-AUC, Brier of the
    action-aware model and of the caller-supplied Day 2 baseline pipeline on
    the SAME rows with its ROC-AUC beside it (D-M7 + review F4), Brier CI,
    the SECONDARY mean |P_hat - integrated TRUE| with its Jensen-floor
    annotation, Pearson r AND scipy-free Spearman rho between P_hat and
    integrated TRUE, and the PRIMARY logit-scale effect recovery:
    mean(logit P_hat_arm) minus mean(logit P_hat_CONTROL) versus
    ``policy.main_effects_logit[arm]`` (gap reported against the
    kind-aware documented gate band), plus interaction-cell contrasts for
    every configured rule owned by the arm -- cell-mean arm-vs-control
    logit difference minus the arm's main effect, compared against the
    rule's configured effect; cells whose configured effect is expected to
    attenuate under gradient-boosting shrinkage at finite n carry an
    ``attenuation_expected`` annotation instead of a gate. A top-level
    micro-averaged block pools ALL randomized test rows, scoring each row
    with its ASSIGNED arm's model (honest pooled view).

    ``bundle_kind`` ("raw" | "calibrated", detected from bundle metadata)
    and the matching ``gate_band_logit`` (+-0.25 raw / +-0.40 calibrated --
    sigmoid calibration shrinks logits) are recorded top-level and per-arm.

    Nothing here mutates its inputs, draws wall-clock time, or consumes the
    forbidden Day-1 ``recovered`` label; identical inputs reproduce a
    byte-identical report under the documented seed discipline. All outputs
    describe OBSERVED SIMULATED OUTCOME / SIMULATED GROUND TRUTH quantities
    in the synthetic world and support nothing causal.
    """
    _require_frame(test_frame, "test_frame")
    _reject_missing_observation_columns(test_frame, "test_frame")
    _reject_missing_feature_columns(test_frame, "test_frame")
    if not isinstance(bundle, ActionModelBundle):
        raise ValueError(
            f"bundle must be an ActionModelBundle, got {type(bundle).__name__}"
        )
    unknown_arms = sorted(set(bundle.arms) - set(CANONICAL_ARMS))
    if unknown_arms:
        raise ValueError(f"bundle carries non-canonical arms: {unknown_arms}")
    if bootstrap_replications < 1:
        raise ValueError(
            f"bootstrap_replications must be >= 1, got {bootstrap_replications}"
        )

    # The ONE sanctioned randomness derivation: named-seed bootstrap stream.
    rng = np.random.default_rng(seed)

    # Review F2: kind-aware documented gate band, recorded in the report.
    bundle_kind = _detect_bundle_kind(bundle)
    gate_band_logit = (
        CALIBRATED_GATE_BAND_LOGIT if bundle_kind == BUNDLE_KIND_CALIBRATED
        else RAW_GATE_BAND_LOGIT
    )

    randomized = test_frame.loc[
        test_frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED
    ]

    arm_blocks = {}
    results = {}
    for arm in bundle.arms:
        rows = randomized.loc[randomized[ACTION_COLUMN] == arm]
        n_rows = int(len(rows))
        entry = {
            "n": n_rows,
            "small_segment": n_rows < SMALL_SEGMENT_THRESHOLD,
            "bundle_kind": bundle_kind,
            "gate_band_logit": gate_band_logit,
            "roc_auc": float("nan"),
            "roc_auc_ci95": [float("nan"), float("nan")],
            "pr_auc": float("nan"),
            "brier_model": float("nan"),
            "brier_ci95": [float("nan"), float("nan")],
            "brier_baseline_day2": float("nan"),
            "roc_auc_baseline_day2": float("nan"),
            "mean_abs_error_vs_integrated_true": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "main_effect_configured_logit": float(policy.main_effects_logit.get(arm, float("nan"))),
            "main_effect_estimated_logit_contrast": float("nan"),
            "main_effect_recovery_gap_logit": float("nan"),
            "interaction_cells": {},
        }
        blocks = {}
        if n_rows > 0:
            p_hat = predict_action_probability(bundle, rows, arm)
            truth = ground_truth_propensity(rows, policy, arm)
            labels = rows[TARGET_COLUMN].astype(int).to_numpy()
            p_baseline = predict_recovery_probability(baseline_model, rows)
            logits = _safe_logit(p_hat)

            entry["roc_auc"] = _roc_auc(labels, p_hat)
            entry["pr_auc"] = _average_precision(labels, p_hat)
            entry["brier_model"] = _brier(labels, p_hat)
            entry["brier_baseline_day2"] = _brier(labels, p_baseline)
            entry["roc_auc_baseline_day2"] = _roc_auc(labels, p_baseline)
            entry["mean_abs_error_vs_integrated_true"] = float(
                np.mean(np.abs(p_hat - truth))
            )
            entry["pearson_r"] = _pearson(p_hat, truth)
            entry["spearman_rho"] = _spearman(p_hat, truth)

            positions = np.arange(n_rows)
            auc_bounds, brier_bounds = _stratified_bootstrap_cis(
                labels, p_hat, [positions], rng, bootstrap_replications
            )
            entry["roc_auc_ci95"] = auc_bounds
            entry["brier_ci95"] = brier_bounds

            blocks = {
                "rows": rows,
                "labels": labels,
                "p_hat": p_hat,
                "p_baseline": p_baseline,
                "logits": logits,
            }
        arm_blocks[arm] = blocks
        results[arm] = entry

    control_block = arm_blocks.get("CONTROL")
    control_logits = control_block["logits"] if control_block else None
    control_rows = control_block["rows"] if control_block else None

    for arm in bundle.arms:
        entry = results[arm]
        block = arm_blocks[arm]
        configured_main = entry["main_effect_configured_logit"]

        if block and control_logits is not None:
            estimated_contrast = float(block["logits"].mean() - control_logits.mean())
            entry["main_effect_estimated_logit_contrast"] = estimated_contrast
            if np.isfinite(configured_main):
                entry["main_effect_recovery_gap_logit"] = (
                    estimated_contrast - float(configured_main)
                )

        for rule in policy.interactions:
            if rule.action != arm:
                continue
            key = _interaction_key(rule)
            # Review F3: weak negative effects (the shipped late-stage
            # fatigue rule) are expected to attenuate toward zero under
            # XGBoost shrinkage at finite n -- annotate instead of gate.
            cell = {
                "column": rule.column,
                "condition": (
                    f"{rule.column}=={rule.equals_value}"
                    if rule.column == "failure_category"
                    else f"{rule.column}>={rule.min_threshold}"
                ),
                "configured_effect_logit": float(rule.effect_logit),
                "estimated_cell_contrast_logit": float("nan"),
                "recovery_gap_logit": float("nan"),
                "attenuation_expected": float(rule.effect_logit) < 0.0,
                "n_cell": 0,
            }
            if block and control_rows is not None:
                arm_mask = _rule_row_mask(block["rows"], rule)
                control_mask = _control_rule_mask(control_rows, rule)
                cell["n_cell"] = int(arm_mask.sum())
                if arm_mask.any() and control_mask.any():
                    cell_difference = float(
                        block["logits"][arm_mask].mean()
                        - control_logits[control_mask].mean()
                    )
                    estimated_interaction = cell_difference - float(configured_main)
                    cell["estimated_cell_contrast_logit"] = estimated_interaction
                    cell["recovery_gap_logit"] = (
                        estimated_interaction - float(rule.effect_logit)
                    )
            entry["interaction_cells"][key] = cell

    pooled_labels = []
    pooled_p_hat = []
    pooled_p_baseline = []
    pooled_strata = []
    for arm in bundle.arms:
        block = arm_blocks[arm]
        if not block:
            continue
        pooled_strata.append(np.arange(block["labels"].size, dtype=int))
        pooled_labels.append(block["labels"])
        pooled_p_hat.append(block["p_hat"])
        pooled_p_baseline.append(block["p_baseline"])

    micro = {
        "n": 0,
        "roc_auc": float("nan"),
        "roc_auc_ci95": [float("nan"), float("nan")],
        "pr_auc": float("nan"),
        "brier_model": float("nan"),
        "brier_ci95": [float("nan"), float("nan")],
        "brier_baseline_day2": float("nan"),
        "roc_auc_baseline_day2": float("nan"),
    }
    if pooled_labels:
        labels = np.concatenate(pooled_labels)
        p_hat = np.concatenate(pooled_p_hat)
        p_baseline = np.concatenate(pooled_p_baseline)
        micro["n"] = int(labels.size)
        micro["roc_auc"] = _roc_auc(labels, p_hat)
        micro["pr_auc"] = _average_precision(labels, p_hat)
        micro["brier_model"] = _brier(labels, p_hat)
        micro["brier_baseline_day2"] = _brier(labels, p_baseline)
        micro["roc_auc_baseline_day2"] = _roc_auc(labels, p_baseline)
        auc_bounds, brier_bounds = _stratified_bootstrap_cis(
            labels, p_hat, pooled_strata, rng, bootstrap_replications
        )
        micro["roc_auc_ci95"] = auc_bounds
        micro["brier_ci95"] = brier_bounds

    return {
        "label": LABEL_OBSERVED_SIMULATED_OUTCOME,
        "truth_label": TRUTH_LABEL_SIMULATED_GROUND_TRUTH,
        "scope_note": (
            "Synthetic-world-only evaluation of action-aware models against "
            "the Day 4 simulator; supports nothing causal about production."
        ),
        "primary_agreement_check": PRIMARY_AGREEMENT_CHECK,
        "secondary_comparison_note": SECONDARY_COMPARISON_NOTE,
        "bootstrap": {
            "replications": int(bootstrap_replications),
            "confidence_level": 0.95,
            "scheme": "stratified_within_arm_row_resampling",
        },
        "seed": int(seed),
        "small_segments_threshold": SMALL_SEGMENT_THRESHOLD,
        "bundle_kind": bundle_kind,
        "gate_band_logit": gate_band_logit,
        "n_randomized_test_rows": int(len(randomized)),
        "micro_averaged": micro,
        "arms": results,
    }
