"""Gate A harness comparing the pooled and per-arm model families (plan Task 3).

Both families were trained on the IDENTICAL randomized train/validation
segments of the same assembled observation frame (D-E1); this module evaluates
them once on the identical randomized test segment and applies the
pre-registered D-E2 preference rule. The rule consumes the CALIBRATED bundles
only -- the shipped Day 5 form -- so ``compare_models`` refuses raw bundles
loudly; per-arm remains the reference production family unless ALL FOUR
pre-registered criteria pass, in which case pooled becomes the preferred
candidate. Either way both families stay available and nothing is deleted.

Reported sections (top-level keys):

* ``predictive`` -- per family ``{micro, arms}`` ROC-AUC/PR-AUC/Brier tables;
  per-arm numbers come straight from
  :func:`ml.action_evaluation.evaluate_action_models`, pooled numbers are
  computed on the identical rows by slicing the shared pipeline's predictions
  per assigned arm and reusing that module's seeded stratified-bootstrap CI
  helper WITH the Day-5 corrected pool offsets (each arm's resample indices
  point at its own segment of the concatenated pool).
* ``ground_truth_agreement`` -- per family per arm the SECONDARY probability-
  scale mean |P_hat - noise-integrated TRUE| via
  ``simulation.outcomes.ground_truth_propensity`` plus Pearson r and
  scipy-free Spearman rho; carries the documented Jensen-floor note (models
  converge to the integrated propensity, never the pre-noise sigmoid) and is
  never a pass/fail gate by itself.
* ``effect_contrasts`` -- per family logit-scale main-effect contrasts versus
  config plus interaction cells carrying the ``attenuation_expected``
  semantics: the weak-negative RETRY_LATER x attempt_number>=3 fatigue cell
  is REPORTED, never gated (Day 5 discipline), while genuinely gated cells
  feed D-E2 criterion 4 against the kind-aware +-0.40 calibrated band.
* ``rule_application`` -- the pure ``_apply_rule`` engine's verdict:
  ``preferred_model`` is "pooled" iff strict micro-Brier CI non-overlap,
  ground-truth agreement no worse, smallest-arm (HUMAN_REVIEW-shaped) Brier
  no worse, and gated interaction-cell recovery within band ALL hold;
  otherwise "per_arm". NaN inputs fail closed toward per-arm.
* ``complexity`` -- fit wall-clock seconds per family measured during a timed
  refit at the 100% train fraction plus parameter-count capacity proxies
  (documented n_estimators x max_depth product per fitted pipeline, a coarse
  capacity comparison, never an exact node count).

Labels ``OBSERVED SIMULATED OUTCOME`` and ``SIMULATED GROUND TRUTH`` are
embedded wherever applicable; every number here describes the synthetic world
only and supports nothing causal about any production system.

Determinism discipline: the ONLY stochastic consumer is the pooled-family
bootstrap, whose generator is derived EXACTLY once from the named ``seed``
parameter via ``np.random.default_rng(seed)`` (default 20260826); every dict
is built in sorted key order and arms iterate in canonical ARM_ORDER, so two
calls with identical inputs produce identical reports EXCEPT the single
documented wall-clock field ``complexity.families.*.fit_seconds``, which is
machine-dependent by nature. The optional keyword-only ``train_frame`` /
``validation_frame`` pair enables that timed refit without changing the
positional contract; when omitted, ``fit_seconds`` records None.

DEVIATION NOTE (reviewer-visible): measuring contracted fit seconds requires
stdlib ``time`` (imported as ``from time import perf_counter``); the module
docstring and tests document this single addition to the otherwise Day-5-style
import whitelist. No other stdlib root is used, dataclasses is unused here,
and no forbidden randomness/wall-clock tokens appear outside the sanctioned
derivation above.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from ml.action_evaluation import (
    BUNDLE_KIND_CALIBRATED,
    CALIBRATED_GATE_BAND_LOGIT,
    LABEL_OBSERVED_SIMULATED_OUTCOME,
    SECONDARY_COMPARISON_NOTE,
    TRUTH_LABEL_SIMULATED_GROUND_TRUTH,
    _average_precision,
    _brier,
    _control_rule_mask,
    _detect_bundle_kind,
    _interaction_key,
    _pearson,
    _reject_missing_feature_columns,
    _reject_missing_observation_columns,
    _require_frame,
    _roc_auc,
    _rule_row_mask,
    _safe_logit,
    _spearman,
    _stratified_bootstrap_cis,
    evaluate_action_models,
)
from ml.action_model import (
    ACTION_COLUMN,
    ARM_ORDER,
    SMALL_SEGMENT_THRESHOLD,
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    TARGET_COLUMN,
    ActionModelBundle,
    calibrate_action_models,
    predict_action_probability,
    train_action_models,
)
from ml.pooled_model import (
    PooledModelBundle,
    calibrate_pooled_model,
    predict_pooled_probability,
    train_pooled_model,
)
from simulation.config import CANONICAL_ARMS, TreatmentPolicy
from simulation.outcomes import ground_truth_propensity

LABEL_OBSERVED = LABEL_OBSERVED_SIMULATED_OUTCOME
TRUTH_LABEL = TRUTH_LABEL_SIMULATED_GROUND_TRUTH

RULE_ID = "D-E2"
RULE_APPLIES_TO = "calibrated_bundles_only"
PREFERRED_POOLED = "pooled"
PREFERRED_PER_ARM = "per_arm"

CRITERION_STRICT_CI = "strict_micro_brier_ci_non_overlap"
CRITERION_AGREEMENT = "ground_truth_agreement_no_worse"
CRITERION_SMALLEST_ARM = "smallest_arm_brier_no_worse"
CRITERION_INTERACTION_BAND = "interaction_recovery_within_band"
CRITERIA_ORDER = (
    CRITERION_STRICT_CI,
    CRITERION_AGREEMENT,
    CRITERION_SMALLEST_ARM,
    CRITERION_INTERACTION_BAND,
)

TREATED_ARMS = tuple(arm for arm in ARM_ORDER if arm != "CONTROL")

BOOTSTRAP_SCHEME = (
    "stratified_within_arm_row_resampling_with_corrected_pool_offsets"
)

SCOPE_NOTE = (
    "Synthetic-world-only Gate A comparison of action-aware model families "
    "on identical randomized strata under the pre-registered D-E2 rule; "
    "supports nothing causal about production."
)


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _population_sd(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def _smallest_test_arm(randomized: pd.DataFrame) -> str:
    """Arm with fewest randomized test rows; ties resolve by ARM_ORDER."""
    counts = {
        arm: int((randomized[ACTION_COLUMN] == arm).sum()) for arm in ARM_ORDER
    }
    return min(ARM_ORDER, key=lambda arm: counts[arm])


def _predict_for_family(
    bundle: object, rows: pd.DataFrame, arm: str
) -> np.ndarray:
    if isinstance(bundle, ActionModelBundle):
        return np.asarray(
            predict_action_probability(bundle, rows, arm), dtype=float
        )
    if isinstance(bundle, PooledModelBundle):
        return np.asarray(
            predict_pooled_probability(bundle, rows, arm), dtype=float
        )
    raise ValueError(
        f"unsupported bundle type {type(bundle).__name__}; expected an "
        "ActionModelBundle or PooledModelBundle"
    )


# ---------------------------------------------------------------------------
# Predictive section builders
# ---------------------------------------------------------------------------


def _metric_entry(
    labels: np.ndarray,
    probabilities: np.ndarray,
    strata: list[np.ndarray],
    rng: np.random.Generator,
    replications: int,
) -> dict:
    """AUC/PR-AUC/Brier plus reused Day-5 stratified-bootstrap CIs."""
    auc_bounds, brier_bounds = _stratified_bootstrap_cis(
        labels, probabilities, strata, rng, replications
    )
    return {
        "auc": float(_roc_auc(labels, probabilities)),
        "auc_ci95": auc_bounds,
        "brier": float(_brier(labels, probabilities)),
        "brier_ci95": brier_bounds,
        "pr_auc": float(_average_precision(labels, probabilities)),
    }


def _empty_metric_entry() -> dict:
    nan = float("nan")
    return {
        "auc": nan,
        "auc_ci95": [nan, nan],
        "brier": nan,
        "brier_ci95": [nan, nan],
        "pr_auc": nan,
    }


def _normalized_per_arm_family(eval_report: dict) -> dict:
    arms = {}
    for arm in ARM_ORDER:
        entry = eval_report["arms"][arm]
        arms[arm] = {
            "auc": float(entry["roc_auc"]),
            "auc_ci95": [float(bound) for bound in entry["roc_auc_ci95"]],
            "brier": float(entry["brier_model"]),
            "brier_ci95": [float(bound) for bound in entry["brier_ci95"]],
            "n": int(entry["n"]),
            "pr_auc": float(entry["pr_auc"]),
            "small_segment": bool(entry["small_segment"]),
        }
    micro_source = eval_report["micro_averaged"]
    micro = {
        "auc": float(micro_source["roc_auc"]),
        "auc_ci95": [float(bound) for bound in micro_source["roc_auc_ci95"]],
        "brier": float(micro_source["brier_model"]),
        "brier_ci95": [float(bound) for bound in micro_source["brier_ci95"]],
        "n": int(micro_source["n"]),
        "pr_auc": float(micro_source["pr_auc"]),
    }
    return {"arms": arms, "micro": micro}


def _pooled_blocks(
    pooled_bundle: PooledModelBundle,
    randomized: pd.DataFrame,
    policy: TreatmentPolicy,
) -> dict:
    blocks = {}
    for arm in ARM_ORDER:
        rows = randomized.loc[randomized[ACTION_COLUMN] == arm]
        n_rows = int(len(rows))
        block = {
            "logits": None,
            "labels": None,
            "n": n_rows,
            "p_hat": None,
            "rows": None,
            "small_segment": n_rows < SMALL_SEGMENT_THRESHOLD,
            "truth": None,
        }
        if n_rows > 0:
            p_hat = np.asarray(
                predict_pooled_probability(pooled_bundle, rows, arm), dtype=float
            )
            labels = rows[TARGET_COLUMN].astype(int).to_numpy()
            truth = ground_truth_propensity(rows, policy, arm)
            block.update(
                {
                    "labels": labels,
                    "logits": _safe_logit(p_hat),
                    "p_hat": p_hat,
                    "rows": rows,
                    "truth": truth,
                }
            )
        blocks[arm] = block
    return blocks


def _pooled_predictive(
    blocks: dict, rng: np.random.Generator, replications: int
) -> dict:
    """Slice pooled predictions per assigned arm; corrected-offset strata.

    Mirrors the Day-5 defect fix: each arm's resample index array points at
    ITS OWN segment of the CONCATENATED micro pool (running offsets kept),
    so micro resampling draws from every arm slice rather than collapsing
    onto the leading segment.
    """
    arms = {}
    pool_labels: list[np.ndarray] = []
    pool_probabilities: list[np.ndarray] = []
    pool_strata: list[np.ndarray] = []
    pool_offset = 0
    for arm in ARM_ORDER:
        block = blocks[arm]
        entry = {"n": block["n"], "small_segment": block["small_segment"]}
        if block["n"] > 0:
            entry.update(
                _metric_entry(
                    block["labels"],
                    block["p_hat"],
                    [np.arange(block["n"], dtype=int)],
                    rng,
                    replications,
                )
            )
            pool_strata.append(
                np.arange(pool_offset, pool_offset + block["n"], dtype=int)
            )
            pool_offset += block["n"]
            pool_labels.append(block["labels"])
            pool_probabilities.append(block["p_hat"])
        else:
            entry.update(_empty_metric_entry())
        arms[arm] = entry
    micro = {"n": int(pool_offset)}
    if pool_labels:
        micro.update(
            _metric_entry(
                np.concatenate(pool_labels),
                np.concatenate(pool_probabilities),
                pool_strata,
                rng,
                replications,
            )
        )
    else:
        micro.update(_empty_metric_entry())
    return {"arms": arms, "micro": micro}


# ---------------------------------------------------------------------------
# Ground-truth agreement section builders
# ---------------------------------------------------------------------------


def _family_agreement_from_blocks(blocks: dict) -> dict:
    arms = {}
    finite_maes: list[float] = []
    for arm in ARM_ORDER:
        block = blocks[arm]
        entry = {
            "mean_abs_error_vs_integrated_true": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
        }
        if block["n"] > 0:
            mae = float(np.mean(np.abs(block["p_hat"] - block["truth"])))
            entry["mean_abs_error_vs_integrated_true"] = mae
            entry["pearson_r"] = float(_pearson(block["p_hat"], block["truth"]))
            entry["spearman_rho"] = float(_spearman(block["p_hat"], block["truth"]))
            finite_maes.append(mae)
        arms[arm] = entry
    return {
        "arm_mean_abs_error": (
            float(np.mean(finite_maes)) if finite_maes else float("nan")
        ),
        "arms": arms,
    }


def _normalized_per_arm_agreement(eval_report: dict) -> dict:
    arms = {}
    finite_maes: list[float] = []
    for arm in ARM_ORDER:
        entry = eval_report["arms"][arm]
        mae = float(entry["mean_abs_error_vs_integrated_true"])
        if _is_finite(mae):
            finite_maes.append(mae)
        arms[arm] = {
            "mean_abs_error_vs_integrated_true": mae,
            "pearson_r": float(entry["pearson_r"]),
            "spearman_rho": float(entry["spearman_rho"]),
        }
    return {
        "arm_mean_abs_error": (
            float(np.mean(finite_maes)) if finite_maes else float("nan")
        ),
        "arms": arms,
    }


# ---------------------------------------------------------------------------
# Effect-contrast section builders
# ---------------------------------------------------------------------------


def _condition_label(rule: object) -> str:
    if rule.column == "failure_category":
        return f"{rule.column}=={rule.equals_value}"
    return f"{rule.column}>={rule.min_threshold}"


def _normalized_per_arm_contrasts(eval_report: dict) -> dict:
    main_effects = {}
    interaction_cells = {}
    for arm in ARM_ORDER:
        entry = eval_report["arms"][arm]
        main_effects[arm] = {
            "configured_logit": float(entry["main_effect_configured_logit"]),
            "estimated_contrast_logit": float(
                entry["main_effect_estimated_logit_contrast"]
            ),
            "gap_logit": float(entry["main_effect_recovery_gap_logit"]),
        }
        for key in sorted(entry["interaction_cells"]):
            interaction_cells[key] = entry["interaction_cells"][key]
    return {"interaction_cells": interaction_cells, "main_effects": main_effects}


def _pooled_effect_contrasts(blocks: dict, policy: TreatmentPolicy) -> dict:
    control = blocks["CONTROL"]
    control_logits = control["logits"]

    main_effects = {}
    for arm in ARM_ORDER:
        block = blocks[arm]
        configured = float(policy.main_effects_logit.get(arm, float("nan")))
        estimated = float("nan")
        gap = float("nan")
        if block["n"] > 0 and control_logits is not None:
            estimated = float(block["logits"].mean() - control_logits.mean())
            if _is_finite(configured):
                gap = estimated - configured
        main_effects[arm] = {
            "configured_logit": configured,
            "estimated_contrast_logit": estimated,
            "gap_logit": gap,
        }

    cells_by_arm: dict[str, dict[str, dict]] = {
        arm: {} for arm in ARM_ORDER
    }
    for rule in policy.interactions:
        key = _interaction_key(rule)
        cell = {
            "attenuation_expected": float(rule.effect_logit) < 0.0,
            "column": rule.column,
            "condition": _condition_label(rule),
            "configured_effect_logit": float(rule.effect_logit),
            "estimated_cell_contrast_logit": float("nan"),
            "n_cell": 0,
            "recovery_gap_logit": float("nan"),
        }
        arm_block = blocks[rule.action]
        has_support = (
            arm_block["n"] > 0
            and control["n"] > 0
            and control_logits is not None
        )
        if has_support:
            arm_mask = _rule_row_mask(arm_block["rows"], rule)
            control_mask = _control_rule_mask(control["rows"], rule)
            cell["n_cell"] = int(arm_mask.sum())
            if arm_mask.any() and control_mask.any():
                configured_main = main_effects[rule.action]["configured_logit"]
                difference = float(
                    arm_block["logits"][arm_mask].mean()
                    - control_logits[control_mask].mean()
                )
                estimated_cell = difference - float(configured_main)
                cell["estimated_cell_contrast_logit"] = estimated_cell
                cell["recovery_gap_logit"] = (
                    estimated_cell - float(rule.effect_logit)
                )
        cells_by_arm[rule.action][key] = cell

    interaction_cells = {}
    for arm in ARM_ORDER:
        for key in sorted(cells_by_arm[arm]):
            interaction_cells[key] = cells_by_arm[arm][key]
    return {"interaction_cells": interaction_cells, "main_effects": main_effects}


# ---------------------------------------------------------------------------
# Pure pre-registered rule engine (D-E2)
# ---------------------------------------------------------------------------


def _apply_rule(
    predictive: dict,
    agreement: dict,
    effect_contrasts: dict,
    smallest_test_arm: str,
) -> dict:
    """Apply the four pre-registered D-E2 criteria to plain metric dicts.

    Pure function over already-computed sections so unit tests can force
    every outcome with synthetic dicts. Comparisons with non-finite inputs
    fail CLOSED toward the per-arm reference. Criterion semantics:

    1. strict micro-Brier CI non-overlap: pooled upper bound strictly below
       the per-arm lower bound (point ordering alone is insufficient);
    2. ground-truth agreement no worse: pooled arm-mean
       |P_hat - integrated TRUE| <= the per-arm arm-mean;
    3. smallest-arm Brier no worse: pooled test Brier on
       ``smallest_test_arm`` <= the per-arm Brier on those rows;
    4. interaction recovery within the kind-aware band for every POOLED
       cell WITHOUT the ``attenuation_expected`` annotation; annotated
       (fatigue-type weak negative) cells are reported, never gated.
    """
    pooled_micro_ci = [
        float(value) for value in predictive["pooled"]["micro"]["brier_ci95"]
    ]
    per_arm_micro_ci = [
        float(value) for value in predictive["per_arm"]["micro"]["brier_ci95"]
    ]
    criterion_1 = all(
        _is_finite(value) for value in pooled_micro_ci + per_arm_micro_ci
    ) and (pooled_micro_ci[1] < per_arm_micro_ci[0])
    entries = [
        {
            "criterion": CRITERION_STRICT_CI,
            "evidence": {
                "comparison": "pooled_upper_strictly_below_per_arm_lower",
                "per_arm_lower": per_arm_micro_ci[0],
                "per_arm_upper": per_arm_micro_ci[1],
                "pooled_lower": pooled_micro_ci[0],
                "pooled_upper": pooled_micro_ci[1],
            },
            "passed": bool(criterion_1),
        }
    ]

    pooled_mae = float(agreement["pooled"]["arm_mean_abs_error"])
    per_arm_mae = float(agreement["per_arm"]["arm_mean_abs_error"])
    criterion_2 = (
        _is_finite(pooled_mae)
        and _is_finite(per_arm_mae)
        and pooled_mae <= per_arm_mae
    )
    entries.append(
        {
            "criterion": CRITERION_AGREEMENT,
            "evidence": {
                "comparison": "pooled_arm_mean_no_worse_than_per_arm",
                "per_arm_mean_abs_error_vs_integrated_true": per_arm_mae,
                "pooled_mean_abs_error_vs_integrated_true": pooled_mae,
            },
            "passed": bool(criterion_2),
        }
    )

    pooled_small = float(
        predictive["pooled"]["arms"][smallest_test_arm]["brier"]
    )
    per_arm_small = float(
        predictive["per_arm"]["arms"][smallest_test_arm]["brier"]
    )
    criterion_3 = (
        _is_finite(pooled_small)
        and _is_finite(per_arm_small)
        and pooled_small <= per_arm_small
    )
    entries.append(
        {
            "criterion": CRITERION_SMALLEST_ARM,
            "evidence": {
                "comparison": "pooled_smallest_arm_brier_no_worse",
                "per_arm_brier": per_arm_small,
                "pooled_brier": pooled_small,
                "smallest_test_arm": smallest_test_arm,
            },
            "passed": bool(criterion_3),
        }
    )

    band = float(effect_contrasts.get("gate_band_logit", float("nan")))
    cells = effect_contrasts["pooled"]["interaction_cells"]
    gated_gaps: dict[str, float] = {}
    annotated_keys: list[str] = []
    for key in sorted(cells):
        cell = cells[key]
        if cell.get("attenuation_expected") is True:
            annotated_keys.append(key)
            continue
        gated_gaps[key] = float(cell.get("recovery_gap_logit", float("nan")))
    criterion_4 = _is_finite(band)
    for gap in gated_gaps.values():
        if not (_is_finite(gap) and abs(gap) <= band):
            criterion_4 = False
    entries.append(
        {
            "criterion": CRITERION_INTERACTION_BAND,
            "evidence": {
                "annotated_not_gated": annotated_keys,
                "band": band,
                "gated_cells": gated_gaps,
            },
            "passed": bool(criterion_4),
        }
    )

    ordered = [
        next(entry for entry in entries if entry["criterion"] == name)
        for name in CRITERIA_ORDER
    ]
    preferred = (
        PREFERRED_POOLED
        if all(entry["passed"] for entry in ordered)
        else PREFERRED_PER_ARM
    )
    return {
        "applies_to": RULE_APPLIES_TO,
        "criteria": ordered,
        "preferred_model": preferred,
        "rule_id": RULE_ID,
    }


# ---------------------------------------------------------------------------
# Complexity section
# ---------------------------------------------------------------------------


def _unwrap_xgb_classifier(model: object) -> object | None:
    """Walk calibration/frozen wrappers down to the underlying XGBClassifier."""
    current = model
    for _ in range(8):
        named_steps = getattr(current, "named_steps", None)
        if isinstance(named_steps, dict) and "classifier" in named_steps:
            step = named_steps["classifier"]
            return step if type(step).__name__ == "XGBClassifier" else None
        calibrated = getattr(current, "calibrated_classifiers_", None)
        if calibrated:
            current = calibrated[0]
            continue
        inner = getattr(current, "estimator", None)
        if inner is not None:
            current = inner
            continue
        return None
    return None


def _capacity_proxy(model: object) -> dict:
    classifier = _unwrap_xgb_classifier(model)
    if classifier is None:
        raise ValueError(
            "could not unwrap an XGBClassifier through the supplied bundle "
            "wrappers; the capacity proxy requires the shipped "
            "Pipeline/XGBClassifier layout"
        )
    params = classifier.get_params()
    n_estimators = int(params["n_estimators"])
    max_depth = int(params["max_depth"])
    return {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "proxy": n_estimators * max_depth,
    }


def _complexity_section(
    per_arm_bundle: ActionModelBundle,
    pooled_bundle: PooledModelBundle,
    train_frame: pd.DataFrame | None,
    validation_frame: pd.DataFrame | None,
    seed: int,
) -> dict:
    trainers = {
        "per_arm": train_action_models,
        "pooled": train_pooled_model,
    }
    fits = {"per_arm": len(ARM_ORDER), "pooled": 1}
    proxies = {
        "per_arm": _capacity_proxy(per_arm_bundle.models[ARM_ORDER[0]]),
        "pooled": _capacity_proxy(pooled_bundle.model),
    }
    families = {}
    for family in ("per_arm", "pooled"):
        fit_seconds = None
        if train_frame is not None and validation_frame is not None:
            started = perf_counter()
            trainers[family](train_frame, validation_frame, seed=seed)
            fit_seconds = round(float(perf_counter() - started), 6)
        families[family] = {
            "fit_seconds": fit_seconds,
            "fits": fits[family],
            "parameter_count_proxy": {
                "per_pipeline": proxies[family],
                "total": proxies[family]["proxy"] * fits[family],
            },
        }
    return {
        "families": {key: families[key] for key in sorted(families)},
        "measurement_note": (
            "fit wall-clock seconds measured with perf_counter during a "
            "timed refit at the 100% train fraction (training fit only, "
            "calibration excluded); machine-dependent by nature and the ONE "
            "field excluded from byte-determinism comparisons; None when "
            "the caller supplies no train/validation frames"
        ),
        "parameter_count_note": (
            "documented capacity proxy: n_estimators * max_depth per fitted "
            "XGBoost pipeline; a coarse capacity comparison, never an exact "
            "node or parameter count"
        ),
    }


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


def compare_models(
    per_arm_calibrated_bundle: ActionModelBundle,
    pooled_calibrated_bundle: PooledModelBundle,
    baseline_model: object,
    test_frame: pd.DataFrame,
    policy: TreatmentPolicy,
    seed: int = 20260826,
    *,
    bootstrap_replications: int = 500,
    train_frame: pd.DataFrame | None = None,
    validation_frame: pd.DataFrame | None = None,
) -> dict:
    """Run the Gate A pooled-vs-per-arm comparison under the D-E2 rule.

    Both supplied bundles MUST be sigmoid-calibrated (detected from bundle
    metadata, mirroring ``ml.action_evaluation``): the pre-registered verdict
    applies to the shipped calibrated form only, and raw bundles raise
    ``ValueError`` naming the offending family. Both families are scored on
    the identical ``randomized`` test segment; the per-arm side reuses
    :func:`ml.action_evaluation.evaluate_action_models` wholesale while the
    pooled side slices its shared pipeline's predictions per assigned arm and
    reuses the same stratified-bootstrap CI helper with corrected pool
    offsets. ``train_frame``/``validation_frame`` are optional and used ONLY
    for the complexity section's timed refit at the 100% fraction (both or
    neither). Nothing here mutates its inputs, the Day 2 baseline stays
    untouched beyond being handed to the evaluator, and identical inputs
    reproduce an identical report apart from the documented fit-seconds
    fields.
    """
    _require_frame(test_frame, "test_frame")
    _reject_missing_observation_columns(test_frame, "test_frame")
    _reject_missing_feature_columns(test_frame, "test_frame")
    if (train_frame is None) != (validation_frame is None):
        raise ValueError(
            "train_frame and validation_frame must be supplied together for "
            "the complexity section's timed refit"
        )
    if not isinstance(per_arm_calibrated_bundle, ActionModelBundle):
        raise ValueError(
            "per_arm_calibrated_bundle must be an ActionModelBundle, got "
            f"{type(per_arm_calibrated_bundle).__name__}"
        )
    if not isinstance(pooled_calibrated_bundle, PooledModelBundle):
        raise ValueError(
            "pooled_calibrated_bundle must be a PooledModelBundle, got "
            f"{type(pooled_calibrated_bundle).__name__}"
        )
    per_arm_kind = _detect_bundle_kind(per_arm_calibrated_bundle)
    if per_arm_kind != BUNDLE_KIND_CALIBRATED:
        raise ValueError(
            "the D-E2 rule applies to calibrated bundles only: the per_arm "
            f"bundle kind is '{per_arm_kind}'; calibrate it before comparing"
        )
    pooled_kind = _detect_bundle_kind(pooled_calibrated_bundle)
    if pooled_kind != BUNDLE_KIND_CALIBRATED:
        raise ValueError(
            "the D-E2 rule applies to calibrated bundles only: the pooled "
            f"bundle kind is '{pooled_kind}'; calibrate it before comparing"
        )
    unknown_arms = sorted(set(per_arm_calibrated_bundle.arms) - set(CANONICAL_ARMS))
    if unknown_arms:
        raise ValueError(
            f"per_arm bundle carries non-canonical arms: {unknown_arms}"
        )
    if not isinstance(policy, TreatmentPolicy):
        raise ValueError(
            f"policy must be a TreatmentPolicy, got {type(policy).__name__}"
        )
    if bootstrap_replications < 1:
        raise ValueError(
            "bootstrap_replications must be >= 1, got "
            f"{bootstrap_replications}"
        )

    randomized = test_frame.loc[test_frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]
    if len(randomized) == 0:
        raise ValueError(
            "test_frame holds zero 'randomized' stratum rows; the comparison "
            "consumes only randomized observations"
        )

    per_arm_eval = evaluate_action_models(
        per_arm_calibrated_bundle,
        baseline_model,
        test_frame,
        policy,
        seed=seed,
        bootstrap_replications=bootstrap_replications,
    )

    # The ONE sanctioned randomness derivation: named-seed bootstrap stream
    # feeding the reused stratified-bootstrap helper on the pooled side.
    rng = np.random.default_rng(seed)

    blocks = _pooled_blocks(pooled_calibrated_bundle, randomized, policy)
    predictive = {
        "per_arm": _normalized_per_arm_family(per_arm_eval),
        "pooled": _pooled_predictive(blocks, rng, bootstrap_replications),
    }
    agreement = {
        "jensen_note": SECONDARY_COMPARISON_NOTE,
        "per_arm": _normalized_per_arm_agreement(per_arm_eval),
        "pooled": _family_agreement_from_blocks(blocks),
        "truth_label": TRUTH_LABEL,
    }
    effect_contrasts = {
        "gate_band_logit": CALIBRATED_GATE_BAND_LOGIT,
        "note": (
            "kind-aware documented band for CALIBRATED bundles; "
            "attenuation_expected cells (weak negative effects such as the "
            "RETRY_LATER late-stage fatigue rule) are reported, never gated"
        ),
        "per_arm": _normalized_per_arm_contrasts(per_arm_eval),
        "pooled": _pooled_effect_contrasts(blocks, policy),
    }
    smallest_test_arm = _smallest_test_arm(randomized)
    rule_application = _apply_rule(
        predictive, agreement, effect_contrasts, smallest_test_arm
    )
    complexity = _complexity_section(
        per_arm_calibrated_bundle,
        pooled_calibrated_bundle,
        train_frame,
        validation_frame,
        seed,
    )

    return {
        "bootstrap": {
            "confidence_level": 0.95,
            "replications": int(bootstrap_replications),
            "scheme": BOOTSTRAP_SCHEME,
        },
        "bundle_kind": BUNDLE_KIND_CALIBRATED,
        "complexity": complexity,
        "effect_contrasts": effect_contrasts,
        "ground_truth_agreement": agreement,
        "label": LABEL_OBSERVED,
        "n_randomized_test_rows": int(len(randomized)),
        "predictive": predictive,
        "rule_application": rule_application,
        "scope_note": SCOPE_NOTE,
        "seed": int(seed),
        "smallest_test_arm": smallest_test_arm,
        "truth_label": TRUTH_LABEL,
    }


# ---------------------------------------------------------------------------
# Standalone protocols (Task 6 calls these explicitly)
# ---------------------------------------------------------------------------


def _fit_and_calibrate_families(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    seed: int,
) -> tuple[ActionModelBundle, PooledModelBundle]:
    per_arm_raw, _ = train_action_models(train_frame, validation_frame, seed=seed)
    pooled_raw, _ = train_pooled_model(train_frame, validation_frame, seed=seed)
    return (
        calibrate_action_models(per_arm_raw, validation_frame),
        calibrate_pooled_model(pooled_raw, validation_frame),
    )


def _point_briers(
    bundle: object, randomized_test: pd.DataFrame, smallest_test_arm: str
) -> dict:
    """Assigned-arm scoring on the randomized test segment, no bootstrap."""
    prediction_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    smallest_predictions: np.ndarray | None = None
    smallest_labels: np.ndarray | None = None
    for arm in ARM_ORDER:
        rows = randomized_test.loc[randomized_test[ACTION_COLUMN] == arm]
        if len(rows) == 0:
            raise ValueError(
                f"test segment holds zero randomized rows for arm '{arm}'; "
                "curve and stability points require every canonical arm"
            )
        predictions = _predict_for_family(bundle, rows, arm)
        labels = rows[TARGET_COLUMN].astype(int).to_numpy()
        prediction_blocks.append(predictions)
        label_blocks.append(labels)
        if arm == smallest_test_arm:
            smallest_predictions = predictions
            smallest_labels = labels
    all_labels = np.concatenate(label_blocks)
    all_predictions = np.concatenate(prediction_blocks)
    assert smallest_labels is not None and smallest_predictions is not None
    return {
        "micro_brier": float(_brier(all_labels, all_predictions)),
        "smallest_arm": smallest_test_arm,
        "smallest_arm_brier": float(_brier(smallest_labels, smallest_predictions)),
        "smallest_arm_n": int(smallest_labels.size),
    }


def _fraction_key(fraction: float) -> str:
    return format(float(fraction), ".2f")


def learning_curves(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    fractions: tuple[float, ...] = (0.25, 0.5, 1.0),
    seed: int = 20260826,
) -> dict:
    """Sample-efficiency curves for BOTH families (D-E6 protocol).

    Fractions apply to the TRAIN segment only, taken as a deterministic
    prefix of its randomized rows (frame order preserved, no reshuffling);
    calibration at EVERY fraction uses the FULL validation segment unchanged
    so the points stay comparable; the per-arm family refits all five arms at
    each fraction (intended and counted). Each point reports micro-Brier and
    smallest-arm Brier on the identical randomized test segment. Fully
    deterministic: no wall clock, no generator derivations.
    """
    _require_frame(train_frame, "train_frame")
    _require_frame(validation_frame, "validation_frame")
    _require_frame(test_frame, "test_frame")
    _reject_missing_observation_columns(train_frame, "train_frame")
    _reject_missing_observation_columns(validation_frame, "validation_frame")
    _reject_missing_observation_columns(test_frame, "test_frame")

    requested = sorted({float(value) for value in fractions})
    for fraction in requested:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"fractions must lie in (0, 1], got {fraction!r}"
            )

    randomized_train = train_frame.loc[
        train_frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED
    ]
    randomized_test = test_frame.loc[test_frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]
    if len(randomized_train) == 0:
        raise ValueError(
            "train_frame holds zero 'randomized' stratum rows; learning "
            "curves need randomized training observations"
        )
    smallest_test_arm = _smallest_test_arm(randomized_test)

    curves = {}
    for fraction in requested:
        row_count = max(1, int(np.ceil(fraction * len(randomized_train))))
        sub_train = randomized_train.iloc[:row_count]
        missing_arms = [
            arm
            for arm in ARM_ORDER
            if not bool((sub_train[ACTION_COLUMN] == arm).any())
        ]
        if missing_arms:
            raise ValueError(
                f"fraction {fraction!r} leaves randomized training rows for "
                f"no arm(s) {missing_arms}; thin the fractions, not the arms"
            )
        per_arm_bundle, pooled_bundle = _fit_and_calibrate_families(
            sub_train, validation_frame, seed
        )
        curves[_fraction_key(fraction)] = {
            "n_train_rows_used": int(len(sub_train)),
            "per_arm": _point_briers(
                per_arm_bundle, randomized_test, smallest_test_arm
            ),
            "pooled": _point_briers(
                pooled_bundle, randomized_test, smallest_test_arm
            ),
        }

    return {
        "curves": {key: curves[key] for key in sorted(curves)},
        "fractions": requested,
        "label": LABEL_OBSERVED,
        "protocol_note": (
            "fractions apply to the TRAIN segment only (deterministic "
            "prefix of its randomized rows); calibration at every fraction "
            "uses the FULL validation segment unchanged; the per-arm family "
            "refits all five arms at each fraction; points score the "
            "identical randomized test segment"
        ),
        "smallest_test_arm": smallest_test_arm,
    }


def _mean_incrementals(bundle: object, randomized_test: pd.DataFrame) -> dict:
    """Counterfactual arm-mean lift P_hat(a) - P_hat(CONTROL) per treated arm.

    Every randomized test row is scored under EVERY arm regardless of its
    assignment (both families support counterfactual queries), so the
    quantity is a uniform-view MODEL ESTIMATE contrast within the randomized
    stratum -- nothing causal.
    """
    control = _predict_for_family(bundle, randomized_test, "CONTROL")
    increments = {}
    for arm in TREATED_ARMS:
        treated = _predict_for_family(bundle, randomized_test, arm)
        increments[arm] = float(np.mean(treated - control))
    return increments


def stability_check(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    seeds: tuple[int, ...] = (20260826, 1),
) -> dict:
    """Multi-seed stability for BOTH families (D-E6 protocol).

    For each seed both families are refit on the FULL randomized train
    segment (seed drives the estimator streams only; the dataset world is
    fixed) and calibrated on the FULL validation segment; the report carries
    per-seed micro-Brier plus counterfactual arm-mean incrementals over the
    identical randomized test segment, and population-sd spreads across
    seeds. At least two distinct seeds are required. Deterministic end to
    end.
    """
    _require_frame(train_frame, "train_frame")
    _require_frame(validation_frame, "validation_frame")
    _require_frame(test_frame, "test_frame")
    _reject_missing_observation_columns(train_frame, "train_frame")
    _reject_missing_observation_columns(validation_frame, "validation_frame")
    _reject_missing_observation_columns(test_frame, "test_frame")

    # Caller order is preserved (deduplicated): the canonical seed first is
    # the documented convention, and the sequence itself stays deterministic.
    unique_seeds = list(dict.fromkeys(int(value) for value in seeds))
    if len(unique_seeds) < 2:
        raise ValueError(
            "stability_check requires at least two DISTINCT seeds, got "
            f"{list(seeds)}"
        )

    randomized_test = test_frame.loc[test_frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED]
    if len(randomized_test) == 0:
        raise ValueError(
            "test_frame holds zero 'randomized' stratum rows; stability "
            "points need randomized test observations"
        )
    smallest_test_arm = _smallest_test_arm(randomized_test)

    per_seed = {}
    micro_values = {"per_arm": [], "pooled": []}
    incremental_values = {"per_arm": {}, "pooled": {}}
    for arm in TREATED_ARMS:
        incremental_values["per_arm"][arm] = []
        incremental_values["pooled"][arm] = []
    for seed in unique_seeds:
        per_arm_bundle, pooled_bundle = _fit_and_calibrate_families(
            train_frame, validation_frame, seed
        )
        family_bundles = {"per_arm": per_arm_bundle, "pooled": pooled_bundle}
        seed_entry = {}
        for family in ("per_arm", "pooled"):
            bundle = family_bundles[family]
            point = _point_briers(bundle, randomized_test, smallest_test_arm)
            increments = _mean_incrementals(bundle, randomized_test)
            micro_values[family].append(point["micro_brier"])
            for arm in TREATED_ARMS:
                incremental_values[family][arm].append(increments[arm])
            seed_entry[family] = {
                "mean_incremental_by_arm": {
                    arm: increments[arm] for arm in TREATED_ARMS
                },
                "micro_brier": point["micro_brier"],
            }
        per_seed[str(seed)] = seed_entry

    stability_block = {
        "mean_incremental_sd_by_arm": {
            family: {
                arm: _population_sd(incremental_values[family][arm])
                for arm in TREATED_ARMS
            }
            for family in ("per_arm", "pooled")
        },
        "micro_brier_sd": {
            family: _population_sd(micro_values[family])
            for family in ("per_arm", "pooled")
        },
    }

    return {
        "incremental_note": (
            "MODEL ESTIMATE counterfactual lifts P_hat(a) - P_hat(CONTROL) "
            "averaged over ALL randomized test rows (uniform view, both "
            "families queried under every arm); nothing causal"
        ),
        "label": LABEL_OBSERVED,
        "metric_note": (
            "per-seed micro-Brier scores the assigned-arm view on the "
            "identical randomized test segment; sd is the population "
            "standard deviation across the supplied seeds; the optimizer "
            "verdict itself stays single-canonical-run (stability is "
            "reported, not certified)"
        ),
        "per_seed": {key: per_seed[key] for key in sorted(per_seed)},
        "seeds": unique_seeds,
        "stability": stability_block,
    }
