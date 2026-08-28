"""Task 7: Leakage-safe portfolio outcome evaluation.

Provides offline evaluation of frozen portfolio allocations against held-out
Day 4 synthetic outcomes. Allocation must be completely fixed before any
outcome data is joined.

Temporal boundary (mandatory ordering):
    decision-time inputs
    -> candidate construction
    -> pre-allocation policy
    -> optimizer / greedy allocation
    -> post-allocation policy authorization
    -> FREEZE ALLOCATION
    -> HELD-OUT OUTCOME JOIN  <-- this module
    -> evaluation metrics

This module never calls:
    - predict_all_actions()
    - build_candidate_universe()
    - solve_portfolio_allocation()
    - optimize_portfolio_greedy()
    - rank_candidate_pairs()
    - model training or fitting functions
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.portfolio_audit import PortfolioAllocation


# =============================================================================
# Validation Helpers
# =============================================================================

def _validate_outcome_frame(
    allocation: PortfolioAllocation,
    outcome_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Validate outcome frame has required columns and matching attempt_ids.

    Raises:
        ValueError: if outcome_frame is empty, missing required columns,
            has duplicate attempt_ids, or allocation attempt_ids are not
            present in outcome_frame.
    """
    if outcome_frame.empty:
        raise ValueError("outcome_frame must not be empty")

    required_cols = {"attempt_id", "amount_inr", "recovered"}
    missing = required_cols - set(outcome_frame.columns)
    if missing:
        raise ValueError(f"outcome_frame missing required columns: {missing}")

    # Check for duplicate attempt_ids
    if outcome_frame["attempt_id"].duplicated().any():
        dupes = outcome_frame[outcome_frame["attempt_id"].duplicated()]["attempt_id"].tolist()
        raise ValueError(f"outcome_frame contains duplicate attempt_ids: {dupes[:5]}")

    # Check allocation attempt_ids are present in outcome_frame
    alloc_ids = {e.attempt_id for e in allocation.entries}
    outcome_ids = set(outcome_frame["attempt_id"])
    missing_ids = alloc_ids - outcome_ids
    if missing_ids:
        raise ValueError(
            f"allocation attempt_ids not found in outcome_frame: "
            f"{sorted(missing_ids)[:10]}"
        )

    return outcome_frame


# =============================================================================
# Core Evaluation
# =============================================================================

def evaluate_portfolio_allocation(
    allocation: PortfolioAllocation,
    outcome_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate a sealed portfolio allocation against held-out synthetic outcomes.

    This function is a scorekeeper. It does NOT:
    - call prediction models
    - rebuild candidates
    - run optimization or greedy allocation
    - mutate the allocation in any way

    The allocation must be fully frozen before this function is called.

    Args:
        allocation: A frozen PortfolioAllocation from optimize_portfolio,
            optimize_portfolio_greedy, or any allocation pipeline.
        outcome_frame: DataFrame with at least columns:
            - attempt_id (str): payment attempt identifier
            - amount_inr (float): payment amount in INR
            - recovered (int/bool): binary synthetic recovery outcome
            Optionally:
            - assigned_action (str): the randomized treatment assignment
              (needed for observational comparison using CONTROL arm rows)

    Returns:
        Deterministic evaluation dictionary with:
        - total_evaluated: number of rows evaluated
        - evaluated_attempt_ids: sorted list of evaluated attempt IDs
        - intervention_count: rows with non-STOP/non-NO_INTERVENTION authorized action
        - stop_count: rows with STOP authorized action
        - no_intervention_count: rows with NO_INTERVENTION authorized action
        - total_recovered: sum of recovered for evaluated rows
        - total_recovered_amount_inr: sum of (recovered * amount_inr)
        - recovery_rate: total_recovered / total_evaluated (0 if none)
        - policy_overrides_evaluated: count of rows where policy overrode optimizer
        - action_metrics: per-action metrics dict
        - comparisons: confounded and observational comparison blocks
        - model_objective_value_inr: sum of selected_net_incremental_value_inr
    """
    # Validate inputs (allocation is NOT mutated)
    _validate_outcome_frame(allocation, outcome_frame)

    # Build outcome lookup (attempt_id -> row data)
    outcome_by_id: dict[str, dict] = {}
    for _, row in outcome_frame.iterrows():
        outcome_by_id[row["attempt_id"]] = {
            "amount_inr": float(row["amount_inr"]),
            "recovered": int(row["recovered"]),
            "recovered_amount_inr": float(row["amount_inr"]) * int(row["recovered"]),
            "assigned_action": row.get("assigned_action"),
            "failure_category": row.get("failure_category"),
        }

    # Process each allocation entry
    total_evaluated = 0
    intervention_count = 0
    stop_count = 0
    no_intervention_count = 0
    total_recovered = 0
    total_recovered_amount_inr = 0.0
    total_amount_inr = 0.0
    policy_overrides_count = 0
    model_objective = 0.0

    action_metrics: dict[str, dict[str, Any]] = {}
    intervention_rows: list[dict] = []
    no_intervention_rows: list[dict] = []

    for entry in allocation.entries:
        aid = entry.attempt_id
        total_evaluated += 1
        outcome = outcome_by_id[aid]
        auth_action = entry.authorized_action
        recovered = outcome["recovered"]
        amount = outcome["amount_inr"]
        recovered_amount = outcome["recovered_amount_inr"]

        total_recovered += recovered
        total_recovered_amount_inr += recovered_amount
        total_amount_inr += amount

        if entry.policy_overrode_recommendation:
            policy_overrides_count += 1

        if entry.selected_net_incremental_value_inr is not None:
            model_objective += entry.selected_net_incremental_value_inr

        # Classify by authorized action
        if auth_action == "STOP":
            stop_count += 1
        elif auth_action == "NO_INTERVENTION":
            no_intervention_count += 1
            no_intervention_rows.append({
                "attempt_id": aid,
                "amount_inr": amount,
                "recovered": recovered,
                "recovered_amount_inr": recovered_amount,
            })
        else:
            intervention_count += 1
            intervention_rows.append({
                "attempt_id": aid,
                "authorized_action": auth_action,
                "amount_inr": amount,
                "recovered": recovered,
                "recovered_amount_inr": recovered_amount,
            })

        # Per-action metrics
        if auth_action not in action_metrics:
            action_metrics[auth_action] = {
                "count": 0,
                "recovered": 0,
                "recovered_amount_inr": 0.0,
                "total_amount_inr": 0.0,
            }
        am = action_metrics[auth_action]
        am["count"] += 1
        am["recovered"] += recovered
        am["recovered_amount_inr"] += recovered_amount
        am["total_amount_inr"] += amount

    # Compute recovery rate
    recovery_rate = total_recovered / total_evaluated if total_evaluated > 0 else 0.0

    # --- Confounded comparison ---
    # Compares intervention rows against NO_INTERVENTION rows within the portfolio
    intervention_recovered = sum(r["recovered"] for r in intervention_rows)
    intervention_count_cmp = len(intervention_rows)
    intervention_recovery_rate = (
        intervention_recovered / intervention_count_cmp if intervention_count_cmp > 0 else 0.0
    )
    intervention_recovered_amount = sum(r["recovered_amount_inr"] for r in intervention_rows)

    no_int_recovered = sum(r["recovered"] for r in no_intervention_rows)
    no_int_count_cmp = len(no_intervention_rows)
    no_int_recovery_rate = (
        no_int_recovered / no_int_count_cmp if no_int_count_cmp > 0 else 0.0
    )
    no_int_recovered_amount = sum(r["recovered_amount_inr"] for r in no_intervention_rows)

    confounded = {
        "label": "CONFOUNDED: portfolio intervention vs NO_INTERVENTION",
        "intervention_count": intervention_count_cmp,
        "intervention_recovery_rate": round(intervention_recovery_rate, 6),
        "intervention_recovered_amount_inr": round(intervention_recovered_amount, 2),
        "no_intervention_count": no_int_count_cmp,
        "no_intervention_recovery_rate": round(no_int_recovery_rate, 6),
        "no_intervention_recovered_amount_inr": round(no_int_recovered_amount, 2),
    }

    # --- Unconfounded comparison ---
    # Compares intervention rows against CONTROL arm rows from the outcome frame
    control_rows = outcome_frame[
        outcome_frame.get("assigned_action", pd.Series(dtype=str)) == "CONTROL"
    ] if "assigned_action" in outcome_frame.columns else pd.DataFrame()

    control_count = len(control_rows)
    if control_count > 0:
        control_recovered = int(control_rows["recovered"].sum())
        control_recovery_rate = control_recovered / control_count
        control_recovered_amount = float(
            (control_rows["recovered"].astype(int) * control_rows["amount_inr"]).sum()
        )
    else:
        control_recovered = 0
        control_recovery_rate = 0.0
        control_recovered_amount = 0.0

    unconfounded = {
        "label": "OBSERVATIONAL: optimizer-selected intervention vs randomized CONTROL",
        "intervention_count": intervention_count_cmp,
        "intervention_recovery_rate": round(intervention_recovery_rate, 6),
        "intervention_recovered_amount_inr": round(intervention_recovered_amount, 2),
        "control_count": control_count,
        "control_recovery_rate": round(control_recovery_rate, 6),
        "control_recovered_amount_inr": round(control_recovered_amount, 2),
    }

    return {
        "total_evaluated": total_evaluated,
        "evaluated_attempt_ids": sorted({e.attempt_id for e in allocation.entries}),
        "intervention_count": intervention_count,
        "stop_count": stop_count,
        "no_intervention_count": no_intervention_count,
        "total_recovered": total_recovered,
        "total_recovered_amount_inr": round(total_recovered_amount_inr, 2),
        "total_amount_inr_at_risk": round(total_amount_inr, 2),
        "recovery_rate": round(recovery_rate, 6),
        "policy_overrides_evaluated": policy_overrides_count,
        "model_objective_value_inr": round(model_objective, 2),
        "action_metrics": action_metrics,
        "comparisons": {
            "confounded": confounded,
            "observational": unconfounded,
        },
    }


# =============================================================================
# Baseline Comparison
# =============================================================================

def compare_portfolio_to_baseline(
    optimizer_eval: dict,
    greedy_eval: dict,
) -> dict[str, Any]:
    """Compare optimizer evaluation metrics against greedy baseline metrics.

    Args:
        optimizer_eval: Evaluation dict from evaluate_portfolio_allocation
            applied to the optimizer allocation.
        greedy_eval: Evaluation dict from evaluate_portfolio_allocation
            applied to the greedy allocation.

    Returns:
        Comparison dictionary with deltas and labels.
    """
    opt_obj = optimizer_eval.get("model_objective_value_inr", 0.0)
    gr_obj = greedy_eval.get("model_objective_value_inr", 0.0)

    opt_recovered_amount = optimizer_eval.get("total_recovered_amount_inr", 0.0)
    gr_recovered_amount = greedy_eval.get("total_recovered_amount_inr", 0.0)

    opt_recovered = optimizer_eval.get("total_recovered", 0)
    gr_recovered = greedy_eval.get("total_recovered", 0)

    objective_delta = round(opt_obj - gr_obj, 2)
    recovered_amount_delta = round(opt_recovered_amount - gr_recovered_amount, 2)
    recovery_count_delta = opt_recovered - gr_recovered

    if objective_delta > 0:
        advantage_label = "PORTFOLIO_ADVANTAGE_OBSERVED"
    elif objective_delta < 0:
        advantage_label = "BASELINE_ADVANTAGE_OBSERVED"
    else:
        advantage_label = "NO_PORTFOLIO_ADVANTAGE_OBSERVED"

    return {
        "objective_delta_inr": objective_delta,
        "recovered_amount_delta_inr": recovered_amount_delta,
        "recovery_count_delta": recovery_count_delta,
        "advantage_label": advantage_label,
        "optimizer_model_objective_value_inr": opt_obj,
        "greedy_model_objective_value_inr": gr_obj,
    }
