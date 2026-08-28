"""Task 6: Fair deterministic greedy portfolio baseline.

Provides a row-first greedy baseline allocation that consumes the same
CandidatePair universe, action costs, constraints, and ranking as the exact
DP solver. The only difference is the allocation algorithm.

Fairness contract:
- Same candidate universe (CandidatePair objects from build_candidate_universe)
- Same canonical ranking (rank_candidate_pairs)
- Same OptimizerConfig constraints (budget_limit_paise, human_review_capacity)
- Same action_cost_paise, net_incremental_value_inr, positive-value gate
- Same per-row exclusivity (at most one action per attempt_id)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ml.action_model import ARM_ORDER
from ml.portfolio_audit import PortfolioAllocation, PortfolioEntry, PortfolioSummary
from ml.portfolio_optimizer import (
    CandidatePair,
    OptimizerConfig,
    TREATED_ARMS,
    rank_candidate_pairs,
)
from recovery.scoring import RETRY_INTERVENTION_COST_INR


_ARM_ORDER_INDEX: dict[str, int] = {arm: i for i, arm in enumerate(ARM_ORDER)}


def _greedy_select(
    candidates: Sequence[CandidatePair],
    config: OptimizerConfig,
) -> tuple[dict[str, CandidatePair], dict[str, str], dict]:
    """Run the deterministic global-pair-first greedy allocation.

    Algorithm:
    1. Rank all candidates by canonical Task 3 ranking (net value desc,
       attempt_id asc, ARM_ORDER asc).
    2. Iterate through ranked candidates.
    3. Select a candidate only if:
       a. Its attempt_id has not already received an intervention.
       b. Its net incremental value is strictly positive.
       c. Adding its action cost does not exceed the monetary budget.
       d. Adding its HR consumption does not exceed human_review_capacity.
    4. Once one candidate is selected for an attempt_id, no other candidate
       for that row may be selected.
    5. Continue until the ranked list is exhausted.

    Returns:
        (allocated, unallocated_reasons, metadata) matching DP solver output.
    """
    # Compute budget limit in integer paise
    budget_limit_paise: int | None = None
    if config.budget_limit_inr is not None:
        budget_limit_paise = int(round(config.budget_limit_inr * 100))

    hr_capacity: int | None = config.human_review_capacity

    # Rank candidates using canonical Task 3 ranking
    ranked = rank_candidate_pairs(candidates)

    allocated: dict[str, CandidatePair] = {}
    used_attempt_ids: set[str] = set()
    budget_spent_paise: int = 0
    hr_used: int = 0

    for cand in ranked:
        aid = cand.attempt_id

        # Per-row exclusivity
        if aid in used_attempt_ids:
            continue

        # Positive net value gate
        if cand.net_incremental_value_inr <= 0.0:
            continue

        # Budget feasibility (integer paise, exact comparison)
        if budget_limit_paise is not None:
            if budget_spent_paise + cand.action_cost_paise > budget_limit_paise:
                continue

        # HR capacity feasibility
        if hr_capacity is not None:
            if cand.arm == "HUMAN_REVIEW":
                if hr_used + 1 > hr_capacity:
                    continue

        # Select this candidate
        allocated[aid] = cand
        used_attempt_ids.add(aid)
        budget_spent_paise += cand.action_cost_paise
        if cand.arm == "HUMAN_REVIEW":
            hr_used += 1

    # Build unallocated reasons
    by_row: dict[str, list[CandidatePair]] = defaultdict(list)
    for c in candidates:
        by_row[c.attempt_id].append(c)

    all_attempt_ids = sorted(set(c.attempt_id for c in candidates))
    unallocated_reasons: dict[str, str] = {}
    for aid in all_attempt_ids:
        if aid not in allocated:
            cands = by_row[aid]
            has_positive = any(c.net_incremental_value_inr > 0.0 for c in cands)
            if not has_positive:
                unallocated_reasons[aid] = "non_positive_net_value"
            elif budget_limit_paise is not None and budget_spent_paise >= budget_limit_paise:
                unallocated_reasons[aid] = "budget_exhausted"
            elif hr_capacity is not None and hr_used >= hr_capacity:
                unallocated_reasons[aid] = "hr_capacity_exhausted"
            else:
                unallocated_reasons[aid] = "budget_exhausted"

    # Compute metadata
    budget_allocated_paise = budget_spent_paise
    budget_allocated_inr = budget_allocated_paise / 100.0
    budget_remaining_paise = (
        budget_limit_paise - budget_allocated_paise if budget_limit_paise is not None else None
    )
    budget_remaining_inr = (
        budget_remaining_paise / 100.0 if budget_remaining_paise is not None else None
    )

    metadata = {
        "budget_allocated_inr": budget_allocated_inr,
        "budget_allocated_paise": budget_allocated_paise,
        "budget_remaining_inr": budget_remaining_inr,
        "budget_remaining_paise": budget_remaining_paise,
        "hr_allocated_count": hr_used,
        "solver_type": "greedy",
        "preflight_stats": {},
    }

    return allocated, unallocated_reasons, metadata


def optimize_portfolio_greedy(
    candidates: tuple[CandidatePair, ...],
    config: OptimizerConfig,
) -> PortfolioAllocation:
    """Run row-first greedy baseline allocation under identical candidate universe,
    action costs, and constraints as the exact DP solver.

    This function accepts pre-built CandidatePair objects and produces a
    PortfolioAllocation compatible with the exact DP solver output.

    Args:
        candidates: Positive net-value CandidatePair objects from
            build_candidate_universe (same universe as exact DP).
        config: Shared OptimizerConfig with budget_limit_inr and
            human_review_capacity.

    Returns:
        PortfolioAllocation with entries, summary, and metadata.
    """
    allocated, unallocated_reasons, meta = _greedy_select(candidates, config)

    # Build candidates lookup by attempt_id
    by_row: dict[str, list[CandidatePair]] = defaultdict(list)
    for c in candidates:
        by_row[c.attempt_id].append(c)

    all_attempt_ids = sorted(set(c.attempt_id for c in candidates))

    entries: list[PortfolioEntry] = []
    total_overrides = 0
    total_stop_overrides = 0
    rec_counts: dict[str, int] = defaultdict(int)
    auth_counts: dict[str, int] = defaultdict(int)

    for aid in all_attempt_ids:
        cands = by_row[aid]

        if aid in allocated:
            cand = allocated[aid]
            gross_by_arm = {c.arm: c.gross_incremental_value_inr for c in cands}
            cost_by_arm = {c.arm: c.action_cost_inr for c in cands}
            net_by_arm = {c.arm: c.net_incremental_value_inr for c in cands}

            # Sort rank
            sorted_cands = rank_candidate_pairs(tuple(cands))
            sort_rank = None
            for rank_idx, sc in enumerate(sorted_cands, 1):
                if sc.arm == cand.arm:
                    sort_rank = rank_idx
                    break

            entry = PortfolioEntry(
                attempt_id=aid,
                payment_id=cand.payment_id,
                row_index=cand.row_index,
                optimizer_recommendation=cand.arm,
                no_intervention_reason=None,
                gross_incremental_value_by_arm=gross_by_arm,
                action_cost_by_arm=cost_by_arm,
                net_incremental_value_by_arm=net_by_arm,
                selected_gross_incremental_value_inr=cand.gross_incremental_value_inr,
                selected_action_cost_inr=cand.action_cost_inr,
                selected_action_cost_paise=cand.action_cost_paise,
                selected_net_incremental_value_inr=cand.net_incremental_value_inr,
                optimizer_sort_rank=sort_rank,
                authorized_action=cand.arm,
                authorization_reason="greedy_allocation",
                matched_rule_id=None,
                policy_overrode_recommendation=False,
            )
        else:
            reason = unallocated_reasons.get(aid, "non_positive_net_value")
            gross_by_arm = {c.arm: c.gross_incremental_value_inr for c in cands}
            cost_by_arm = {c.arm: c.action_cost_inr for c in cands}
            net_by_arm = {c.arm: c.net_incremental_value_inr for c in cands}
            row_idx = cands[0].row_index if cands else 0
            payment_id = cands[0].payment_id if cands else ""

            entry = PortfolioEntry(
                attempt_id=aid,
                payment_id=payment_id,
                row_index=row_idx,
                optimizer_recommendation="NO_INTERVENTION",
                no_intervention_reason=reason,
                gross_incremental_value_by_arm=gross_by_arm,
                action_cost_by_arm=cost_by_arm,
                net_incremental_value_by_arm=net_by_arm,
                selected_gross_incremental_value_inr=None,
                selected_action_cost_inr=None,
                selected_action_cost_paise=None,
                selected_net_incremental_value_inr=None,
                optimizer_sort_rank=None,
                authorized_action="NO_INTERVENTION",
                authorization_reason="greedy_no_allocation",
                matched_rule_id=None,
                policy_overrode_recommendation=False,
            )

        entries.append(entry)
        rec_counts[entry.optimizer_recommendation] += 1
        auth_counts[entry.authorized_action] += 1

    # Build summary
    optimizer_allocated_count = len(allocated)
    no_intervention_count = len(all_attempt_ids) - optimizer_allocated_count

    summary = PortfolioSummary(
        total_rows=len(all_attempt_ids),
        pre_screen_stopped_count=0,
        invalid_prediction_count=0,
        optimizer_allocated_count=optimizer_allocated_count,
        no_intervention_count=no_intervention_count,
        eligible_candidate_count=len(all_attempt_ids),
        budget_limit_inr=config.budget_limit_inr,
        budget_limit_paise=int(round(config.budget_limit_inr * 100)) if config.budget_limit_inr is not None else None,
        budget_allocated_inr=meta["budget_allocated_inr"],
        budget_allocated_paise=meta["budget_allocated_paise"],
        budget_remaining_inr=meta["budget_remaining_inr"],
        budget_remaining_paise=meta["budget_remaining_paise"],
        human_review_capacity_limit=config.human_review_capacity,
        human_review_allocated_count=meta["hr_allocated_count"],
        post_policy_net_authorized_count=optimizer_allocated_count,
        total_policy_overrides=total_overrides,
        total_policy_stop_overrides=total_stop_overrides,
        optimizer_objective_value_inr=sum(
            c.net_incremental_value_inr for c in allocated.values()
        ),
        optimizer_status="success" if allocated or not all_attempt_ids else "empty_portfolio",
        action_recommendation_counts=dict(rec_counts),
        action_authorized_counts=dict(auth_counts),
    )

    return PortfolioAllocation(
        entries=tuple(entries),
        summary=summary,
        metadata={"solver_type": "greedy"},
    )
