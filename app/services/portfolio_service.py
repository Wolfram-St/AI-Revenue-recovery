"""Portfolio service: orchestrate the Day 7 optimizer pipeline.

The service calls the existing core functions in the correct order:
1. build_candidate_universe (pre-screen + model predictions)
2. solve_portfolio_allocation (exact 2D DP)
3. authorize_post_allocation (post-allocation policy)

The HTTP layer preserves Day 7 semantics: allocation and authorization
are separate stages, STOP rules remain dominant, and the portfolio
allocator's accounting is frozen before authorization.
"""

from __future__ import annotations

from ml.portfolio_optimizer import (
    OptimizerConfig,
    build_candidate_universe,
    solve_portfolio_allocation,
    authorize_post_allocation,
)
from ml.portfolio_audit import PortfolioAllocation
from recovery.policy import load_policy_config

from app.errors import AnalysisError
from app.services.data_bootstrap import get_bootstrap


def optimize_portfolio(
    budget_inr: float,
    human_review_capacity: int,
) -> PortfolioAllocation:
    """Run the full Day 7 portfolio optimization pipeline."""
    bootstrap = get_bootstrap()

    if bootstrap.action_bundle is None:
        raise AnalysisError("Action models are not available for portfolio optimization")

    policy = load_policy_config()

    try:
        candidates, pre_screened_entries, metadata = build_candidate_universe(
            bootstrap.dataset,
            bootstrap.action_bundle,
            policy,
        )
    except Exception as exc:
        raise AnalysisError(f"Candidate universe construction failed: {exc}") from exc

    eligible_attempt_ids = set(
        bootstrap.dataset["attempt_id"]
    ) - set(pre_screened_entries.keys())

    config = OptimizerConfig(
        budget_limit_inr=budget_inr,
        human_review_capacity=human_review_capacity,
        max_supported_rows=1000,
        max_supported_budget_units=500,
        max_supported_hr_capacity=200,
    )

    try:
        allocated, unallocated_reasons, solver_metadata = solve_portfolio_allocation(
            candidates,
            eligible_attempt_ids,
            config,
        )
    except Exception as exc:
        raise AnalysisError(f"Portfolio allocation failed: {exc}") from exc

    allocation = authorize_post_allocation(
        allocated=allocated,
        unallocated_reasons=unallocated_reasons,
        pre_screened_entries=pre_screened_entries,
        eligible_attempt_ids=eligible_attempt_ids,
        all_candidates=candidates,
        policy=policy,
        candidate_frame=bootstrap.dataset,
    )

    merged_metadata = {**solver_metadata, **allocation.metadata}
    return PortfolioAllocation(
        entries=allocation.entries,
        summary=allocation.summary,
        metadata=merged_metadata,
    )
