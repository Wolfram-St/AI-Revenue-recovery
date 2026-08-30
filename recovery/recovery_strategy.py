"""Recovery strategy engine — maps failure root causes to bounded actions.

The strategy engine is the **decision layer** between classification and
execution.  Given a ``ClassificationResult`` and a ``StrategyContext``
containing budget/HR constraints and retry history, it produces a
``RecoveryAction`` that is safe to execute.

Design principles
-----------------
1. **Deterministic** — identical inputs always produce the same action.
2. **Budget-aware** — never recommends an action that exceeds the remaining
   monetary budget or HR capacity.
3. **Retry-capped** — enforces a configurable maximum retry count per
   payment before escalating to human review or stopping.
4. **Closed action vocabulary** — the set of possible actions is frozen
   to ensure downstream executors handle every case.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from recovery.failure_classifier import ClassificationResult, FailureCategory


# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------

class RecoveryActionType(str, enum.Enum):
    """Closed set of recovery actions the agent can take."""

    RETRY_NOW = "retry_now"
    RETRY_LATER = "retry_later"
    CHANGE_PAYMENT_METHOD = "change_payment_method"
    NOTIFY_CUSTOMER = "notify_customer"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    STOP = "stop"


@dataclass(frozen=True)
class RecoveryAction:
    """A concrete, executable recovery decision.

    Attributes
    ----------
    action_type:
        The action to take.
    reason:
        Human-readable explanation of *why* this action was chosen.
    retry_count:
        How many times this payment has already been retried (before this
        action).
    estimated_cost_paise:
        Estimated cost of executing this action in paise.  Zero for
        notifications and escalations.
    """

    action_type: RecoveryActionType
    reason: str
    retry_count: int
    estimated_cost_paise: int


from recovery.compliance import evaluate_all_compliance_rules


# ---------------------------------------------------------------------------
# Strategy context — everything the engine needs to decide
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyContext:
    """Immutable context for strategy selection.

    Attributes
    ----------
    classification:
        Root-cause classification of the failure.
    amount_paise:
        Original payment amount in paise.
    retry_count:
        How many retries have already been attempted for this payment.
    max_retries:
        Maximum retries allowed before escalating (default 3).
    remaining_budget_paise:
        Remaining recovery budget in paise.
    hr_capacity_remaining:
        Number of human-review slots still available.
    timestamp:
        Current evaluation timestamp (for TRAI hours and cooling-off checks).
    contact_history:
        Sequence of past contact timestamps for this customer.
    last_refusal_timestamp:
        Timestamp of last customer refusal or dispute if any.
    ptp_status:
        Active Promise-to-Pay status (e.g. 'PTP_ACTIVE').
    ptp_promised_date:
        Promised payment date for active PTP.
    dispute_status:
        Account dispute status ('NONE', 'UNDER_REVIEW', 'CONFIRMED').
    enforce_compliance:
        Whether to enforce TRAI/RBI compliance checks (default False for baseline tests).
    """

    classification: ClassificationResult
    amount_paise: int
    retry_count: int = 0
    max_retries: int = 3
    remaining_budget_paise: int = 100_000  # ₹1000 default
    hr_capacity_remaining: int = 10
    timestamp: Any | None = None
    contact_history: Sequence[Any] = ()
    last_refusal_timestamp: Any | None = None
    ptp_status: str | None = None
    ptp_promised_date: Any | None = None
    dispute_status: str | None = None
    enforce_compliance: bool = False


# ---------------------------------------------------------------------------
# Strategy rules — category → action mapping
# ---------------------------------------------------------------------------

def _retry_now_action(ctx: StrategyContext) -> RecoveryAction | None:
    """Attempt immediate retry if budget allows."""
    cost = ctx.amount_paise  # retry costs the payment amount
    if cost > ctx.remaining_budget_paise:
        return None
    return RecoveryAction(
        action_type=RecoveryActionType.RETRY_NOW,
        reason="Immediate retry feasible — budget permits and failure is transient",
        retry_count=ctx.retry_count,
        estimated_cost_paise=cost,
    )


def _retry_later_action(ctx: StrategyContext) -> RecoveryAction | None:
    """Schedule a deferred retry (no immediate budget impact)."""
    return RecoveryAction(
        action_type=RecoveryActionType.RETRY_LATER,
        reason="Deferred retry recommended — customer may resolve root cause",
        retry_count=ctx.retry_count,
        estimated_cost_paise=0,  # no cost until actual retry
    )


def _change_payment_method_action(ctx: StrategyContext) -> RecoveryAction:
    """Ask customer to switch payment method."""
    return RecoveryAction(
        action_type=RecoveryActionType.CHANGE_PAYMENT_METHOD,
        reason=f"Root cause is {ctx.classification.category.value} — "
               f"changing payment method may resolve the issue",
        retry_count=ctx.retry_count,
        estimated_cost_paise=0,
    )


def _escalate_action(ctx: StrategyContext, reason: str) -> RecoveryAction:
    """Escalate to human review."""
    return RecoveryAction(
        action_type=RecoveryActionType.ESCALATE_TO_HUMAN,
        reason=reason,
        retry_count=ctx.retry_count,
        estimated_cost_paise=0,
    )


def _stop_action(ctx: StrategyContext, reason: str) -> RecoveryAction:
    """Stop all recovery attempts."""
    return RecoveryAction(
        action_type=RecoveryActionType.STOP,
        reason=reason,
        retry_count=ctx.retry_count,
        estimated_cost_paise=0,
    )


# ---------------------------------------------------------------------------
# Main strategy function
# ---------------------------------------------------------------------------

def select_recovery_action(ctx: StrategyContext) -> RecoveryAction:
    """Select the optimal recovery action for a failed payment.

    The decision tree is evaluated in priority order:

    0. **Compliance guard** — if enforce_compliance is set, check TRAI/RBI rules.
    1. **Retry-cap guard** — if ``retry_count >= max_retries``, escalate.
    2. **Budget guard** — if budget is exhausted, escalate or stop.
    3. **Category-specific rules** — map root cause to action.

    Parameters
    ----------
    ctx:
        The full decision context including classification, retry history,
        and remaining budget/HR capacity.

    Returns
    -------
    RecoveryAction
        The recommended action (never None — always produces a decision).
    """
    # ── Guard 0: Regulatory Compliance ────────────────────────────────
    if ctx.enforce_compliance:
        comp_context = {
            "dispute_status": ctx.dispute_status,
            "ptp_status": ctx.ptp_status,
            "ptp_promised_date": ctx.ptp_promised_date,
            "last_refusal_timestamp": ctx.last_refusal_timestamp,
            "contact_history_timestamps": ctx.contact_history,
            "event_timestamp": ctx.timestamp,
        }
        audit = evaluate_all_compliance_rules(comp_context, current_time=ctx.timestamp)
        if not audit.all_passed and audit.blocking_rule:
            block = audit.blocking_rule
            return _stop_action(
                ctx,
                f"Regulatory compliance block [{block.rule_id}]: {block.reason}",
            )

    cat = ctx.classification.category
    retries_exhausted = ctx.retry_count >= ctx.max_retries
    budget_exhausted = ctx.remaining_budget_paise <= 0
    hr_exhausted = ctx.hr_capacity_remaining <= 0

    # ── Guard: retries exhausted ───────────────────────────────────────
    if retries_exhausted:
        if cat in (FailureCategory.FRAUD_SUSPECTED, FailureCategory.UNKNOWN):
            return _stop_action(
                ctx,
                f"Max retries ({ctx.max_retries}) exhausted with "
                f"{cat.value} root cause — stopping recovery",
            )
        if hr_capacity(ctx) > 0:
            return _escalate_action(
                ctx,
                f"Max retries ({ctx.max_retries}) exhausted — "
                f"escalating to human review",
            )
        return _stop_action(
            ctx,
            f"Max retries exhausted and no HR capacity — stopping",
        )

    # ── Guard: budget exhausted ────────────────────────────────────────
    if budget_exhausted and cat not in (
        FailureCategory.FRAUD_SUSPECTED,
        FailureCategory.BUSINESS_ERROR,
    ):
        if hr_capacity(ctx) > 0:
            return _escalate_action(
                ctx,
                "Recovery budget exhausted — escalating to human review",
            )
        return _stop_action(
            ctx,
            "Recovery budget exhausted and no HR capacity — stopping",
        )

    # ── Category-specific strategy ─────────────────────────────────────

    if cat == FailureCategory.NETWORK_ERROR:
        # Network errors are transient — retry immediately
        action = _retry_now_action(ctx)
        if action:
            return action
        return _retry_later_action(ctx)

    if cat == FailureCategory.INSUFFICIENT_FUNDS:
        # Customer may add funds — defer retry
        return _retry_later_action(ctx)

    if cat == FailureCategory.AUTHENTICATION_FAILURE:
        # OTP/PIN failure — customer can retry with correct credentials
        if ctx.retry_count < 1:
            action = _retry_now_action(ctx)
            if action:
                return action
        return _change_payment_method_action(ctx)

    if cat == FailureCategory.CARD_ISSUE:
        # Card problems — ask customer to update card
        return _change_payment_method_action(ctx)

    if cat == FailureCategory.MANDATE_ISSUE:
        # Mandate problem — customer must re-authorize
        return _change_payment_method_action(ctx)

    if cat == FailureCategory.LIMIT_EXCEEDED:
        # Limits hit — defer retry
        return _retry_later_action(ctx)

    if cat == FailureCategory.DO_NOT_HONOR:
        # Generic decline — try again, may be transient
        action = _retry_now_action(ctx)
        if action:
            return action
        return _change_payment_method_action(ctx)

    if cat == FailureCategory.FRAUD_SUSPECTED:
        # Fraud — escalate immediately, do not retry
        return _escalate_action(
            ctx,
            "Fraud suspected — escalating to human review for manual investigation",
        )

    if cat == FailureCategory.BUSINESS_ERROR:
        # Merchant config issue — escalate to fix the root cause
        return _escalate_action(
            ctx,
            "Business/configuration error — requires merchant-side fix",
        )

    # ── UNKNOWN fallback ───────────────────────────────────────────────
    if hr_capacity(ctx) > 0:
        return _escalate_action(
            ctx,
            "Unknown failure root cause — escalating to human review",
        )
    return _stop_action(
        ctx,
        "Unknown root cause and no HR capacity — stopping recovery",
    )


def hr_capacity(ctx: StrategyContext) -> int:
    """Remaining HR capacity (0 if exhausted)."""
    return max(0, ctx.hr_capacity_remaining)
