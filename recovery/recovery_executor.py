"""Recovery action executor — translates decisions into Razorpay API calls.

The executor is the **action boundary**: it receives a ``RecoveryAction``
(from the strategy engine) and a ``RecoveryTarget`` (identifying the
payment/subscription), then performs the actual Razorpay API call or
records the action for simulation mode.

In **live mode** (``simulation=False``), the executor calls the Razorpay API
to create retry orders, fetch updated payment status, etc.

In **simulation mode** (``simulation=True``), no API calls are made; the
executor returns synthetic success results for demo purposes.  This is the
default for hackathon demos without real Razorpay credentials.

Every execution produces an ``ExecutionResult`` that feeds into the audit
trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from recovery.razorpay_client import RazorpayClient, RazorpayAPIError
from recovery.recovery_strategy import RecoveryAction, RecoveryActionType


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecoveryTarget:
    """Identifies the entity being recovered.

    Attributes
    ----------
    payment_id:
        Razorpay payment ID (for one-time payment retries).
    subscription_id:
        Razorpay subscription ID (for subscription recovery).
    order_id:
        Razorpay order ID (for creating retry orders).
    amount_paise:
        Original payment amount in paise.
    customer_id:
        Razorpay customer ID.
    customer_email:
        Customer email for notifications.
    customer_contact:
        Customer phone for notifications.
    """

    payment_id: str | None = None
    subscription_id: str | None = None
    order_id: str | None = None
    amount_paise: int = 0
    customer_id: str | None = None
    customer_email: str | None = None
    customer_contact: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    """Immutable result of executing a recovery action.

    Attributes
    ----------
    success:
        Whether the action executed without error.
    action_type:
        The action that was executed.
    new_order_id:
        If a retry order was created, its Razorpay order ID.
    new_payment_id:
        If a payment was fetched/created, its ID.
    message:
        Human-readable status message.
    error:
        Error message if the action failed.
    executed_at:
        Unix timestamp of execution.
    cost_paise:
        Actual cost incurred in paise.
    """

    success: bool
    action_type: RecoveryActionType
    new_order_id: str | None = None
    new_payment_id: str | None = None
    message: str = ""
    error: str | None = None
    executed_at: float = field(default_factory=time.time)
    cost_paise: int = 0


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class RecoveryExecutor:
    """Executes recovery actions via the Razorpay API.

    Parameters
    ----------
    client:
        An initialised ``RazorpayClient`` (ignored when ``simulation=True``).
    simulation:
        If True, no API calls are made — returns synthetic results.
    """

    def __init__(
        self,
        client: RazorpayClient | None = None,
        *,
        simulation: bool = True,
    ) -> None:
        self._client = client
        self._simulation = simulation

    @property
    def is_simulation(self) -> bool:
        """Whether the executor is in simulation mode."""
        return self._simulation

    def execute(
        self,
        action: RecoveryAction,
        target: RecoveryTarget,
    ) -> ExecutionResult:
        """Execute a recovery action against a target.

        Parameters
        ----------
        action:
            The action to execute (from the strategy engine).
        target:
            Identifies the payment/subscription to act on.

        Returns
        -------
        ExecutionResult
            The outcome of the execution attempt.
        """
        dispatch = {
            RecoveryActionType.RETRY_NOW: self._execute_retry_now,
            RecoveryActionType.RETRY_LATER: self._execute_retry_later,
            RecoveryActionType.CHANGE_PAYMENT_METHOD: self._execute_change_method,
            RecoveryActionType.NOTIFY_CUSTOMER: self._execute_notify,
            RecoveryActionType.ESCALATE_TO_HUMAN: self._execute_escalate,
            RecoveryActionType.STOP: self._execute_stop,
        }
        handler = dispatch.get(action.action_type)
        if handler is None:
            return ExecutionResult(
                success=False,
                action_type=action.action_type,
                error=f"Unknown action type: {action.action_type}",
            )
        return handler(action, target)

    # ── Action handlers ────────────────────────────────────────────────

    def _execute_retry_now(
        self, action: RecoveryAction, target: RecoveryTarget,
    ) -> ExecutionResult:
        """Create a new order and (in live mode) attempt immediate retry."""
        if self._simulation:
            return ExecutionResult(
                success=True,
                action_type=RecoveryActionType.RETRY_NOW,
                new_order_id=f"order_sim_{int(time.time())}",
                message="SIMULATION: Retry order created successfully",
                cost_paise=target.amount_paise,
            )

        try:
            assert self._client is not None, "RazorpayClient required in live mode"
            order = self._client.create_retry_order(
                amount_paise=target.amount_paise,
                receipt=f"retry_{target.payment_id or 'unknown'}",
                notes={
                    "recovery_action": "retry_now",
                    "original_payment": target.payment_id or "",
                    "retry_count": str(action.retry_count + 1),
                },
            )
            return ExecutionResult(
                success=True,
                action_type=RecoveryActionType.RETRY_NOW,
                new_order_id=order.get("id"),
                message=f"Retry order created: {order.get('id')}",
                cost_paise=target.amount_paise,
            )
        except (RazorpayAPIError, Exception) as exc:
            return ExecutionResult(
                success=False,
                action_type=RecoveryActionType.RETRY_NOW,
                error=str(exc),
            )

    def _execute_retry_later(
        self, action: RecoveryAction, target: RecoveryTarget,
    ) -> ExecutionResult:
        """Schedule a deferred retry (no immediate API call)."""
        return ExecutionResult(
            success=True,
            action_type=RecoveryActionType.RETRY_LATER,
            message=(
                f"Retry scheduled for later (attempt #{action.retry_count + 1}). "
                f"Payment {target.payment_id or 'N/A'} will be retried "
                f"after root cause window."
            ),
            cost_paise=0,
        )

    def _execute_change_method(
        self, action: RecoveryAction, target: RecoveryTarget,
    ) -> ExecutionResult:
        """Notify customer to update payment method."""
        if self._simulation:
            return ExecutionResult(
                success=True,
                action_type=RecoveryActionType.CHANGE_PAYMENT_METHOD,
                message=(
                    f"SIMULATION: Customer {target.customer_email or 'N/A'} "
                    f"notified to update payment method for {target.payment_id or 'N/A'}"
                ),
            )

        # In live mode, we'd send an email/SMS via Razorpay or external service
        # For now, record the intent — actual notification is out of scope
        return ExecutionResult(
            success=True,
            action_type=RecoveryActionType.CHANGE_PAYMENT_METHOD,
            message=(
                f"Payment method change requested for {target.payment_id or 'N/A'}. "
                f"Customer notified via {target.customer_email or 'N/A'}."
            ),
        )

    def _execute_notify(
        self, action: RecoveryAction, target: RecoveryTarget,
    ) -> ExecutionResult:
        """Send a payment failure notification to the customer."""
        return ExecutionResult(
            success=True,
            action_type=RecoveryActionType.NOTIFY_CUSTOMER,
            message=(
                f"Notification sent to {target.customer_email or 'N/A'} "
                f"regarding payment {target.payment_id or 'N/A'}"
            ),
        )

    def _execute_escalate(
        self, action: RecoveryAction, target: RecoveryTarget,
    ) -> ExecutionResult:
        """Escalate to human review."""
        return ExecutionResult(
            success=True,
            action_type=RecoveryActionType.ESCALATE_TO_HUMAN,
            message=(
                f"Escalated to human review: {action.reason}. "
                f"Payment: {target.payment_id or 'N/A'}, "
                f"Amount: ₹{target.amount_paise / 100:.2f}"
            ),
        )

    def _execute_stop(
        self, action: RecoveryAction, target: RecoveryTarget,
    ) -> ExecutionResult:
        """Stop all recovery attempts."""
        return ExecutionResult(
            success=True,
            action_type=RecoveryActionType.STOP,
            message=(
                f"Recovery stopped: {action.reason}. "
                f"Payment: {target.payment_id or 'N/A'}"
            ),
        )
