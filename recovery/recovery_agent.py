"""Recovery agent orchestrator — the full detect → diagnose → decide → execute pipeline.

This is the **main entry point** for the recovery agent.  It wires together:

1. **Detect** — fetch failed payments from Razorpay (or use synthetic data)
2. **Diagnose** — classify each failure's root cause
3. **Decide** — select a recovery action within budget constraints
4. **Execute** — perform the action via the Razorpay API
5. **Audit** — record every decision for compliance and dashboard display

The orchestrator is stateless per invocation but maintains budget counters
across a batch run.  It supports both **live mode** (real Razorpay API) and
**simulation mode** (synthetic data for hackathon demos).

Usage
-----
::

    # Simulation mode (no Razorpay credentials needed)
    agent = RecoveryAgent(simulation=True)
    result = agent.run_batch(synthetic_failures)

    # Live mode (requires RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)
    agent = RecoveryAgent.from_env()
    result = agent.run_live(from_ts=..., to_ts=...)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from recovery.razorpay_client import RazorpayClient, RazorpayConfig, load_razorpay_client_from_env
from recovery.failure_classifier import classify_failure, ClassificationResult, FailureCategory
from recovery.recovery_strategy import (
    RecoveryAction,
    RecoveryActionType,
    StrategyContext,
    select_recovery_action,
)
from recovery.recovery_executor import RecoveryExecutor, RecoveryTarget, ExecutionResult
from recovery.audit_trail import AuditTrail, AuditEntry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Configuration for a recovery agent run.

    Attributes
    ----------
    budget_limit_paise:
        Maximum total budget for retry actions in paise.
    max_retries_per_payment:
        Maximum retry attempts per individual payment.
    human_review_capacity:
        Maximum number of payments that can be escalated to human review.
    simulation:
        If True, use synthetic data and skip real API calls.
    """

    budget_limit_paise: int = 100_000  # ₹1000
    max_retries_per_payment: int = 3
    human_review_capacity: int = 10
    simulation: bool = True


# ---------------------------------------------------------------------------
# Batch result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchResult:
    """Result of running the agent on a batch of failed payments.

    Attributes
    ----------
    total_processed:
        Number of payments processed.
    total_amount_paise:
        Sum of all payment amounts in paise.
    recovered_count:
        Number of payments where retry was attempted.
    recovered_paise:
        Sum of amounts where retry was attempted.
    escalated_count:
        Number of payments escalated to human review.
    stopped_count:
        Number of payments where recovery was stopped.
    budget_used_paise:
        Total budget consumed by retry actions.
    budget_remaining_paise:
        Remaining budget after the run.
    audit_entries:
        Full audit trail for every processed payment.
    """

    total_processed: int
    total_amount_paise: int
    recovered_count: int
    recovered_paise: int
    escalated_count: int
    stopped_count: int
    budget_used_paise: int
    budget_remaining_paise: int
    audit_entries: list[AuditEntry]

    @property
    def recovery_rate(self) -> float:
        """Fraction of total amount where retry was attempted."""
        if self.total_amount_paise == 0:
            return 0.0
        return self.recovered_paise / self.total_amount_paise

    def to_dict(self) -> dict:
        """Serialize for JSON response."""
        trail = AuditTrail()
        trail._entries = list(self.audit_entries)
        return {
            "total_processed": self.total_processed,
            "total_amount_inr": round(self.total_amount_paise / 100.0, 2),
            "recovered_count": self.recovered_count,
            "recovered_inr": round(self.recovered_paise / 100.0, 2),
            "recovery_rate_pct": round(self.recovery_rate * 100, 1),
            "escalated_count": self.escalated_count,
            "stopped_count": self.stopped_count,
            "budget_used_inr": round(self.budget_used_paise / 100.0, 2),
            "budget_remaining_inr": round(self.budget_remaining_paise / 100.0, 2),
            "audit_summary": trail.summary(),
        }


# ---------------------------------------------------------------------------
# Recovery Agent
# ---------------------------------------------------------------------------

class RecoveryAgent:
    """Orchestrates the full recovery pipeline.

    Parameters
    ----------
    config:
        Agent configuration (budget, retries, etc.).
    client:
        Razorpay client (ignored in simulation mode).
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        client: RazorpayClient | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._client = client
        self._executor = RecoveryExecutor(
            client=client,
            simulation=self._config.simulation,
        )
        self._audit = AuditTrail()

    @classmethod
    def from_env(cls, config: AgentConfig | None = None) -> RecoveryAgent:
        """Create a live-mode agent from environment variables."""
        cfg = config or AgentConfig(simulation=False)
        client = load_razorpay_client_from_env()
        return cls(config=cfg, client=client)

    @classmethod
    def simulation(cls, config: AgentConfig | None = None) -> RecoveryAgent:
        """Create a simulation-mode agent (no Razorpay credentials needed)."""
        cfg = config or AgentConfig(simulation=True)
        return cls(config=cfg, client=None)

    @property
    def audit(self) -> AuditTrail:
        """The audit trail for this agent run."""
        return self._audit

    # ── Core pipeline ──────────────────────────────────────────────────

    def _process_one(
        self,
        payment: dict,
        *,
        retry_count: int = 0,
    ) -> AuditEntry:
        """Run the full pipeline on a single payment dict.

        Parameters
        ----------
        payment:
            Razorpay payment object (dict) with at minimum:
            ``id``, ``amount``, ``error_reason``, ``method``.
        retry_count:
            How many times this payment has already been retried.

        Returns
        -------
        AuditEntry
            The audit entry for this payment's recovery cycle.
        """
        start = time.time()

        payment_id = payment.get("id", "unknown")
        amount_paise = payment.get("amount", 0)

        # Step 1: Classify
        classification = classify_failure(payment)

        # Step 2: Build strategy context
        budget_remaining = max(
            0,
            self._config.budget_limit_paise - self._audit.total_recovered_paise,
        )
        ctx = StrategyContext(
            classification=classification,
            amount_paise=amount_paise,
            retry_count=retry_count,
            max_retries=self._config.max_retries_per_payment,
            remaining_budget_paise=budget_remaining,
            hr_capacity_remaining=self._config.human_review_capacity,
        )

        # Step 3: Select action
        action = select_recovery_action(ctx)

        # Step 4: Build target
        target = RecoveryTarget(
            payment_id=payment_id,
            amount_paise=amount_paise,
            customer_id=payment.get("customer_id"),
            customer_email=payment.get("email"),
            customer_contact=payment.get("contact"),
        )

        # Step 5: Execute
        execution = self._executor.execute(action, target)

        # Step 6: Record audit
        elapsed_ms = (time.time() - start) * 1000
        entry = self._audit.record(
            payment_id=payment_id,
            amount_paise=amount_paise,
            classification=classification,
            action=action,
            execution=execution,
            cycle_duration_ms=elapsed_ms,
        )

        return entry

    def run_batch(
        self,
        payments: list[dict],
    ) -> BatchResult:
        """Process a batch of failed payments.

        Parameters
        ----------
        payments:
            List of Razorpay payment dicts (each with ``id``, ``amount``,
            ``error_reason``, ``method``, etc.).

        Returns
        -------
        BatchResult
            Aggregate results including audit trail and budget summary.
        """
        initial_budget = self._config.budget_limit_paise

        for payment in payments:
            self._process_one(payment)

        audit_entries = self._audit.entries
        recovered_paise = self._audit.total_recovered_paise
        budget_used = recovered_paise  # only retry actions consume budget

        return BatchResult(
            total_processed=len(payments),
            total_amount_paise=self._audit.total_amount_paise,
            recovered_count=sum(
                1 for e in audit_entries
                if e.action_type == RecoveryActionType.RETRY_NOW.value
                and e.execution_success
            ),
            recovered_paise=recovered_paise,
            escalated_count=sum(
                1 for e in audit_entries
                if e.action_type == RecoveryActionType.ESCALATE_TO_HUMAN.value
            ),
            stopped_count=sum(
                1 for e in audit_entries
                if e.action_type == RecoveryActionType.STOP.value
            ),
            budget_used_paise=budget_used,
            budget_remaining_paise=max(0, initial_budget - budget_used),
            audit_entries=audit_entries,
        )

    def run_live(
        self,
        *,
        from_ts: int | None = None,
        to_ts: int | None = None,
    ) -> BatchResult:
        """Fetch failed payments from Razorpay and run the recovery pipeline.

        Parameters
        ----------
        from_ts:
            Start of time window (Unix timestamp).
        to_ts:
            End of time window (Unix timestamp).

        Returns
        -------
        BatchResult
            Aggregate results.

        Raises
        ------
        RuntimeError
            If called in simulation mode.
        """
        if self._config.simulation:
            raise RuntimeError("run_live() requires live mode — use run_batch() for simulation")

        assert self._client is not None
        payments = self._client.fetch_failed_payments(from_ts=from_ts, to_ts=to_ts)
        return self.run_batch(payments)


# ---------------------------------------------------------------------------
# Synthetic data for demos
# ---------------------------------------------------------------------------

def generate_synthetic_failures(count: int = 20) -> list[dict]:
    """Generate realistic synthetic failed payment data for hackathon demos.

    Returns a list of payment dicts that mimic the Razorpay API response
    format, covering all major failure categories.

    Parameters
    ----------
    count:
        Number of synthetic failures to generate (distributed across
        categories).
    """
    import random

    templates = [
        # Network errors — should trigger RETRY_NOW
        {
            "method": "card",
            "error_code": "GATEWAY_ERROR",
            "error_source": "gateway",
            "error_step": "payment_processing",
            "error_reason": "timeout",
            "error_description": "Payment gateway timed out",
            "bank": "HDFC",
        },
        # Insufficient funds — should trigger RETRY_LATER
        {
            "method": "card",
            "error_code": None,
            "error_source": "customer",
            "error_step": "payment_authorization",
            "error_reason": "insufficient_funds",
            "error_description": "Insufficient funds in account",
            "bank": "SBIN",
        },
        # OTP failure — should trigger CHANGE_PAYMENT_METHOD
        {
            "method": "card",
            "error_code": None,
            "error_source": "customer",
            "error_step": "payment_authentication",
            "error_reason": "incorrect_otp",
            "error_description": "Payment failed due to incorrect OTP",
            "bank": "ICIC",
        },
        # Card expired — should trigger CHANGE_PAYMENT_METHOD
        {
            "method": "card",
            "error_code": None,
            "error_source": "customer",
            "error_step": "payment_authorization",
            "error_reason": "expired_card",
            "error_description": "Card has expired",
            "bank": "HDFC",
        },
        # UPI auth failure — should trigger CHANGE_PAYMENT_METHOD
        {
            "method": "upi",
            "error_code": None,
            "error_source": "customer",
            "error_step": "payment_authentication",
            "error_reason": "incorrect_upi_pin",
            "error_description": "Incorrect UPI PIN entered",
            "vpa": "user@paytm",
        },
        # Fraud — should trigger ESCALATE
        {
            "method": "card",
            "error_code": None,
            "error_source": "risk",
            "error_step": "risk_assessment",
            "error_reason": "suspected_fraud",
            "error_description": "Transaction flagged as suspected fraud",
            "bank": "HDFC",
        },
        # Daily limit — should trigger RETRY_LATER
        {
            "method": "card",
            "error_code": None,
            "error_source": "customer",
            "error_step": "payment_authorization",
            "error_reason": "daily_limit_exceeded",
            "error_description": "Daily transaction limit exceeded",
            "bank": "SBIN",
        },
        # Generic decline — should trigger RETRY_NOW
        {
            "method": "card",
            "error_code": None,
            "error_source": "customer",
            "error_step": "payment_authorization",
            "error_reason": "do_not_honor",
            "error_description": "Transaction declined by bank",
            "bank": "ICIC",
        },
    ]

    amounts = [5000, 10000, 15000, 20000, 50000, 100000, 250000, 500000]
    customers = [
        {"customer_id": f"cust_{i:04d}", "email": f"customer{i}@example.com", "contact": f"+9190000{i:04d}"}
        for i in range(1, 51)
    ]

    random.seed(42)  # Deterministic for demo
    failures = []
    for i in range(count):
        template = random.choice(templates)
        amount = random.choice(amounts)
        cust = customers[i % len(customers)]
        failures.append({
            **template,
            "id": f"pay_synthetic_{i+1:04d}",
            "amount": amount,
            "currency": "INR",
            "status": "failed",
            "created_at": int(time.time()) - random.randint(0, 86400),
            **cust,
        })

    return failures
