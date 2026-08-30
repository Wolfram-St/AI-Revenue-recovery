"""Audit trail for the recovery agent — immutable decision log and persistence layer.

Every detect → classify → decide → execute cycle produces an ``AuditEntry`` that
captures the full decision chain for compliance, debugging, and the
money-recovered dashboard.

The audit trail is append-only and provides both in-memory buffering and
parameterised PostgreSQL mapping matching ``db/schema.sql`` for persistent durability.
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

from recovery.failure_classifier import ClassificationResult, FailureCategory
from recovery.recovery_strategy import RecoveryAction, RecoveryActionType
from recovery.recovery_executor import ExecutionResult


# ---------------------------------------------------------------------------
# Audit Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditEntry:
    """Immutable log entry for a single recovery attempt.

    Attributes
    ----------
    entry_id:
        Unique identifier for this audit entry.
    payment_id:
        Razorpay payment ID being recovered.
    subscription_id:
        Razorpay subscription ID (if applicable).
    amount_paise:
        Original payment amount in paise.
    failure_category:
        Root-cause category from the classifier.
    classification_confidence:
        Classifier confidence (0.0–1.0).
    classification_reason:
        Human-readable classification explanation.
    raw_error_code:
        Razorpay error code (if any).
    raw_error_reason:
        Razorpay error reason (if any).
    action_type:
        The recovery action that was selected.
    strategy_reason:
        Why this action was chosen.
    retry_count:
        How many retries have been attempted (before this one).
    estimated_cost_paise:
        Estimated cost of the action.
    execution_success:
        Whether the action executed successfully.
    new_order_id:
        Razorpay order ID created for retry (if applicable).
    execution_message:
        Human-readable execution status.
    execution_error:
        Error message if execution failed.
    cost_paise:
        Actual cost incurred in paise.
    created_at:
        Unix timestamp when this entry was created.
    cycle_duration_ms:
        Wall-clock time for the full detect→classify→decide→execute cycle.
    """

    entry_id: str
    payment_id: str
    subscription_id: str
    amount_paise: int

    # Classification
    failure_category: str
    classification_confidence: float
    classification_reason: str
    raw_error_code: str
    raw_error_reason: str

    # Strategy
    action_type: str
    strategy_reason: str
    retry_count: int
    estimated_cost_paise: int

    # Execution
    execution_success: bool
    new_order_id: str
    execution_message: str
    execution_error: str
    cost_paise: int

    # Metadata
    created_at: float = field(default_factory=time.time)
    cycle_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class AuditLogRecord:
    """Represents an immutable record mapping to the PostgreSQL audit_logs table."""

    recovery_case_id: str | None
    event_type: str
    actor_type: str
    action: str | None
    decision_reason: str | None
    event_payload: dict[str, Any]
    created_at: str  # Canonical ISO-8601 UTC string


# ---------------------------------------------------------------------------
# SQL Persistence Helpers (matching db/schema.sql)
# ---------------------------------------------------------------------------

def build_sql_insert_for_audit_entry(entry: AuditEntry) -> tuple[str, tuple[Any, ...]]:
    """Generate parameterised SQL INSERT for an AuditEntry matching db/schema.sql audit_logs table."""
    query = """
    INSERT INTO audit_logs (
        recovery_case_id,
        event_type,
        actor_type,
        action,
        decision_reason,
        event_payload,
        created_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s);
    """.strip()

    dt = datetime.datetime.fromtimestamp(entry.created_at, tz=datetime.timezone.utc)
    created_iso = dt.isoformat()
    payload_json = entry.to_json()

    params = (
        entry.payment_id,
        "RECOVERY_EXECUTION",
        "SYSTEM",
        entry.action_type,
        f"{entry.strategy_reason} (Category: {entry.failure_category})",
        payload_json,
        created_iso,
    )
    return query, params


# ---------------------------------------------------------------------------
# Audit trail store
# ---------------------------------------------------------------------------

class AuditTrail:
    """Append-only audit log supporting in-memory buffers and SQL persistence export."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._counter: int = 0

    def record(
        self,
        *,
        payment_id: str,
        subscription_id: str = "",
        amount_paise: int,
        classification: ClassificationResult,
        action: RecoveryAction,
        execution: ExecutionResult,
        cycle_duration_ms: float = 0.0,
    ) -> AuditEntry:
        """Create and append an audit entry for a recovery cycle."""
        self._counter += 1
        entry = AuditEntry(
            entry_id=f"audit_{self._counter:06d}",
            payment_id=payment_id,
            subscription_id=subscription_id,
            amount_paise=amount_paise,
            failure_category=classification.category.value,
            classification_confidence=classification.confidence,
            classification_reason=classification.reason,
            raw_error_code=classification.raw_error_code or "",
            raw_error_reason=classification.raw_error_reason or "",
            action_type=execution.action_type.value,
            strategy_reason=action.reason,
            retry_count=action.retry_count,
            estimated_cost_paise=action.estimated_cost_paise,
            execution_success=execution.success,
            new_order_id=execution.new_order_id or "",
            execution_message=execution.message,
            execution_error=execution.error or "",
            cost_paise=execution.cost_paise,
            created_at=execution.executed_at,
            cycle_duration_ms=cycle_duration_ms,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[AuditEntry]:
        """All audit entries in chronological order."""
        return list(self._entries)

    @property
    def count(self) -> int:
        """Total number of audit entries."""
        return len(self._entries)

    def to_sql_inserts(self) -> list[tuple[str, tuple[Any, ...]]]:
        """Convert all buffered entries into parameterised SQL insert statements."""
        return [build_sql_insert_for_audit_entry(e) for e in self._entries]

    # ── Aggregate statistics ───────────────────────────────────────────

    @property
    def total_amount_paise(self) -> int:
        """Total amount of all failed payments in the trail."""
        return sum(e.amount_paise for e in self._entries)

    @property
    def total_recovered_paise(self) -> int:
        """Total amount where execution succeeded and action was retry_now."""
        return sum(
            e.cost_paise
            for e in self._entries
            if e.execution_success and e.action_type == RecoveryActionType.RETRY_NOW.value
        )

    @property
    def recovery_rate(self) -> float:
        """Fraction of total amount that was successfully retried."""
        total = self.total_amount_paise
        if total == 0:
            return 0.0
        return self.total_recovered_paise / total

    @property
    def action_distribution(self) -> dict[str, int]:
        """Count of each action type across all entries."""
        dist: dict[str, int] = {}
        for e in self._entries:
            dist[e.action_type] = dist.get(e.action_type, 0) + 1
        return dist

    @property
    def category_distribution(self) -> dict[str, int]:
        """Count of each failure category across all entries."""
        dist: dict[str, int] = {}
        for e in self._entries:
            dist[e.failure_category] = dist.get(e.failure_category, 0) + 1
        return dist

    @property
    def success_rate(self) -> float:
        """Fraction of executions that succeeded."""
        if not self._entries:
            return 0.0
        return sum(1 for e in self._entries if e.execution_success) / len(self._entries)

    def summary(self) -> dict[str, Any]:
        """Produce a summary dict for the dashboard API."""
        return {
            "total_entries": self.count,
            "total_amount_inr": self.total_amount_paise / 100.0,
            "total_recovered_inr": self.total_recovered_paise / 100.0,
            "recovery_rate_pct": round(self.recovery_rate * 100, 1),
            "success_rate_pct": round(self.success_rate * 100, 1),
            "action_distribution": [
                {"action": k, "count": v}
                for k, v in sorted(self.action_distribution.items())
            ],
            "category_distribution": [
                {"category": k, "count": v}
                for k, v in sorted(self.category_distribution.items())
            ],
        }

    def clear(self) -> None:
        """Clear all entries (for testing only)."""
        self._entries.clear()
        self._counter = 0
