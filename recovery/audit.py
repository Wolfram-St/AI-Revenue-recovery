"""Deterministic, explainable decision and audit traces per recovery case.

A trace preserves BOTH the scoring recommendation and the policy
authorization as separate concepts, and the two MAY legitimately diverge --
for example a case can carry ``scoring_recommendation == "INTERVENE"``
while policy authorizes ``STOP``. Nothing reconciles them after the fact:
the trace records what scoring proposed and what policy authorized,
verbatim, next to the decision-time facts and the frozen model contract
marker ``P(recovered | context)``.

Field names mirror ``db/schema.sql`` vocabulary so a later persistence
layer maps one-to-one onto ``recovery_cases.*``,
``recovery_actions.authorization_reason``, and ``audit_logs.event_payload``.
This module performs no database writes, uses no ORM, exposes no API, reads
no wall clock (the only timestamp is the row's own ``event_timestamp``
metadata), and generates no identifiers of its own: identical inputs always
produce identical traces and byte-identical JSON.
"""

from __future__ import annotations

import datetime
import json
import math
import numbers
from dataclasses import dataclass, fields
from typing import Any, Sequence

import pandas as pd

from recovery.policy import CANONICAL_ACTIONS, PolicyDecision
from recovery.scoring import INTERVENE, NO_INTERVENTION, OpportunityScore

MODEL_CONTRACT = "P(recovered | context)"

REQUIRED_ROW_KEYS = (
    "attempt_id",
    "payment_id",
    "customer_id",
    "event_timestamp",
    "amount_inr",
    "failure_category",
)


def _validated_finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{name} must be a real finite number, got {value!r} of type "
            f"{type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _validated_probability(name: str, value: Any) -> float:
    number = _validated_finite_real(name, value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0], got {value!r}")
    return number


def _validated_non_negative_real(name: str, value: Any) -> float:
    number = _validated_finite_real(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return number


def _validated_text(name: str, value: Any) -> str:
    """Validate and normalize text; accepted values are stored stripped."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value.strip()


def _validated_optional_text(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _validated_text(name, value)


def _validated_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def _coerced_evaluated_rules(value: Any) -> tuple[tuple[str, bool], ...]:
    pairs = []
    try:
        entries = list(value)
    except TypeError as exc:
        raise ValueError(
            f"evaluated_rules must be an iterable of (rule_id, matched) "
            f"pairs, got {value!r}"
        ) from exc
    for entry in entries:
        try:
            pair = tuple(entry)
        except TypeError as exc:
            raise ValueError(
                f"evaluated_rules entries must be (rule_id, matched) pairs, "
                f"got {entry!r}"
            ) from exc
        if len(pair) != 2:
            raise ValueError(
                f"evaluated_rules entries must be (rule_id, matched) pairs, "
                f"got {entry!r}"
            )
        rule_id, matched = pair
        pairs.append((_validated_text("evaluated_rules rule id", rule_id),
                      _validated_bool("evaluated_rules matched flag", matched)))
    return tuple(pairs)


def _normalized_event_timestamp(value: Any) -> str:
    """Normalize once to a canonical ISO-8601 UTC string.

    Accepts ``pd.Timestamp``, ``datetime``, or ISO-8601 string; naive
    inputs are interpreted as UTC. Derived only from row metadata.
    """
    if isinstance(value, pd.Timestamp):
        stamp = value
    elif isinstance(value, datetime.datetime):
        stamp = pd.Timestamp(value)
    elif isinstance(value, str):
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"event_timestamp {value!r} is not a parsable timestamp"
            ) from exc
    else:
        raise ValueError(
            f"event_timestamp must be a pd.Timestamp, datetime, or ISO-8601 "
            f"string, got {value!r} of type {type(value).__name__}"
        )
    if pd.isna(stamp):
        raise ValueError(f"event_timestamp must be a real timestamp, got {value!r}")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


@dataclass(frozen=True)
class DecisionTrace:
    """Frozen, JSON-safe record of one scored and authorized case."""

    attempt_id: str
    payment_id: str
    customer_id: str
    event_timestamp: str
    amount_inr: float
    failure_category: str
    recovery_probability: float
    expected_recovery_value_inr: float
    scoring_recommendation: str
    authorized_action: str
    authorization_reason: str
    matched_rule_id: str | None
    matched_rule_name: str | None
    rule_priority: int | None
    is_stop: bool
    evaluated_rules: tuple[tuple[str, bool], ...]
    model_contract: str
    probability_is_action_conditional: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id",
                           _validated_text("attempt_id", self.attempt_id))
        object.__setattr__(self, "payment_id",
                           _validated_text("payment_id", self.payment_id))
        object.__setattr__(self, "customer_id",
                           _validated_text("customer_id", self.customer_id))
        _validated_text("event_timestamp", self.event_timestamp)
        object.__setattr__(
            self,
            "amount_inr",
            round(_validated_non_negative_real("amount_inr", self.amount_inr), 2),
        )
        object.__setattr__(
            self,
            "failure_category",
            _validated_text("failure_category", self.failure_category),
        )
        object.__setattr__(
            self,
            "recovery_probability",
            _validated_probability("recovery_probability", self.recovery_probability),
        )
        object.__setattr__(
            self,
            "expected_recovery_value_inr",
            _validated_finite_real(
                "expected_recovery_value_inr", self.expected_recovery_value_inr
            ),
        )
        if self.scoring_recommendation not in (INTERVENE, NO_INTERVENTION):
            raise ValueError(
                f"scoring_recommendation {self.scoring_recommendation!r} must be "
                f"exactly {INTERVENE!r} or {NO_INTERVENTION!r}"
            )
        if self.authorized_action not in CANONICAL_ACTIONS:
            raise ValueError(
                f"authorized_action {self.authorized_action!r} is outside the "
                f"canonical vocabulary {sorted(CANONICAL_ACTIONS)}"
            )
        object.__setattr__(
            self,
            "authorization_reason",
            _validated_text("authorization_reason", self.authorization_reason),
        )
        object.__setattr__(self, "matched_rule_id",
                           _validated_optional_text("matched_rule_id", self.matched_rule_id))
        object.__setattr__(self, "matched_rule_name",
                           _validated_optional_text("matched_rule_name",
                                                    self.matched_rule_name))
        if self.rule_priority is not None and type(self.rule_priority) is not int:
            raise ValueError(
                f"rule_priority must be an integer or None, got {self.rule_priority!r}"
            )
        object.__setattr__(self, "is_stop", _validated_bool("is_stop", self.is_stop))
        object.__setattr__(self, "evaluated_rules",
                           _coerced_evaluated_rules(self.evaluated_rules))
        if self.model_contract != MODEL_CONTRACT:
            raise ValueError(
                f"model_contract must be exactly {MODEL_CONTRACT!r}, got "
                f"{self.model_contract!r}"
            )
        if self.probability_is_action_conditional is not False:
            raise ValueError(
                "probability_is_action_conditional must be exactly False; this "
                "system estimates P(recovered | context) only"
            )
        if self.is_stop != (self.authorized_action == "STOP"):
            raise ValueError(
                f"is_stop={self.is_stop!r} contradicts authorized_action "
                f"{self.authorized_action!r}: is_stop must equal "
                "(authorized_action == 'STOP')"
            )


def build_decision_trace(
    row, score: OpportunityScore, decision: PolicyDecision
) -> DecisionTrace:
    """Build one frozen decision trace from decision-time facts.

    ``row`` supplies identity and failure metadata; ``score`` supplies the
    scoring recommendation ``INTERVENE``/``NO_INTERVENTION`` plus ERV;
    ``decision`` supplies the policy authorization. The recommendation and
    authorization are stored verbatim as separate concepts and may diverge
    (e.g., INTERVENE recommended while STOP authorized). The only timestamp
    is the row's own ``event_timestamp``, normalized once to a canonical
    ISO-8601 UTC string; nothing here reads any clock.
    """
    missing = [key for key in REQUIRED_ROW_KEYS if key not in row]
    if missing:
        raise ValueError(f"row is missing required decision-time keys: {missing}")

    return DecisionTrace(
        attempt_id=row["attempt_id"],
        payment_id=row["payment_id"],
        customer_id=row["customer_id"],
        event_timestamp=_normalized_event_timestamp(row["event_timestamp"]),
        amount_inr=row["amount_inr"],
        failure_category=row["failure_category"],
        recovery_probability=score.recovery_probability,
        expected_recovery_value_inr=score.expected_recovery_value_inr,
        scoring_recommendation=score.scoring_recommendation,
        authorized_action=decision.authorized_action,
        authorization_reason=decision.reason,
        matched_rule_id=decision.matched_rule_id,
        matched_rule_name=decision.matched_rule_name,
        rule_priority=decision.priority,
        is_stop=decision.is_stop,
        evaluated_rules=tuple(decision.evaluated_rules),
        model_contract=MODEL_CONTRACT,
        probability_is_action_conditional=False,
    )


def _jsonify_tuples(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonify_tuples(item) for item in value]
    return value


def trace_to_dict(trace: DecisionTrace) -> dict:
    """Return a plain JSON-safe dict preserving field order."""
    payload = {}
    for field in fields(trace):
        payload[field.name] = _jsonify_tuples(getattr(trace, field.name))
    return payload


def traces_to_json(traces: Sequence[DecisionTrace]) -> str:
    """Serialize traces deterministically; identical inputs yield identical bytes."""
    return json.dumps([trace_to_dict(trace) for trace in traces], sort_keys=False)
