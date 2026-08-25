"""Deterministic composition of probability, scoring, policy, and trace.

The recovery engine chains the existing pure modules into one decision path
per submitted row: ``predict_recovery_probability`` produces the calibrated
estimate ``P(recovered | context)``, ``score_opportunities`` combines that
estimate with the row amount into an expected-recovery-value ranking signal,
and ``build_decision_trace`` freezes the outcome per row. In itself the
engine trains nothing, calls no external service, writes no database,
executes no payment, calls no LLM, and uses no LangGraph;
it exposes no HTTP endpoint, reads no wall clock, performs no I/O, and
prints nothing, so identical inputs always produce identical results.

Candidate rule (the core composition semantic, stated once here): only rows
whose ``OpportunityScore.worth_intervening`` is True are submitted to the
policy engine as intervention candidates. Rows that are not worth
intervening are terminal no-op cases: the engine records
authorized_action="STOP", is_stop=True, matched_rule_id=None,
evaluated_rules=(), and an authorization reason stating that the expected
recovery value was not positive while quoting the exact ERV amount, because
there is no economic case for intervention and manufacturing a positive
authorization would be dishonest. The policy layer is deliberately NOT
consulted for such rows. Two invariants therefore hold mechanically:
(a) a negative or zero expected recovery value can never become an automated
positive action, and (b) every trace's authorized action comes either from a
real policy decision or from this documented no-candidate terminal.

The scoring recommendation and the policy authorization remain
separate fields from scoring through trace, and they may legitimately
diverge; the policy authorization is the final action decision
FOR SUBMITTED CANDIDATES.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd

from ml.train import predict_recovery_probability
from recovery.audit import DecisionTrace, build_decision_trace
from recovery.policy import (
    CANONICAL_ACTIONS,
    PolicyConfig,
    PolicyDecision,
    decide_action,
    load_policy_config,
)
from recovery.scoring import OpportunityScore, score_opportunities

_SCORE_FIELDS = tuple(field.name for field in fields(OpportunityScore))


@dataclass(frozen=True)
class EngineResult:
    """Frozen bundle of per-row traces in dataframe order plus a summary."""

    traces: tuple[DecisionTrace, ...]
    summary: dict[str, object]


def _terminal_no_candidate_decision(erv_inr: float) -> PolicyDecision:
    reason = (
        "No intervention candidate: the expected recovery value of "
        f"{erv_inr:.2f} INR was not positive, so the policy engine was not "
        "consulted and no automated action is authorized."
    )
    return PolicyDecision(
        authorized_action="STOP",
        matched_rule_id=None,
        matched_rule_name=None,
        priority=None,
        reason=reason,
        is_stop=True,
        evaluated_rules=(),
    )


def _score_at(scored: pd.DataFrame, position: int) -> OpportunityScore:
    values = scored.iloc[position]
    return OpportunityScore(*(values[name] for name in _SCORE_FIELDS))


def _empty_summary() -> dict[str, object]:
    return {
        "case_count": 0,
        "candidate_count": 0,
        "action_counts": {action: 0 for action in sorted(CANONICAL_ACTIONS)},
        "stop_count": 0,
        "human_review_count": 0,
        "total_candidate_erv_inr": 0.0,
        "noop_count": 0,
    }


def run_recovery_engine(
    df: pd.DataFrame,
    model,
    policy_config: PolicyConfig | None = None,
) -> EngineResult:
    """Compose probability, scoring, policy, and trace for every row of ``df``.

    Uses the shipped business rules when ``policy_config`` is None. Rows stay
    in dataframe order; one frozen trace is emitted per row. See the module
    docstring for the candidate rule and its two mechanical invariants.
    Rows carrying NaN in policy-referenced columns may silently skip matching
    rules (NaN comparisons are False); the shipped numeric pipeline fails
    loudly before policy when core features are NaN.
    """
    if len(df) == 0:
        return EngineResult(traces=(), summary=_empty_summary())

    config = load_policy_config() if policy_config is None else policy_config
    probabilities = predict_recovery_probability(model, df)
    scored = score_opportunities(df, probabilities)

    traces: list[DecisionTrace] = []
    candidate_count = 0
    noop_count = 0
    total_candidate_erv_inr = 0.0
    action_counts: dict[str, int] = {
        action: 0 for action in sorted(CANONICAL_ACTIONS)
    }

    for position in range(len(df)):
        row = df.iloc[position]
        score = _score_at(scored, position)
        if bool(scored["worth_intervening"].iloc[position]):
            context = dict(row)
            context["recovery_probability"] = float(probabilities[position])
            decision = decide_action(context, config)
            candidate_count += 1
            total_candidate_erv_inr += float(score.expected_recovery_value_inr)
        else:
            decision = _terminal_no_candidate_decision(
                score.expected_recovery_value_inr
            )
            noop_count += 1
        trace = build_decision_trace(row, score, decision)
        action_counts[trace.authorized_action] += 1
        traces.append(trace)

    summary = {
        "case_count": len(traces),
        "candidate_count": candidate_count,
        "action_counts": action_counts,
        "stop_count": action_counts["STOP"],
        "human_review_count": action_counts["HUMAN_REVIEW"],
        "total_candidate_erv_inr": round(total_candidate_erv_inr, 2),
        "noop_count": noop_count,
    }
    return EngineResult(traces=tuple(traces), summary=summary)
