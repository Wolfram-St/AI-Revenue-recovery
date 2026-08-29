"""Analysis service: invoke real core logic for a single case."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.errors import CaseNotFoundError, AnalysisError
from app.services.data_bootstrap import get_bootstrap
from app.api.schemas.analysis import AnalysisResponse, PolicyInfo, CandidateAction
from ml.action_model import ARM_ORDER, predict_all_actions
from recovery.audit import trace_to_dict
from recovery.policy import load_policy_config, decide_action
from recovery.scoring import score_opportunity


def analyze_case(case_id: str) -> AnalysisResponse:
    bootstrap = get_bootstrap()

    trace = None
    row_idx = None
    for i, t in enumerate(bootstrap.traces):
        if t.attempt_id == case_id:
            trace = t
            row_idx = i
            break

    if trace is None:
        raise CaseNotFoundError(case_id)

    try:
        row = bootstrap.dataset.iloc[row_idx]
    except IndexError:
        raise AnalysisError(f"Cannot locate row for case {case_id}")

    row_dict = dict(row)
    row_dict["recovery_probability"] = trace.recovery_probability

    config = load_policy_config()
    decision = decide_action(row_dict, config)

    candidate_actions = []
    if trace.scoring_recommendation == "INTERVENE":
        single_row = bootstrap.dataset.iloc[[row_idx]]
        try:
            action_probs = predict_all_actions(bootstrap.model, single_row)
            for arm in ARM_ORDER:
                if arm == "CONTROL":
                    continue
                prob = float(action_probs.iloc[0][arm])
                erv = prob * trace.amount_inr - 10.0
                candidate_actions.append(
                    CandidateAction(
                        arm=arm,
                        probability=round(prob, 6),
                        expected_recovery_value_inr=round(erv, 2),
                    )
                )
        except Exception:
            pass

    policy_info = PolicyInfo(
        decision=decision.authorized_action,
        authorized_action=decision.authorized_action,
        reason=decision.reason,
        matched_rule_id=decision.matched_rule_id,
        matched_rule_name=decision.matched_rule_name,
        is_stop=decision.is_stop,
    )

    return AnalysisResponse(
        attempt_id=trace.attempt_id,
        amount_inr=trace.amount_inr,
        failure_category=trace.failure_category,
        recovery_probability=trace.recovery_probability,
        scoring_recommendation=trace.scoring_recommendation,
        expected_recovery_value_inr=trace.expected_recovery_value_inr,
        worth_intervening=trace.scoring_recommendation == "INTERVENE",
        candidate_actions=candidate_actions,
        policy=policy_info,
        audit_context={
            "evaluated_rules": [
                {"rule_id": rule_id, "matched": matched}
                for rule_id, matched in decision.evaluated_rules
            ],
            "model_contract": trace.model_contract,
            "authorization_reason": trace.authorization_reason,
        },
    )
