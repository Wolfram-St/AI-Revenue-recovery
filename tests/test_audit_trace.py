"""Tests for deterministic decision traces and audit records.

These tests pin the trace contract end to end: every field maps from real
``score_opportunity`` and ``decide_action`` outputs plus row metadata; the
scoring recommendation and the policy authorization are preserved as
separate concepts that may legitimately diverge; timestamps normalize once
to canonical ISO-8601 UTC strings derived only from row data; serialization
is byte-identical across independent builds; and field names mirror
``db/schema.sql`` so a later persistence layer maps one-to-one.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recovery.audit import (
    MODEL_CONTRACT,
    DecisionTrace,
    build_decision_trace,
    trace_to_dict,
    traces_to_json,
)
from recovery.policy import (
    CANONICAL_ACTIONS,
    PolicyDecision,
    decide_action,
    load_policy_config,
)
from recovery.scoring import (
    INTERVENE,
    NO_INTERVENTION,
    OpportunityScore,
    score_opportunity,
)

FIELD_ORDER = [
    "attempt_id",
    "payment_id",
    "customer_id",
    "event_timestamp",
    "amount_inr",
    "failure_category",
    "recovery_probability",
    "expected_recovery_value_inr",
    "scoring_recommendation",
    "authorized_action",
    "authorization_reason",
    "matched_rule_id",
    "matched_rule_name",
    "rule_priority",
    "is_stop",
    "evaluated_rules",
    "model_contract",
    "probability_is_action_conditional",
]

REQUIRED_ROW_KEYS = [
    "attempt_id",
    "payment_id",
    "customer_id",
    "event_timestamp",
    "amount_inr",
    "failure_category",
]

FORBIDDEN_POST_INTERVENTION_COLUMNS = {
    "recovered",
    "recovery_time_hours",
    "recovery_action",
    "action_outcome",
    "recovered_amount_inr",
}

ALLOWED_AUDIT_IMPORTS = {
    "__future__",
    "dataclasses",
    "datetime",
    "json",
    "math",
    "numbers",
    "pandas",
    "recovery",
    "typing",
}

RETRY_NOW_EVALUATED_RULES = (
    ("R001", False),
    ("R002", False),
    ("R003", False),
    ("R004", False),
    ("R005", False),
    ("R006", False),
    ("R007", True),
    ("R008", False),
)


def _row(**overrides):
    values = {
        "attempt_id": "att_000042",
        "payment_id": "pay_000017",
        "customer_id": "cus_000009",
        "event_timestamp": pd.Timestamp("2026-03-14T09:30:00Z"),
        "amount_inr": 1000.0,
        "failure_category": "temporary_decline",
    }
    values.update(overrides)
    return pd.Series(values)


def _score(**overrides):
    scored = score_opportunity(0.80, 1000.0, "temporary_decline")
    values = {
        "scoring_recommendation": scored.scoring_recommendation,
        "expected_recovery_value_inr": scored.expected_recovery_value_inr,
        "recovery_probability": scored.recovery_probability,
        "payment_amount_inr": scored.payment_amount_inr,
        "risk_penalty_inr": scored.risk_penalty_inr,
        "intervention_cost_inr": scored.intervention_cost_inr,
        "worth_intervening": scored.worth_intervening,
    }
    values.update(overrides)
    return OpportunityScore(**values)


def _retry_now_decision():
    return decide_action(
        {
            "customer_opted_out": False,
            "fraud_risk": False,
            "failure_category": "temporary_decline",
            "attempt_number": 1,
            "amount_inr": 1000.0,
            "recovery_probability": 0.80,
        },
        load_policy_config(),
    )


def _stop_decision():
    return decide_action(
        {
            "customer_opted_out": False,
            "fraud_risk": True,
            "failure_category": "temporary_decline",
            "attempt_number": 2,
            "amount_inr": 5000.0,
            "recovery_probability": 0.75,
        },
        load_policy_config(),
    )


def _residual_decision():
    return decide_action(
        {
            "customer_opted_out": False,
            "fraud_risk": False,
            "failure_category": "authentication_required",
            "attempt_number": 1,
            "amount_inr": 500.0,
            "recovery_probability": 0.45,
        },
        load_policy_config(),
    )


def _trace(row=None, score=None, decision=None):
    return build_decision_trace(
        row if row is not None else _row(),
        score if score is not None else _score(),
        decision if decision is not None else _retry_now_decision(),
    )


def _trace_kwargs(**overrides):
    values = {
        "attempt_id": "att_000042",
        "payment_id": "pay_000017",
        "customer_id": "cus_000009",
        "event_timestamp": "2026-03-14T09:30:00+00:00",
        "amount_inr": 1000.0,
        "failure_category": "temporary_decline",
        "recovery_probability": 0.80,
        "expected_recovery_value_inr": 790.0,
        "scoring_recommendation": INTERVENE,
        "authorized_action": "RETRY_NOW",
        "authorization_reason": (
            "Temporary failure with sufficiently high predicted recovery probability."
        ),
        "matched_rule_id": "R007",
        "matched_rule_name": "retry_now_eligible",
        "rule_priority": 50,
        "is_stop": False,
        "evaluated_rules": RETRY_NOW_EVALUATED_RULES,
        "model_contract": "P(recovered | context)",
        "probability_is_action_conditional": False,
    }
    values.update(overrides)
    return values


def test_complete_valid_trace_maps_every_field_from_real_modules():
    trace = _trace()
    assert trace.attempt_id == "att_000042"
    assert trace.payment_id == "pay_000017"
    assert trace.customer_id == "cus_000009"
    assert trace.event_timestamp == "2026-03-14T09:30:00+00:00"
    assert trace.amount_inr == 1000.0
    assert trace.failure_category == "temporary_decline"
    assert trace.recovery_probability == 0.80
    assert trace.expected_recovery_value_inr == 790.0
    assert trace.scoring_recommendation == INTERVENE
    assert trace.authorized_action == "RETRY_NOW"
    assert trace.authorization_reason == (
        "Temporary failure with sufficiently high predicted recovery probability."
    )
    assert trace.matched_rule_id == "R007"
    assert trace.matched_rule_name == "retry_now_eligible"
    assert trace.rule_priority == 50
    assert trace.is_stop is False
    assert trace.evaluated_rules == RETRY_NOW_EVALUATED_RULES
    assert all(isinstance(pair, tuple) for pair in trace.evaluated_rules)


@pytest.mark.parametrize("missing_key", REQUIRED_ROW_KEYS)
def test_each_missing_required_row_key_is_named_in_the_error(missing_key):
    row = _row()
    del row[missing_key]
    with pytest.raises(ValueError) as excinfo:
        build_decision_trace(row, _score(), _retry_now_decision())
    assert missing_key in str(excinfo.value)


def test_several_missing_row_keys_are_all_named_in_one_message():
    row = _row()
    del row["payment_id"]
    del row["customer_id"]
    del row["event_timestamp"]
    with pytest.raises(ValueError) as excinfo:
        build_decision_trace(row, _score(), _retry_now_decision())
    message = str(excinfo.value)
    assert "payment_id" in message
    assert "customer_id" in message
    assert "event_timestamp" in message


@pytest.mark.parametrize("bad_probability", [float("nan"), 1.5, True])
def test_invalid_recovery_probability_raises_value_error(bad_probability):
    with pytest.raises(ValueError, match="recovery_probability"):
        _trace(score=_score(recovery_probability=bad_probability))


@pytest.mark.parametrize("bad_erv", [float("nan"), float("inf"), "12"])
def test_invalid_expected_recovery_value_raises_value_error(bad_erv):
    with pytest.raises(ValueError, match="expected_recovery_value_inr"):
        _trace(score=_score(expected_recovery_value_inr=bad_erv))


@pytest.mark.parametrize("bad_amount", [-1.0, float("nan"), float("inf"), "50"])
def test_invalid_row_amount_raises_value_error(bad_amount):
    with pytest.raises(ValueError, match="amount_inr"):
        _trace(row=_row(amount_inr=bad_amount))


def test_amount_inr_is_rounded_to_two_decimals():
    trace = _trace(row=_row(amount_inr=1234.567))
    assert trace.amount_inr == 1234.57


def test_noncanonical_authorized_action_is_rejected_citing_vocabulary():
    bypassed = PolicyDecision(
        authorized_action="AUTO_RETRY",
        matched_rule_id=None,
        matched_rule_name=None,
        priority=None,
        reason="Constructed directly, bypassing the policy engine.",
        is_stop=False,
        evaluated_rules=(("R001", False),),
    )
    with pytest.raises(ValueError) as excinfo:
        build_decision_trace(_row(), _score(), bypassed)
    message = str(excinfo.value)
    assert "AUTO_RETRY" in message
    assert "canonical vocabulary" in message
    for action in sorted(CANONICAL_ACTIONS):
        assert action in message


def test_stop_authorization_path_is_preserved_from_real_policy_decision():
    score = score_opportunity(0.75, 5000.0, "temporary_decline")
    trace = _trace(row=_row(amount_inr=5000.0), score=score, decision=_stop_decision())
    assert trace.authorized_action == "STOP"
    assert trace.is_stop is True
    assert trace.matched_rule_id == "R002"
    assert trace.matched_rule_name == "fraud_risk"
    assert trace.rule_priority == 95
    assert trace.evaluated_rules == (
        ("R001", False),
        ("R002", True),
        ("R003", False),
        ("R004", False),
        ("R005", False),
        ("R006", False),
        ("R007", True),
        ("R008", False),
    )


def test_intervene_recommendation_with_stop_authorization_both_survive_verbatim():
    score = score_opportunity(0.75, 5000.0, "temporary_decline")
    assert score.scoring_recommendation == INTERVENE
    trace = _trace(row=_row(amount_inr=5000.0), score=score, decision=_stop_decision())
    assert trace.scoring_recommendation == "INTERVENE"
    assert trace.authorized_action == "STOP"
    assert trace.is_stop is True


def test_no_intervention_with_residual_retry_later_both_survive_verbatim():
    score = score_opportunity(0.45, 20.0, "authentication_required")
    assert score.scoring_recommendation == NO_INTERVENTION
    decision = _residual_decision()
    assert decision.authorized_action == "RETRY_LATER"
    assert decision.matched_rule_id is None
    trace = _trace(row=_row(amount_inr=20.0), score=score, decision=decision)
    assert trace.scoring_recommendation == "NO_INTERVENTION"
    assert trace.authorized_action == "RETRY_LATER"
    assert trace.matched_rule_id is None
    assert trace.matched_rule_name is None
    assert trace.rule_priority is None
    assert trace.is_stop is False


def test_serialization_is_byte_identical_across_independent_builds():
    first_json = traces_to_json([_trace()])
    second_json = traces_to_json([_trace()])
    assert first_json == second_json
    first_dict = trace_to_dict(_trace())
    second_dict = trace_to_dict(_trace())
    assert first_dict == second_dict
    assert list(first_dict.keys()) == FIELD_ORDER


def test_traces_to_json_round_trips_a_sequence_of_traces():
    stop_trace = _trace(decision=_stop_decision())
    payload = traces_to_json([_trace(), stop_trace])
    parsed = json.loads(payload)
    assert parsed == [trace_to_dict(_trace()), trace_to_dict(stop_trace)]
    assert traces_to_json([_trace(), stop_trace]) == payload


def test_trace_to_dict_preserves_exact_field_order_and_tuple_conversion():
    payload = trace_to_dict(_trace())
    assert list(payload.keys()) == FIELD_ORDER
    assert payload["evaluated_rules"] == [list(pair) for pair in RETRY_NOW_EVALUATED_RULES]
    assert isinstance(payload["is_stop"], bool)
    assert isinstance(payload["probability_is_action_conditional"], bool)


def test_model_contract_marker_is_the_exact_literal():
    assert MODEL_CONTRACT == "P(recovered | context)"
    assert _trace().model_contract == "P(recovered | context)"


def test_probability_is_action_conditional_is_exactly_false_and_serialized():
    trace = _trace()
    assert trace.probability_is_action_conditional is False
    payload = trace_to_dict(trace)
    assert "probability_is_action_conditional" in payload
    assert payload["probability_is_action_conditional"] is False


def test_directly_constructed_false_claim_is_rejected():
    with pytest.raises(ValueError, match="probability_is_action_conditional"):
        DecisionTrace(
            **_trace_kwargs(probability_is_action_conditional=True),
        )


def test_event_timestamp_normalization_pins_the_exact_iso_output():
    trace = _trace(row=_row(event_timestamp=pd.Timestamp("2026-01-01T00:15:00Z")))
    assert trace.event_timestamp == "2026-01-01T00:15:00+00:00"


def test_event_timestamp_accepts_datetime_string_and_offsets_identically():
    from_timestamp = _trace(
        row=_row(event_timestamp=pd.Timestamp("2026-01-01T00:15:00Z"))
    ).event_timestamp
    from_datetime = _trace(
        row=_row(
            event_timestamp=datetime.datetime(
                2026, 1, 1, 0, 15, 0, tzinfo=datetime.timezone.utc
            )
        )
    ).event_timestamp
    from_string = _trace(
        row=_row(event_timestamp="2026-01-01T00:15:00Z")
    ).event_timestamp
    from_offset = _trace(
        row=_row(event_timestamp=pd.Timestamp("2026-01-01T05:45:00+05:30"))
    ).event_timestamp
    assert from_timestamp == "2026-01-01T00:15:00+00:00"
    assert from_datetime == from_timestamp
    assert from_string == from_timestamp
    assert from_offset == "2026-01-01T00:15:00+00:00"
    assert _trace(
        row=_row(event_timestamp=pd.Timestamp("2026-01-01T00:15:00Z"))
    ).event_timestamp == from_timestamp


def test_naive_event_timestamp_is_interpreted_as_utc_deterministically():
    first = _trace(row=_row(event_timestamp="2026-01-01T00:15:00")).event_timestamp
    second = _trace(row=_row(event_timestamp=pd.Timestamp("2026-01-01T00:15:00"))).event_timestamp
    assert first == "2026-01-01T00:15:00+00:00"
    assert first == second


@pytest.mark.parametrize(
    "bad_timestamp",
    [
        "not-a-timestamp",
        "2026-13-45",
        42,
        None,
        float("nan"),
        pd.NaT,
        np.datetime64("NaT", "ns"),
    ],
)
def test_unparsable_event_timestamp_raises_value_error(bad_timestamp):
    with pytest.raises(ValueError, match="event_timestamp"):
        _trace(row=_row(event_timestamp=bad_timestamp))


@pytest.mark.parametrize("bad_entry", [42, None, (42,)])
def test_malformed_evaluated_rules_entries_raise_named_value_error(bad_entry):
    with pytest.raises(ValueError, match="evaluated_rules entries"):
        DecisionTrace(**_trace_kwargs(evaluated_rules=[bad_entry]))


def test_stop_action_with_false_is_stop_is_rejected():
    with pytest.raises(ValueError, match="is_stop"):
        DecisionTrace(
            **_trace_kwargs(
                authorized_action="STOP",
                matched_rule_id="R002",
                matched_rule_name="fraud_risk",
                rule_priority=95,
                is_stop=False,
            )
        )


def test_non_stop_action_with_true_is_stop_is_rejected():
    with pytest.raises(ValueError, match="is_stop"):
        DecisionTrace(**_trace_kwargs(is_stop=True))


def test_text_fields_store_the_stripped_validated_values():
    trace = _trace(
        row=_row(
            attempt_id="  att_1  ",
            payment_id="\tpay_1\n",
            customer_id=" cus_1 ",
            failure_category=" temporary_decline ",
        )
    )
    assert trace.attempt_id == "att_1"
    assert trace.payment_id == "pay_1"
    assert trace.customer_id == "cus_1"
    assert trace.failure_category == "temporary_decline"
    annotated = DecisionTrace(
        **_trace_kwargs(
            authorization_reason="  reason text  ",
            matched_rule_id=" R007 ",
            matched_rule_name=" retry_now_eligible ",
        )
    )
    assert annotated.authorization_reason == "reason text"
    assert annotated.matched_rule_id == "R007"
    assert annotated.matched_rule_name == "retry_now_eligible"


def test_whitespace_only_text_fields_are_still_rejected():
    with pytest.raises(ValueError, match="attempt_id"):
        _trace(row=_row(attempt_id="   "))
    with pytest.raises(ValueError, match="authorization_reason"):
        DecisionTrace(**_trace_kwargs(authorization_reason=" \t "))


_SUBPROCESS_TRACE_SCRIPT = """
import hashlib
import pandas as pd

from recovery.audit import build_decision_trace, traces_to_json
from recovery.policy import decide_action, load_policy_config
from recovery.scoring import score_opportunity

policy = load_policy_config()
row = pd.Series({
    "attempt_id": "att_000042",
    "payment_id": "pay_000017",
    "customer_id": "cus_000009",
    "event_timestamp": pd.Timestamp("2026-03-14T09:30:00Z"),
    "amount_inr": 1000.0,
    "failure_category": "temporary_decline",
})
score = score_opportunity(0.80, 1000.0, "temporary_decline")
decision = decide_action(
    {
        "customer_opted_out": False,
        "fraud_risk": False,
        "failure_category": "temporary_decline",
        "attempt_number": 1,
        "amount_inr": 1000.0,
        "recovery_probability": 0.80,
    },
    policy,
)
first = build_decision_trace(row, score, decision)
second = build_decision_trace(row, score, decision)
payload = traces_to_json([first, second])
print(hashlib.sha256(payload.encode("utf-8")).hexdigest())
"""


def test_serialization_digest_matches_across_separate_subprocess_runs():
    repo_root = Path(__file__).resolve().parents[1]
    digests = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_TRACE_SCRIPT],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=120,
            check=True,
        )
        digests.append(completed.stdout.strip())
    assert len(digests) == 2
    assert digests[0] == digests[1]
    assert len(digests[0]) == 64
    int(digests[0], 16)


def test_trace_is_frozen_against_mutation():
    trace = _trace()
    with pytest.raises(dataclasses.FrozenInstanceError):
        trace.attempt_id = "att_mutated"


def test_evaluated_rules_stored_as_tuples_even_when_constructed_with_lists():
    trace = DecisionTrace(
        **_trace_kwargs(evaluated_rules=[["R001", False], ["R007", True]])
    )
    assert isinstance(trace.evaluated_rules, tuple)
    assert all(isinstance(pair, tuple) for pair in trace.evaluated_rules)
    assert trace.evaluated_rules == (("R001", False), ("R007", True))
    assert trace_to_dict(trace)["evaluated_rules"] == [["R001", False], ["R007", True]]


def test_dataclass_fields_are_exactly_eighteen_in_fixed_order():
    names = [field.name for field in dataclasses.fields(DecisionTrace)]
    assert names == FIELD_ORDER


def test_trace_field_names_align_with_db_schema_columns():
    field_names = {field.name for field in dataclasses.fields(DecisionTrace)}
    assert "expected_recovery_value_inr" in field_names
    assert "authorization_reason" in field_names
    assert "authorized_action" in field_names
    assert "recovery_probability" in field_names
    assert "failure_category" in field_names


def test_forbidden_post_intervention_columns_never_appear_in_a_trace():
    payload = trace_to_dict(_trace())
    assert not set(payload) & FORBIDDEN_POST_INTERVENTION_COLUMNS


def test_rows_accepted_as_series_or_plain_dict_build_equal_traces():
    from_series = _trace(row=_row())
    from_dict = _trace(row=dict(_row()))
    assert from_series == from_dict


def test_module_docstring_states_the_divergence_honesty_boundary():
    import recovery.audit as audit_module

    assert audit_module.__doc__ is not None
    assert "separate concepts" in audit_module.__doc__
    assert "diverge" in audit_module.__doc__
    assert "P(recovered | context)" in audit_module.__doc__


def test_module_never_reads_wall_clock_randomness_or_writes_anywhere():
    import recovery.audit as audit_module

    source = inspect.getsource(audit_module)
    assert "now(" not in source
    assert "utcnow" not in source
    assert ".today(" not in source
    assert "time.time" not in source
    assert "uuid" not in source
    assert "random" not in source
    assert "sklearn" not in source
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= ALLOWED_AUDIT_IMPORTS
