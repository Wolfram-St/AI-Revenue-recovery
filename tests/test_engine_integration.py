"""Integration tests for the composed recovery decision engine.

These tests pin the whole pipeline end to end on real data slices and the
real trained baseline: probabilities from ``ml.train`` feed
``score_opportunities``, only rows whose scorer marked ``worth_intervening``
reach ``decide_action``, every row yields exactly one frozen
``DecisionTrace`` in dataframe order, and the summary arithmetic closes over
the canonical action vocabulary. The candidate rule is enforced mechanically:
no-candidate rows are terminal STOP records built without consulting the
policy layer, proven here with a spying double that fails the test if the
policy engine ever sees such a row.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re

import pandas as pd
import pytest

import recovery.engine as engine_module
from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.train import predict_recovery_probability, train_baseline
from recovery.audit import trace_to_dict
from recovery.engine import EngineResult, run_recovery_engine
from recovery.policy import CANONICAL_ACTIONS, RESIDUAL_DEFAULT_ACTION
from recovery.scoring import (
    INTERVENE,
    NO_INTERVENTION,
    RETRY_INTERVENTION_COST_INR,
)

N_ROWS = 600
DATA_SEED = 42
MODEL_SEED = 42

SUMMARY_KEYS = [
    "case_count",
    "candidate_count",
    "action_counts",
    "stop_count",
    "human_review_count",
    "total_candidate_erv_inr",
    "noop_count",
]

TRACE_FIELDS = [
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

ALLOWED_ENGINE_IMPORT_ROOTS = {
    "__future__",
    "dataclasses",
    "typing",
    "numpy",
    "pandas",
    "recovery",
    "ml",
    "data",
}

FORBIDDEN_ENGINE_TOKENS = [
    "datetime.now",
    "random",
    "requests",
    "sqlalchemy",
    "langgraph",
]

ALL_RULES_UNMATCHED = tuple((f"R{i:03d}", False) for i in range(1, 9))

CONFLICT_CONFIG_TEMPLATE = """version: "9.9"
project: "RecoverAI"
principle: "AI recommends; policy engine authorizes."
rule_resolution:
  matching_rule: highest_priority_wins
  stop_precedence: true
  deterministic: true

action_vocabulary:
  - RETRY_NOW
  - RETRY_LATER
  - REQUEST_UPDATE
  - HUMAN_REVIEW
  - STOP

rules:
  - id: P099
    name: dominant_positive
    priority: 99
    condition: "attempt_number >= 1"
    action: RETRY_NOW
    reason: "Synthetic positive rule with the numerically dominant rank."

  - id: S010
    name: rescued_stop
    priority: 10
    condition: "attempt_number >= 1"
    action: STOP
    reason: "Synthetic stop rule that dominance must rescue."
"""


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_rows=N_ROWS, seed=DATA_SEED)


@pytest.fixture(scope="module")
def split_frames(dataset):
    return chronological_split(dataset)


@pytest.fixture(scope="module")
def model(split_frames):
    trained, _ = train_baseline(split_frames[0], split_frames[1], seed=MODEL_SEED)
    return trained


@pytest.fixture(scope="module")
def validation_run(model, split_frames):
    return run_recovery_engine(split_frames[1], model)


@pytest.fixture(scope="module")
def test_run(model, split_frames):
    return run_recovery_engine(split_frames[2], model)


def _is_noop_trace(trace):
    return (
        trace.authorized_action == "STOP"
        and trace.is_stop is True
        and trace.matched_rule_id is None
        and trace.evaluated_rules == ()
    )


def _candidate_traces(run):
    return [trace for trace in run.traces if not _is_noop_trace(trace)]


def _divergent_traces(*runs):
    return [
        trace
        for run in runs
        for trace in run.traces
        if trace.scoring_recommendation == INTERVENE
        and trace.authorized_action == "STOP"
    ]


def _crafted_rows(source, overrides_per_row):
    rows = source.iloc[: len(overrides_per_row)].copy().reset_index(drop=True)
    for position, overrides in enumerate(overrides_per_row):
        for column, value in overrides.items():
            rows.loc[position, column] = value
    return rows


def test_high_probability_eligible_payment_gets_consistent_positive_authorization(
    validation_run,
):
    consistent = [
        trace
        for trace in _candidate_traces(validation_run)
        if trace.scoring_recommendation == INTERVENE
        and trace.authorized_action in {"RETRY_NOW", "REQUEST_UPDATE"}
        and not trace.is_stop
    ]
    assert consistent, "slice must contain at least one positively authorized candidate"
    for trace in consistent:
        assert trace.expected_recovery_value_inr > 0.0
        assert trace.model_contract == "P(recovered | context)"
        assert trace.probability_is_action_conditional is False


def test_fraud_candidate_is_recommended_intervene_but_stopped_by_policy(
    model, split_frames
):
    frame = _crafted_rows(
        split_frames[1],
        [
            {
                "fraud_risk": True,
                "customer_opted_out": False,
                "attempt_number": 1,
                "amount_inr": 8000.0,
                "failure_category": "temporary_decline",
                "failure_code": "T001",
                "issuer_response": "issuer_temporary_decline",
            }
        ],
    )
    result = run_recovery_engine(frame, model)
    assert result.summary["case_count"] == 1
    trace = result.traces[0]
    assert trace.scoring_recommendation == INTERVENE
    assert trace.authorized_action == "STOP"
    assert trace.is_stop is True
    assert trace.matched_rule_id == "R002"
    assert trace.matched_rule_name == "fraud_risk"
    assert trace.rule_priority == 95
    assert trace.authorization_reason == (
        "Fraud-risk cases are excluded from automated recovery."
    )
    assert trace.scoring_recommendation != trace.authorized_action


def test_policy_not_consulted_for_no_candidate_rows_but_is_for_candidates(
    monkeypatch, model, split_frames
):
    frame = _crafted_rows(
        split_frames[1],
        [
            {
                "customer_opted_out": False,
                "fraud_risk": False,
                "attempt_number": 1,
                "amount_inr": 5000.0,
                "failure_category": "temporary_decline",
                "failure_code": "T001",
                "issuer_response": "issuer_temporary_decline",
            },
            {
                "amount_inr": 5.0,
            },
        ],
    )
    calls = []
    real_decide_action = engine_module.decide_action

    def spying_decide_action(context, config):
        calls.append(dict(context))
        return real_decide_action(context, config)

    monkeypatch.setattr(engine_module, "decide_action", spying_decide_action)

    result = run_recovery_engine(frame, model)

    assert len(calls) == 1, "policy layer must see exactly the candidate row"
    assert calls[0]["amount_inr"] == 5000.0
    assert "recovery_probability" in calls[0]
    assert "customer_opted_out" in calls[0]
    assert "fraud_risk" in calls[0]
    assert "failure_category" in calls[0]
    assert "attempt_number" in calls[0]

    candidate_trace, noop_trace = result.traces
    assert candidate_trace.amount_inr == 5000.0
    assert not _is_noop_trace(candidate_trace)

    assert noop_trace.scoring_recommendation == NO_INTERVENTION
    assert noop_trace.authorized_action == "STOP"
    assert noop_trace.is_stop is True
    assert noop_trace.matched_rule_id is None
    assert noop_trace.matched_rule_name is None
    assert noop_trace.rule_priority is None
    assert noop_trace.evaluated_rules == ()
    assert noop_trace.expected_recovery_value_inr < 0.0
    assert "not positive" in noop_trace.authorization_reason
    assert (
        f"{noop_trace.expected_recovery_value_inr:.2f}"
        in noop_trace.authorization_reason
    )
    assert result.summary["candidate_count"] == 1
    assert result.summary["noop_count"] == 1


def test_fraud_cases_stop_dominates_with_no_positive_action_anywhere(
    model, split_frames
):
    frame = _crafted_rows(
        split_frames[1],
        [
            {
                "fraud_risk": True,
                "customer_opted_out": False,
                "attempt_number": number,
                "amount_inr": amount,
                "failure_category": "temporary_decline",
                "failure_code": "T002",
                "issuer_response": "network_timeout",
            }
            for number, amount in ((1, 9000.0), (2, 12000.0), (3, 7000.0))
        ],
    )
    result = run_recovery_engine(frame, model)
    assert len(result.traces) == 3
    for trace in result.traces:
        assert trace.scoring_recommendation == INTERVENE
        assert trace.authorized_action == "STOP"
        assert trace.matched_rule_id == "R002"
        assert trace.rule_priority == 95
        assert trace.is_stop is True
        payload = trace_to_dict(trace)
        assert payload["authorized_action"] == "STOP"
        assert payload["scoring_recommendation"] == INTERVENE
    assert result.summary["stop_count"] == 3
    assert result.summary["action_counts"]["STOP"] == 3
    assert result.summary["action_counts"]["RETRY_NOW"] == 0
    assert result.summary["human_review_count"] == 0


def test_customer_opt_out_case_is_stopped_by_shipped_opt_out_rule(model, split_frames):
    frame = _crafted_rows(
        split_frames[1],
        [
            {
                "customer_opted_out": True,
                "fraud_risk": False,
                "attempt_number": 1,
                "amount_inr": 6000.0,
                "failure_category": "temporary_decline",
                "failure_code": "T001",
                "issuer_response": "issuer_temporary_decline",
            }
        ],
    )
    result = run_recovery_engine(frame, model)
    trace = result.traces[0]
    assert trace.authorized_action == "STOP"
    assert trace.matched_rule_id == "R001"
    assert trace.matched_rule_name == "customer_opt_out"
    assert trace.rule_priority == 100
    assert trace.is_stop is True
    assert trace.authorization_reason == (
        "Customer has opted out of recovery communications."
    )
    assert trace.scoring_recommendation == INTERVENE


def test_unknown_failure_category_pays_risk_penalty_versus_clean_twin_same_run(
    model, split_frames
):
    shared = {
        "customer_opted_out": False,
        "fraud_risk": False,
        "attempt_number": 1,
        "amount_inr": 20000.0,
    }
    frame = _crafted_rows(
        split_frames[1],
        [
            {
                **shared,
                "failure_category": "unknown",
                "failure_code": "U001",
                "issuer_response": "unknown_issuer_error",
            },
            {
                **shared,
                "failure_category": "temporary_decline",
                "failure_code": "T001",
                "issuer_response": "issuer_temporary_decline",
            },
        ],
    )
    result = run_recovery_engine(frame, model)
    unknown_trace, clean_trace = result.traces
    assert unknown_trace.failure_category == "unknown"
    assert clean_trace.failure_category == "temporary_decline"
    for trace in (unknown_trace, clean_trace):
        implied_penalty = (
            trace.recovery_probability * 20000.0
            - RETRY_INTERVENTION_COST_INR
            - trace.expected_recovery_value_inr
        )
        if trace.failure_category == "unknown":
            assert implied_penalty > 0.0
        else:
            assert abs(implied_penalty) <= 0.01
    assert (
        unknown_trace.expected_recovery_value_inr
        < clean_trace.expected_recovery_value_inr
    )


def test_candidate_matching_no_rule_gets_residual_retry_later(model, split_frames):
    search = _crafted_rows(
        split_frames[1],
        [
            {
                "customer_opted_out": False,
                "fraud_risk": False,
                "attempt_number": 1,
                "amount_inr": 4000.0,
                "failure_category": "authentication_required",
                "failure_code": "A001",
                "issuer_response": "authentication_required",
            }
        ]
        * 30,
    )
    probabilities = predict_recovery_probability(model, search)
    eligible = [
        position
        for position, probability in enumerate(probabilities)
        if 0.20 <= float(probability) < 0.70
    ]
    assert eligible, "seeded slice must contain a mid-probability authentication case"
    frame = search.iloc[[eligible[0]]]

    result = run_recovery_engine(frame, model)

    trace = result.traces[0]
    assert trace.scoring_recommendation == INTERVENE
    assert trace.authorized_action == RESIDUAL_DEFAULT_ACTION == "RETRY_LATER"
    assert trace.matched_rule_id is None
    assert trace.matched_rule_name is None
    assert trace.rule_priority is None
    assert trace.is_stop is False
    assert trace.evaluated_rules == ALL_RULES_UNMATCHED
    assert not _is_noop_trace(trace)
    assert result.summary["candidate_count"] == 1
    assert result.summary["noop_count"] == 0
    assert result.summary["total_candidate_erv_inr"] == round(
        trace.expected_recovery_value_inr, 2
    )


def test_conflicting_rules_resolve_to_the_policy_engine_precedence_winner(
    tmp_path, model, split_frames
):
    config_path = tmp_path / "conflict_rules.yaml"
    config_path.write_text(CONFLICT_CONFIG_TEMPLATE, encoding="utf-8")
    from recovery.policy import load_policy_config

    conflicting = load_policy_config(str(config_path))
    frame = _crafted_rows(
        split_frames[1],
        [
            {
                "customer_opted_out": False,
                "fraud_risk": False,
                "attempt_number": 1,
                "amount_inr": amount,
                "failure_category": "temporary_decline",
                "failure_code": "T001",
                "issuer_response": "issuer_temporary_decline",
            }
            for amount in (5000.0, 8000.0)
        ],
    )

    result = run_recovery_engine(frame, model, policy_config=conflicting)

    assert len(result.traces) == 2
    for trace in result.traces:
        assert trace.authorized_action == "STOP"
        assert trace.matched_rule_id == "S010"
        assert trace.matched_rule_name == "rescued_stop"
        assert trace.rule_priority == 10
        assert trace.is_stop is True
        assert trace.evaluated_rules == (("P099", True), ("S010", True))
    assert result.summary["stop_count"] == 2
    assert result.summary["action_counts"]["RETRY_NOW"] == 0


def test_recommendation_and_authorization_survive_verbatim_across_whole_runs(
    validation_run, test_run
):
    for run in (validation_run, test_run):
        for trace in run.traces:
            payload = trace_to_dict(trace)
            assert list(payload.keys()) == TRACE_FIELDS
            assert trace.scoring_recommendation in (INTERVENE, NO_INTERVENTION)
            assert trace.authorized_action in sorted(CANONICAL_ACTIONS)
            if trace.scoring_recommendation == NO_INTERVENTION:
                assert trace.authorized_action == "STOP"
                assert trace.is_stop is True
    divergent = _divergent_traces(validation_run, test_run)
    assert divergent, "expected policy-overrides-AI divergence in the seeded slices"
    for trace in divergent:
        assert trace.scoring_recommendation == INTERVENE
        assert trace.authorized_action == "STOP"


def test_end_to_end_runs_are_identical_for_fixed_inputs(model, split_frames):
    frame = split_frames[1]
    first = run_recovery_engine(frame, model)
    second = run_recovery_engine(frame, model)
    assert first.summary == second.summary
    first_payload = json.dumps([trace_to_dict(t) for t in first.traces])
    second_payload = json.dumps([trace_to_dict(t) for t in second.traces])
    assert first_payload == second_payload


@pytest.mark.parametrize("slice_position", [1, 2])
def test_summary_reports_exact_keys_and_arithmetic_invariants(
    split_frames, validation_run, test_run, slice_position
):
    run = validation_run if slice_position == 1 else test_run
    summary = run.summary
    assert list(summary.keys()) == SUMMARY_KEYS
    frame = split_frames[slice_position]
    assert summary["case_count"] == len(frame) == len(run.traces)
    assert sum(summary["action_counts"].values()) == summary["case_count"]
    assert summary["candidate_count"] + summary["noop_count"] == summary["case_count"]
    assert list(summary["action_counts"].keys()) == sorted(CANONICAL_ACTIONS)
    assert summary["stop_count"] == summary["action_counts"]["STOP"]
    assert summary["human_review_count"] == summary["action_counts"]["HUMAN_REVIEW"]
    candidates = _candidate_traces(run)
    assert len(candidates) == summary["candidate_count"]
    recomputed_total = round(
        sum(trace.expected_recovery_value_inr for trace in candidates), 2
    )
    assert summary["total_candidate_erv_inr"] == recomputed_total
    observed_counts = {}
    for trace in run.traces:
        observed_counts[trace.authorized_action] = (
            observed_counts.get(trace.authorized_action, 0) + 1
        )
    expected_counts = {action: 0 for action in sorted(CANONICAL_ACTIONS)}
    expected_counts.update(observed_counts)
    assert summary["action_counts"] == expected_counts


def test_traces_align_one_to_one_with_dataframe_row_order(model, split_frames):
    frame = split_frames[1]
    result = run_recovery_engine(frame, model)
    assert len(result.traces) == len(frame)
    assert [trace.attempt_id for trace in result.traces] == list(frame["attempt_id"])
    assert [trace.payment_id for trace in result.traces] == list(frame["payment_id"])


def test_every_trace_json_round_trips(validation_run, test_run):
    for run in (validation_run, test_run):
        for trace in run.traces:
            payload = trace_to_dict(trace)
            assert json.loads(json.dumps(payload)) == payload


def test_empty_dataframe_returns_valid_zeroed_result(model, split_frames):
    for empty_frame in (pd.DataFrame(), split_frames[1].iloc[[]]):
        result = run_recovery_engine(empty_frame, model)
        assert isinstance(result, EngineResult)
        assert result.traces == ()
        assert list(result.summary.keys()) == SUMMARY_KEYS
        assert result.summary["case_count"] == 0
        assert result.summary["candidate_count"] == 0
        assert result.summary["noop_count"] == 0
        assert result.summary["stop_count"] == 0
        assert result.summary["human_review_count"] == 0
        assert result.summary["total_candidate_erv_inr"] == 0.0
        assert result.summary["action_counts"] == {
            action: 0 for action in sorted(CANONICAL_ACTIONS)
        }


def test_engine_result_is_frozen(model, split_frames):
    result = run_recovery_engine(split_frames[1].iloc[[]], model)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.traces = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.summary = {}


def test_engine_docstring_states_the_full_boundary():
    assert engine_module.__doc__ is not None
    docstring = engine_module.__doc__
    for phrase in (
        "trains nothing",
        "calls no external service",
        "writes no database",
        "executes no payment",
        "calls no LLM",
        "uses no LangGraph",
        "exposes no HTTP endpoint",
        "separate fields",
        "FOR SUBMITTED CANDIDATES",
        "worth_intervening",
        "terminal no-op",
        'authorized_action="STOP"',
        "matched_rule_id=None",
        "evaluated_rules=()",
        "not positive",
    ):
        assert phrase in docstring, f"engine docstring must state: {phrase}"


def test_engine_imports_only_allowed_module_roots():
    source = inspect.getsource(engine_module)
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= ALLOWED_ENGINE_IMPORT_ROOTS


def test_engine_source_contains_no_forbidden_runtime_tokens():
    source = inspect.getsource(engine_module)
    for token in FORBIDDEN_ENGINE_TOKENS:
        assert token not in source, f"engine source must not contain {token!r}"


def test_engine_contains_no_duplicated_scoring_or_precedence_logic():
    source = inspect.getsource(engine_module)
    assert "risk_penalty" not in source
    assert "* amount" not in source
    comparison = r"(?:==|!=|>=|<=|>|<)"
    assert not re.search(rf"\bpriority\b\s*{comparison}", source)
    assert not re.search(rf"{comparison}\s*\bpriority\b", source)
