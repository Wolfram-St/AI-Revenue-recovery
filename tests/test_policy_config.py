"""Tests for policy configuration loading and safe condition parsing.

Security regression cases assert that hostile condition strings raise
ValueError from ``parse_condition`` -- they must never parse successfully and
never merely evaluate to False.
"""

from __future__ import annotations

import ast

import pytest

from recovery.policy import (
    CANONICAL_ACTIONS,
    PolicyRule,
    evaluate_condition,
    load_policy_config,
    parse_condition,
)

CONFIG_PATH = "config/business_rules.yaml"

REJECTED_CONDITIONS = [
    '__import__("os")',
    '__import__("os").system("echo hi")',
    "some_function(amount_inr)",
    "foo.__class__",
    "amount_inr.__class__.__bases__",
    "(lambda: 1)()",
    "[x for x in [1]]",
    "amount_inr if fraud_risk else 0",
    '{"a": 1}',
    "amount_inr * 1000000",
    "-amount_inr > 0",
    "1 < amount_inr < 50000",
    "fraud_risk == true or attempt_number >= 4",
    "attempt_number >= 4 and (amount_inr == 1 or fraud_risk == true)",
    "recovery_cases[0] == 1",
    "(x := 1) == 1",
    "",
    "   ",
    "x" * 300,
]

BASE_CONFIG = """version: "9.9"
project: "RecoverAI"
principle: "AI recommends; policy engine authorizes."
rule_resolution:
  matching_rule: highest_priority_wins
  stop_precedence: false
  deterministic: true

action_vocabulary:
  - RETRY_NOW
  - RETRY_LATER
  - REQUEST_UPDATE
  - HUMAN_REVIEW
  - STOP

rules:
  - id: X001
    name: probe_rule
    priority: 10
    condition: "attempt_number >= 4"
    action: STOP
    reason: "Probe rule for config mutation tests."
"""

DUPLICATE_ID_CONFIG = BASE_CONFIG + (
    "  - id: X001\n"
    "    name: probe_rule_again\n"
    "    priority: 20\n"
    '    condition: "fraud_risk == true"\n'
    "    action: STOP\n"
    '    reason: "Second rule reusing the same id."\n'
)


def _rule_by_id(policy, rule_id):
    return {rule.id: rule for rule in policy.rules}[rule_id]


def _write_config(tmp_path, text):
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_canonical_action_vocabulary_is_exposed():
    assert CANONICAL_ACTIONS == frozenset(
        {"RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW", "STOP"}
    )


def test_loads_shipped_business_rules():
    policy = load_policy_config(CONFIG_PATH)
    assert [rule.id for rule in policy.rules] == [f"R{i:03d}" for i in range(1, 9)]
    assert policy.version == "1.1"
    assert policy.stop_precedence is True


def test_loaded_config_exposes_canonical_actions():
    policy = load_policy_config(CONFIG_PATH)
    assert frozenset(policy.canonical_actions) == CANONICAL_ACTIONS


def test_every_shipped_condition_round_trips_through_parser():
    policy = load_policy_config(CONFIG_PATH)
    for rule in policy.rules:
        reparsed = parse_condition(rule.condition_text)
        assert ast.dump(reparsed) == ast.dump(rule.condition_ast)


def test_r004_retry_limit_truth_table():
    rule = _rule_by_id(load_policy_config(CONFIG_PATH), "R004")
    assert evaluate_condition(rule.condition_ast, {"attempt_number": 4}) is True
    assert evaluate_condition(rule.condition_ast, {"attempt_number": 3}) is False


def test_r005_payment_method_issue_truth_table():
    rule = _rule_by_id(load_policy_config(CONFIG_PATH), "R005")
    matched = {"failure_category": "payment_method_issue"}
    other = {"failure_category": "temporary_decline"}
    assert evaluate_condition(rule.condition_ast, matched) is True
    assert evaluate_condition(rule.condition_ast, other) is False


def test_r003_hard_decline_truth_table():
    rule = _rule_by_id(load_policy_config(CONFIG_PATH), "R003")
    matched = {"failure_category": "hard_decline"}
    other = {"failure_category": "network_timeout"}
    assert evaluate_condition(rule.condition_ast, matched) is True
    assert evaluate_condition(rule.condition_ast, other) is False


def test_r001_opted_out_truth_table():
    rule = _rule_by_id(load_policy_config(CONFIG_PATH), "R001")
    assert evaluate_condition(rule.condition_ast, {"customer_opted_out": True}) is True
    assert evaluate_condition(rule.condition_ast, {"customer_opted_out": False}) is False


def test_r002_fraud_risk_truth_table():
    rule = _rule_by_id(load_policy_config(CONFIG_PATH), "R002")
    assert evaluate_condition(rule.condition_ast, {"fraud_risk": True}) is True
    assert evaluate_condition(rule.condition_ast, {"fraud_risk": False}) is False


def test_r007_retry_now_eligible_truth_table():
    rule = _rule_by_id(load_policy_config(CONFIG_PATH), "R007")
    eligible = {
        "failure_category": "temporary_decline",
        "recovery_probability": 0.75,
        "attempt_number": 2,
    }
    below_threshold = {
        "failure_category": "temporary_decline",
        "recovery_probability": 0.65,
        "attempt_number": 2,
    }
    assert evaluate_condition(rule.condition_ast, eligible) is True
    assert evaluate_condition(rule.condition_ast, below_threshold) is False


def test_lowercase_boolean_literals_are_reserved_words():
    opted_out = parse_condition("customer_opted_out == true")
    not_fraud = parse_condition("fraud_risk == false")
    assert evaluate_condition(opted_out, {"customer_opted_out": True}) is True
    assert evaluate_condition(not_fraud, {"fraud_risk": False}) is True
    assert evaluate_condition(not_fraud, {"fraud_risk": True}) is False


@pytest.mark.parametrize("condition_text", REJECTED_CONDITIONS)
def test_hostile_or_invalid_conditions_raise_valueerror(condition_text):
    with pytest.raises(ValueError):
        parse_condition(condition_text)


def test_unknown_operator_is_rejected_not_executed():
    with pytest.raises(ValueError):
        parse_condition("attempt_number =~ '4'")


def test_python_style_boolean_constant_is_rejected():
    with pytest.raises(ValueError):
        parse_condition("customer_opted_out == True")


def test_equality_threshold_is_exact():
    condition_ast = parse_condition("amount_inr == 25000")
    assert evaluate_condition(condition_ast, {"amount_inr": 25000}) is True
    assert evaluate_condition(condition_ast, {"amount_inr": 24999.999}) is False
    assert evaluate_condition(condition_ast, {"amount_inr": 25000.001}) is False


def test_strict_greater_threshold_is_exact():
    condition_ast = parse_condition("amount_inr > 25000")
    assert evaluate_condition(condition_ast, {"amount_inr": 25000}) is False
    assert evaluate_condition(condition_ast, {"amount_inr": 25000.01}) is True


def test_float_literal_equality_is_exact_binary_comparison():
    condition_ast = parse_condition("recovery_probability == 0.70")
    assert evaluate_condition(condition_ast, {"recovery_probability": 0.70}) is True
    assert evaluate_condition(condition_ast, {"recovery_probability": 0.699999}) is False
    assert evaluate_condition(condition_ast, {"recovery_probability": 0.700001}) is False


def test_gte_threshold_is_exact():
    condition_ast = parse_condition("recovery_probability >= 0.70")
    assert evaluate_condition(condition_ast, {"recovery_probability": 0.70}) is True
    assert evaluate_condition(condition_ast, {"recovery_probability": 0.6999999}) is False


def test_unknown_action_is_rejected(tmp_path):
    mutated = BASE_CONFIG.replace("action: STOP", "action: AUTO_RETRY")
    with pytest.raises(ValueError):
        load_policy_config(_write_config(tmp_path, mutated))


def test_duplicate_rule_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_policy_config(_write_config(tmp_path, DUPLICATE_ID_CONFIG))


def test_non_integer_priority_is_rejected(tmp_path):
    string_priority = BASE_CONFIG.replace("priority: 10", "priority: high")
    float_priority = BASE_CONFIG.replace("priority: 10", "priority: 10.5")
    bool_priority = BASE_CONFIG.replace("priority: 10", "priority: true")
    with pytest.raises(ValueError):
        load_policy_config(_write_config(tmp_path, string_priority))
    with pytest.raises(ValueError):
        load_policy_config(_write_config(tmp_path, float_priority))
    with pytest.raises(ValueError):
        load_policy_config(_write_config(tmp_path, bool_priority))


def test_non_boolean_stop_precedence_is_rejected(tmp_path):
    mutated = BASE_CONFIG.replace("stop_precedence: false", 'stop_precedence: "yes"')
    with pytest.raises(ValueError):
        load_policy_config(_write_config(tmp_path, mutated))


def test_bare_column_as_whole_condition_is_rejected():
    with pytest.raises(ValueError):
        parse_condition("fraud_risk")


def test_action_vocabulary_mismatch_is_rejected(tmp_path):
    mutated = BASE_CONFIG.replace("  - STOP\n", "")
    with pytest.raises(ValueError):
        load_policy_config(_write_config(tmp_path, mutated))


def test_missing_column_raises_keyerror_at_evaluation():
    condition_ast = parse_condition("attempt_number >= 4")
    with pytest.raises(KeyError, match="attempt_number"):
        evaluate_condition(condition_ast, {})


def test_policy_rule_dataclass_holds_parsed_and_raw_condition():
    condition_ast = parse_condition("attempt_number >= 4")
    rule = PolicyRule(
        id="T001",
        name="template_rule",
        priority=1,
        action="STOP",
        reason="Shape probe.",
        condition_ast=condition_ast,
        condition_text="attempt_number >= 4",
    )
    assert rule.priority == 1
    assert rule.action == "STOP"
    assert rule.condition_text == "attempt_number >= 4"
