"""Tests for the deterministic policy decision layer (``decide_action``).

The policy layer separates recommendation from authorization:
``decide_action`` consumes only decision-time context facts and the calibrated
probability ``P(recovered | context)``; it never sees or overrides any external
AI recommendation because none exists at this layer. Whatever action it returns
is the authorized action, derived purely from rule precedence.

Also covers the single permitted Task 1 integration change: the optional
per-rule ``enabled`` attribute (absent means True; non-bool values rejected).
"""

from __future__ import annotations

import dataclasses

import pytest

from recovery.policy import (
    CANONICAL_ACTIONS,
    RESIDUAL_DEFAULT_ACTION,
    PolicyConfig,
    PolicyRule,
    decide_action,
    evaluate_condition,
    load_policy_config,
    parse_condition,
)

CONFIG_PATH = "config/business_rules.yaml"

# A decision-time context that satisfies no shipped rule: every column any
# shipped condition references is present so evaluation never KeyErrors.
NEUTRAL_CONTEXT = {
    "customer_opted_out": False,
    "fraud_risk": False,
    "failure_category": "network_timeout",
    "attempt_number": 1,
    "amount_inr": 5000,
    "recovery_probability": 0.45,
}

SHIPPED_RULES = {
    "R001": ("customer_opt_out", 100, "STOP"),
    "R002": ("fraud_risk", 95, "STOP"),
    "R003": ("hard_decline", 90, "STOP"),
    "R004": ("retry_limit", 85, "STOP"),
    "R005": ("payment_method_update", 80, "REQUEST_UPDATE"),
    "R006": ("high_value_uncertain", 75, "HUMAN_REVIEW"),
    "R007": ("retry_now_eligible", 50, "RETRY_NOW"),
    "R008": ("low_probability_stop", 40, "STOP"),
}

SHIPPED_REASONS = {rule.id: rule.reason for rule in load_policy_config(CONFIG_PATH).rules}

TRUTH_TABLE = [
    ("R001", {"customer_opted_out": True}),
    ("R002", {"fraud_risk": True}),
    ("R003", {"failure_category": "hard_decline"}),
    ("R004", {"attempt_number": 4}),
    ("R005", {"failure_category": "payment_method_issue"}),
    ("R006", {"amount_inr": 30000}),
    (
        "R007",
        {
            "failure_category": "temporary_decline",
            "recovery_probability": 0.85,
            "attempt_number": 2,
        },
    ),
    ("R008", {"recovery_probability": 0.10}),
]

SYNTHETIC_TEMPLATE = """version: "9.9"
project: "RecoverAI"
principle: "AI recommends; policy engine authorizes."
rule_resolution:
  matching_rule: highest_priority_wins
  stop_precedence: {stop_precedence}
  deterministic: true

action_vocabulary:
  - RETRY_NOW
  - RETRY_LATER
  - REQUEST_UPDATE
  - HUMAN_REVIEW
  - STOP

rules:
{rules}"""

POSITIVE_HIGH_PRIORITY = """  - id: P099
    name: high_priority_positive
    priority: 99
    condition: "attempt_number >= 1"
    action: RETRY_NOW
    reason: "Synthetic positive rule with numerically dominant priority."
"""

STOP_LOW_PRIORITY = """  - id: S010
    name: low_priority_stop
    priority: 10
    condition: "attempt_number >= 1"
    action: STOP
    reason: "Synthetic stop rule that dominance must rescue."
"""

EQUAL_POSITIVE_HIGH_ID = """  - id: Y010
    name: equal_positive_high_id
    priority: 50
    condition: "attempt_number >= 1"
    action: RETRY_LATER
    reason: "Equal-priority candidate carrying the higher id."
"""

EQUAL_POSITIVE_LOW_ID = """  - id: Y002
    name: equal_positive_low_id
    priority: 50
    condition: "attempt_number >= 1"
    action: REQUEST_UPDATE
    reason: "Equal-priority candidate carrying the lower id."
"""

STOP_EQUAL_LOW_ID = """  - id: W001
    name: equal_stop_low_id
    priority: 55
    condition: "attempt_number >= 1"
    action: STOP
    reason: "Equal-priority stop whose id sorts lower."
"""

POSITIVE_EQUAL_HIGH_ID = """  - id: W002
    name: equal_positive_high_id
    priority: 55
    condition: "attempt_number >= 1"
    action: RETRY_NOW
    reason: "Equal-priority positive whose id sorts higher."
"""

POSITIVE_EQUAL_LOW_ID = """  - id: V001
    name: equal_positive_low_id
    priority: 55
    condition: "attempt_number >= 1"
    action: RETRY_NOW
    reason: "Equal-priority positive whose id sorts lower."
"""

STOP_EQUAL_HIGH_ID = """  - id: V002
    name: equal_stop_high_id
    priority: 55
    condition: "attempt_number >= 1"
    action: STOP
    reason: "Equal-priority stop whose id sorts higher."
"""

DISABLED_STOP_GUARD = """  - id: D001
    name: disabled_stop_guard
    priority: 90
    condition: "fraud_risk == true"
    action: STOP
    reason: "Would stop everything if it were enabled."
    enabled: false
"""

FALLBACK_RETRY = """  - id: D002
    name: fallback_retry
    priority: 40
    condition: "attempt_number >= 2"
    action: RETRY_NOW
    reason: "Positive rule that wins while the guard sleeps."
{enabled_line}"""

ENABLED_PROBE_TEMPLATE = """  - id: N001
    name: enabled_probe
    priority: 10
    condition: "attempt_number >= 4"
    action: STOP
    reason: "Probe for the optional enabled attribute."
    enabled: {value}
"""

ID_PROBE_TEMPLATE = """  - id: {value}
    name: id_probe
    priority: 10
    condition: "attempt_number >= 4"
    action: STOP
    reason: "Probe for rule id typing."
"""


def _synthetic(stop_precedence: bool, *rule_blocks: str) -> str:
    return SYNTHETIC_TEMPLATE.format(
        stop_precedence=str(stop_precedence).lower(), rules="\n".join(rule_blocks)
    )


def _write_config(tmp_path, text):
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _expected_trace(rule_id):
    return tuple(
        (f"R{i:03d}", f"R{i:03d}" == rule_id) for i in range(1, 9)
    )


def test_residual_default_constant_is_retry_later():
    assert RESIDUAL_DEFAULT_ACTION == "RETRY_LATER"


@pytest.mark.parametrize("rule_id,overrides", TRUTH_TABLE)
def test_every_shipped_rule_fires_on_a_crafted_context(rule_id, overrides):
    name, priority, action = SHIPPED_RULES[rule_id]
    decision = decide_action({**NEUTRAL_CONTEXT, **overrides}, load_policy_config(CONFIG_PATH))
    assert decision.authorized_action == action
    assert decision.matched_rule_id == rule_id
    assert decision.matched_rule_name == name
    assert decision.priority == priority
    assert decision.reason == SHIPPED_REASONS[rule_id]
    assert decision.is_stop is (action == "STOP")
    assert decision.evaluated_rules == _expected_trace(rule_id)
    assert len(decision.evaluated_rules) == 8


def test_stop_overrides_numerically_higher_positive_priority(tmp_path):
    config = load_policy_config(
        _write_config(tmp_path, _synthetic(True, POSITIVE_HIGH_PRIORITY, STOP_LOW_PRIORITY))
    )
    decision = decide_action({"attempt_number": 1}, config)
    assert decision.authorized_action == "STOP"
    assert decision.matched_rule_id == "S010"
    assert decision.priority == 10
    assert decision.is_stop is True


def test_without_stop_precedence_the_higher_positive_priority_wins(tmp_path):
    config = load_policy_config(
        _write_config(tmp_path, _synthetic(False, POSITIVE_HIGH_PRIORITY, STOP_LOW_PRIORITY))
    )
    decision = decide_action({"attempt_number": 1}, config)
    assert decision.authorized_action == "RETRY_NOW"
    assert decision.matched_rule_id == "P099"
    assert decision.priority == 99
    assert decision.is_stop is False


@pytest.mark.parametrize("swap_order", [False, True])
def test_equal_priority_ties_break_to_lowest_rule_id_regardless_of_insertion(tmp_path, swap_order):
    blocks = [EQUAL_POSITIVE_HIGH_ID, EQUAL_POSITIVE_LOW_ID]
    if swap_order:
        blocks.reverse()
    config = load_policy_config(_write_config(tmp_path, _synthetic(False, *blocks)))
    decision = decide_action({"attempt_number": 1}, config)
    assert decision.matched_rule_id == "Y002"
    assert decision.authorized_action == "REQUEST_UPDATE"
    assert decision.priority == 50


@pytest.mark.parametrize("stop_id_sorts_lower", [True, False])
def test_equal_priority_stop_beats_positive_under_stop_precedence(tmp_path, stop_id_sorts_lower):
    if stop_id_sorts_lower:
        blocks = [STOP_EQUAL_LOW_ID, POSITIVE_EQUAL_HIGH_ID]
        stop_id = "W001"
    else:
        blocks = [POSITIVE_EQUAL_LOW_ID, STOP_EQUAL_HIGH_ID]
        stop_id = "V002"
    config = load_policy_config(_write_config(tmp_path, _synthetic(True, *blocks)))
    decision = decide_action({"attempt_number": 1}, config)
    assert decision.authorized_action == "STOP"
    assert decision.matched_rule_id == stop_id
    assert decision.is_stop is True


def test_residual_default_is_retry_later_when_nothing_matches():
    context = {**NEUTRAL_CONTEXT, "failure_category": "authentication_required"}
    decision = decide_action(context, load_policy_config(CONFIG_PATH))
    assert decision.authorized_action == RESIDUAL_DEFAULT_ACTION
    assert decision.matched_rule_id is None
    assert decision.matched_rule_name is None
    assert decision.priority is None
    assert decision.is_stop is False
    assert "default" in decision.reason.lower()
    assert RESIDUAL_DEFAULT_ACTION in decision.reason
    assert decision.evaluated_rules == _expected_trace(None)


def test_missing_column_raises_keyerror():
    policy = load_policy_config(CONFIG_PATH)
    incomplete = {k: v for k, v in NEUTRAL_CONTEXT.items() if k != "recovery_probability"}
    with pytest.raises(KeyError, match="recovery_probability"):
        decide_action(incomplete, policy)
    r008 = {rule.id: rule for rule in policy.rules}["R008"]
    with pytest.raises(KeyError, match="recovery_probability"):
        evaluate_condition(r008.condition_ast, {})


def test_disabled_top_priority_stop_lets_lower_positive_win(tmp_path):
    config = load_policy_config(_write_config(tmp_path, _synthetic(True, DISABLED_STOP_GUARD, FALLBACK_RETRY.format(enabled_line=""))))
    decision = decide_action({"fraud_risk": True, "attempt_number": 2}, config)
    assert decision.authorized_action == "RETRY_NOW"
    assert decision.matched_rule_id == "D002"
    assert decision.is_stop is False
    assert decision.evaluated_rules == (("D001", False), ("D002", True))


def test_all_rules_disabled_degrades_to_residual_default(tmp_path):
    everything_off = _synthetic(
        True,
        DISABLED_STOP_GUARD,
        FALLBACK_RETRY.format(enabled_line="    enabled: false\n"),
    )
    config = load_policy_config(_write_config(tmp_path, everything_off))
    decision = decide_action({"fraud_risk": True, "attempt_number": 2}, config)
    assert decision.authorized_action == RESIDUAL_DEFAULT_ACTION
    assert decision.matched_rule_id is None
    assert decision.evaluated_rules == (("D001", False), ("D002", False))


def test_empty_rules_tuple_behaves_like_no_match():
    empty = PolicyConfig(
        version="empty", stop_precedence=True, rules=(), canonical_actions=CANONICAL_ACTIONS
    )
    decision = decide_action({}, empty)
    assert decision.authorized_action == RESIDUAL_DEFAULT_ACTION
    assert decision.matched_rule_id is None
    assert decision.is_stop is False
    assert decision.evaluated_rules == ()
    assert "default" in decision.reason.lower()


@pytest.mark.parametrize("value", ["1", "0", '"yes"'])
def test_loader_rejects_non_boolean_enabled(tmp_path, value):
    text = SYNTHETIC_TEMPLATE.format(
        stop_precedence="false", rules=ENABLED_PROBE_TEMPLATE.format(value=value)
    )
    with pytest.raises(ValueError, match="N001"):
        load_policy_config(_write_config(tmp_path, text))


def test_enabled_false_parses_and_absent_defaults_to_true(tmp_path):
    probe = load_policy_config(
        _write_config(
            tmp_path,
            SYNTHETIC_TEMPLATE.format(
                stop_precedence="false",
                rules=ENABLED_PROBE_TEMPLATE.format(value="false"),
            ),
        )
    )
    assert probe.rules[0].enabled is False
    shipped = load_policy_config(CONFIG_PATH)
    assert all(rule.enabled is True for rule in shipped.rules)


def test_loader_rejects_unquoted_numeric_rule_id(tmp_path):
    text = SYNTHETIC_TEMPLATE.format(
        stop_precedence="false", rules=ID_PROBE_TEMPLATE.format(value="001")
    )
    with pytest.raises(ValueError, match="rule id must be a non-empty string"):
        load_policy_config(_write_config(tmp_path, text))


def test_loader_rejects_mixed_quoted_and_numeric_rule_ids(tmp_path):
    mixed = ID_PROBE_TEMPLATE.format(value="'R009'") + ID_PROBE_TEMPLATE.format(value="010")
    text = SYNTHETIC_TEMPLATE.format(stop_precedence="false", rules=mixed)
    with pytest.raises(ValueError, match="rule id must be a non-empty string"):
        load_policy_config(_write_config(tmp_path, text))


def test_loader_rejects_empty_rule_id(tmp_path):
    text = SYNTHETIC_TEMPLATE.format(
        stop_precedence="false", rules=ID_PROBE_TEMPLATE.format(value='""')
    )
    with pytest.raises(ValueError, match="rule id must be a non-empty string"):
        load_policy_config(_write_config(tmp_path, text))


def test_noncanonical_action_bypassing_loader_trips_defensive_check():
    rogue = PolicyRule(
        id="V666",
        name="rogue",
        priority=99,
        action="AUTO_RETRY",
        reason="Constructed directly in Python to bypass loader validation.",
        condition_ast=parse_condition("attempt_number >= 1"),
        condition_text="attempt_number >= 1",
    )
    config = PolicyConfig(
        version="rogue", stop_precedence=False, rules=(rogue,), canonical_actions=CANONICAL_ACTIONS
    )
    with pytest.raises(ValueError):
        decide_action({"attempt_number": 1}, config)


def test_amount_boundary_exactly_25000_does_not_fire_r006():
    context = {**NEUTRAL_CONTEXT, "amount_inr": 25000, "recovery_probability": 0.69}
    decision = decide_action(context, load_policy_config(CONFIG_PATH))
    assert decision.authorized_action == RESIDUAL_DEFAULT_ACTION
    assert dict(decision.evaluated_rules)["R006"] is False


def test_probability_boundary_exactly_070_fires_r007():
    context = {
        **NEUTRAL_CONTEXT,
        "failure_category": "temporary_decline",
        "recovery_probability": 0.70,
    }
    decision = decide_action(context, load_policy_config(CONFIG_PATH))
    assert decision.authorized_action == "RETRY_NOW"
    assert decision.matched_rule_id == "R007"
    assert dict(decision.evaluated_rules)["R007"] is True


def test_decide_action_is_deterministic_for_identical_inputs():
    policy = load_policy_config(CONFIG_PATH)
    first = decide_action(NEUTRAL_CONTEXT, policy)
    second = decide_action(dict(NEUTRAL_CONTEXT), policy)
    assert first == second
    for field in dataclasses.fields(first):
        assert getattr(first, field.name) == getattr(second, field.name)
