"""Tests for the Day 4 synthetic treatment policy configuration loader."""

from __future__ import annotations

import dataclasses
import math

import pytest
import yaml

from simulation.config import (
    SEED_STREAM_ASSIGNMENT,
    SEED_STREAM_OUTCOMES,
    SEED_STREAM_TEMPORAL,
    CANONICAL_ARMS,
    InteractionRule,
    TreatmentPolicy,
    load_treatment_policy,
)

EXPECTED_ARM_PROBABILITIES = {
    "CONTROL": 0.20,
    "RETRY_NOW": 0.30,
    "RETRY_LATER": 0.25,
    "REQUEST_UPDATE": 0.15,
    "HUMAN_REVIEW": 0.10,
}
EXPECTED_MAIN_EFFECTS = {
    "CONTROL": 0.0,
    "RETRY_NOW": 0.60,
    "RETRY_LATER": 0.35,
    "REQUEST_UPDATE": 0.45,
    "HUMAN_REVIEW": 0.25,
}
EXPECTED_CATEGORY_EFFECTS = {
    "temporary_decline": 0.95,
    "payment_method_issue": 0.25,
    "authentication_required": 0.05,
    "unknown": -0.15,
    "hard_decline": -1.45,
}
DEFAULT_PATH = "config/treatment_policy.yaml"

REQUIRED_TOP_LEVEL_KEYS = [
    "version",
    "master_seed",
    "arm_probabilities",
    "main_effects_logit",
    "interactions_logit",
    "noise_sigma_logit",
    "treatment_delay_hours",
    "resolution_window_hours",
    "base_propensity_terms",
]


def load_default() -> TreatmentPolicy:
    return load_treatment_policy(DEFAULT_PATH)


def write_variant(tmp_path, mutate) -> str:
    """Write a mutated copy of the shipped YAML and return its path."""
    raw = yaml.safe_load(open(DEFAULT_PATH, encoding="utf-8").read())
    mutate(raw)
    variant = tmp_path / "variant.yaml"
    variant.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return str(variant)


def expect_value_error(tmp_path, mutate, match=None):
    path = write_variant(tmp_path, mutate)
    with pytest.raises(ValueError) as excinfo:
        load_treatment_policy(path)
    if match is not None:
        assert match in str(excinfo.value)
    return excinfo


# ---------------------------------------------------------------------------
# Shipped configuration: verbatim values
# ---------------------------------------------------------------------------


def test_shipped_yaml_loads_with_verbatim_values():
    policy = load_default()
    assert policy.version == "1.0"
    assert policy.master_seed == 20260826
    assert policy.arm_probabilities == EXPECTED_ARM_PROBABILITIES
    assert policy.main_effects_logit == EXPECTED_MAIN_EFFECTS
    assert policy.interactions == (
        InteractionRule(
            action="RETRY_NOW",
            column="failure_category",
            equals_value="temporary_decline",
            min_threshold=None,
            effect_logit=0.40,
        ),
        InteractionRule(
            action="RETRY_LATER",
            column="attempt_number",
            equals_value=None,
            min_threshold=3,
            effect_logit=-0.25,
        ),
    )
    assert policy.noise_sigma_logit == 0.5
    assert policy.treatment_delay_hours == {
        "RETRY_NOW": 0.25,
        "REQUEST_UPDATE": 2.0,
        "HUMAN_REVIEW": 4.0,
        "RETRY_LATER": 24.0,
    }
    assert policy.resolution_window_hours == (1.0, 48.0)
    terms = policy.base_propensity_terms
    assert terms.intercept == -0.35
    assert terms.category_effects == EXPECTED_CATEGORY_EFFECTS
    assert terms.successful_payment_count_log1p == 0.11
    assert terms.historical_recovery_count_min5 == 0.16
    assert terms.attempt_number_prior_offset == -0.28
    assert terms.fraud_risk == -0.22
    assert terms.amount_log1p_per_k == -0.10
    assert terms.method_upi == 0.12
    assert terms.device_android == 0.10


def test_loaded_objects_and_nested_rules_are_frozen():
    policy = load_default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.master_seed = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.base_propensity_terms.intercept = 9.9
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.interactions[0].effect_logit = 9.9


def test_double_load_is_deterministic():
    first = load_default()
    second = load_default()
    assert first == second


def test_canonical_arm_vocabulary_exposed():
    assert CANONICAL_ARMS == frozenset(
        {"CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW"}
    )


# ---------------------------------------------------------------------------
# arm_probabilities validation
# ---------------------------------------------------------------------------


def test_probability_sum_violation_rejected(tmp_path):
    def bump_control(raw):
        raw["arm_probabilities"]["CONTROL"] = 0.30

    expect_value_error(tmp_path, bump_control, match="sum")


def test_extra_arm_key_rejected(tmp_path):
    def add_arm(raw):
        probabilities = raw["arm_probabilities"]
        probabilities["EXTRA_ARM"] = 0.05

    expect_value_error(
        tmp_path, add_arm, match="EXTRA_ARM"
    )


def test_missing_arm_key_rejected(tmp_path):
    def drop_arm(raw):
        del raw["arm_probabilities"]["HUMAN_REVIEW"]

    expect_value_error(tmp_path, drop_arm, match="HUMAN_REVIEW")


# ---------------------------------------------------------------------------
# main_effects_logit validation
# ---------------------------------------------------------------------------


def test_main_effect_out_of_range_rejected(tmp_path):
    def inflate(raw):
        raw["main_effects_logit"]["RETRY_NOW"] = 3.5

    expect_value_error(tmp_path, inflate, match="RETRY_NOW")


def test_nonzero_control_main_effect_rejected(tmp_path):
    def shift(raw):
        raw["main_effects_logit"]["CONTROL"] = 0.1

    expect_value_error(tmp_path, shift, match="CONTROL")


def test_main_effects_extra_key_rejected(tmp_path):
    def extra(raw):
        raw["main_effects_logit"]["EXTRA_EFFECT"] = 0.1

    expect_value_error(tmp_path, extra, match="EXTRA_EFFECT")


# ---------------------------------------------------------------------------
# interactions validation
# ---------------------------------------------------------------------------


def test_interaction_unknown_action_rejected(tmp_path):
    def stop_action(raw):
        raw["interactions_logit"].append(
            {
                "action": "STOP",
                "column": "failure_category",
                "equals_value": "temporary_decline",
                "effect_logit": 0.10,
            }
        )

    expect_value_error(tmp_path, stop_action, match="STOP")


def test_interaction_wrong_column_shape_rejected(tmp_path):
    def wrong_shape(raw):
        raw["interactions_logit"] = [
            {
                "action": "RETRY_NOW",
                "column": "attempt_number",
                "equals_value": 3,
                "effect_logit": 0.40,
            }
        ]

    expect_value_error(tmp_path, wrong_shape, match="attempt_number")


def test_interaction_unknown_column_rejected(tmp_path):
    def unknown_column(raw):
        raw["interactions_logit"] = [
            {
                "action": "RETRY_NOW",
                "column": "issuer_response",
                "equals_value": "network_timeout",
                "effect_logit": 0.40,
            }
        ]

    expect_value_error(tmp_path, unknown_column, match="issuer_response")


def test_interaction_missing_shape_field_rejected(tmp_path):
    def no_shape(raw):
        raw["interactions_logit"][0].pop("equals_value")

    expect_value_error(tmp_path, no_shape, match="RETRY_NOW")


def test_interaction_effect_out_of_range_rejected(tmp_path):
    def out_of_range(raw):
        raw["interactions_logit"][0]["effect_logit"] = -3.5

    expect_value_error(tmp_path, out_of_range, match="effect_logit")


def test_interaction_unknown_failure_category_rejected(tmp_path):
    def typo(raw):
        raw["interactions_logit"][0]["equals_value"] = "tempory_declne"

    expect_value_error(tmp_path, typo, match="unknown failure_category")


def test_empty_interactions_list_loads_with_zero_rules(tmp_path):
    def clear(raw):
        raw["interactions_logit"] = []

    path = write_variant(tmp_path, clear)
    policy = load_treatment_policy(path)
    assert policy.interactions == ()


# ---------------------------------------------------------------------------
# noise / delays / resolution window validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.0, 5.5])
def test_noise_sigma_bounds_rejected(tmp_path, sigma):
    def set_sigma(raw):
        raw["noise_sigma_logit"] = sigma

    expect_value_error(tmp_path, set_sigma, match="noise_sigma_logit")


def test_delay_missing_arm_rejected(tmp_path):
    def drop_delay(raw):
        del raw["treatment_delay_hours"]["HUMAN_REVIEW"]

    expect_value_error(tmp_path, drop_delay, match="HUMAN_REVIEW")


def test_negative_delay_rejected(tmp_path):
    def negative(raw):
        raw["treatment_delay_hours"]["RETRY_NOW"] = -1.0

    expect_value_error(tmp_path, negative, match="RETRY_NOW")


def test_delay_above_168_hours_rejected(tmp_path):
    def too_long(raw):
        raw["treatment_delay_hours"]["RETRY_LATER"] = 200.0

    expect_value_error(tmp_path, too_long, match="RETRY_LATER")


@pytest.mark.parametrize(
    "window",
    [[0.0, 48.0], [48.0, 1.0], [1.0, 721.0]],
)
def test_resolution_window_bounds_rejected(tmp_path, window):
    def set_window(raw):
        raw["resolution_window_hours"] = window

    expect_value_error(tmp_path, set_window, match="resolution_window_hours")


# ---------------------------------------------------------------------------
# base_propensity_terms validation
# ---------------------------------------------------------------------------


def test_base_term_missing_rejected(tmp_path):
    def drop_intercept(raw):
        del raw["base_propensity_terms"]["intercept"]

    expect_value_error(tmp_path, drop_intercept, match="intercept")


def test_category_effects_missing_category_rejected(tmp_path):
    def drop_unknown(raw):
        del raw["base_propensity_terms"]["category_effects"]["unknown"]

    expect_value_error(tmp_path, drop_unknown, match="unknown")


def test_base_term_non_finite_rejected(tmp_path):
    import math as _math

    def infinite(raw):
        raw["base_propensity_terms"]["fraud_risk"] = _math.inf

    expect_value_error(tmp_path, infinite, match="fraud_risk")


# ---------------------------------------------------------------------------
# top-level schema + scalar validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", REQUIRED_TOP_LEVEL_KEYS)
def test_missing_top_level_section_raises_value_error(tmp_path, missing_key):
    def drop_section(raw):
        del raw[missing_key]

    expect_value_error(
        tmp_path,
        drop_section,
        match=f"missing required configuration section: {missing_key}",
    )


def test_unknown_top_level_key_rejected(tmp_path):
    def add_llm(raw):
        raw["llm_assignment"] = True

    expect_value_error(tmp_path, add_llm, match="llm_assignment")


def test_master_seed_bool_rejected(tmp_path):
    def boolify(raw):
        raw["master_seed"] = True

    expect_value_error(tmp_path, boolify, match="master_seed")


def test_master_seed_string_rejected(tmp_path):
    def stringify(raw):
        raw["master_seed"] = "42"

    expect_value_error(tmp_path, stringify, match="master_seed")


def test_empty_version_rejected(tmp_path):
    def blank(raw):
        raw["version"] = "   "

    expect_value_error(tmp_path, blank, match="version")


def test_non_mapping_document_rejected(tmp_path):
    path = tmp_path / "scalar.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_treatment_policy(str(path))


# ---------------------------------------------------------------------------
# InteractionRule direct-construction XOR contract
# ---------------------------------------------------------------------------


def test_interaction_rule_rejects_both_shape_fields():
    with pytest.raises(ValueError, match="exactly one"):
        InteractionRule(
            action="RETRY_NOW",
            column="failure_category",
            equals_value="temporary_decline",
            min_threshold=3,
            effect_logit=0.40,
        )


def test_interaction_rule_rejects_neither_shape_field():
    with pytest.raises(ValueError, match="exactly one"):
        InteractionRule(
            action="RETRY_NOW",
            column="failure_category",
            equals_value=None,
            min_threshold=None,
            effect_logit=0.40,
        )


def test_interaction_rule_valid_constructions_both_shapes():
    equality_rule = InteractionRule(
        action="RETRY_NOW",
        column="failure_category",
        equals_value="temporary_decline",
        min_threshold=None,
        effect_logit=0.40,
    )
    threshold_rule = InteractionRule(
        action="RETRY_LATER",
        column="attempt_number",
        equals_value=None,
        min_threshold=3,
        effect_logit=-0.25,
    )
    assert equality_rule.equals_value == "temporary_decline"
    assert equality_rule.min_threshold is None
    assert threshold_rule.equals_value is None
    assert threshold_rule.min_threshold == 3


# ---------------------------------------------------------------------------
# seed-stream constants (plan D1b spawn order)
# ---------------------------------------------------------------------------


def test_seed_stream_constants_are_distinct_ordered_ints():
    assert SEED_STREAM_ASSIGNMENT == 0
    assert SEED_STREAM_OUTCOMES == 1
    assert SEED_STREAM_TEMPORAL == 2
    assert len(
        {SEED_STREAM_ASSIGNMENT, SEED_STREAM_OUTCOMES, SEED_STREAM_TEMPORAL}
    ) == 3


# ---------------------------------------------------------------------------
# no executable content guard
# ---------------------------------------------------------------------------


def _assert_plain_types(node) -> None:
    if isinstance(node, bool) or isinstance(node, (int, float, str)):
        return
    if isinstance(node, list):
        for item in node:
            _assert_plain_types(item)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            assert isinstance(key, str)
            _assert_plain_types(value)
        return
    raise AssertionError(f"non-plain type in YAML document: {type(node)!r}")


def test_yaml_contains_only_declarative_plain_types():
    text = open(DEFAULT_PATH, encoding="utf-8").read()
    document = yaml.safe_load(text)
    _assert_plain_types(document)


def test_yaml_has_no_code_like_substrings():
    text = open(DEFAULT_PATH, encoding="utf-8").read()
    assert "${" not in text
    assert "exec" not in text
