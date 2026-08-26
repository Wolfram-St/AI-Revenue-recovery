"""Tests for Day 4 ground-truth/evaluation reporting (plan Task 7, decision D6)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

from simulation.config import load_treatment_policy
from simulation.reporting import (
    BASELINE_ARM,
    LABEL_GROUND_TRUTH,
    LABEL_OBSERVED,
    ground_truth_table,
    observed_differences,
    overlap_diagnostics,
    render_summary,
    summarize_arms,
)

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "reporting.py"

OBSERVED_LABEL = "OBSERVED SIMULATED OUTCOME"
GROUND_TRUTH_LABEL = "SIMULATED GROUND TRUTH"

REQUIRED_JOINED_COLUMNS = (
    "amount_inr",
    "assigned_action",
    "arm_source",
    "simulated_recovered",
    "simulated_recovered_amount_inr",
)

ARM_ORDER = ("CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")

TREATED_ARMS = ("RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")


def joined_frame() -> pd.DataFrame:
    """Hand-computed 8-row fixture.

    Rows 0-1: randomized CONTROL (one recovered @1000, one lost @500).
    Row 2:    safety-censored CONTROL, lost @800 (never entered stage 2).
    Rows 3-4: randomized RETRY_NOW, both recovered (2000 + 1000).
    Row 5:    randomized RETRY_LATER, lost @400.
    Row 6:    randomized REQUEST_UPDATE, lost @300.
    Row 7:    randomized HUMAN_REVIEW, recovered @250.
    Every asserted number below is derivable from this table by hand.
    """
    return pd.DataFrame(
        {
            "attempt_id": [f"ATT-{position:06d}" for position in range(8)],
            "amount_inr": [1000.0, 500.0, 800.0, 2000.0, 1000.0, 400.0, 300.0, 250.0],
            "assigned_action": [
                "CONTROL",
                "CONTROL",
                "CONTROL",
                "RETRY_NOW",
                "RETRY_NOW",
                "RETRY_LATER",
                "REQUEST_UPDATE",
                "HUMAN_REVIEW",
            ],
            "arm_source": [
                "randomized",
                "randomized",
                "safety_censored",
                "randomized",
                "randomized",
                "randomized",
                "randomized",
                "randomized",
            ],
            "assignment_probability": [0.20, 0.20, 0.00, 0.30, 0.30, 0.25, 0.15, 0.10],
            "simulated_recovered": [1, 0, 0, 1, 1, 0, 0, 1],
            "simulated_recovered_amount_inr": [
                1000.0, 0.0, 0.0, 2000.0, 1000.0, 0.0, 0.0, 250.0,
            ],
            "propensity_under_assignment": [0.40, 0.60, 0.05, 0.70, 0.90, 0.30, 0.20, 0.80],
        }
    )


def empty_joined_frame() -> pd.DataFrame:
    columns = REQUIRED_JOINED_COLUMNS + (
        "assignment_probability",
        "propensity_under_assignment",
    )
    return pd.DataFrame({name: pd.Series(dtype="float64") for name in columns})


def stable_difference_frame() -> pd.DataFrame:
    """30 randomized controls (rate 0.50, avg revenue 50.0) vs 35 HUMAN_REVIEW
    rows (rate 0.80, avg revenue 160.0): both sides clear the 30-case floor."""
    recovered = [1, 0] * 15 + [1] * 28 + [0] * 7
    return pd.DataFrame(
        {
            "amount_inr": [100.0] * 30 + [200.0] * 35,
            "assigned_action": ["CONTROL"] * 30 + ["HUMAN_REVIEW"] * 35,
            "arm_source": ["randomized"] * 65,
            "simulated_recovered": recovered,
            "simulated_recovered_amount_inr": [
                100.0 if flag else 0.0 for flag in recovered[:30]
            ]
            + [200.0 if flag else 0.0 for flag in recovered[30:]],
        }
    )


# ---------------------------------------------------------------------------
# 1. Hand-computed fixtures: summarize_arms
# ---------------------------------------------------------------------------


def test_summarize_arms_matches_hand_computed_fixture_exactly():
    result = summarize_arms(joined_frame())

    assert result["label"] == OBSERVED_LABEL
    assert result["case_count"] == 8
    assert set(result) == {"label", "case_count", "arms"}
    assert list(result["arms"]) == list(ARM_ORDER)

    control = result["arms"]["CONTROL"]
    assert set(control) == {
        "count",
        "randomized_count",
        "safety_censored_count",
        "recovery_rate",
        "recovered_amount_inr_total",
        "randomized_recovery_rate",
    }
    assert control["count"] == 3
    assert control["randomized_count"] == 2
    assert control["safety_censored_count"] == 1
    assert round(control["recovery_rate"], 4) == 0.3333
    assert control["recovered_amount_inr_total"] == 1000.00
    # Randomized controls only (rows 0-1): 1 of 2 recovered -- excludes the
    # safety-censored row folded into the aggregate 0.3333 rate above.
    assert round(control["randomized_recovery_rate"], 4) == 0.5000

    retry_now = result["arms"]["RETRY_NOW"]
    assert retry_now["count"] == 2
    assert retry_now["randomized_count"] == 2
    assert retry_now["safety_censored_count"] == 0
    assert round(retry_now["recovery_rate"], 4) == 1.0000
    assert retry_now["recovered_amount_inr_total"] == 3000.00

    retry_later = result["arms"]["RETRY_LATER"]
    assert retry_later["count"] == 1
    assert retry_later["randomized_count"] == 1
    assert retry_later["safety_censored_count"] == 0
    assert round(retry_later["recovery_rate"], 4) == 0.0000
    assert retry_later["recovered_amount_inr_total"] == 0.00

    request_update = result["arms"]["REQUEST_UPDATE"]
    assert request_update["count"] == 1
    assert round(request_update["recovery_rate"], 4) == 0.0000
    assert request_update["recovered_amount_inr_total"] == 0.00

    human_review = result["arms"]["HUMAN_REVIEW"]
    assert human_review["count"] == 1
    assert round(human_review["recovery_rate"], 4) == 1.0000
    assert human_review["recovered_amount_inr_total"] == 250.00


# ---------------------------------------------------------------------------
# 2. Labels: exact strings on all four outputs
# ---------------------------------------------------------------------------


def test_all_four_outputs_carry_the_exact_label_strings():
    frame = joined_frame()

    assert LABEL_OBSERVED == "OBSERVED SIMULATED OUTCOME"
    assert LABEL_GROUND_TRUTH == "SIMULATED GROUND TRUTH"
    assert summarize_arms(frame)["label"] == "OBSERVED SIMULATED OUTCOME"
    assert observed_differences(frame)["label"] == "OBSERVED SIMULATED OUTCOME"
    assert overlap_diagnostics(frame)["label"] == "OBSERVED SIMULATED OUTCOME"
    assert ground_truth_table(POLICY)["label"] == "SIMULATED GROUND TRUTH"


# ---------------------------------------------------------------------------
# 3. Hand-computed fixtures: observed_differences
# ---------------------------------------------------------------------------


def test_observed_differences_hand_checked_rate_and_revenue_per_case():
    result = observed_differences(joined_frame())

    assert set(result) == {"label", "note", "baseline_arm", "treated_arms"}
    assert result["baseline_arm"] == "CONTROL"
    assert list(result["treated_arms"]) == list(TREATED_ARMS)

    # Baseline = RANDOMIZED controls only (rows 0-1): n=2, rate 0.5, avg 500.0.
    # The safety-censored row 2 (lost, amount 800) must NOT enter the baseline.
    retry_now = result["treated_arms"]["RETRY_NOW"]
    assert round(retry_now["recovery_rate_difference"], 4) == 0.5000
    assert retry_now["revenue_per_case_difference_inr"] == 1000.00

    retry_later = result["treated_arms"]["RETRY_LATER"]
    assert round(retry_later["recovery_rate_difference"], 4) == -0.5000
    assert retry_later["revenue_per_case_difference_inr"] == -500.00

    request_update = result["treated_arms"]["REQUEST_UPDATE"]
    assert round(request_update["recovery_rate_difference"], 4) == -0.5000
    assert request_update["revenue_per_case_difference_inr"] == -500.00

    human_review = result["treated_arms"]["HUMAN_REVIEW"]
    assert round(human_review["recovery_rate_difference"], 4) == 0.5000
    assert human_review["revenue_per_case_difference_inr"] == -250.00


def test_difference_note_states_naive_confounded_by_eligibility_not_causal():
    note = observed_differences(joined_frame())["note"]

    lowered = note.lower()
    assert "naive" in lowered
    assert "confounded" in lowered
    assert "eligibility" in lowered
    assert "not causal" in lowered


def test_count_caveat_fires_below_threshold_naming_both_counts():
    caveat = observed_differences(joined_frame())["treated_arms"]["RETRY_NOW"][
        "count_caveat"
    ]

    assert isinstance(caveat, str)
    assert "RETRY_NOW" in caveat
    assert "CONTROL" in caveat
    assert "30" in caveat


def test_count_caveat_fires_for_every_small_fixture_arm():
    treated = observed_differences(joined_frame())["treated_arms"]

    for arm in TREATED_ARMS:
        assert isinstance(treated[arm]["count_caveat"], str), arm


def test_count_caveat_stays_null_when_both_sides_clear_threshold():
    result = observed_differences(stable_difference_frame())

    entry = result["treated_arms"]["HUMAN_REVIEW"]
    assert entry["count_caveat"] is None
    # Hand-check at scale: 28/35 - 15/30 = 0.8 - 0.5; 160.0 - 50.0.
    assert round(entry["recovery_rate_difference"], 4) == 0.3000
    assert entry["revenue_per_case_difference_inr"] == 110.00


# ---------------------------------------------------------------------------
# 4. Ground-truth table matches the loaded policy exactly
# ---------------------------------------------------------------------------


def test_ground_truth_table_matches_loaded_policy_exactly():
    table = ground_truth_table(POLICY)

    assert set(table) == {
        "label",
        "master_seed",
        "noise_sigma_logit",
        "arm_probabilities",
        "main_effects_logit",
        "interactions",
        "base_propensity_terms",
    }
    assert table["master_seed"] == 20260826
    assert table["noise_sigma_logit"] == 0.5
    assert table["arm_probabilities"] == dict(POLICY.arm_probabilities)
    assert table["arm_probabilities"] == {
        "CONTROL": 0.20,
        "RETRY_NOW": 0.30,
        "RETRY_LATER": 0.25,
        "REQUEST_UPDATE": 0.15,
        "HUMAN_REVIEW": 0.10,
    }
    assert table["main_effects_logit"] == dict(POLICY.main_effects_logit)
    assert table["interactions"] == [
        {
            "action": "RETRY_NOW",
            "column": "failure_category",
            "condition": "failure_category == temporary_decline",
            "effect_logit": 0.40,
        },
        {
            "action": "RETRY_LATER",
            "column": "attempt_number",
            "condition": "attempt_number >= 3",
            "effect_logit": -0.25,
        },
    ]

    terms = table["base_propensity_terms"]
    assert terms["intercept"] == -0.35
    assert terms["category_effects"] == {
        "temporary_decline": 0.95,
        "payment_method_issue": 0.25,
        "authentication_required": 0.05,
        "unknown": -0.15,
        "hard_decline": -1.45,
    }
    assert terms["successful_payment_count_log1p"] == 0.11
    assert terms["historical_recovery_count_min5"] == 0.16
    assert terms["attempt_number_prior_offset"] == -0.28
    assert terms["fraud_risk"] == -0.22
    assert terms["amount_log1p_per_k"] == -0.10
    assert terms["method_upi"] == 0.12
    assert terms["device_android"] == 0.10


def test_ground_truth_table_returns_copies_never_policy_internals():
    table = ground_truth_table(POLICY)
    table["arm_probabilities"]["CONTROL"] = 9.9
    table["main_effects_logit"]["RETRY_NOW"] = 9.9
    table["base_propensity_terms"]["category_effects"]["unknown"] = 9.9

    fresh = ground_truth_table(POLICY)

    assert fresh["arm_probabilities"]["CONTROL"] == 0.20
    assert fresh["main_effects_logit"]["RETRY_NOW"] == 0.60
    assert fresh["base_propensity_terms"]["category_effects"]["unknown"] == -0.15
    assert POLICY.arm_probabilities["CONTROL"] == 0.20


# ---------------------------------------------------------------------------
# 5. Overlap diagnostics: censored rows never inflate randomized ranges
# ---------------------------------------------------------------------------


def test_overlap_diagnostics_split_by_source_with_distinct_control_entries():
    result = overlap_diagnostics(joined_frame())

    assert set(result) == {
        "label",
        "eligible_count",
        "safety_censored_count",
        "assignment_probability_ranges",
        "propensity_under_assignment_ranges",
        "positivity_note",
    }
    assert result["label"] == OBSERVED_LABEL
    assert result["eligible_count"] == 7
    assert result["safety_censored_count"] == 1

    ranges = result["assignment_probability_ranges"]
    assert ranges["CONTROL"]["randomized"] == {"min": 0.20, "max": 0.20}
    assert ranges["CONTROL"]["safety_censored"] == {"min": 0.0, "max": 0.0}, (
        "the safety-censored assignment_probability of 0.0 must be reflected, "
        "never folded into the randomized range"
    )
    assert ranges["RETRY_NOW"] == {"randomized": {"min": 0.30, "max": 0.30}}
    assert "safety_censored" not in ranges["RETRY_NOW"]
    assert ranges["RETRY_LATER"]["randomized"] == {"min": 0.25, "max": 0.25}
    assert ranges["REQUEST_UPDATE"]["randomized"] == {"min": 0.15, "max": 0.15}
    assert ranges["HUMAN_REVIEW"]["randomized"] == {"min": 0.10, "max": 0.10}

    propensity = result["propensity_under_assignment_ranges"]
    assert propensity["CONTROL"] == {"min": 0.05, "max": 0.60}
    assert propensity["RETRY_NOW"] == {"min": 0.70, "max": 0.90}
    assert propensity["RETRY_LATER"] == {"min": 0.30, "max": 0.30}
    assert propensity["REQUEST_UPDATE"] == {"min": 0.20, "max": 0.20}
    assert propensity["HUMAN_REVIEW"] == {"min": 0.80, "max": 0.80}


def test_positivity_note_states_stratum_positivity_and_population_limit():
    note = overlap_diagnostics(joined_frame())["positivity_note"]

    lowered = note.lower()
    assert "positivity" in lowered
    assert "by construction" in lowered
    assert "eligible stratum" in lowered
    assert "not supported" in lowered


# ---------------------------------------------------------------------------
# 6. Determinism: identical inputs -> identical outputs
# ---------------------------------------------------------------------------


def test_render_summary_byte_identical_across_calls_and_carries_all_labels():
    frame = joined_frame()

    first = render_summary(frame, POLICY)
    second = render_summary(frame, POLICY)

    assert isinstance(first, str)
    assert first == second
    assert first.count("OBSERVED SIMULATED OUTCOME") >= 3
    assert first.count("SIMULATED GROUND TRUTH") >= 1


def test_dict_outputs_equal_across_calls():
    frame = joined_frame()

    assert summarize_arms(frame) == summarize_arms(frame)
    assert observed_differences(frame) == observed_differences(frame)
    assert overlap_diagnostics(frame) == overlap_diagnostics(frame)
    assert ground_truth_table(POLICY) == ground_truth_table(POLICY)


# ---------------------------------------------------------------------------
# 7. Missing columns rejected loudly; empty frame -> zeroed valid structures
# ---------------------------------------------------------------------------


def test_missing_base_column_raises_value_error_naming_each_missing_one():
    with pytest.raises(ValueError) as excinfo:
        summarize_arms(joined_frame().drop(columns=["amount_inr"]))
    assert "amount_inr" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        observed_differences(
            joined_frame().drop(columns=["assigned_action", "arm_source"])
        )
    message = str(excinfo.value)
    assert "assigned_action" in message
    assert "arm_source" in message


def test_overlap_names_probability_and_propensity_when_missing():
    frame = joined_frame().drop(
        columns=["assignment_probability", "propensity_under_assignment"]
    )

    with pytest.raises(ValueError) as excinfo:
        overlap_diagnostics(frame)

    message = str(excinfo.value)
    assert "assignment_probability" in message
    assert "propensity_under_assignment" in message


def test_empty_frame_yields_zeroed_valid_structures_with_labels_intact():
    empty = empty_joined_frame()

    arms = summarize_arms(empty)
    assert arms["label"] == OBSERVED_LABEL
    assert arms["case_count"] == 0
    assert list(arms["arms"]) == list(ARM_ORDER)
    for stats in arms["arms"].values():
        assert stats["count"] == 0
        assert stats["randomized_count"] == 0
        assert stats["safety_censored_count"] == 0
        assert stats["recovery_rate"] == 0.0
        assert stats["recovered_amount_inr_total"] == 0.0
    assert arms["arms"]["CONTROL"]["randomized_recovery_rate"] == 0.0

    differences = observed_differences(empty)
    assert differences["label"] == OBSERVED_LABEL
    assert differences["baseline_arm"] == "CONTROL"
    for entry in differences["treated_arms"].values():
        assert entry["recovery_rate_difference"] == 0.0
        assert entry["revenue_per_case_difference_inr"] == 0.0
        assert isinstance(entry["count_caveat"], str)

    overlap = overlap_diagnostics(empty)
    assert overlap["label"] == OBSERVED_LABEL
    assert overlap["eligible_count"] == 0
    assert overlap["safety_censored_count"] == 0
    for ranges in overlap["assignment_probability_ranges"].values():
        for bounds in ranges.values():
            assert bounds == {"min": 0.0, "max": 0.0}
    for bounds in overlap["propensity_under_assignment_ranges"].values():
        assert bounds == {"min": 0.0, "max": 0.0}

    text = render_summary(empty, POLICY)
    assert "OBSERVED SIMULATED OUTCOME" in text
    assert "SIMULATED GROUND TRUTH" in text


# ---------------------------------------------------------------------------
# 8. Purity: import whitelist, no rng, no wall clock, honesty guard
# ---------------------------------------------------------------------------


def _source_text() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def _source_without_docstring() -> str:
    """Executable-source view: token scans ignore the module docstring."""
    source = _source_text()
    docstring = ast.get_docstring(ast.parse(source))
    assert docstring is not None
    assert source.count(docstring) == 1
    return source.replace(docstring, " ", 1)


def _import_root_modules(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


ALLOWED_IMPORT_ROOTS = frozenset({"__future__", "numpy", "pandas", "simulation"})


def test_reporting_import_roots_match_whitelist_exactly():
    roots = _import_root_modules(_source_text())

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )


def test_no_default_rng_occurs_anywhere_in_source():
    assert "default_rng" not in _source_text()


def test_no_wall_clock_or_stdlib_randomness_tokens_in_executable_source():
    code = _source_without_docstring()

    forbidden_patterns = (
        r"(?<![\w.])datetime\.now",
        r"(?<![\w.])time\s*\.",
        r"(?<![\w.])secrets?\b",
        r"(?<![\w.])uuid\b",
        r"(?<![\w.])random\b",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"


def test_honesty_guard_disclaimer_required_estimate_language_banned_outside_docstring():
    full = _source_text()
    assert "not causal" in full, "the disclaimer fragment must be present"

    executable = _source_without_docstring()
    assert "causal estimate" not in executable, (
        "'causal estimate' may appear only inside the module-docstring disclaimer"
    )


def test_baseline_arm_constant_pins_control():
    assert BASELINE_ARM == "CONTROL"
