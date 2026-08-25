"""Tests for pure opportunity scoring and Expected Recovery Value.

Scoring consumes only the existing baseline estimate ``P(recovered | context)``
plus deterministic cost/risk constants. These tests pin the frozen formula,
the strict unrounded positivity rule, input validation, the batch contract
(index alignment, exact columns, dense ranks starting at 1, ``Int64`` nulls
for non-worth rows), determinism under row shuffling with stable tie-breaking
by original index order, input immutability, and the honesty boundary that
Expected Recovery Value is a prioritization signal only.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from recovery.scoring import (
    INTERVENE,
    NO_INTERVENTION,
    UNKNOWN_CATEGORY_RISK_FRACTION,
    RETRY_INTERVENTION_COST_INR,
    OpportunityScore,
    score_opportunities,
    score_opportunity,
)

VERBATIM_BOUNDARY_STATEMENT = (
    "Expected Recovery Value is a prioritization score, not an "
    "action-conditional causal estimate and not realized revenue."
)

BATCH_COLUMNS = [
    "scoring_recommendation",
    "expected_recovery_value_inr",
    "recovery_probability",
    "payment_amount_inr",
    "risk_penalty_inr",
    "intervention_cost_inr",
    "worth_intervening",
    "opportunity_rank",
]

ALLOWED_SCORING_IMPORTS = {
    "__future__",
    "dataclasses",
    "math",
    "numbers",
    "numpy",
    "typing",
    "pandas",
}


def test_module_constants_are_the_documented_simulation_parameters():
    assert RETRY_INTERVENTION_COST_INR == 10.0
    assert isinstance(RETRY_INTERVENTION_COST_INR, float)
    assert UNKNOWN_CATEGORY_RISK_FRACTION == 0.05
    assert isinstance(UNKNOWN_CATEGORY_RISK_FRACTION, float)


def test_recommendation_vocabulary_constants():
    assert INTERVENE == "INTERVENE"
    assert NO_INTERVENTION == "NO_INTERVENTION"


def test_docstring_states_the_prioritization_boundary_verbatim():
    import recovery.scoring as scoring_module

    assert VERBATIM_BOUNDARY_STATEMENT in scoring_module.__doc__
    assert VERBATIM_BOUNDARY_STATEMENT in score_opportunity.__doc__


def test_scoring_imports_only_pure_foundations_never_ml_or_policy():
    source = inspect.getsource(inspect.getmodule(score_opportunity))
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= ALLOWED_SCORING_IMPORTS
    assert "sklearn" not in source


def test_zero_probability_is_valid_and_never_worth_intervening():
    score = score_opportunity(0.0, 10000.0, "unknown")
    assert score.scoring_recommendation == NO_INTERVENTION
    assert score.worth_intervening is False
    assert score.expected_recovery_value_inr == round(
        0.0 * 10000.0 - RETRY_INTERVENTION_COST_INR - 0.05 * 10000.0, 2
    )
    assert score.expected_recovery_value_inr == -510.0
    assert score.risk_penalty_inr == 500.0


def test_full_probability_pins_exact_monetary_math_on_unknown_category():
    score = score_opportunity(1.0, 10000.0, "unknown")
    assert score.expected_recovery_value_inr == 9490.0
    assert score.risk_penalty_inr == 500.0
    assert score.intervention_cost_inr == 10.0
    assert score.payment_amount_inr == 10000.0
    assert score.worth_intervening is True
    assert score.scoring_recommendation == INTERVENE


def test_normal_probability_matches_hand_computed_frozen_formula():
    score = score_opportunity(0.55, 20000.0, "temporary_decline")
    assert score.expected_recovery_value_inr == 10990.0
    assert score.risk_penalty_inr == 0.0
    assert score.recovery_probability == 0.55
    assert score.payment_amount_inr == 20000.0
    assert score.intervention_cost_inr == 10.0
    assert score.worth_intervening is True
    assert score.scoring_recommendation == INTERVENE


def test_zero_amount_is_allowed_scores_negative_and_does_not_crash():
    score = score_opportunity(0.90, 0.0, "temporary_decline")
    assert score.payment_amount_inr == 0.0
    assert score.risk_penalty_inr == 0.0
    assert score.expected_recovery_value_inr == -10.0
    assert score.worth_intervening is False
    assert score.scoring_recommendation == NO_INTERVENTION


@pytest.mark.parametrize(
    "bad_probability",
    [-0.1, 1.0000001, float("nan"), float("inf"), True, False, "0.5", None],
)
def test_invalid_recovery_probability_raises_value_error(bad_probability):
    with pytest.raises(ValueError, match="recovery_probability"):
        score_opportunity(bad_probability, 1000.0, "temporary_decline")


def test_probability_error_message_names_the_offending_value():
    with pytest.raises(ValueError) as excinfo:
        score_opportunity(-0.1, 1000.0, "temporary_decline")
    assert "-0.1" in str(excinfo.value)


@pytest.mark.parametrize(
    "bad_amount", [-1, -0.01, float("nan"), float("inf"), "100", True, None]
)
def test_invalid_amount_raises_value_error(bad_amount):
    with pytest.raises(ValueError, match="amount_inr"):
        score_opportunity(0.5, bad_amount, "temporary_decline")


@pytest.mark.parametrize(
    "bad_cost", [-10.0, float("nan"), float("inf"), "free", None]
)
def test_invalid_intervention_cost_raises_value_error(bad_cost):
    with pytest.raises(ValueError, match="intervention_cost_inr"):
        score_opportunity(0.5, 1000.0, "temporary_decline", bad_cost)


@pytest.mark.parametrize("bad_category", ["", None, 7])
def test_invalid_failure_category_raises_value_error(bad_category):
    with pytest.raises(ValueError, match="failure_category"):
        score_opportunity(0.5, 1000.0, bad_category)


def test_unknown_category_carries_rounded_risk_penalty_reflected_in_erv():
    clean = score_opportunity(0.60, 4000.0, "temporary_decline")
    murky = score_opportunity(0.60, 4000.0, "unknown")
    assert murky.risk_penalty_inr == round(UNKNOWN_CATEGORY_RISK_FRACTION * 4000.0, 2)
    assert murky.risk_penalty_inr == 200.0
    assert murky.expected_recovery_value_inr == 2190.0
    assert murky.expected_recovery_value_inr == clean.expected_recovery_value_inr - 200.0


def test_known_category_temporary_decline_has_exactly_zero_penalty():
    score = score_opportunity(0.60, 4000.0, "temporary_decline")
    assert score.risk_penalty_inr == 0.0
    assert score.expected_recovery_value_inr == 2390.0


def test_negative_erv_case_is_not_worth_intervening():
    score = score_opportunity(0.01, 50.0, "temporary_decline")
    assert score.expected_recovery_value_inr == -9.5
    assert score.worth_intervening is False
    assert score.scoring_recommendation == NO_INTERVENTION


def test_zero_erv_boundary_is_strictly_not_worth_intervening():
    score = score_opportunity(0.5, 20.0, "temporary_decline")
    assert score.expected_recovery_value_inr == 0.0
    assert score.worth_intervening is False
    assert score.scoring_recommendation == NO_INTERVENTION


def test_positive_erv_is_worth_intervening():
    score = score_opportunity(0.80, 1000.0, "temporary_decline")
    assert score.expected_recovery_value_inr == 790.0
    assert score.worth_intervening is True
    assert score.scoring_recommendation == INTERVENE


def test_unrounded_sign_decides_even_when_displayed_erv_rounds_to_zero():
    barely_positive = score_opportunity(0.5, 20.008, "temporary_decline")
    barely_negative = score_opportunity(0.5, 19.996, "temporary_decline")
    assert barely_positive.expected_recovery_value_inr == 0.0
    assert barely_negative.expected_recovery_value_inr == 0.0
    assert barely_positive.worth_intervening is True
    assert barely_negative.worth_intervening is False
    assert barely_positive.scoring_recommendation == INTERVENE
    assert barely_negative.scoring_recommendation == NO_INTERVENTION


def test_representative_score_matches_frozen_formula_field_by_field():
    score = score_opportunity(0.42, 1234.56, "unknown", intervention_cost_inr=25.0)
    assert score.scoring_recommendation == "INTERVENE"
    assert score.expected_recovery_value_inr == 431.79
    assert score.recovery_probability == 0.42
    assert score.payment_amount_inr == 1234.56
    assert score.risk_penalty_inr == 61.73
    assert score.intervention_cost_inr == 25.0
    assert score.worth_intervening is True
    expected = OpportunityScore(
        scoring_recommendation="INTERVENE",
        expected_recovery_value_inr=431.79,
        recovery_probability=0.42,
        payment_amount_inr=1234.56,
        risk_penalty_inr=61.73,
        intervention_cost_inr=25.0,
        worth_intervening=True,
    )
    assert score == expected


def test_score_opportunity_is_repeatable_for_identical_inputs():
    first = score_opportunity(0.33, 777.77, "unknown")
    second = score_opportunity(0.33, 777.77, "unknown")
    assert first == second
    assert first is not second


def test_opportunityscore_fields_are_exactly_seven_in_order():
    names = [field.name for field in dataclasses.fields(OpportunityScore)]
    assert names == [
        "scoring_recommendation",
        "expected_recovery_value_inr",
        "recovery_probability",
        "payment_amount_inr",
        "risk_penalty_inr",
        "intervention_cost_inr",
        "worth_intervening",
    ]


def test_opportunityscore_is_frozen_against_mutation():
    score = score_opportunity(0.5, 100.0, "unknown")
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.worth_intervening = True


@pytest.mark.parametrize(
    "probability,amount,category,cost",
    [
        (0.95, 50000.0, "temporary_decline", RETRY_INTERVENTION_COST_INR),
        (0.55, 50000.0, "temporary_decline", RETRY_INTERVENTION_COST_INR),
        (0.01, 50.0, "temporary_decline", RETRY_INTERVENTION_COST_INR),
        (0.5, 20.0, "temporary_decline", RETRY_INTERVENTION_COST_INR),
        (0.42, 1234.56, "unknown", RETRY_INTERVENTION_COST_INR),
        (0.0, 9999.0, "unknown", RETRY_INTERVENTION_COST_INR),
        (1.0, 10000.0, "unknown", RETRY_INTERVENTION_COST_INR),
        (0.80, 1000.0, "temporary_decline", 0.0),
        (0.60, 4000.0, "unknown", 5.0),
        (0.42, 1234.56, "temporary_decline", 25.0),
        (1.0, 100.0, "temporary_decline", 1000000.0),
    ],
)
def test_recommendation_equals_intervene_iff_worth_intervening(
    probability, amount, category, cost
):
    score = score_opportunity(probability, amount, category, cost)
    assert score.scoring_recommendation == (
        INTERVENE if score.worth_intervening else NO_INTERVENTION
    )
    assert isinstance(score.worth_intervening, bool)


def _tie_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount_inr": [500.0, 3000.0, 3000.0, 50.0],
            "failure_category": [
                "gateway_timeout",
                "temporary_decline",
                "temporary_decline",
                "temporary_decline",
            ],
        }
    )


_TIE_PROBABILITIES = [0.95, 0.70, 0.70, 0.02]


def test_batch_output_columns_are_exact_and_ordered():
    out = score_opportunities(_tie_frame(), _TIE_PROBABILITIES)
    assert list(out.columns) == BATCH_COLUMNS


def test_batch_index_is_aligned_to_input_index():
    frame = _tie_frame()
    frame.index = ["case-d", "case-c", "case-b", "case-a"]
    out = score_opportunities(frame, _TIE_PROBABILITIES)
    assert out.index.equals(frame.index)
    assert list(out.index) == ["case-d", "case-c", "case-b", "case-a"]


def test_batch_does_not_mutate_input_frame_or_probability_array():
    frame = _tie_frame()
    snapshot = frame.copy(deep=True)
    probabilities = np.array([0.95, 0.70, 0.70, 0.02])
    probabilities_snapshot = probabilities.copy()
    score_opportunities(frame, probabilities)
    pd.testing.assert_frame_equal(frame, snapshot)
    np.testing.assert_array_equal(probabilities, probabilities_snapshot)


def test_batch_accepts_list_series_and_ndarray_identically():
    frame = _tie_frame()
    as_list = score_opportunities(frame, [0.95, 0.70, 0.70, 0.02])
    as_series = score_opportunities(frame, pd.Series([0.95, 0.70, 0.70, 0.02]))
    as_array = score_opportunities(frame, np.array([0.95, 0.70, 0.70, 0.02]))
    pd.testing.assert_frame_equal(as_list, as_series)
    pd.testing.assert_frame_equal(as_list, as_array)


@pytest.mark.parametrize(
    "wrong_length", [[], [0.5], [0.5, 0.6], [0.5, 0.6, 0.7], [0.5, 0.6, 0.7, 0.8, 0.9]]
)
def test_batch_length_mismatch_raises_value_error(wrong_length):
    with pytest.raises(ValueError, match="probabilit"):
        score_opportunities(_tie_frame(), wrong_length)


def test_batch_missing_required_columns_raise_value_error():
    no_amount = pd.DataFrame({"failure_category": ["temporary_decline"]})
    with pytest.raises(ValueError, match="amount_inr"):
        score_opportunities(no_amount, [0.5])
    no_category = pd.DataFrame({"amount_inr": [1000.0]})
    with pytest.raises(ValueError, match="failure_category"):
        score_opportunities(no_category, [0.5])


def test_batch_rank_is_dense_from_one_over_eligible_rows_only():
    out = score_opportunities(_tie_frame(), _TIE_PROBABILITIES)
    worth = out["worth_intervening"].to_numpy()
    assert int(worth.sum()) == 3
    ranked = out.loc[worth, "opportunity_rank"]
    assert sorted(ranked.tolist()) == [1, 2, 3]
    assert ranked.is_unique
    assert out["opportunity_rank"].dtype == "Int64"
    assert out.loc[~worth, "opportunity_rank"].isna().all()


def test_negative_erv_row_receives_no_rank_in_batch():
    out = score_opportunities(_tie_frame(), _TIE_PROBABILITIES)
    assert pd.isna(out["opportunity_rank"].iloc[3])
    assert not out["worth_intervening"].iloc[3]
    assert out["scoring_recommendation"].iloc[3] == NO_INTERVENTION


def test_value_dominates_probability_for_prioritization():
    frame = pd.DataFrame(
        {
            "amount_inr": [50000.0, 500.0],
            "failure_category": ["card_decline", "card_decline"],
        }
    )
    out = score_opportunities(frame, [0.55, 0.95])
    assert out["expected_recovery_value_inr"].tolist() == [27490.0, 465.0]
    assert out["opportunity_rank"].tolist() == [1, 2]
    assert out.loc[0, "recovery_probability"] < out.loc[1, "recovery_probability"]


def test_unknown_category_ranks_lower_than_identical_clean_row():
    frame = pd.DataFrame(
        {
            "amount_inr": [4000.0, 4000.0],
            "failure_category": ["temporary_decline", "unknown"],
        }
    )
    out = score_opportunities(frame, [0.60, 0.60])
    assert out["worth_intervening"].tolist() == [True, True]
    assert out["opportunity_rank"].tolist() == [1, 2]
    assert out.loc[0, "expected_recovery_value_inr"] == 2390.0
    assert out.loc[1, "expected_recovery_value_inr"] == 2190.0


def test_shuffled_rows_keep_identical_scores_and_stable_tie_ranks():
    frame = _tie_frame()
    base = score_opportunities(frame, _TIE_PROBABILITIES)
    permutation = [2, 0, 3, 1]
    moved_frame = frame.iloc[permutation]
    moved_probs = [_TIE_PROBABILITIES[position] for position in permutation]
    moved = score_opportunities(moved_frame, moved_probs)

    pd.testing.assert_frame_equal(base.sort_index(), moved.sort_index())

    assert base.loc[1, "expected_recovery_value_inr"] == base.loc[
        2, "expected_recovery_value_inr"
    ]
    assert base.loc[1, "opportunity_rank"] == 1
    assert base.loc[2, "opportunity_rank"] == 2
    assert moved.loc[1, "opportunity_rank"] == 1
    assert moved.loc[2, "opportunity_rank"] == 2


def test_tie_breaks_by_lexicographic_index_label_regardless_of_insertion():
    tied_rows = {
        "amount_inr": [3000.0, 3000.0],
        "failure_category": ["temporary_decline", "temporary_decline"],
    }
    probabilities = [0.70, 0.70]
    label_b_first = score_opportunities(
        pd.DataFrame(tied_rows, index=["case-b", "case-a"]), probabilities
    )
    assert label_b_first["expected_recovery_value_inr"].tolist() == [2090.0, 2090.0]
    assert label_b_first.loc["case-a", "opportunity_rank"] == 1
    assert label_b_first.loc["case-b", "opportunity_rank"] == 2
    label_a_first = score_opportunities(
        pd.DataFrame(tied_rows, index=["case-a", "case-b"]), probabilities
    )
    assert label_a_first.loc["case-a", "opportunity_rank"] == 1
    assert label_a_first.loc["case-b", "opportunity_rank"] == 2


def test_sub_paise_erv_differences_round_to_a_label_precedence_tie():
    frame = pd.DataFrame(
        {
            "amount_inr": [40.008, 40.004],
            "failure_category": ["temporary_decline", "temporary_decline"],
        },
        index=["row-b", "row-a"],
    )
    out = score_opportunities(frame, [0.5, 0.5])
    assert out["expected_recovery_value_inr"].tolist() == [10.0, 10.0]
    assert out["worth_intervening"].tolist() == [True, True]
    assert out["opportunity_rank"].tolist() == [2, 1]
    assert sorted(out["opportunity_rank"].dropna().tolist()) == [1, 2]


def test_mixed_type_index_labels_fail_loudly_with_named_value_error():
    frame = pd.DataFrame(
        {
            "amount_inr": [1000.0, 2000.0],
            "failure_category": ["temporary_decline", "temporary_decline"],
        },
        index=["a", 1],
    )
    with pytest.raises(ValueError) as excinfo:
        score_opportunities(frame, [0.9, 0.8])
    assert (
        str(excinfo.value)
        == "opportunity_rank requires uniformly typed index labels, found mixed types"
    )


def test_batch_is_deterministic_across_repeated_calls():
    frame = _tie_frame()
    first = score_opportunities(frame, _TIE_PROBABILITIES)
    second = score_opportunities(frame, _TIE_PROBABILITIES)
    pd.testing.assert_frame_equal(first, second)


def test_empty_frame_yields_empty_frame_with_exact_contract_columns():
    empty = pd.DataFrame({"amount_inr": [], "failure_category": []})
    out = score_opportunities(empty, [])
    assert len(out) == 0
    assert list(out.columns) == BATCH_COLUMNS
    assert out["opportunity_rank"].dtype == "Int64"


def test_zero_erv_row_in_batch_gets_no_rank():
    frame = pd.DataFrame(
        {
            "amount_inr": [20.0, 1000.0],
            "failure_category": ["temporary_decline", "temporary_decline"],
        }
    )
    out = score_opportunities(frame, [0.5, 0.8])
    assert out["expected_recovery_value_inr"].tolist() == [0.0, 790.0]
    assert out["worth_intervening"].tolist() == [False, True]
    assert pd.isna(out["opportunity_rank"].iloc[0])
    assert out["opportunity_rank"].tolist()[1] == 1


def test_batch_validation_errors_fire_before_any_scoring():
    frame = pd.DataFrame({"amount_inr": [1.0, 2.0]})
    with pytest.raises(ValueError):
        score_opportunities(frame, [0.5, 0.5, 0.5])
