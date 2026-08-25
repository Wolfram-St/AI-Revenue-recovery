"""Tests for Day 4 simulated action-aware outcomes (plan Task 3, decisions D1b/D2/D3)."""

from __future__ import annotations

import ast
import dataclasses
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulation import outcomes as outcomes_module
from simulation.config import CANONICAL_ARMS, load_treatment_policy
from simulation.outcomes import RESULT_COLUMNS, simulate_outcomes

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "outcomes.py"

ARM_CYCLE = ("CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")
CONSUMED_COLUMNS = (
    "amount_inr",
    "attempt_number",
    "assigned_action",
    "device_type",
    "failure_category",
    "fraud_risk",
    "historical_recovery_count",
    "payment_method",
    "successful_payment_count",
)
CATEGORIES = (
    "temporary_decline",
    "payment_method_issue",
    "authentication_required",
    "unknown",
    "hard_decline",
)
METHODS = ("card", "upi", "netbanking", "wallet")
DEVICES = ("android", "ios", "web")


def balanced_frame(n_rows: int) -> pd.DataFrame:
    """Decision-time covariate cycles perfectly balanced across the five arms.

    ``assigned_action`` is set directly (allowed: ``simulate_outcomes`` consumes
    only that assignment-metadata column). The arm cycle advances every five
    rows so it stays independent of the per-row category/method/device cycles.
    The fraud cycle uses period 47 -- coprime to the 25-row arm block period --
    so flagged rows reach every arm instead of aliasing into one arm (the
    original %50 period is a multiple of 25 and collapsed fraud onto
    HUMAN_REVIEW).
    """
    positions = np.arange(n_rows)
    return pd.DataFrame(
        {
            "amount_inr": 3000.0 + (positions % 7) * 250.0,
            "failure_category": np.array(CATEGORIES, dtype=object)[positions % 5],
            "attempt_number": (positions % 4) + 1,
            "successful_payment_count": positions % 5,
            "historical_recovery_count": positions % 7,
            "fraud_risk": (positions % 47) == 46,
            "payment_method": np.array(METHODS, dtype=object)[positions % 4],
            "device_type": np.array(DEVICES, dtype=object)[positions % 3],
            "assigned_action": np.array(ARM_CYCLE, dtype=object)[(positions // 5) % 5],
        }
    )


def analytic_base_logit(frame: pd.DataFrame) -> np.ndarray:
    """Recompute the documented base logit family straight from policy terms."""
    terms = POLICY.base_propensity_terms
    return (
        float(terms.intercept)
        + frame["failure_category"].map(terms.category_effects).to_numpy(dtype=float)
        + float(terms.successful_payment_count_log1p)
        * np.log1p(frame["successful_payment_count"].to_numpy(dtype=float))
        + float(terms.historical_recovery_count_min5)
        * np.minimum(frame["historical_recovery_count"].to_numpy(dtype=float), 5.0)
        + float(terms.attempt_number_prior_offset)
        * np.maximum(frame["attempt_number"].to_numpy(dtype=float) - 1.0, 0.0)
        + float(terms.fraud_risk) * frame["fraud_risk"].astype(float).to_numpy()
        + float(terms.amount_log1p_per_k)
        * np.log1p(frame["amount_inr"].to_numpy(dtype=float) / 1000.0)
        + float(terms.method_upi) * (frame["payment_method"] == "upi").astype(float).to_numpy()
        + float(terms.device_android) * (frame["device_type"] == "android").astype(float).to_numpy()
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def configured_interaction_effect(action: str, column: str, equals_value: str) -> float:
    return math.fsum(
        float(rule.effect_logit)
        for rule in POLICY.interactions
        if rule.action == action
        and rule.column == column
        and rule.equals_value == equals_value
    )


@pytest.fixture(scope="module")
def monte_carlo_run() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = balanced_frame(30000)
    # F1 structural guard: the arm x fraud crosstab must show flagged rows in
    # every arm, so no directional comparison below is fraud-confounded.
    fraud_by_arm = pd.crosstab(frame["assigned_action"], frame["fraud_risk"])
    assert sorted(fraud_by_arm.index) == sorted(ARM_CYCLE)
    assert (fraud_by_arm[True] > 0).all(), (
        "fixture regression: fraud_risk cycle aliases into a subset of arms: "
        f"{fraud_by_arm[True].to_dict()}"
    )
    return frame, simulate_outcomes(frame, POLICY)


# ---------------------------------------------------------------------------
# 1. CONTROL observed mean ≈ analytic mean(sigmoid(base_logit)), n=30000
# ---------------------------------------------------------------------------


def test_control_mean_matches_analytic_base_propensity(monte_carlo_run):
    frame, result = monte_carlo_run
    control_mask = frame["assigned_action"].to_numpy() == "CONTROL"
    expected_mean = float(_sigmoid(analytic_base_logit(frame))[control_mask].mean())
    observed_mean = float(result["simulated_recovered"].to_numpy()[control_mask].mean())

    # Realized gap under master_seed 20260826: 0.0059 (tolerance 0.02).
    assert abs(observed_mean - expected_mean) <= 0.02, (
        f"observed control rate {observed_mean:.4f} vs analytic "
        f"{expected_mean:.4f}"
    )

    # Plumbing beyond statistics: stored pre-noise propensity reproduces the
    # analytic curve up to the documented 6-decimal rounding.
    propensity_mean = float(
        result["propensity_under_assignment"].to_numpy()[control_mask].mean()
    )
    assert abs(propensity_mean - expected_mean) <= 1e-6


# ---------------------------------------------------------------------------
# 2. Direction/magnitude sanity: known effects show up in observed rates
# ---------------------------------------------------------------------------


def test_observed_rates_move_with_known_effects_and_interaction(monte_carlo_run):
    frame, result = monte_carlo_run
    recovered = result["simulated_recovered"].to_numpy()

    def rate(arm: str) -> float:
        return float(recovered[(frame["assigned_action"] == arm).to_numpy()].mean())

    def arm_category_rate(arm: str, category: str) -> float:
        mask = ((frame["assigned_action"] == arm) & (frame["failure_category"] == category)).to_numpy()
        return float(recovered[mask].mean())

    # Realized margins are deterministic under master_seed 20260826 and this
    # fixture; comments record them so future edits notice drift.
    assert rate("RETRY_NOW") > rate("CONTROL"), "main effect +0.60 must lift RETRY_NOW"  # realized +0.133
    assert rate("RETRY_LATER") > rate("CONTROL"), "main effect +0.35 must lift RETRY_LATER"  # realized +0.038
    assert rate("REQUEST_UPDATE") > rate("HUMAN_REVIEW"), (
        "effect ordering 0.45 > 0.25 must hold"
    )  # realized +0.046

    # Matched other-columns inside one arm: temporary_decline benefits from
    # the +0.40 RETRY_NOW interaction on top of its category main effect.
    retry_temporary = arm_category_rate("RETRY_NOW", "temporary_decline")
    retry_unknown = arm_category_rate("RETRY_NOW", "unknown")
    assert retry_temporary > retry_unknown, "interaction/category lift not visible"  # realized +0.290

    # Difference-in-differences isolates the interaction itself: the
    # RETRY_NOW-minus-REQUEST_UPDATE gap must be larger for temporary_decline
    # (where the rule fires) than for unknown (where it cannot fire).
    update_temporary = arm_category_rate("REQUEST_UPDATE", "temporary_decline")
    update_unknown = arm_category_rate("REQUEST_UPDATE", "unknown")
    difference_in_differences = (retry_temporary - update_temporary) - (
        retry_unknown - update_unknown
    )
    assert difference_in_differences > 0.0, (
        f"interaction DiD {difference_in_differences:.4f} should be positive"
    )  # realized +0.086


# ---------------------------------------------------------------------------
# 3. Ground-truth exactness on crafted rows (one interaction firing)
# ---------------------------------------------------------------------------


def test_ground_truth_columns_match_hand_computation():
    crafted = pd.DataFrame(
        {
            "amount_inr": [2000.0, 2000.0],
            "failure_category": ["temporary_decline", "temporary_decline"],
            "attempt_number": [2, 2],
            "successful_payment_count": [3, 3],
            "historical_recovery_count": [2, 2],
            "fraud_risk": [True, True],
            "payment_method": ["upi", "upi"],
            "device_type": ["android", "android"],
            "assigned_action": ["RETRY_NOW", "CONTROL"],
        }
    )
    terms = POLICY.base_propensity_terms
    base_logit = (
        float(terms.intercept)
        + float(terms.category_effects["temporary_decline"])
        + float(terms.successful_payment_count_log1p) * math.log1p(3.0)
        + float(terms.historical_recovery_count_min5) * min(2, 5)
        + float(terms.attempt_number_prior_offset) * max(2 - 1, 0)
        + float(terms.fraud_risk) * 1.0
        + float(terms.amount_log1p_per_k) * math.log1p(2000.0 / 1000.0)
        + float(terms.method_upi) * 1.0
        + float(terms.device_android) * 1.0
    )
    effect_logit = float(POLICY.main_effects_logit["RETRY_NOW"]) + configured_interaction_effect(
        "RETRY_NOW", "failure_category", "temporary_decline"
    )
    assert effect_logit > float(POLICY.main_effects_logit["RETRY_NOW"]), (
        "crafted row must actually fire the interaction"
    )
    sigmoid = lambda value: 1.0 / (1.0 + math.exp(-value))

    result = simulate_outcomes(crafted, POLICY)

    assert result["base_recovery_propensity"].iloc[0] == pytest.approx(
        round(sigmoid(base_logit), 6), abs=1e-9
    )
    assert result["action_effect_logit"].iloc[0] == pytest.approx(
        round(effect_logit, 6), abs=1e-9
    )
    assert result["propensity_under_assignment"].iloc[0] == pytest.approx(
        round(sigmoid(base_logit + effect_logit), 6), abs=1e-9
    )
    assert int(result["simulated_recovered"].iloc[0]) in (0, 1)

    # CONTROL carries exactly zero effect: pre-noise propensity equals base.
    assert result["action_effect_logit"].iloc[1] == 0.0
    assert result["propensity_under_assignment"].iloc[1] == (
        result["base_recovery_propensity"].iloc[1]
    )


# ---------------------------------------------------------------------------
# 3b. Numerical safety: overflow-stable sigmoid + loud logit-magnitude guard
# ---------------------------------------------------------------------------


def test_pathological_base_terms_raise_instead_of_silent_zeroes():
    extreme_terms = dataclasses.replace(
        POLICY.base_propensity_terms, amount_log1p_per_k=-1000.0
    )
    pathological_policy = dataclasses.replace(
        POLICY, base_propensity_terms=extreme_terms
    )
    frame = balanced_frame(2000)
    frame["amount_inr"] = np.linspace(1000.0, 1_000_000.0, len(frame))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError) as excinfo:
            simulate_outcomes(frame, pathological_policy)

    message = str(excinfo.value)
    assert "representable synthetic range" in message
    assert "base_propensity_terms" in message


def test_stable_sigmoid_survives_extreme_inputs_without_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        saturated = outcomes_module._sigmoid(np.array([-800.0, 800.0]))
        moderate = outcomes_module._sigmoid(np.array([-30.0, 30.0]))

    # At +/-800 binary64 exp underflows to exact 0.0/1.0 saturation -- the
    # guarantee there is no overflow warning and bounded output. Strict
    # (0, 1) membership is asserted at +/-30: beyond |x| ~ 37 the nearest
    # float64 to the true value IS 1.0 (spacing 1.1e-16), so no honest
    # implementation can return a value strictly inside the unit interval.
    assert saturated[0] == 0.0 and saturated[1] == 1.0
    assert np.isfinite(saturated).all()
    assert 0.0 < moderate[0] < 1.0 and 0.0 < moderate[1] < 1.0


def test_shipped_configuration_run_emits_no_runtime_warnings():
    frame = balanced_frame(500)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = simulate_outcomes(frame, POLICY)

    assert len(result) == len(frame)


# ---------------------------------------------------------------------------
# 4. Boundaries and dtypes
# ---------------------------------------------------------------------------


def test_propensities_strictly_bounded_and_labels_binary(monte_carlo_run):
    _, result = monte_carlo_run

    for column in ("base_recovery_propensity", "propensity_under_assignment"):
        series = result[column]
        assert (series > 0.0).all(), f"{column} hit the lower boundary"
        assert (series < 1.0).all(), f"{column} hit the upper boundary"
        assert series.dtype == np.dtype("float64")

    assert result["simulated_recovered"].isin([0, 1]).all()
    assert result["simulated_recovered"].dtype == np.dtype("int8")
    assert result.notna().all().all()


# ---------------------------------------------------------------------------
# 5. Determinism: same seed identical; different master seed differs
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_frames():
    frame = balanced_frame(500)

    first = simulate_outcomes(frame, POLICY)
    second = simulate_outcomes(frame, POLICY)

    assert first.equals(second)


def test_different_master_seed_changes_draws_not_truth():
    frame = balanced_frame(500)
    mutated_policy = dataclasses.replace(POLICY, master_seed=POLICY.master_seed + 1)

    baseline = simulate_outcomes(frame, POLICY)
    alternative = simulate_outcomes(frame, mutated_policy)

    assert (
        baseline["simulated_recovered"].to_numpy()
        != alternative["simulated_recovered"].to_numpy()
    ).any(), "different master_seed must resample the Bernoulli draws"
    # Ground truth is a function of context and policy numbers, never of draws.
    for column in (
        "base_recovery_propensity",
        "action_effect_logit",
        "propensity_under_assignment",
    ):
        assert baseline[column].equals(alternative[column]), column


# ---------------------------------------------------------------------------
# 6. Loud validation; inputs never mutated
# ---------------------------------------------------------------------------


def test_missing_required_columns_rejected_naming_each():
    frame = balanced_frame(8)

    with pytest.raises(ValueError) as excinfo:
        simulate_outcomes(frame.drop(columns=["payment_method"]), POLICY)
    assert "payment_method" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        simulate_outcomes(frame.drop(columns=["amount_inr", "device_type"]), POLICY)
    message = str(excinfo.value)
    assert "amount_inr" in message
    assert "device_type" in message


@pytest.mark.parametrize("column", ["amount_inr", "failure_category", "assigned_action"])
def test_nan_in_consumed_column_rejected_naming_it(column):
    frame = balanced_frame(8)
    frame.loc[3, column] = np.nan

    with pytest.raises(ValueError) as excinfo:
        simulate_outcomes(frame, POLICY)

    assert column in str(excinfo.value)


def test_unknown_assigned_actions_rejected_naming_offenders():
    frame = balanced_frame(8)
    frame.loc[0, "assigned_action"] = "STOP"
    frame.loc[5, "assigned_action"] = "FOO"

    with pytest.raises(ValueError) as excinfo:
        simulate_outcomes(frame, POLICY)

    message = str(excinfo.value)
    assert "STOP" in message
    assert "FOO" in message


def test_typoed_failure_category_rejected_naming_the_value():
    frame = balanced_frame(6)
    frame.loc[1, "failure_category"] = "tempory_declne"

    with pytest.raises(ValueError) as excinfo:
        simulate_outcomes(frame, POLICY)

    assert "tempory_declne" in str(excinfo.value)


def test_inputs_not_mutated_by_valid_or_invalid_calls():
    frame = balanced_frame(6)
    snapshot = frame.copy(deep=True)
    corrupted = frame.copy()
    corrupted.loc[1, "assigned_action"] = "STOP"

    with pytest.raises(ValueError):
        simulate_outcomes(corrupted, POLICY)

    simulate_outcomes(frame, POLICY)

    pd.testing.assert_frame_equal(frame, snapshot)
    assert frame.dtypes.equals(snapshot.dtypes)


# ---------------------------------------------------------------------------
# 7. Empty df -> empty valid result
# ---------------------------------------------------------------------------


def test_empty_frame_yields_empty_result_with_correct_schema():
    empty = pd.DataFrame(
        {
            "amount_inr": pd.Series(dtype="float64"),
            "failure_category": pd.Series(dtype="object"),
            "attempt_number": pd.Series(dtype="int64"),
            "successful_payment_count": pd.Series(dtype="int64"),
            "historical_recovery_count": pd.Series(dtype="int64"),
            "fraud_risk": pd.Series(dtype="bool"),
            "payment_method": pd.Series(dtype="object"),
            "device_type": pd.Series(dtype="object"),
            "assigned_action": pd.Series(dtype="object"),
        }
    )

    result = simulate_outcomes(empty, POLICY)

    assert len(result) == 0
    assert list(result.columns) == list(RESULT_COLUMNS)
    assert result.index.equals(empty.index)
    assert result["simulated_recovered"].dtype == np.dtype("int8")
    for column in (
        "base_recovery_propensity",
        "action_effect_logit",
        "propensity_under_assignment",
    ):
        assert result[column].dtype == np.dtype("float64")


# ---------------------------------------------------------------------------
# 8. Purity: import whitelist, single spawned child stream, no wall clock
# ---------------------------------------------------------------------------


def _import_root_modules(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


ALLOWED_IMPORT_ROOTS = frozenset({"__future__", "math", "numpy", "pandas", "simulation"})


def _source_without_docstring() -> str:
    """Executable-source view: token counts ignore the module docstring."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))
    assert docstring is not None
    assert source.count(docstring) == 1
    return source.replace(docstring, " ", 1)


def test_simulation_outcomes_import_roots_whitelisted_without_recovery():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )
    assert "recovery" not in roots, "outcome generation must not depend on recovery/*"


def test_seed_stream_child_used_exactly_once_via_spawn():
    code = _source_without_docstring()

    occurrences = [match.start() for match in re.finditer(r"np\.random\.default_rng", code)]
    assert len(occurrences) == 1, "exactly one master-seed rng derivation allowed"
    spawn_pattern = (
        r"np\.random\.default_rng\(\s*policy\.master_seed\s*\)\s*\.\s*spawn\s*\("
        r"\s*SEED_STREAM_OUTCOMES\s*\+\s*1\s*\)\s*\[\s*SEED_STREAM_OUTCOMES\s*\]"
    )
    assert re.search(spawn_pattern, code) is not None, (
        "the single derivation must be default_rng(policy.master_seed).spawn("
        "SEED_STREAM_OUTCOMES + 1)[SEED_STREAM_OUTCOMES]"
    )


def test_source_has_no_wall_clock_or_stdlib_randomness_tokens():
    code = _source_without_docstring()

    # Word-bounded like the Task 2 purity tests so attribute access such as
    # np.random.default_rng or outcome_rng.random never trips the scan while
    # genuine stdlib-randomness usage still fails loudly.
    forbidden_patterns = (
        r"(?<![\w.])datetime\b",
        r"(?<![\w.])time\s*\.",
        r"(?<![\w.])random\b",
        r"(?<![\w.])secrets?\b",
        r"(?<![\w.])uuid\b",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"


# ---------------------------------------------------------------------------
# 9. Statistical honesty in language
# ---------------------------------------------------------------------------


def test_module_language_never_claims_causal_estimates():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "causal estimate" not in source
    assert "SIMULATED" in source
    docstring = ast.get_docstring(ast.parse(source))
    assert "SYNTHETIC GROUND TRUTH" in docstring
    assert "decision-time" in docstring
