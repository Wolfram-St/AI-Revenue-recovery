"""Tests for Day 4 two-stage treatment assignment (plan Task 2, decisions D1/D1b)."""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.train import predict_recovery_probability, train_baseline
from recovery.policy import decide_action, load_policy_config
from simulation.config import CANONICAL_ARMS, SEED_STREAM_ASSIGNMENT, load_treatment_policy
from simulation.treatment import assign_treatments

RESULT_COLUMNS = ["assigned_action", "arm_source", "assignment_probability"]
POLICY = load_treatment_policy("config/treatment_policy.yaml")
POLICY_CONFIG = load_policy_config()
CONTEXT_COLUMNS = [
    "customer_opted_out",
    "fraud_risk",
    "failure_category",
    "attempt_number",
    "amount_inr",
]
SOURCE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "treatment.py"


def eligible_frame(n_rows: int, index=None, **column_overrides) -> pd.DataFrame:
    """Synthetic frame whose rows all pass the stage-1 gate at p=0.5."""
    columns = {
        "customer_opted_out": np.full(n_rows, False),
        "fraud_risk": np.full(n_rows, False),
        "failure_category": np.full(n_rows, "temporary_decline"),
        "attempt_number": np.ones(n_rows, dtype=int),
        "amount_inr": np.full(n_rows, 5000.0),
    }
    columns.update(column_overrides)
    return pd.DataFrame(columns, index=index)


# ---------------------------------------------------------------------------
# 1. Safety censoring: every STOP rule forces CONTROL / safety_censored / 0.0
# ---------------------------------------------------------------------------


def _stop_case(name: str) -> tuple[pd.DataFrame, list[float]]:
    n_rows = 12
    frame = eligible_frame(n_rows)
    probabilities = [0.5] * n_rows
    if name == "fraud_risk":
        frame["fraud_risk"] = True
    elif name == "customer_opted_out":
        frame["customer_opted_out"] = True
    elif name == "hard_decline":
        frame["failure_category"] = np.full(n_rows, "hard_decline")
    elif name == "retry_limit":
        frame["attempt_number"] = np.full(n_rows, 4)
    elif name == "low_probability":
        probabilities = [0.19] * n_rows
    else:
        raise AssertionError(f"unknown stop case {name}")
    return frame, probabilities


@pytest.mark.parametrize(
    "case",
    ["fraud_risk", "customer_opted_out", "hard_decline", "retry_limit", "low_probability"],
)
def test_stop_rows_are_forced_safety_censored_controls(case):
    frame, probabilities = _stop_case(case)

    result = assign_treatments(frame, probabilities, POLICY)

    assert list(result.columns) == RESULT_COLUMNS
    assert (result["assigned_action"] == "CONTROL").all()
    assert (result["arm_source"] == "safety_censored").all()
    assert (result["assignment_probability"] == 0.0).all()


def test_result_index_matches_input_index():
    frame, probabilities = _stop_case("fraud_risk")
    custom_index = [1000 + 7 * position for position in range(len(frame))]
    frame.index = custom_index

    result = assign_treatments(frame, probabilities, POLICY)

    assert result.index.equals(frame.index)


# ---------------------------------------------------------------------------
# 2. Eligible rows only ever draw from the configured arm set
# ---------------------------------------------------------------------------


def test_eligible_rows_receive_only_configured_randomized_arms():
    n_rows = 400
    categories = np.array(
        ["temporary_decline", "payment_method_issue", "authentication_required", "unknown"]
    )
    frame = eligible_frame(
        n_rows,
        failure_category=categories[np.arange(n_rows) % len(categories)],
    )
    probabilities = [0.5] * n_rows

    result = assign_treatments(frame, probabilities, POLICY)

    assert (result["arm_source"] == _randomized_source()).all()
    drawn = set(result["assigned_action"])
    assert drawn <= set(POLICY.arm_probabilities)
    assert "STOP" not in drawn
    assert drawn <= set(CANONICAL_ARMS)
    recorded = result["assignment_probability"].to_numpy()
    expected = np.array([POLICY.arm_probabilities[arm] for arm in result["assigned_action"]])
    assert np.array_equal(recorded, expected)


def _randomized_source() -> str:
    return "randomized"


# ---------------------------------------------------------------------------
# 3. Empirical arm frequencies match configured probabilities (n=20000)
# ---------------------------------------------------------------------------


def test_empirical_arm_frequencies_within_tolerance():
    n_rows = 20000
    frame = eligible_frame(n_rows)
    probabilities = [0.5] * n_rows

    result = assign_treatments(frame, probabilities, POLICY)

    counts = result["assigned_action"].value_counts()
    assert int(counts.sum()) == n_rows
    for arm, configured in POLICY.arm_probabilities.items():
        observed = float(counts.get(arm, 0)) / n_rows
        assert abs(observed - configured) <= 0.03, (
            f"arm {arm}: observed {observed:.4f} vs configured {configured}"
        )


# ---------------------------------------------------------------------------
# 4. Positivity holds from config and on every eligible row
# ---------------------------------------------------------------------------


def test_positivity_config_and_recorded_probabilities():
    for arm in sorted(CANONICAL_ARMS):
        assert arm in POLICY.arm_probabilities
        assert POLICY.arm_probabilities[arm] > 0.0

    n_rows = 800
    frame = eligible_frame(n_rows)
    result = assign_treatments(frame, [0.5] * n_rows, POLICY)

    assert (result["assignment_probability"] > 0.0).all()
    assert (result["assignment_probability"] <= 1.0).all()


# ---------------------------------------------------------------------------
# 5. Determinism under fixed seed; different seeds differ stochastically
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_frames():
    n_rows = 600
    frame = eligible_frame(n_rows)
    probabilities = [0.42] * n_rows

    first = assign_treatments(frame, probabilities, POLICY)
    second = assign_treatments(frame, probabilities, POLICY)

    assert first.equals(second)


def test_different_master_seed_changes_assignment_column():
    n_rows = 500
    frame = eligible_frame(n_rows)
    probabilities = [0.42] * n_rows
    mutated_policy = dataclasses.replace(POLICY, master_seed=POLICY.master_seed + 1)

    baseline = assign_treatments(frame, probabilities, POLICY)
    alternative = assign_treatments(frame, probabilities, mutated_policy)

    assert (baseline["assigned_action"].to_numpy() != alternative["assigned_action"].to_numpy()).any()


# ---------------------------------------------------------------------------
# 6. Loud validation; input df and probabilities never mutated
# ---------------------------------------------------------------------------


def test_length_mismatch_rejected():
    frame = eligible_frame(5)
    with pytest.raises(ValueError):
        assign_treatments(frame, [0.5] * 4, POLICY)
    with pytest.raises(ValueError):
        assign_treatments(frame, [0.5] * 6, POLICY)


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf"), 1.5, -0.1, True, "0.5"],
)
def test_invalid_probability_values_rejected(bad_value):
    frame = eligible_frame(5)
    bad_probabilities = [0.5] * 5
    bad_probabilities[2] = bad_value

    with pytest.raises(ValueError):
        assign_treatments(frame, bad_probabilities, POLICY)


def test_inputs_not_mutated_by_valid_or_invalid_calls():
    frame = eligible_frame(5)
    snapshot = frame.copy(deep=True)
    probabilities = [0.25, 0.5, 0.75, 0.9, 0.2]
    probability_snapshot = list(probabilities)

    invalid_probabilities = [0.5] * 5
    invalid_probabilities[1] = True
    with pytest.raises(ValueError):
        assign_treatments(frame, invalid_probabilities, POLICY)

    assign_treatments(frame, probabilities, POLICY)

    pd.testing.assert_frame_equal(frame, snapshot)
    assert frame.dtypes.equals(snapshot.dtypes)
    assert probabilities == probability_snapshot


# ---------------------------------------------------------------------------
# 6b. NaN context guard protects the safety gate
# ---------------------------------------------------------------------------


def test_nan_attempt_number_rejected_naming_column():
    frame = eligible_frame(6, attempt_number=np.array([1.0] * 6))
    frame.loc[2, "attempt_number"] = np.nan

    with pytest.raises(ValueError) as excinfo:
        assign_treatments(frame, [0.5] * 6, POLICY)

    assert "attempt_number" in str(excinfo.value)


def test_nan_guard_names_every_offending_column_and_row_count():
    frame = eligible_frame(
        8,
        attempt_number=np.array([1.0, 1.0, np.nan, 1.0, 1.0, np.nan, 1.0, 1.0]),
        amount_inr=np.array([5000.0] * 7 + [np.nan]),
    )

    with pytest.raises(ValueError) as excinfo:
        assign_treatments(frame, [0.5] * 8, POLICY)

    message = str(excinfo.value)
    assert "attempt_number" in message
    assert "amount_inr" in message
    assert "2" in message
    assert "1" in message
    assert "safety censoring" in message


def test_clean_frames_pass_the_nan_guard():
    frame = eligible_frame(5)

    result = assign_treatments(frame, [0.5] * 5, POLICY)

    assert len(result) == 5
    assert (result["arm_source"] == "randomized").all()


def test_absent_gate_columns_are_skipped_by_the_guard():
    minimal = pd.DataFrame({"failure_category": ["temporary_decline"] * 3})

    with pytest.raises(KeyError):
        assign_treatments(minimal, [0.5] * 3, POLICY)


# ---------------------------------------------------------------------------
# 7. Empty df -> empty valid result
# ---------------------------------------------------------------------------


def test_empty_frame_yields_empty_result_with_correct_columns():
    empty = pd.DataFrame({column: pd.Series(dtype=object) for column in CONTEXT_COLUMNS})

    result = assign_treatments(empty, [], POLICY)

    assert len(result) == 0
    assert list(result.columns) == RESULT_COLUMNS
    assert result.index.equals(empty.index)
    assert result["assignment_probability"].dtype == np.dtype("float64")


# ---------------------------------------------------------------------------
# 8. Integration honesty vs an independent decide_action recomputation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_model_artifacts() -> tuple[pd.DataFrame, np.ndarray]:
    train_df, validation_df, _ = chronological_split(generate_dataset(600, seed=42))
    model, _metadata = train_baseline(train_df, validation_df, seed=42)
    probabilities = predict_recovery_probability(model, validation_df)
    return validation_df.reset_index(drop=True), probabilities


def test_real_model_run_matches_independent_stop_computation(real_model_artifacts):
    frame, probabilities = real_model_artifacts

    result = assign_treatments(frame, list(probabilities), POLICY)

    independent_stop = []
    for position in range(len(frame)):
        context = dict(frame.iloc[position])
        context["recovery_probability"] = float(probabilities[position])
        decision = decide_action(context, POLICY_CONFIG)
        independent_stop.append(decision.authorized_action == "STOP")

    censored_mask = result["arm_source"] == "safety_censored"
    randomized_mask = ~censored_mask

    assert int(censored_mask.sum()) == sum(independent_stop)
    assert int(censored_mask.sum()) > 0
    assert (result.loc[censored_mask, "assigned_action"] == "CONTROL").all()
    assert (result.loc[censored_mask, "assignment_probability"] == 0.0).all()
    randomized_stop_flags = np.asarray(independent_stop, dtype=bool)[
        randomized_mask.to_numpy()
    ]
    assert not randomized_stop_flags.any()
    assert (result.loc[randomized_mask, "arm_source"] == "randomized").all()
    assert set(result.loc[randomized_mask, "assigned_action"]) <= set(POLICY.arm_probabilities)


# ---------------------------------------------------------------------------
# 9. Purity: import whitelist + forbidden nondeterminism tokens
# ---------------------------------------------------------------------------


def _import_root_modules(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "math", "numpy", "pandas", "recovery", "simulation"}
)


def test_simulation_treatment_import_roots_whitelisted():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )


def test_source_has_no_wall_clock_or_stdlib_randomness_tokens():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    for token in ("datetime.now", "time.time", "time.", "secrets", "uuid"):
        assert token not in source, f"forbidden token {token!r} found"
    # Word-boundary match exempting attribute access: catches `import random`,
    # `random.seed(...)`, and __import__("random") while leaving both the
    # contract vocabulary arm_source == "randomized" and the mandated
    # np.random.default_rng spawn call untouched.
    assert re.search(r"(?<![\w.])random\b", source) is None


def test_seed_stream_child_used_exactly_once_via_spawn():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert source.count("SEED_STREAM_ASSIGNMENT") >= 1
    assert source.count("datetime.now") == 0
    assert source.count("time.time") == 0
    occurrences = [match.start() for match in re.finditer(r"np\.random\.default_rng", source)]
    assert len(occurrences) == 1
    window = source[occurrences[0]:]
    spawn_match = re.search(r"np\.random\.default_rng\([^)]*\)\s*\.\s*spawn\s*\(", window)
    assert spawn_match is not None, "the single default_rng call must be followed by .spawn"
    assert len(re.findall(r"default_rng\s*\(", source)) == 1
