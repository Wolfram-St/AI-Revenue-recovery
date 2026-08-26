"""Tests for Day 5 observation assembly + stratified chronological splits."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_matrix,
)
from ml.train import predict_recovery_probability, train_baseline
from recovery.policy import decide_action, load_policy_config
from simulation.config import load_treatment_policy
from simulation.observations import (
    assemble_observations,
    randomized_subset,
    safety_censored_count,
    split_observations,
)

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "observations.py"

ATTEMPT_ROWS = 600

NEW_COLUMNS = (
    "assigned_action",
    "arm_source",
    "assignment_probability",
    "simulated_recovered",
    "simulated_recovered_amount_inr",
    "base_recovery_propensity",
    "action_effect_logit",
    "propensity_under_assignment",
    "treatment_timestamp",
    "outcome_timestamp",
    "stratum",
)


@pytest.fixture(scope="module")
def assembled_bundle():
    """Full chain artifact: 600-row attempts -> baseline model -> observations."""
    attempts = generate_dataset(ATTEMPT_ROWS, seed=42).reset_index(drop=True)
    train_df, validation_df, _ = chronological_split(attempts)
    model, _metadata = train_baseline(train_df, validation_df, seed=42)
    probabilities = [
        float(value) for value in predict_recovery_probability(model, attempts)
    ]
    assembled = assemble_observations(attempts, probabilities, POLICY)
    return attempts, probabilities, assembled


# ---------------------------------------------------------------------------
# 1. Column contract: exact names/order, row identity preserved
# ---------------------------------------------------------------------------


def test_column_contract_is_exact_names_and_order(assembled_bundle):
    attempts, _, assembled = assembled_bundle

    assert list(assembled.columns) == list(attempts.columns) + list(NEW_COLUMNS)


def test_row_count_and_attempt_id_sequence_are_preserved(assembled_bundle):
    attempts, _, assembled = assembled_bundle

    assert len(assembled) == ATTEMPT_ROWS
    assert assembled["attempt_id"].tolist() == attempts["attempt_id"].tolist()


# ---------------------------------------------------------------------------
# 2. Stratum correctness vs independent decide_action recomputation
# ---------------------------------------------------------------------------


def test_stratum_matches_independent_decide_action_recomputation(assembled_bundle):
    attempts, probabilities, assembled = assembled_bundle
    policy_config = load_policy_config()

    expected = []
    for position in range(len(attempts)):
        context = dict(attempts.iloc[position])
        context["recovery_probability"] = probabilities[position]
        stopped = decide_action(context, policy_config).authorized_action == "STOP"
        expected.append("safety_censored" if stopped else "randomized")

    assert assembled["stratum"].tolist() == expected


def test_both_strata_present_at_fixture_scale(assembled_bundle):
    _, _, assembled = assembled_bundle

    counts = assembled["stratum"].value_counts()
    assert set(counts.index) == {"randomized", "safety_censored"}


# ---------------------------------------------------------------------------
# 3. Chronological splitting: strict ordering, fractions respected
# ---------------------------------------------------------------------------


def test_split_segments_are_strictly_chronological_and_sized(assembled_bundle):
    _, _, assembled = assembled_bundle

    train_df, validation_df, test_df = split_observations(assembled)

    assert (len(train_df), len(validation_df), len(test_df)) == (420, 90, 90)
    assert train_df["event_timestamp"].max() < validation_df["event_timestamp"].min()
    assert validation_df["event_timestamp"].max() < test_df["event_timestamp"].min()


def test_split_segments_partition_the_frame_without_overlap(assembled_bundle):
    _, _, assembled = assembled_bundle

    segments = split_observations(assembled)

    combined_ids = pd.concat([segment["attempt_id"] for segment in segments]).tolist()
    assert sorted(combined_ids) == sorted(assembled["attempt_id"].tolist())
    for segment in segments:
        assert list(segment.columns) == list(assembled.columns)


# ---------------------------------------------------------------------------
# 4. Determinism under identical inputs + policy
# ---------------------------------------------------------------------------


def test_two_assemblies_are_identical_including_timestamps(assembled_bundle):
    attempts, probabilities, first = assembled_bundle

    second = assemble_observations(attempts, probabilities, POLICY)

    assert first.equals(second)


# ---------------------------------------------------------------------------
# 5. Non-mutation of inputs
# ---------------------------------------------------------------------------


def test_attempts_frame_and_probabilities_are_never_mutated(assembled_bundle):
    attempts, probabilities, _ = assembled_bundle
    attempts_snapshot = attempts.copy(deep=True)
    probabilities_snapshot = list(probabilities)

    assemble_observations(attempts, probabilities, POLICY)

    pd.testing.assert_frame_equal(attempts, attempts_snapshot)
    assert attempts.dtypes.equals(attempts_snapshot.dtypes)
    assert probabilities == probabilities_snapshot


# ---------------------------------------------------------------------------
# 6. Empty behavior
# ---------------------------------------------------------------------------


def test_empty_attempts_yield_empty_result_with_exact_columns(assembled_bundle):
    attempts, _, _ = assembled_bundle
    empty = attempts.head(0)

    result = assemble_observations(empty, [], POLICY)

    assert len(result) == 0
    assert list(result.columns) == list(attempts.columns) + list(NEW_COLUMNS)


def test_split_observations_on_empty_result_returns_three_empties(assembled_bundle):
    attempts, _, _ = assembled_bundle
    result = assemble_observations(attempts.head(0), [], POLICY)

    segments = split_observations(result)

    assert len(segments) == 3
    for segment in segments:
        assert len(segment) == 0
        assert list(segment.columns) == list(result.columns)


# ---------------------------------------------------------------------------
# 7. Missing stratum column rejected loudly
# ---------------------------------------------------------------------------


def test_split_observations_requires_the_stratum_column(assembled_bundle):
    _, _, assembled = assembled_bundle

    with pytest.raises(ValueError, match="stratum"):
        split_observations(assembled.drop(columns=["stratum"]))


# ---------------------------------------------------------------------------
# 8. Leakage guard regression pin: whitelisted feature builder stays clean
# ---------------------------------------------------------------------------


def test_feature_builder_consumes_assembled_frame_without_leakage(assembled_bundle):
    _, _, assembled = assembled_bundle

    X, y = build_feature_matrix(assembled)

    assert len(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == 14
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert set(X.columns).isdisjoint(set(NEW_COLUMNS))
    assert y.tolist() == assembled["recovered"].astype(int).tolist()


# ---------------------------------------------------------------------------
# 9. Purity: import whitelist, no local randomness, no wall clock
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset({"__future__", "numpy", "pandas", "simulation", "data"})

FORBIDDEN_PATTERNS = (
    r"(?<![\w.])datetime\s*\.\s*now\b",
    r"(?<![\w.])time\s*\.",
    r"(?<![\w.])secrets?\b",
    r"(?<![\w.])uuid\b",
    r"(?<![\w.])random\b",
)


def _import_root_modules(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _source_without_docstring() -> str:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))
    assert docstring is not None
    assert source.count(docstring) == 1
    return source.replace(docstring, " ", 1)


def test_observations_import_roots_whitelisted_without_recovery():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )
    assert "recovery" not in roots, "assembly must not depend on recovery/*"


def test_no_rng_derivation_exists_in_source():
    code = _source_without_docstring()

    assert code.count("default_rng") == 0, (
        "randomness must live exclusively in the composed simulator stages"
    )


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_no_wall_clock_or_stdlib_randomness_token(pattern):
    code = _source_without_docstring()

    assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"


@pytest.mark.parametrize(
    "forbidden_substring",
    ["RandomState", "np.random.random"],
    ids=["randomstate", "np-random-random"],
)
def test_no_legacy_numpy_randomness_entry_points(forbidden_substring):
    code = _source_without_docstring()

    assert forbidden_substring not in code, (
        f"forbidden substring {forbidden_substring!r} found"
    )


# ---------------------------------------------------------------------------
# 10. randomized_subset / safety_censored_count hand checks
# ---------------------------------------------------------------------------


def test_randomized_subset_returns_exact_randomized_rows_as_a_copy():
    frame = pd.DataFrame(
        {
            "attempt_id": ["a", "b", "c"],
            "stratum": ["randomized", "safety_censored", "randomized"],
        }
    )

    subset = randomized_subset(frame)
    subset.loc[subset.index[0], "stratum"] = "tampered"

    assert subset["attempt_id"].tolist() == ["a", "c"]
    assert frame["stratum"].tolist() == [
        "randomized",
        "safety_censored",
        "randomized",
    ]


def test_safety_censored_count_matches_hand_count():
    frame = pd.DataFrame(
        {
            "attempt_id": ["a", "b", "c"],
            "stratum": ["randomized", "safety_censored", "randomized"],
        }
    )

    assert safety_censored_count(frame) == 1


def test_helpers_handle_empty_frames_with_stratum_column():
    empty = pd.DataFrame({"stratum": pd.Series(dtype="object")})

    assert len(randomized_subset(empty)) == 0
    assert safety_censored_count(empty) == 0


@pytest.mark.parametrize(
    "helper",
    [randomized_subset, safety_censored_count],
    ids=["randomized_subset", "safety_censored_count"],
)
def test_helpers_require_the_stratum_column(helper):
    frame = pd.DataFrame({"attempt_id": ["a", "b"]})

    with pytest.raises(ValueError, match="stratum"):
        helper(frame)


# ---------------------------------------------------------------------------
# 11. Delegated probability validation surfaces cleanly at this layer
# ---------------------------------------------------------------------------


def test_wrong_length_probabilities_surface_from_the_delegated_validator(
    assembled_bundle,
):
    attempts, _, _ = assembled_bundle

    with pytest.raises(ValueError, match="does not match df row count"):
        assemble_observations(attempts, [0.5] * (ATTEMPT_ROWS + 1), POLICY)


def test_non_finite_probability_surfaces_from_the_delegated_validator(
    assembled_bundle,
):
    attempts, _, _ = assembled_bundle
    probabilities = [0.5] * ATTEMPT_ROWS
    probabilities[ATTEMPT_ROWS // 2] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        assemble_observations(attempts, probabilities, POLICY)
