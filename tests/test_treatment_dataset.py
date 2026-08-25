"""Tests for the Day 4 treatment/outcome dataset contract (plan Task 5, decision D5)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.generate_dataset import TIME_COLUMN, generate_dataset
from data.splits import chronological_split
from ml.features import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURES,
    NUMERIC_FEATURES,
    build_feature_matrix,
)
from ml.train import predict_recovery_probability, train_baseline
from simulation.config import CANONICAL_ARMS, load_treatment_policy
from simulation.dataset import (
    CANONICAL_COLUMNS,
    FIELD_CLASSIFICATION,
    build_treatment_dataset,
    dataset_contract,
    validate_treatment_dataset,
)
from simulation.outcomes import RESULT_COLUMNS as OUTCOME_RESULT_COLUMNS
from simulation.outcomes import simulate_outcomes
from simulation.treatment import RESULT_COLUMNS as ASSIGNMENT_RESULT_COLUMNS
from simulation.treatment import assign_treatments

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "dataset.py"

CANONICAL_COLUMN_ORDER = [
    "attempt_id",
    "event_timestamp",
    "assigned_action",
    "arm_source",
    "assignment_probability",
    "treatment_timestamp",
    "outcome_timestamp",
    "simulated_recovered",
    "simulated_recovered_amount_inr",
    "base_recovery_propensity",
    "action_effect_logit",
    "propensity_under_assignment",
]

CLASSIFICATION_VOCABULARY = {
    "identifier",
    "decision_time_feature",
    "treatment_metadata",
    "outcome_timing",
    "outcome",
    "ground_truth",
    "evaluation_metadata",
}

# Fields whose presence in a model feature matrix would be leakage. The
# "decision_time_feature" class is excluded because context legitimately stays
# in the attempts frame -- though today NO field carries that class.
LEAKAGE_FIELDS = frozenset(
    name
    for name, classification in FIELD_CLASSIFICATION.items()
    if classification != "decision_time_feature"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def handcrafted_frames(n_rows: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Minimal valid (attempts, assignments, outcomes) trio, all randomized."""
    arms = ["CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW"]
    probabilities = [0.20, 0.30, 0.25, 0.15, 0.10]
    ids = [f"ATT-{position:06d}" for position in range(n_rows)]
    return (
        pd.DataFrame(
            {
                "attempt_id": ids,
                "event_timestamp": pd.date_range(
                    "2026-01-01T00:00:00Z", periods=n_rows, freq="15min"
                ),
            }
        ),
        pd.DataFrame(
            {
                "attempt_id": ids,
                "assigned_action": arms[:n_rows],
                "arm_source": ["randomized"] * n_rows,
                "assignment_probability": probabilities[:n_rows],
            },
            index=pd.RangeIndex(n_rows),
        ),
        pd.DataFrame(
            {
                "attempt_id": ids,
                "simulated_recovered": np.array([0, 1, 1, 0, 1][:n_rows], dtype="int8"),
                "simulated_recovered_amount_inr": [0.0, 1500.0, 2200.5, 0.0, 900.0][:n_rows],
                "base_recovery_propensity": [0.40, 0.50, 0.45, 0.35, 0.42][:n_rows],
                "action_effect_logit": [0.0, 0.60, 0.35, 0.45, 0.25][:n_rows],
                "propensity_under_assignment": [0.40, 0.66, 0.58, 0.51, 0.50][:n_rows],
            },
            index=pd.RangeIndex(n_rows),
        ),
    )


def build_handcrafted_dataset(n_rows: int = 4) -> pd.DataFrame:
    attempts, assignments, outcomes = handcrafted_frames(n_rows)
    return build_treatment_dataset(attempts, assignments, outcomes)


@pytest.fixture(scope="module")
def real_pipeline_run() -> dict[str, pd.DataFrame]:
    """Run the REAL Day 2 -> Day 4 pipeline on a generate_dataset(200) slice."""
    frame = generate_dataset(200, seed=42).reset_index(drop=True)
    train_df, validation_df, _test_df = chronological_split(frame)
    model, _metadata = train_baseline(train_df, validation_df, seed=42)
    probabilities = predict_recovery_probability(model, validation_df)

    attempts = validation_df.loc[:, ["attempt_id", "event_timestamp"]]
    assignments = assign_treatments(validation_df, list(map(float, probabilities)), POLICY)
    assignments.insert(0, "attempt_id", validation_df["attempt_id"].to_numpy())
    outcomes = simulate_outcomes(
        pd.concat([validation_df, assignments[list(ASSIGNMENT_RESULT_COLUMNS)]], axis=1),
        POLICY,
    )
    outcomes.insert(0, "attempt_id", validation_df["attempt_id"].to_numpy())
    dataset = build_treatment_dataset(attempts, assignments, outcomes)
    return {
        "context": validation_df,
        "attempts": attempts,
        "assignments": assignments,
        "outcomes": outcomes,
        "dataset": dataset,
    }


# ---------------------------------------------------------------------------
# 1. Field classification + dataset contract (D5)
# ---------------------------------------------------------------------------


def test_field_classification_covers_canonical_columns_exactly():
    assert list(FIELD_CLASSIFICATION) == CANONICAL_COLUMN_ORDER
    assert list(CANONICAL_COLUMNS) == CANONICAL_COLUMN_ORDER


def test_field_classification_values_within_closed_vocabulary():
    assert set(FIELD_CLASSIFICATION.values()) <= CLASSIFICATION_VOCABULARY
    assert FIELD_CLASSIFICATION["attempt_id"] == "identifier"
    assert FIELD_CLASSIFICATION["event_timestamp"] == "evaluation_metadata"
    for name in ("assigned_action", "arm_source", "assignment_probability"):
        assert FIELD_CLASSIFICATION[name] == "treatment_metadata"
    for name in ("treatment_timestamp", "outcome_timestamp"):
        assert FIELD_CLASSIFICATION[name] == "outcome_timing"
    for name in ("simulated_recovered", "simulated_recovered_amount_inr"):
        assert FIELD_CLASSIFICATION[name] == "outcome"
    for name in (
        "base_recovery_propensity",
        "action_effect_logit",
        "propensity_under_assignment",
    ):
        assert FIELD_CLASSIFICATION[name] == "ground_truth"


def test_dataset_contract_pins_targets_and_copies_classification():
    contract = dataset_contract()

    assert contract["rows"] == 5000
    assert contract["join_key"] == "attempt_id"
    assert contract["synthetic"] is True
    assert contract["baseline_contract_untouched"] is True
    assert contract["field_classification"] == FIELD_CLASSIFICATION
    # Copy semantics: mutating the returned mapping must not touch the constant.
    contract["field_classification"]["attempt_id"] = "VANDALIZED"
    contract["rows"] = 1
    assert FIELD_CLASSIFICATION["attempt_id"] == "identifier"
    assert dataset_contract()["rows"] == 5000


def test_day2_label_recovered_absent_from_treatment_contract():
    # Name-collision honesty: "recovered" is the Day 2 label and must never
    # appear as a treatment-dataset column or classified field.
    assert "recovered" not in FIELD_CLASSIFICATION
    assert "recovered" not in CANONICAL_COLUMNS
    assert "recovered" not in dataset_contract()["field_classification"]
    built = build_handcrafted_dataset()
    assert "recovered" not in built.columns


# ---------------------------------------------------------------------------
# 2. Happy-path construction from real pipeline outputs
# ---------------------------------------------------------------------------


def test_real_pipeline_run_builds_canonical_dataset(real_pipeline_run):
    dataset = real_pipeline_run["dataset"]

    assert list(dataset.columns) == CANONICAL_COLUMN_ORDER
    assert len(dataset) == len(real_pipeline_run["attempts"])
    assert isinstance(dataset.index, pd.RangeIndex)
    assert dataset.index.start == 0 and dataset.index.step == 1
    assert set(dataset["assigned_action"]) <= set(CANONICAL_ARMS)
    assert set(dataset["arm_source"]) <= {"randomized", "safety_censored"}

    report = validate_treatment_dataset(dataset)
    assert report["valid"], report["violations"]
    assert report["violations"] == []
    assert report["row_count"] == len(dataset)
    assert report["column_count"] == len(CANONICAL_COLUMNS)
    assert report["classification_complete"] is True


def test_built_fields_align_with_source_frames_even_when_inputs_shuffled():
    attempts, assignments, outcomes = handcrafted_frames(5)
    shuffled_assignments = assignments.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled_outcomes = outcomes.sample(frac=1.0, random_state=13).reset_index(drop=True)

    dataset = build_treatment_dataset(attempts, shuffled_assignments, shuffled_outcomes)

    # Rows are aligned by attempt_id, so the result always follows ATTEMPTS
    # row order -- here ATT-000000..ATT-000004 -- regardless of input shuffle.
    expected_arms = ["CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW"]
    expected_recovered = np.array([0, 1, 1, 0, 1], dtype="int8")
    assert dataset["assigned_action"].tolist() == expected_arms
    assert dataset["simulated_recovered"].tolist() == expected_recovered.tolist()
    expected_probabilities = pd.Series([0.20, 0.30, 0.25, 0.15, 0.10])
    pd.testing.assert_series_equal(
        dataset["assignment_probability"].reset_index(drop=True),
        expected_probabilities,
        check_names=False,
    )


def test_inputs_not_mutated_by_build():
    attempts, assignments, outcomes = handcrafted_frames(4)
    timeline = pd.DataFrame(
        {
            "attempt_id": attempts["attempt_id"],
            "treatment_timestamp": pd.Series([None] * 4, dtype=object),
            "outcome_timestamp": pd.Series(
                [pd.Timestamp("2026-01-01T02:00:00Z")] * 4, dtype=object
            ),
        }
    )
    frames = {
        "attempts_df": attempts,
        "assignments_df": assignments,
        "outcomes_df": outcomes,
        "timeline_df": timeline,
    }
    snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}

    build_treatment_dataset(attempts, assignments, outcomes, timeline)

    for name, frame in frames.items():
        snapshot = snapshots[name]
        pd.testing.assert_frame_equal(frame, snapshot)
        assert frame.dtypes.equals(snapshot.dtypes), f"{name} dtypes changed"


# ---------------------------------------------------------------------------
# 3. Temporal placeholders (timeline optional until Task 6)
# ---------------------------------------------------------------------------


def test_missing_timestamps_become_none_object_placeholders():
    built = build_handcrafted_dataset()

    for column in ("treatment_timestamp", "outcome_timestamp"):
        assert built[column].dtype == object
        assert built[column].isna().all()


def test_timeline_columns_carried_when_provided_and_validate_passes():
    attempts, assignments, outcomes = handcrafted_frames(4)
    timeline = pd.DataFrame(
        {
            "attempt_id": attempts["attempt_id"],
            "treatment_timestamp": pd.Series(
                [
                    None,
                    pd.Timestamp("2026-01-01T00:15:00Z"),
                    pd.Timestamp("2026-01-01T01:00:00Z"),
                    pd.Timestamp("2026-01-01T04:00:00Z"),
                ],
                dtype=object,
            ),
            "outcome_timestamp": pd.Series(
                [
                    pd.Timestamp("2026-01-01T02:00:00Z"),
                    pd.Timestamp("2026-01-01T05:00:00Z"),
                    pd.Timestamp("2026-01-01T09:00:00Z"),
                    pd.Timestamp("2026-01-01T23:30:00Z"),
                ],
                dtype=object,
            ),
        }
    )

    built = build_treatment_dataset(attempts, assignments, outcomes, timeline)

    assert built["treatment_timestamp"].tolist()[0] is None
    assert built["treatment_timestamp"].tolist()[1] == pd.Timestamp("2026-01-01T00:15:00Z")
    assert built["outcome_timestamp"].tolist()[3] == pd.Timestamp("2026-01-01T23:30:00Z")
    report = validate_treatment_dataset(built)
    assert report["valid"], report["violations"]


def test_timeline_missing_required_column_rejected():
    attempts, assignments, outcomes = handcrafted_frames(4)
    timeline = pd.DataFrame(
        {"attempt_id": attempts["attempt_id"], "treatment_timestamp": [None] * 4}
    )

    with pytest.raises(ValueError) as excinfo:
        build_treatment_dataset(attempts, assignments, outcomes, timeline)

    assert "outcome_timestamp" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Join validation: nulls, uniqueness, identical id sets, collisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("culprit", ["attempts", "assignments", "outcomes"])
def test_duplicate_attempt_ids_rejected(culprit):
    attempts, assignments, outcomes = handcrafted_frames(4)
    frames = {"attempts": attempts, "assignments": assignments, "outcomes": outcomes}
    frames[culprit].loc[2, "attempt_id"] = frames[culprit].loc[0, "attempt_id"]

    with pytest.raises(ValueError) as excinfo:
        build_treatment_dataset(frames["attempts"], frames["assignments"], frames["outcomes"])

    message = str(excinfo.value)
    assert culprit in message
    assert "unique" in message


def test_id_set_mismatch_names_count_and_examples():
    attempts, assignments, outcomes = handcrafted_frames(4)
    dropped_outcomes = outcomes.iloc[1:].reset_index(drop=True)

    with pytest.raises(ValueError) as excinfo:
        build_treatment_dataset(attempts, assignments, dropped_outcomes)

    message = str(excinfo.value)
    assert "1" in message
    assert "'ATT-000000'" in message


def test_extra_timeline_ids_rejected():
    attempts, assignments, outcomes = handcrafted_frames(4)
    timeline = pd.DataFrame(
        {
            "attempt_id": list(attempts["attempt_id"]) + ["ATT-999999"],
            "treatment_timestamp": [None] * 5,
            "outcome_timestamp": [pd.Timestamp("2026-01-02T00:00:00Z")] * 5,
        }
    )

    with pytest.raises(ValueError) as excinfo:
        build_treatment_dataset(attempts, assignments, outcomes, timeline)

    message = str(excinfo.value)
    assert "ATT-999999" in message


@pytest.mark.parametrize("culprit", ["attempts", "outcomes"])
def test_null_attempt_ids_rejected(culprit):
    attempts, assignments, outcomes = handcrafted_frames(4)
    frames = {"attempts": attempts, "outcomes": outcomes}
    frames[culprit]["attempt_id"] = frames[culprit]["attempt_id"].astype(object)
    frames[culprit].loc[1, "attempt_id"] = None

    with pytest.raises(ValueError):
        build_treatment_dataset(attempts, assignments, outcomes)


def test_missing_attempt_id_column_rejected():
    attempts, assignments, outcomes = handcrafted_frames(4)

    with pytest.raises(ValueError):
        build_treatment_dataset(
            attempts.drop(columns=["attempt_id"]), assignments, outcomes
        )


@pytest.mark.parametrize(
    ("overlapping_column", "donor"),
    [("assigned_action", "outcomes"), ("event_timestamp", "assignments")],
)
def test_column_collision_across_sources_rejected(overlapping_column, donor):
    attempts, assignments, outcomes = handcrafted_frames(4)
    if donor == "outcomes":
        outcomes = outcomes.copy()
        outcomes[overlapping_column] = assignments[overlapping_column].to_numpy()
    else:
        assignments = assignments.copy()
        assignments[overlapping_column] = attempts[overlapping_column].to_numpy()

    with pytest.raises(ValueError) as excinfo:
        build_treatment_dataset(attempts, assignments, outcomes)

    message = str(excinfo.value)
    assert overlapping_column in message
    assert "collision" in message


# ---------------------------------------------------------------------------
# 5. Empty frames behavior
# ---------------------------------------------------------------------------


def test_empty_frames_build_valid_empty_schema():
    attempts = pd.DataFrame(
        {
            "attempt_id": pd.Series(dtype=object),
            "event_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )
    assignments = pd.DataFrame(
        {
            "attempt_id": pd.Series(dtype=object),
            **{name: pd.Series(dtype=object if name != "assignment_probability" else "float64")
               for name in ASSIGNMENT_RESULT_COLUMNS},
        }
    )
    outcomes = pd.DataFrame(
        {
            "attempt_id": pd.Series(dtype=object),
            "simulated_recovered": pd.Series(dtype="int8"),
            **{
                name: pd.Series(dtype="float64")
                for name in OUTCOME_RESULT_COLUMNS[1:]
            },
        }
    )

    built = build_treatment_dataset(attempts, assignments, outcomes)

    assert len(built) == 0
    assert list(built.columns) == CANONICAL_COLUMN_ORDER
    assert built["simulated_recovered"].dtype == np.dtype("int8")
    for column in (
        "assignment_probability",
        "simulated_recovered_amount_inr",
        "base_recovery_propensity",
        "action_effect_logit",
        "propensity_under_assignment",
    ):
        assert built[column].dtype == np.dtype("float64")
    for column in ("treatment_timestamp", "outcome_timestamp"):
        assert built[column].dtype == object

    report = validate_treatment_dataset(built)
    assert report["valid"], report["violations"]
    assert report["row_count"] == 0
    assert report["classification_complete"] is True


# ---------------------------------------------------------------------------
# 6. Validator on good and tampered data
# ---------------------------------------------------------------------------


def test_validator_accepts_good_data_with_clean_report():
    built = build_handcrafted_dataset()

    report = validate_treatment_dataset(built)

    assert report == {
        "valid": True,
        "violations": [],
        "row_count": 4,
        "column_count": len(CANONICAL_COLUMNS),
        "classification_complete": True,
    }


@pytest.mark.parametrize(
    ("tampering", "expected_fragment"),
    [
        pytest.param(
            lambda df: df.__setitem__("assigned_action", ["STOP"] + df["assigned_action"][1:].tolist()),
            "STOP",
            id="stop-arm-flip",
        ),
        pytest.param(
            lambda df: df.__setitem__(
                "simulated_recovered_amount_inr",
                [-100.0] + df["simulated_recovered_amount_inr"][1:].tolist(),
            ),
            "negative",
            id="negative-amount",
        ),
        pytest.param(
            lambda df: df.__setitem__(
                "base_recovery_propensity",
                [1.5] + df["base_recovery_propensity"][1:].tolist(),
            ),
            "base_recovery_propensity",
            id="propensity-above-one",
        ),
    ],
)
def test_tampered_datasets_are_invalid_with_specific_violations(tampering, expected_fragment):
    built = build_handcrafted_dataset()
    tampering(built)

    report = validate_treatment_dataset(built)

    assert report["valid"] is False
    assert report["violations"], "a tampered frame must carry at least one violation"
    joined_violations = "\n".join(report["violations"])
    assert expected_fragment in joined_violations


def test_unknown_arm_source_value_flagged():
    built = build_handcrafted_dataset()
    built.loc[2, "arm_source"] = "auto_assigned"

    report = validate_treatment_dataset(built)

    joined = "\n".join(report["violations"])
    assert report["valid"] is False
    assert "auto_assigned" in joined


def test_probability_rules_per_arm_source_enforced():
    built = build_handcrafted_dataset()
    built.loc[0, "arm_source"] = "safety_censored"
    built.loc[0, "assignment_probability"] = 0.3
    built.loc[1, "assignment_probability"] = 0.0

    report = validate_treatment_dataset(built)

    joined = "\n".join(report["violations"])
    assert report["valid"] is False
    assert "must carry assignment_probability" in joined
    assert "must be positive" in joined


def test_non_binary_simulated_recovered_flagged():
    built = build_handcrafted_dataset()
    built.loc[2, "simulated_recovered"] = 2

    report = validate_treatment_dataset(built)

    assert report["valid"] is False
    assert any("simulated_recovered" in violation for violation in report["violations"])


def test_effect_logit_outside_bounds_flagged():
    built = build_handcrafted_dataset()
    built.loc[1, "action_effect_logit"] = 3.5

    report = validate_treatment_dataset(built)

    joined = "\n".join(report["violations"])
    assert report["valid"] is False
    assert "action_effect_logit" in joined


# ---------------------------------------------------------------------------
# 6b. Review round: validator must degrade to violations, never raise (F1)
#     and never silently pass nulls (F2)
# ---------------------------------------------------------------------------


def test_string_amounts_degrade_to_numeric_violation_without_raising():
    built = build_handcrafted_dataset(4)
    built["simulated_recovered_amount_inr"] = ["100", "200", "-300", "400"]

    report = validate_treatment_dataset(built)

    joined = "\n".join(report["violations"])
    assert report["valid"] is False
    assert "must be numeric" in joined
    assert "4 non-numeric row(s)" in joined
    # All four entries are non-numeric strings; none reaches the <0 check,
    # so no negative violation may appear for this frame.
    assert "negative" not in joined


def test_negative_check_applies_only_to_recognized_numerics():
    built = build_handcrafted_dataset(4)
    built["simulated_recovered_amount_inr"] = [-50.0, "100", "200", "300"]

    report = validate_treatment_dataset(built)

    joined = "\n".join(report["violations"])
    assert report["valid"] is False
    assert "must be numeric" in joined
    assert "3 non-numeric row(s)" in joined
    assert "negative" in joined


@pytest.mark.parametrize(
    "column",
    ["assigned_action", "arm_source", "simulated_recovered"],
)
def test_null_entries_in_categorization_and_label_fields_flagged(column):
    built = build_handcrafted_dataset(4)
    assert validate_treatment_dataset(built)["valid"] is True

    built.loc[0, column] = np.nan

    report = validate_treatment_dataset(built)

    joined = "\n".join(report["violations"])
    assert report["valid"] is False
    assert f"{column} has 1 missing value(s)" in joined


def test_field_names_never_collide_with_forbidden_or_model_features():
    # Name-level guard (F6): the ONLY FIELD_CLASSIFICATION names shared with
    # FORBIDDEN_FEATURES are the two Day-2-governed metadata names whose
    # exclusion ml/features.py already enforces by whitelist construction;
    # every NEW Day 4 field is disjoint from FORBIDDEN_FEATURES, and the full
    # classification set is disjoint from the model feature lists.
    assert set(FIELD_CLASSIFICATION) & set(FORBIDDEN_FEATURES) == {
        "attempt_id",
        TIME_COLUMN,
    }
    new_fields = set(FIELD_CLASSIFICATION) - {"attempt_id", TIME_COLUMN}
    assert new_fields.isdisjoint(set(FORBIDDEN_FEATURES))
    assert set(FIELD_CLASSIFICATION).isdisjoint(
        set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    )


def test_structural_timestamp_variants_accepted_without_ordering_claims():
    built = build_handcrafted_dataset()
    built["treatment_timestamp"] = [
        None,
        pd.Timestamp("2026-01-01T00:15:00Z"),
        "2026-01-01T01:00:00+00:00",
        pd.NaT,
    ]
    built["outcome_timestamp"] = [
        "not-a-timestamp",
        pd.Timestamp("2026-01-02T00:00:00Z"),
        1700000000,
        pd.Timestamp("2026-01-03T00:00:00Z"),
    ]

    report = validate_treatment_dataset(built)

    assert report["valid"] is False
    joined = "\n".join(report["violations"])
    assert "not-a-timestamp" in joined
    assert "1700000000" in joined
    assert "ordering" in joined.lower() or "Task 6" in joined


def test_all_none_timestamps_remain_structurally_valid():
    built = build_handcrafted_dataset()

    report = validate_treatment_dataset(built)

    assert report["valid"] is True


def test_renamed_column_breaks_classification_completeness():
    built = build_handcrafted_dataset().rename(columns={"arm_source": "arm_label"})

    report = validate_treatment_dataset(built)

    assert report["valid"] is False
    assert report["classification_complete"] is False
    joined = "\n".join(report["violations"])
    assert "FIELD_CLASSIFICATION" in joined
    assert "arm_source" in joined


# ---------------------------------------------------------------------------
# 7. LEAKAGE GUARDS: structural immunity of the Day 2 feature builder
# ---------------------------------------------------------------------------


def test_feature_builder_structurally_immune_to_hostile_merged_frame(real_pipeline_run):
    context = real_pipeline_run["context"]
    dataset = real_pipeline_run["dataset"]

    poisoned = dataset.copy()
    poisoned["recovered_amount_leak"] = poisoned["simulated_recovered_amount_inr"]

    merged = pd.concat([context.reset_index(drop=True), poisoned.reset_index(drop=True)], axis=1)

    X, y = build_feature_matrix(merged)

    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    non_context_fields = {
        name
        for name, classification in FIELD_CLASSIFICATION.items()
        if classification != "decision_time_feature"
    }
    assert set(X.columns).isdisjoint(non_context_fields)
    assert set(X.columns).isdisjoint(LEAKAGE_FIELDS)
    assert "recovered_amount_leak" not in set(X.columns)
    assert y.tolist() == context["recovered"].astype(int).tolist()


def test_new_field_names_never_equal_existing_model_features_regression_14():
    historical_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(historical_features) == 14

    assert set(FIELD_CLASSIFICATION).isdisjoint(historical_features)
    assert not set(FIELD_CLASSIFICATION) & set(historical_features)

    context = generate_dataset(120, seed=7).reset_index(drop=True)
    dataset = build_handcrafted_dataset(4)
    merged = pd.concat([context, dataset], axis=1)
    X, _y = build_feature_matrix(merged)

    assert list(X.columns) == historical_features, (
        "build_feature_matrix output drifted from its historical 14 features"
    )
    # FORBIDDEN_FEATURES still excludes identifiers/time/labels/outcome fields;
    # our additive names are guarded by the whitelist itself, so they appear in
    # neither the selected features nor the forbidden set under these names.
    assert not set(X.columns) & set(FORBIDDEN_FEATURES)


# ---------------------------------------------------------------------------
# 8. Purity: import whitelist, no wall clock, language audit
# ---------------------------------------------------------------------------


def _import_root_modules(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


ALLOWED_IMPORT_ROOTS = frozenset({"__future__", "datetime", "numpy", "pandas", "simulation"})


def test_simulation_dataset_import_roots_whitelisted():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )


def test_source_has_no_wall_clock_or_stdlib_randomness_tokens():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    for token in ("datetime.now", "utcnow", "time.time", "secrets", "uuid"):
        assert token not in source, f"forbidden token {token!r} found"
    assert re.search(r"(?<![\w.])random\b", source) is None


def test_module_language_pins_additive_no_dbwrite_documentation():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))
    assert docstring is not None

    assert "causal estimate" not in source
    assert "ADDITIVE" in docstring
    assert "attempt_id" in docstring
    assert "Day 2 baseline" in docstring
    assert "ml/features.py" in docstring
    assert "no DB writes" in docstring
    # F3: deliberate NaT-passthrough divergence from recovery.audit, pinned.
    assert "Unlike recovery.audit (which rejects NaT loudly)" in docstring
