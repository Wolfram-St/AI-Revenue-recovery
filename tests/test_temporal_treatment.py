"""Tests for Day 4 temporal timeline stamping (plan Task 6, decisions D1b/D4)."""

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
from simulation.config import SEED_STREAM_TEMPORAL, load_treatment_policy
from simulation.outcomes import simulate_outcomes
from simulation.temporal import RESULT_COLUMNS, stamp_treatment_timeline
from simulation.treatment import assign_treatments

POLICY = load_treatment_policy("config/treatment_policy.yaml")
SOURCE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "temporal.py"

REQUIRED_COLUMNS = ("attempt_id", "event_timestamp", "assigned_action", "arm_source")
NEW_COLUMNS = ("treatment_timestamp", "outcome_timestamp")
TZ_UTC = "datetime64[ns, UTC]"

ARM_CYCLE = ("CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")


def crafted_frame(arms, index=None) -> pd.DataFrame:
    """Small frame carrying exactly the columns the stamper consumes."""
    arms = list(arms)
    n_rows = len(arms)
    return pd.DataFrame(
        {
            "attempt_id": pd.Series(
                [f"ATT-{position:06d}" for position in range(n_rows)], dtype="object"
            ),
            "event_timestamp": pd.Series(
                pd.date_range("2026-01-01T00:00:00Z", periods=n_rows, freq="15min"),
                index=index,
            ),
            "assigned_action": pd.Series(arms, dtype="object", index=index),
            "arm_source": pd.Series(
                ["randomized"] * n_rows, dtype="object", index=index
            ),
            "amount_inr": pd.Series(
                3000.0 + np.arange(n_rows) * 250.0, index=index
            ),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def stamped_real_frame() -> pd.DataFrame:
    """Full pipeline artifact: model probabilities -> assignments -> outcomes -> stamps."""
    dataset = generate_dataset(300, seed=42)
    train_df, validation_df, _ = chronological_split(dataset)
    model, _metadata = train_baseline(train_df, validation_df, seed=42)
    probabilities = predict_recovery_probability(model, validation_df)
    frame = validation_df.reset_index(drop=True)
    assignments = assign_treatments(frame, list(probabilities), POLICY)
    outcomes = simulate_outcomes(frame.join(assignments), POLICY)
    full = frame.join(assignments).join(outcomes)
    return stamp_treatment_timeline(full, POLICY)


# ---------------------------------------------------------------------------
# 1. Ordering invariant on a mixed real-ish frame
# ---------------------------------------------------------------------------


def test_treated_rows_order_strictly_failure_then_treatment_then_outcome(
    stamped_real_frame,
):
    result = stamped_real_frame
    treated = (result["assigned_action"] != "CONTROL").to_numpy()

    assert set(result["assigned_action"]) == set(ARM_CYCLE), (
        "fixture regression: every canonical arm must appear"
    )

    failure = result["event_timestamp"]
    treatment = result["treatment_timestamp"]
    outcome = result["outcome_timestamp"]

    assert not treatment[pd.Series(treated, index=result.index)].isna().any(), (
        "every treated row must carry a real treatment timestamp"
    )
    assert (failure[treated].to_numpy() < treatment[treated].to_numpy()).all(), (
        "treatment must land strictly after its own failure"
    )
    assert (treatment[treated].to_numpy() < outcome[treated].to_numpy()).all(), (
        "outcome must land strictly after treatment for treated rows"
    )


def test_control_rows_carry_null_treatment_and_post_failure_outcome(
    stamped_real_frame,
):
    result = stamped_real_frame
    control = (result["assigned_action"] == "CONTROL").to_numpy()
    assert control.any(), "fixture regression: no CONTROL rows survived"

    treatment = result["treatment_timestamp"].to_numpy()[control]
    assert pd.isna(treatment).all(), "CONTROL treatment_timestamp must be null"

    failure = result["event_timestamp"].to_numpy()[control]
    outcome = result["outcome_timestamp"].to_numpy()[control]
    assert (failure < outcome).all(), "control horizon must start at its failure"


def test_result_preserves_original_columns_and_index(stamped_real_frame):
    result = stamped_real_frame

    assert list(result.columns)[-2:] == list(NEW_COLUMNS)
    assert list(result.columns)[:-2][:1] == ["attempt_id"]
    assert "event_timestamp" in result.columns
    assert len(result) == 45


def test_result_index_matches_input_index_exactly():
    frame = crafted_frame(["RETRY_NOW", "CONTROL"], index=[10, 20])

    result = stamp_treatment_timeline(frame, POLICY)

    assert result.index.equals(frame.index)


# ---------------------------------------------------------------------------
# 2. Arm-specific delay exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm", "expected_hours"),
    [
        pytest.param("RETRY_NOW", 0.25, id="retry-now-15min"),
        pytest.param("REQUEST_UPDATE", 2.0, id="request-update-2h"),
        pytest.param("HUMAN_REVIEW", 4.0, id="human-review-4h"),
        pytest.param("RETRY_LATER", 24.0, id="retry-later-24h"),
    ],
)
def test_arm_specific_delay_matches_configuration_exactly(arm, expected_hours):
    frame = crafted_frame([arm, arm, arm])

    result = stamp_treatment_timeline(frame, POLICY)

    # Explicit ns unit sidesteps pandas' generic-unit Timedelta constructor,
    # which trips a NumPy 2.5 deprecation for this exact-hours arithmetic.
    expected_delta = np.timedelta64(int(round(expected_hours * 3.6e12)), "ns")
    configured = float(POLICY.treatment_delay_hours[arm])
    assert configured == expected_hours
    deltas = result["treatment_timestamp"] - result["event_timestamp"]
    assert (deltas == expected_delta).all(), (
        f"{arm} delay drifted from the configured {expected_hours}h"
    )


# ---------------------------------------------------------------------------
# 3. Resolution-window bounds hold across a large run
# ---------------------------------------------------------------------------


def test_resolution_windows_stay_within_configured_bounds_inclusively():
    n_rows = 20000
    frame = crafted_frame(
        list(np.array(ARM_CYCLE, dtype=object)[np.arange(n_rows) % len(ARM_CYCLE)])
    )

    result = stamp_treatment_timeline(frame, POLICY)

    low, high = POLICY.resolution_window_hours
    low_delta = np.timedelta64(int(round(float(low) * 3.6e12)), "ns")
    high_delta = np.timedelta64(int(round(float(high) * 3.6e12)), "ns")
    treated = (result["assigned_action"] != "CONTROL").to_numpy()

    treated_windows = (result["outcome_timestamp"] - result["treatment_timestamp"]).loc[
        pd.Series(treated, index=result.index)
    ]
    control_windows = (result["outcome_timestamp"] - result["event_timestamp"]).loc[
        pd.Series(~treated, index=result.index)
    ]

    assert len(treated_windows) > 0 and len(control_windows) > 0
    assert (treated_windows >= low_delta).all() and (
        treated_windows <= high_delta
    ).all(), "treated resolution window escaped [low, high]"
    assert (control_windows >= low_delta).all() and (
        control_windows <= high_delta
    ).all(), "control observation horizon escaped [low, high]"


# ---------------------------------------------------------------------------
# 4. Determinism: same seed identical; different master seed moves only windows
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_frames():
    frame = crafted_frame(
        list(np.array(ARM_CYCLE, dtype=object)[np.arange(500) % len(ARM_CYCLE)])
    )

    first = stamp_treatment_timeline(frame, POLICY)
    second = stamp_treatment_timeline(frame, POLICY)

    assert first.equals(second)


def test_different_master_seed_changes_windows_not_delays():
    frame = crafted_frame(
        list(np.array(ARM_CYCLE, dtype=object)[np.arange(500) % len(ARM_CYCLE)])
    )
    mutated_policy = dataclasses.replace(POLICY, master_seed=POLICY.master_seed + 1)

    baseline = stamp_treatment_timeline(frame, POLICY)
    alternative = stamp_treatment_timeline(frame, mutated_policy)

    treated = (frame["assigned_action"] != "CONTROL").to_numpy()
    base_windows = (
        baseline["outcome_timestamp"] - baseline["treatment_timestamp"]
    ).to_numpy()[treated]
    alt_windows = (
        alternative["outcome_timestamp"] - alternative["treatment_timestamp"]
    ).to_numpy()[treated]
    assert (base_windows != alt_windows).any(), (
        "different master_seed must resample the resolution windows"
    )

    base_delays = (
        baseline["treatment_timestamp"] - baseline["event_timestamp"]
    ).to_numpy()[treated]
    alt_delays = (
        alternative["treatment_timestamp"] - alternative["event_timestamp"]
    ).to_numpy()[treated]
    np.testing.assert_array_equal(base_delays, alt_delays), (
        "arm delays are configuration, never draws"
    )


# ---------------------------------------------------------------------------
# 5. Impossibles rejected loudly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value", [pytest.param("not-a-date", id="unparsable"), pytest.param(pd.NaT, id="nat")]
)
def test_unparsable_or_nat_event_timestamp_rejected_naming_the_row(bad_value):
    frame = crafted_frame(["RETRY_NOW", "RETRY_NOW", "CONTROL"])
    frame["event_timestamp"] = frame["event_timestamp"].astype(object)
    frame.loc[1, "event_timestamp"] = bad_value

    with pytest.raises(ValueError) as excinfo:
        stamp_treatment_timeline(frame, POLICY)

    message = str(excinfo.value)
    assert "event_timestamp" in message
    assert "1" in message, "the offending row index must be named"


def test_every_unparsable_row_index_is_named_at_once():
    frame = crafted_frame(["CONTROL", "CONTROL", "CONTROL", "CONTROL"])
    frame["event_timestamp"] = frame["event_timestamp"].astype(object)
    frame.loc[0, "event_timestamp"] = "garbage"
    frame.loc[2, "event_timestamp"] = pd.NaT

    with pytest.raises(ValueError) as excinfo:
        stamp_treatment_timeline(frame, POLICY)

    assert "[0, 2]" in str(excinfo.value)


def test_unknown_assigned_action_rejected_naming_the_offender():
    frame = crafted_frame(["CONTROL", "STOP", "RETRY_NOW"])

    with pytest.raises(ValueError) as excinfo:
        stamp_treatment_timeline(frame, POLICY)

    message = str(excinfo.value)
    assert "assigned_action" in message
    assert "STOP" in message


def test_mixed_nan_and_unknown_arm_values_rejected_naming_both():
    frame = crafted_frame(["RETRY_NOW", "CONTROL", "CONTROL"])
    frame.loc[1, "assigned_action"] = np.nan
    frame.loc[2, "assigned_action"] = "FOO"

    with pytest.raises(ValueError) as excinfo:
        stamp_treatment_timeline(frame, POLICY)

    message = str(excinfo.value)
    assert "assigned_action" in message
    assert "FOO" in message
    assert "nan" in message


def test_missing_required_columns_rejected_naming_each():
    frame = crafted_frame(["RETRY_NOW"])

    with pytest.raises(ValueError) as excinfo:
        stamp_treatment_timeline(frame.drop(columns=["attempt_id"]), POLICY)
    assert "attempt_id" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        stamp_treatment_timeline(
            frame.drop(columns=["event_timestamp", "arm_source"]), POLICY
        )
    message = str(excinfo.value)
    assert "event_timestamp" in message
    assert "arm_source" in message


# ---------------------------------------------------------------------------
# 6. Non-mutation of input; output dtype discipline
# ---------------------------------------------------------------------------


def test_input_frame_never_mutated_values_and_dtypes():
    frame = crafted_frame(["CONTROL", "RETRY_NOW", "HUMAN_REVIEW"])
    snapshot = frame.copy(deep=True)

    result = stamp_treatment_timeline(frame, POLICY)

    pd.testing.assert_frame_equal(frame, snapshot)
    assert frame.dtypes.equals(snapshot.dtypes)
    assert list(result.columns) == list(frame.columns) + list(NEW_COLUMNS)


def test_output_columns_are_timezone_aware_utc_and_offsets_normalize():
    frame = crafted_frame(["RETRY_NOW"])
    frame["event_timestamp"] = ["2026-03-14T09:30:00+05:30"]

    result = stamp_treatment_timeline(frame, POLICY)

    for column in NEW_COLUMNS:
        assert str(result[column].dtype) == TZ_UTC, column
    # 09:30+05:30 is 04:00Z; RETRY_NOW stamps 15 minutes later.
    assert result["treatment_timestamp"].iloc[0] == pd.Timestamp(
        "2026-03-14T04:15:00Z"
    )


def test_naive_string_event_timestamp_is_interpreted_as_utc():
    naive_frame = crafted_frame(["RETRY_NOW"])
    naive_frame["event_timestamp"] = ["2026-10-25T02:30:00"]
    zulu_frame = crafted_frame(["RETRY_NOW"])
    zulu_frame["event_timestamp"] = ["2026-10-25T02:30:00Z"]

    naive_result = stamp_treatment_timeline(naive_frame, POLICY)
    zulu_result = stamp_treatment_timeline(zulu_frame, POLICY)

    # Naive wall time reads as UTC by design: identical stamps to the
    # explicit-Z equivalent, RETRY_NOW landing 15 minutes after 02:30Z.
    for column in NEW_COLUMNS:
        assert str(naive_result[column].dtype) == TZ_UTC
        np.testing.assert_array_equal(
            naive_result[column].to_numpy(), zulu_result[column].to_numpy()
        )
    assert naive_result["treatment_timestamp"].iloc[0] == pd.Timestamp(
        "2026-10-25T02:45:00Z"
    )


def test_dst_gap_naive_string_parses_cleanly_as_utc_without_exception():
    frame = crafted_frame(["CONTROL"])
    # 02:30 on the US spring-forward date never exists as a local wall time;
    # interpreted as plain UTC it must parse without any exception.
    frame["event_timestamp"] = ["2026-03-08T02:30:00"]

    result = stamp_treatment_timeline(frame, POLICY)

    assert result["outcome_timestamp"].iloc[0] > pd.Timestamp("2026-03-08T02:30:00Z")


def test_result_columns_constant_lists_the_two_stamps():
    assert list(RESULT_COLUMNS) == ["treatment_timestamp", "outcome_timestamp"]


# ---------------------------------------------------------------------------
# 7. Empty df -> empty valid result
# ---------------------------------------------------------------------------


def test_empty_frame_yields_empty_result_with_utc_placeholder_dtypes():
    empty = pd.DataFrame(
        {column: pd.Series(dtype="object") for column in REQUIRED_COLUMNS}
    )

    result = stamp_treatment_timeline(empty, POLICY)

    assert len(result) == 0
    assert result.index.equals(empty.index)
    assert list(result.columns) == list(REQUIRED_COLUMNS) + list(NEW_COLUMNS)
    for column in NEW_COLUMNS:
        assert str(result[column].dtype) == TZ_UTC, column


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


ALLOWED_IMPORT_ROOTS = frozenset({"__future__", "numpy", "pandas", "simulation"})


def _source_without_docstring() -> str:
    """Executable-source view: token counts ignore the module docstring."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))
    assert docstring is not None
    assert source.count(docstring) == 1
    return source.replace(docstring, " ", 1)


def test_simulation_temporal_import_roots_whitelisted_without_recovery():
    roots = _import_root_modules(SOURCE_PATH.read_text(encoding="utf-8"))

    assert roots == ALLOWED_IMPORT_ROOTS, (
        f"import roots drifted from the exact whitelist: {sorted(roots)}"
    )
    assert "recovery" not in roots, "stamping must not depend on recovery/*"


def test_seed_stream_temporal_child_used_exactly_once_via_spawn():
    code = _source_without_docstring()

    occurrences = [match.start() for match in re.finditer(r"np\.random\.default_rng", code)]
    assert len(occurrences) == 1, "exactly one master-seed rng derivation allowed"
    spawn_pattern = (
        r"np\.random\.default_rng\(\s*policy\.master_seed\s*\)\s*\.\s*spawn\s*\("
        r"\s*SEED_STREAM_TEMPORAL\s*\+\s*1\s*\)\s*\[\s*SEED_STREAM_TEMPORAL\s*\]"
    )
    assert re.search(spawn_pattern, code) is not None, (
        "the single derivation must be default_rng(policy.master_seed).spawn("
        "SEED_STREAM_TEMPORAL + 1)[SEED_STREAM_TEMPORAL]"
    )


def test_source_has_no_wall_clock_or_stdlib_randomness_tokens():
    code = _source_without_docstring()

    forbidden_patterns = (
        r"(?<![\w.])datetime\b",
        r"(?<![\w.])time\s*\.",
        r"(?<![\w.])random\b",
        r"(?<![\w.])secrets?\b",
        r"(?<![\w.])uuid\b",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, code) is None, f"forbidden pattern {pattern!r} found"
