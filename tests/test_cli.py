"""Tests for the Day 6 treatment/outcome CLI artifact (plan Task 1, decision D-E3)."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from data.generate_dataset import FEATURE_COLUMNS, LABEL_COLUMNS, TIME_COLUMN
from simulation.observations import APPENDIX_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "simulation" / "cli.py"
POLICY_MASTER_SEED = 20260826

# Independent reconstruction of the assembled observation contract: the 19
# attempts columns in generator order followed by the simulator appendix.
ATTEMPTS_COLUMNS = (
    tuple(FEATURE_COLUMNS[:3]) + (TIME_COLUMN,) + tuple(FEATURE_COLUMNS[3:]) + tuple(LABEL_COLUMNS)
)
EXPECTED_OBSERVATION_COLUMNS = ATTEMPTS_COLUMNS + tuple(APPENDIX_COLUMNS)


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "simulation.cli", *arguments],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def last_json_line(stdout: str) -> dict:
    return json.loads(stdout.strip().splitlines()[-1])


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated_42(tmp_path_factory) -> tuple[Path, subprocess.CompletedProcess[str]]:
    out = tmp_path_factory.mktemp("seed42") / "treatment_outcomes.csv"
    result = run_cli("generate", "--rows", "300", "--seed", "42", "--out", str(out))
    return out, result


@pytest.fixture(scope="module")
def generated_43(tmp_path_factory) -> tuple[Path, subprocess.CompletedProcess[str]]:
    out = tmp_path_factory.mktemp("seed43") / "treatment_outcomes_alt.csv"
    result = run_cli("generate", "--rows", "300", "--seed", "43", "--out", str(out))
    return out, result


# ---------------------------------------------------------------------------
# 1. Generate happy path: file, header, row count, seed-semantics payload
# ---------------------------------------------------------------------------


def test_generate_happy_path_writes_contract_csv_and_distinct_seeds(generated_42):
    out, result = generated_42

    assert result.returncode == 0, result.stderr
    assert out.exists()

    frame = pd.read_csv(out)
    assert len(frame) == 300

    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(EXPECTED_OBSERVATION_COLUMNS)

    payload = last_json_line(result.stdout)
    assert set(payload) == {"dataset_seed", "policy_master_seed", "rows", "path", "sha256"}
    assert payload["dataset_seed"] == 42
    assert payload["policy_master_seed"] == POLICY_MASTER_SEED
    assert payload["dataset_seed"] != payload["policy_master_seed"]
    assert payload["rows"] == 300
    assert payload["path"] == str(out)
    assert payload["sha256"] == sha256_of(out)


# ---------------------------------------------------------------------------
# 2. Reproducibility: same dataset seed -> byte-identical artifact
# ---------------------------------------------------------------------------


def test_same_dataset_seed_is_byte_identical(tmp_path, generated_42):
    out, first_result = generated_42
    assert first_result.returncode == 0, first_result.stderr

    second_out = tmp_path / "again.csv"
    second_result = run_cli("generate", "--rows", "300", "--seed", "42", "--out", str(second_out))

    assert second_result.returncode == 0, second_result.stderr
    first_payload = last_json_line(first_result.stdout)
    second_payload = last_json_line(second_result.stdout)
    assert first_payload["sha256"] == second_payload["sha256"]
    assert out.read_bytes() == second_out.read_bytes()


# ---------------------------------------------------------------------------
# 3. Different dataset seed: different bytes, still a valid dataset
# ---------------------------------------------------------------------------


def test_different_dataset_seed_changes_bytes_but_validates(generated_42, generated_43):
    out_42, result_42 = generated_42
    out_43, result_43 = generated_43

    assert result_42.returncode == 0, result_42.stderr
    assert result_43.returncode == 0, result_43.stderr
    assert last_json_line(result_42.stdout)["sha256"] != last_json_line(result_43.stdout)["sha256"]
    assert out_42.read_bytes() != out_43.read_bytes()
    assert last_json_line(result_43.stdout)["dataset_seed"] == 43

    validation = run_cli("validate", "--csv", str(out_43))
    assert validation.returncode == 0, validation.stderr
    assert last_json_line(validation.stdout)["valid"] is True


# ---------------------------------------------------------------------------
# 3b. Fail-closed: an invalid request must never leave an artifact behind.
# --rows 0 is the cleanest deterministic invalid trigger: the attempts-frame
# generator raises ValueError("n_rows must be positive") before any assembly,
# validation, or write happens, so the command must exit non-zero with no
# output file -- pinning that nothing is written ahead of the validation gate
# (a regression writing the CSV before the pipeline/gate completes would fail).
# ---------------------------------------------------------------------------


def test_generate_zero_rows_fails_without_writing_artifact(tmp_path):
    out = tmp_path / "should_not_exist.csv"

    result = run_cli("generate", "--rows", "0", "--seed", "42", "--out", str(out))

    assert result.returncode != 0
    assert "n_rows must be positive" in result.stderr, (
        "the failure must be the documented zero-rows rejection, not an unrelated error"
    )
    assert not out.exists()


# ---------------------------------------------------------------------------
# 4. Validate: good CSV passes; tampered label fails
# ---------------------------------------------------------------------------


def test_validate_good_csv_reports_valid_true(generated_42):
    out, _ = generated_42

    result = run_cli("validate", "--csv", str(out))

    assert result.returncode == 0, result.stderr
    report = last_json_line(result.stdout)
    assert report["valid"] is True
    assert report["violations"] == []
    assert report["row_count"] == 300


def test_validate_tampered_simulated_recovery_fails(tmp_path, generated_42):
    source, _ = generated_42
    tampered = tmp_path / "tampered.csv"
    frame = pd.read_csv(source)
    recovered_rows = frame.index[frame["simulated_recovered"] == 1]
    assert len(recovered_rows) > 0, "fixture regression: no recovered rows to tamper"
    frame.loc[recovered_rows[0], "simulated_recovered"] = 5
    frame.to_csv(tampered, index=False)

    result = run_cli("validate", "--csv", str(tampered))

    assert result.returncode != 0
    report = last_json_line(result.stdout)
    assert report["valid"] is False
    assert report["violations"]


# ---------------------------------------------------------------------------
# 5. Summary: parseable JSON carrying the OBSERVED label
# ---------------------------------------------------------------------------


def test_summary_parses_with_observed_label(generated_42):
    out, _ = generated_42

    result = run_cli("summary", "--csv", str(out))

    assert result.returncode == 0, result.stderr
    payload = last_json_line(result.stdout)
    rendered = json.dumps(payload)
    assert "OBSERVED SIMULATED OUTCOME" in rendered
    assert payload["arms_summary"]["label"] == "OBSERVED SIMULATED OUTCOME"
    assert payload["overlap_diagnostics"]["label"] == "OBSERVED SIMULATED OUTCOME"
    assert set(payload["arms_summary"]["arms"]) == {
        "CONTROL",
        "RETRY_NOW",
        "RETRY_LATER",
        "REQUEST_UPDATE",
        "HUMAN_REVIEW",
    }


# ---------------------------------------------------------------------------
# 6. CLI surface: help lists subcommands; unknown subcommand rejected
# ---------------------------------------------------------------------------


def test_help_lists_all_subcommands():
    result = run_cli("--help")

    assert result.returncode == 0
    for name in ("generate", "validate", "summary"):
        assert name in result.stdout


def test_unknown_subcommand_exits_nonzero():
    result = run_cli("explode")

    assert result.returncode != 0


# ---------------------------------------------------------------------------
# 7. Purity: exact import roots, three named ml functions, no rng/wall clock
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "sys",
        "pathlib",
        "pandas",
        "data",
        "simulation",
        "ml",
    }
)

FORBIDDEN_MODULE_TOKENS = (
    "action_model",
    "action_evaluation",
    "incremental",
    "pooled_model",
    "model_comparison",
    "decision_policy",
)

ALLOWED_ML_IMPORTS = frozenset(
    {
        ("ml.train", "train_baseline"),
        ("ml.train", "predict_recovery_probability"),
        ("ml.evaluate", "calibrate_model"),
    }
)


def cli_source() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def executable_code_without_docstring() -> str:
    source = cli_source()
    docstring = ast.get_docstring(ast.parse(source))
    assert docstring is not None
    assert source.count(docstring) == 1
    return source.replace(docstring, " ", 1)


def test_cli_import_roots_match_whitelist_exactly():
    roots = set()
    for node in ast.walk(ast.parse(cli_source())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    assert roots == ALLOWED_IMPORT_ROOTS, f"import roots drifted: {sorted(roots)}"
    assert "recovery" not in roots


def test_cli_ml_imports_limited_to_three_named_functions():
    seen = set()
    for node in ast.walk(ast.parse(cli_source())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "ml":
            for alias in node.names:
                seen.add((node.module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "ml":
                    seen.add((alias.name, ""))

    assert seen == ALLOWED_ML_IMPORTS


def test_cli_source_contains_no_forbidden_module_tokens():
    code = executable_code_without_docstring()

    for token in FORBIDDEN_MODULE_TOKENS:
        assert token not in code, f"forbidden module token {token!r} found"


def test_cli_spawns_no_rng_and_no_wall_clock_or_stdlib_randomness():
    code = executable_code_without_docstring()

    assert "default_rng" not in code
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
# 8. The generated artifact stays untracked
# ---------------------------------------------------------------------------


def test_gitignore_covers_treatment_outcomes_csv():
    gitignore = REPO_ROOT / ".gitignore"
    if not gitignore.exists():
        # The Docker image strips .gitignore (and git itself) by design; the
        # never-committed guarantee only applies to real checkouts.
        pytest.skip(".gitignore absent from this context (e.g. Docker image)")

    lines = gitignore.read_text(encoding="utf-8").splitlines()

    assert "data/treatment_outcomes.csv" in [line.strip() for line in lines]
