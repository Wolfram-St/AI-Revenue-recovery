"""Reproducible synthetic treatment/outcome dataset artifact (Day 6 Gate B).

This CLI wraps the frozen Day 4 chain -- the Day 2 baseline probability
model (trained on the chronologically earliest segment, sigmoid-calibrated
on the next), the policy-seeded assignment/outcome/timeline simulator over
the canonical attempts frame, and the dataset validator -- behind three
subcommands:

``generate`` assembles the full observation frame, validates it against the
column contract plus every treatment-dataset rule, writes it to CSV, and
prints one JSON line recording BOTH seeds distinctly: ``--seed`` is the
DATASET-generation seed while ``policy_master_seed`` always comes from the
treatment-policy YAML. Same rows + seed reproduce a byte-identical file.

``validate`` re-loads a written CSV (timestamps parsed back) and prints the
structured report; exit code reflects validity. ``summary`` prints per-arm
counts/rates plus overlap diagnostics, labeled OBSERVED SIMULATED OUTCOME.

No wall clock enters any output: the only fingerprint is the SHA-256 of the
written file itself. Purity per plan D-E3: stdlib/pandas plus this repo's
simulation/data packages, and EXACTLY the three documented baseline-chain
functions from ml.train/ml.evaluate -- no action-model family modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from data.generate_dataset import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    TIME_COLUMN,
    generate_dataset,
)
from data.splits import chronological_split
from ml.evaluate import calibrate_model
from ml.train import predict_recovery_probability, train_baseline
from simulation.config import DEFAULT_POLICY_PATH, TreatmentPolicy, load_treatment_policy
from simulation.dataset import CANONICAL_COLUMNS, validate_treatment_dataset
from simulation.observations import (
    APPENDIX_COLUMNS,
    STRATUM_COLUMN,
    STRATUM_SAFETY_CENSORED,
    STRATUM_RANDOMIZED,
    assemble_observations,
)
from simulation.reporting import overlap_diagnostics, summarize_arms

DEFAULT_OUT_PATH = "data/treatment_outcomes.csv"
BASELINE_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

_ATTEMPTS_COLUMNS = (
    tuple(FEATURE_COLUMNS[:3])
    + (TIME_COLUMN,)
    + tuple(FEATURE_COLUMNS[3:])
    + tuple(LABEL_COLUMNS)
)
EXPECTED_OBSERVATION_COLUMNS = _ATTEMPTS_COLUMNS + tuple(APPENDIX_COLUMNS)

_TIMESTAMP_COLUMNS = ("event_timestamp", "treatment_timestamp", "outcome_timestamp")

_READ_CHUNK_BYTES = 65536


def build_observation_frame(
    rows: int, dataset_seed: int, policy_path: str | Path
) -> tuple[pd.DataFrame, TreatmentPolicy]:
    """Run the frozen chain end to end: baseline fit -> calibrated
    probabilities over EVERY attempt row -> assembled observations."""
    attempts = generate_dataset(rows, seed=dataset_seed)
    train_df, validation_df, _test_df = chronological_split(
        attempts, TRAIN_FRACTION, VALIDATION_FRACTION
    )
    model, _metadata = train_baseline(train_df, validation_df, seed=BASELINE_SEED)
    calibrated = calibrate_model(model, validation_df)
    probabilities = predict_recovery_probability(calibrated, attempts)
    policy = load_treatment_policy(policy_path)
    observed = assemble_observations(attempts, probabilities, policy)
    return observed, policy


def validate_observation_frame(frame: pd.DataFrame) -> dict:
    """Column-contract check for the assembled frame, then every treatment-
    dataset rule applied to its canonical projection, then the stratum
    vocabulary that only the assembled frame carries."""
    violations: list[str] = []
    expected = list(EXPECTED_OBSERVATION_COLUMNS)
    actual = list(frame.columns)
    missing = [column for column in expected if column not in set(actual)]
    unexpected = [column for column in actual if column not in set(expected)]
    if missing or unexpected or actual != expected:
        violations.append(
            "columns deviate from the assembled observation contract: "
            f"unexpected={unexpected}, missing={missing}, order={actual}"
        )
        return {
            "valid": False,
            "violations": violations,
            "row_count": int(len(frame)),
            "column_count": int(frame.shape[1]),
            "classification_complete": False,
            "column_contract_valid": False,
        }

    stratum = frame[STRATUM_COLUMN]
    offenders = sorted(
        set(stratum.dropna().tolist()) - {STRATUM_RANDOMIZED, STRATUM_SAFETY_CENSORED}
    )
    if offenders:
        violations.append(
            f"stratum must be one of {[STRATUM_RANDOMIZED, STRATUM_SAFETY_CENSORED]}; "
            f"offenders: {offenders}"
        )

    projection = frame.loc[:, list(CANONICAL_COLUMNS)]
    report = validate_treatment_dataset(projection)
    violations.extend(report["violations"])
    return {
        "valid": not violations,
        "violations": violations,
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "classification_complete": report["classification_complete"],
        "column_contract_valid": True,
    }


def load_observation_csv(path: str | Path) -> pd.DataFrame:
    """Read a written observation CSV, restoring timezone-aware timestamps
    (empty CONTROL treatment cells become NaT)."""
    frame = pd.read_csv(path)
    for column in _TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
    return frame


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_generate(args: argparse.Namespace) -> int:
    observed, policy = build_observation_frame(args.rows, args.seed, args.policy)
    report = validate_observation_frame(observed)
    if not report["valid"]:
        print(json.dumps(report))
        print(json.dumps({"error": "generated frame failed validation"}), file=sys.stderr)
        return 1
    destination = Path(args.out)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    observed.to_csv(destination, index=False)
    payload = {
        "dataset_seed": int(args.seed),
        "policy_master_seed": int(policy.master_seed),
        "rows": int(len(observed)),
        "path": str(destination),
        "sha256": _sha256_of_file(destination),
    }
    print(json.dumps(payload))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    frame = load_observation_csv(args.csv)
    report = validate_observation_frame(frame)
    print(json.dumps(report))
    return 0 if report["valid"] else 1


def command_summary(args: argparse.Namespace) -> int:
    frame = load_observation_csv(args.csv)
    report = {
        "arms_summary": summarize_arms(frame),
        "overlap_diagnostics": overlap_diagnostics(frame),
    }
    print(json.dumps(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simulation.cli",
        description=(
            "Reproducible synthetic treatment/outcome dataset artifact "
            "(Day 6 evidence gate)."
        ),
    )
    commands = parser.add_subparsers(
        dest="command",
        metavar="{generate,validate,summary}",
        required=True,
    )

    generate = commands.add_parser(
        "generate",
        help="assemble the observation frame, validate it, write CSV, print seed/sha256 JSON",
    )
    generate.add_argument("--rows", type=int, default=5000, help="attempt count (default 5000)")
    generate.add_argument(
        "--seed",
        type=int,
        default=42,
        help="DATASET-generation seed (default 42 = canonical world)",
    )
    generate.add_argument("--out", default=DEFAULT_OUT_PATH, help="destination CSV path")
    generate.add_argument(
        "--policy",
        default=DEFAULT_POLICY_PATH,
        help="treatment-policy YAML supplying the master seed",
    )
    generate.set_defaults(handler=command_generate)

    validate = commands.add_parser(
        "validate", help="run contract+dataset checks on a written CSV and print the report"
    )
    validate.add_argument("--csv", default=DEFAULT_OUT_PATH, help="observation CSV path")
    validate.set_defaults(handler=command_validate)

    summary = commands.add_parser(
        "summary", help="print per-arm OBSERVED outcome summary and overlap diagnostics"
    )
    summary.add_argument("--csv", default=DEFAULT_OUT_PATH, help="observation CSV path")
    summary.set_defaults(handler=command_summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    sys.exit(main())
