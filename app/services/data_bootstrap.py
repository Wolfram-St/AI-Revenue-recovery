"""Data bootstrap: generate synthetic data and run the recovery engine on startup.

This module creates a reproducible in-memory dataset by calling the existing
data generation and model training functions. The result is cached and used
by the dashboard and case services as a read-only source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from data.generate_dataset import generate_dataset
from data.splits import chronological_split
from ml.evaluate import calibrate_model
from ml.action_model import ActionModelBundle, train_action_models, calibrate_action_models
from ml.train import predict_recovery_probability, train_baseline
from recovery.engine import EngineResult, run_recovery_engine
from recovery.audit import DecisionTrace
from simulation.cli import build_observation_frame
from simulation.config import DEFAULT_POLICY_PATH


TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
BASELINE_SEED = 42
DATASET_SEED = 42
DATASET_ROWS = 1000


@dataclass
class BootstrapResult:
    """Cached result of data bootstrap."""
    traces: tuple[DecisionTrace, ...]
    summary: dict[str, object]
    dataset: pd.DataFrame
    model: Any = field(default=None, repr=False)
    action_bundle: ActionModelBundle | None = None
    observation_frame: pd.DataFrame | None = None


_BOOTSTRAP: BootstrapResult | None = None


def _run_bootstrap() -> BootstrapResult:
    global _BOOTSTRAP
    if _BOOTSTRAP is not None:
        return _BOOTSTRAP

    attempts = generate_dataset(n_rows=DATASET_ROWS, seed=DATASET_SEED)
    train_df, validation_df, _test_df = chronological_split(
        attempts, TRAIN_FRACTION, VALIDATION_FRACTION
    )
    model, _metadata = train_baseline(train_df, validation_df, seed=BASELINE_SEED)
    calibrated = calibrate_model(model, validation_df)
    result = run_recovery_engine(attempts, calibrated)

    observed, _policy = build_observation_frame(DATASET_ROWS, DATASET_SEED, DEFAULT_POLICY_PATH)
    obs_train, obs_val, _obs_test = chronological_split(observed, TRAIN_FRACTION, VALIDATION_FRACTION)
    raw_bundle, _act_meta = train_action_models(obs_train, obs_val)
    action_bundle = calibrate_action_models(raw_bundle, obs_val)

    _BOOTSTRAP = BootstrapResult(
        traces=result.traces,
        summary=result.summary,
        dataset=attempts,
        model=calibrated,
        action_bundle=action_bundle,
        observation_frame=observed,
    )
    return _BOOTSTRAP


def get_bootstrap() -> BootstrapResult:
    """Get or initialize the bootstrap data. Thread-safe for single-worker."""
    return _run_bootstrap()


def reset_bootstrap() -> None:
    """Reset the cached bootstrap (for testing)."""
    global _BOOTSTRAP
    _BOOTSTRAP = None
