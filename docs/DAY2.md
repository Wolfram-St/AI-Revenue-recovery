# Day 2 — Baseline Recovery Model

Status: implemented on branch `feature/day2-baseline-model`.

## What was built

1. `data/validate_dataset.py` — structured contract validation (19-column
   contract, nulls, duplicate identifiers, binary label, allowed failure
   categories, nondecreasing UTC timestamps).
2. `data/splits.py` — chronological 70/15/15 split; never shuffles.
3. `ml/features.py` — decision-time feature matrix: 8 numeric + 6 categorical
   features; identifiers, `event_timestamp`, `recovered`,
   `recovery_time_hours`, and reserved action-aware columns are excluded by
   construction (`FORBIDDEN_FEATURES`).
4. `ml/train.py` — XGBoost pipeline estimating
   `P(recovered | decision-time context)`; preprocessing fitted on train only;
   explicit seed; returns reproducibility metadata.
5. `ml/evaluate.py` — ROC-AUC / PR-AUC / Brier, thresholded-simulation
   revenue metrics in INR, and sigmoid calibration via a frozen fitted
   pipeline (validation data only, test labels untouched).

## Why

The Day 2 goal is the smallest credible predictive core of the recovery
loop: a leakage-safe probability estimate plus honest measurement of what it
is worth in rupees. XGBoost is the specified strong tabular baseline.

## Assumptions and decisions

- **Sigmoid calibration** over isotonic: the validation segment is only 750
  rows; isotonic needs more data to avoid step artifacts.
- **FrozenEstimator** wraps the fitted pipeline so calibration cannot refit
  model weights (requires scikit-learn >= 1.7).
- **No early stopping**: fixed rounds keep training bit-reproducible across
  runs for a fixed seed.
- **Thresholded simulation is labeled as such** everywhere revenue numbers
  appear; no causal per-action claims are made from the general label.
- **ROC-AUC ≈ 0.64 accepted**: consistent with the documented irreducible
  noise in the synthetic outcome mechanism (see DAY2_RESULTS.md §sanity).

## Measured (full canonical run)

See `DAY2_RESULTS.md` for exact metrics. Headlines: calibrated Brier
0.2331 vs raw 0.2440; selected-case recovery rate 57.7% vs 46.1% base rate
on the latest 15% holdout.

## Known limitations

- The label answers "did this payment eventually recover", not "would action
  A have recovered it". Action-aware modeling remains blocked until the
  dataset gains action-assigned observations or an explicitly simulated
  treatment policy.
- Synthetic-data caveat: metrics measure fit to the simulator, not to any
  production environment.
- Revenue metrics depend on the chosen threshold (0.5); threshold
  optimization is deferred until policy constraints exist downstream.

## What comes next

Day 3 candidate work: deterministic policy engine, opportunity scoring by
expected recovered value with bounded actions, and the controlled workflow —
each gated on this baseline's contracts.
