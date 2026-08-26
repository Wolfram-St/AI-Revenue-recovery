# Day 5 — Action-Aware Recovery Model

Status: implemented on branch `feature/day5-action-aware-model`.

## What was built

Day 5 fits, calibrates, and honestly evaluates the first action-aware
recovery model — `P(recovered | decision-time context, action)` — on the
randomized observations produced by the Day 4 simulator:

1. `simulation/observations.py` — assembles the joined observation frame
   (attempts context ⋈ assignments ⋈ outcomes ⋈ timeline, one row per
   attempt), stratifies every row into `randomized` (the experimental sample)
   vs `safety_censored`, and exposes chronological split wrappers over
   `data.splits.chronological_split`.
2. `ml/action_model.py` — one Day-2-shaped pipeline per arm
   (`CONTROL`, `RETRY_NOW`, `RETRY_LATER`, `REQUEST_UPDATE`, `HUMAN_REVIEW`)
   fit ONLY on `train ∩ randomized ∩ arm` rows with target EXPLICITLY
   `simulated_recovered` (never the Day-1 `recovered` label; a purity test
   pins this), plus per-arm sigmoid calibration and
   `predict_action_probability` / `predict_all_actions` prediction surfaces.
3. `simulation/outcomes.py` (additive only) — public
   `ground_truth_propensity(df, policy, action)`: deterministic Gauss–Hermite
   (k=20) replay of the NOISE-INTEGRATED true propensity, sharing one logit
   helper with `simulate_outcomes` (behavior pinned by the existing suite).
4. `ml/action_evaluation.py` — per-arm predictive metrics with seeded
   stratified bootstrap 95% CIs (B=500), the Day 2 baseline scored on the
   SAME rows beside each arm slice, ground-truth agreement statistics, and
   the PRIMARY logit-scale effect-contrast recovery checks.
5. `ml/incremental.py` — `IncrementalRecovery(a)` / `IncrementalRevenue(a)`
   reported strictly as `MODEL ESTIMATE` next to their
   `SIMULATED GROUND TRUTH` twins, computed on ONE shared randomized row
   block (fingerprint-checked).

All numbers quoted in DAY5.md/DAY5_RESULTS.md come from one fresh canonical
run: dataset seed 42, chronological 70/15/15, baseline seed 42 calibrated on
validation, policy `master_seed 20260826`, action-model/bootstrap seed
20260826.

## Why randomized-only training data (D-M1)

The Day 4 mechanism gives unconfounded action variation ONLY inside the
randomized stratum. Safety-censored rows are all-CONTROL by rule — they never
received a stage-2 draw (`assignment_probability = 0.0`) — so including them
would teach the models that certain contexts "are control", conflating
eligibility with treatment effect. Consequence, repeated everywhere: the
action-conditional estimates apply to the ELIGIBLE population;
safety-censored contexts have no modeled counterfactual and appear in results
only as out-of-scope counts (canonical run: 1,188 of 5,000 rows full-frame —
830 train / 166 validation / 192 test).

## Per-arm form tradeoff (D-M2)

Five independent calibrated pipelines estimate `P_a(recovered | context)`
directly, which makes `IncrementalRecovery(a) = P̂_a − P̂_CONTROL` transparent
and lets per-arm ground-truth checks isolate estimation error per arm. The
price is sample sharing: no arm's model sees another arm's rows. Canonical
randomized-only fit sizes — CONTROL 528 · RETRY_NOW 785 · RETRY_LATER 686 ·
REQUEST_UPDATE 400 · HUMAN_REVIEW 271 train rows; HUMAN_REVIEW is the
smallest class end to end (60 calibration / 52 test rows). A pooled
action-feature model would share statistical strength across arms and is the
obvious sample-efficiency follow-up (Day 6 option); it was deliberately NOT
built in Day 5.

## Features and label discipline

Model inputs come exclusively through the Day 2 `build_feature_matrix`
whitelist (14 numeric/categorical features, numeric passthrough + one-hot
`handle_unknown="ignore"`). Timestamps, assignment metadata, ground-truth
columns (`base_recovery_propensity`, `action_effect_logit`,
`propensity_under_assignment`), and `assignment_probability`/`arm_source`
are never features — enforced by construction and by tests. The regression
and calibration target is EXPLICITLY `simulated_recovered`. Reporting labels:
`MODEL ESTIMATE` (derived from fitted bundles), `OBSERVED SIMULATED OUTCOME`
(measured on simulated labels), `SIMULATED GROUND TRUTH` (replayed from the
declarative policy); "causal" appears only inside disclaimers.

## Calibration discipline

Each arm's raw pipeline is sealed in `FrozenEstimator` and wrapped in
sigmoid `CalibratedClassifierCV` fitted on exactly
`validation ∩ randomized ∩ arm` rows (mirroring the Day 2 discipline,
one arm at a time). Segments under 100 randomized rows are flagged in
metadata (`small_segments`: validation CONTROL / REQUEST_UPDATE /
HUMAN_REVIEW, plus test REQUEST_UPDATE / HUMAN_REVIEW at evaluation time).
This discipline is necessary but not sufficient at tiny slices: the
canonical run shows sigmoid calibration on 99 rows INVERTED the CONTROL
arm's test ranking (raw ROC-AUC 0.607750 → calibrated 0.392250; Pearson r
flipped +0.513728 → −0.513613). Reported plainly; see Limitations.

## Ground-truth replay methodology (D-M5)

Because the simulator is fully known, the TRUE noiseless arm propensity is
reproducible. The correct population target of P̂ is the noise-integrated
probability `E_ε[sigmoid(base_logit + effect_logit + ε)]` — NOT the pre-noise
`sigmoid(base+effect)` — because a perfectly fitted model converges to the
former. `ground_truth_propensity` computes it via deterministic Gauss–Hermite
quadrature (k=20 nodes) sharing one logit implementation with
`simulate_outcomes`.

- PRIMARY agreement checks are logit-scale contrasts: per-arm
  mean(logit P̂_a) − mean(logit P̂_CONTROL) versus the configured main effect,
  plus interaction-cell contrasts for both configured rules — where the
  Normal(0, 0.5) logit noise is symmetric and additive. Each report records
  its `bundle_kind` and matching documented gate band (±0.25 raw /
  ±0.40 calibrated; sigmoid calibration shrinks logits).
- SECONDARY probability-scale checks report mean |P̂ − integrated TRUE| with
  the Jensen-floor annotation and are NEVER pass/fail gates: even a perfect
  model keeps an irreducible gap to any pre-noise reference, and raw
  pre-noise comparison would carry an arm-dependent Jensen bias up to ~±4 pp.
- Agreement battery per arm: Pearson r AND Spearman ρ between P̂ and
  integrated TRUE — they diverge exactly where the nonlinear transform bites.

These checks validate whether the model RECOVERS THE SYNTHETIC STRUCTURE.
They say nothing about production.

## Baseline comparison findings (D-M7)

Per arm, the evaluator scores the Day 2 baseline pipeline on the SAME test
rows beside the action-aware model:

- Brier (calibrated bundle): the action-aware model beats the Day 2 baseline
  on RETRY_NOW (0.190238 vs 0.247613), REQUEST_UPDATE (0.211347 vs
  0.239075), and HUMAN_REVIEW (0.226339 vs 0.234493); the baseline stays
  ahead on CONTROL (0.254817 vs 0.241738) and RETRY_LATER (0.252379 vs
  0.244136). Micro-averaged: 0.225168 vs 0.242994.
- ROC-AUC (calibrated bundle): the baseline ranks better on most slices;
  only RETRY_NOW improves (0.575986 vs 0.521505), and calibrated CONTROL
  degrades (see above). Raw-bundle AUC improvements are likewise arm-dependent
  (CONTROL raw AUC 0.607750 vs baseline 0.583607 improves, while RETRY_LATER,
  REQUEST_UPDATE, and HUMAN_REVIEW raw AUCs sit below their baselines).
  Per-arm AUC CIs are wide (±0.09–±0.17 half-widths) — none of these deltas
  is individually decisive at n=52–169.

This comparison does NOT isolate the value of action conditioning: the
baseline was trained on Day-1 labels while the action models fit
`simulated_recovered`, so every delta also absorbs the Day-1-to-Day-4 DGP
transfer mismatch (different label-generating process), plus very different
fit sizes (3,500 baseline rows vs 271–785 per arm). Honest reading: action
conditioning sharpens calibration on treated arms; ranking gains are not
established.

## Residual eligibility drift

Rules R006/R007/R008 consume stage-1 recovery probabilities that were
predicted once over the FULL frame, so train-segment probabilities are mildly
in-sample; the eligible boundary therefore shifts slightly between
chronological segments. Canonical evidence: randomized share of segment —
train 2,670/3,500 (76.3%), validation 584/750 (77.9%), test 558/750 (74.4%)
— a drift of a few points across segments, small but nonzero, and part of the
documented selection confounding rather than a new defect.

## Incremental quantities (D-M6)

`IncrementalRecovery(a) = mean P̂_a − mean P̂_CONTROL` over ONE shared
randomized test block (both columns from the same bundle on identical rows;
per-row paired differences also reported; row-set fingerprint proves
alignment). The truth twin replays integrated propensities from the policy.
`IncrementalRevenue(a) = IncrementalRecovery(a) × amount −
RETRY_INTERVENTION_COST_INR − risk_penalty(unknown-category)`, constants
imported from `recovery.scoring` (never restated). Disclosed simplification,
embedded verbatim in every revenue table: a single retry-cost constant is
applied uniformly to ALL treated arms including REQUEST_UPDATE/HUMAN_REVIEW
whose true economics differ. Captured model-vs-truth numbers live in
DAY5_RESULTS.md; headline honesty note: model estimates OVERSHOOT truth on
every treated arm (+0.038 to +0.077 absolute) and mis-order RETRY_LATER vs
REQUEST_UPDATE.

## What this does NOT prove

Nothing here estimates real-world uplift, incremental causal revenue, or
production treatment effects.

The Day 2 baseline remains P(recovered | context) and was not modified.

All action-aware estimates describe the synthetic world defined by the Day 4 simulator; no real-world causal or production claim is made.

No baseline or simulator behavior file changed except the sanctioned additive
public function in `simulation/outcomes.py`; the incremental tables are NOT
an optimization target in Day 5 — no optimizer, bandit, uplift estimator, or
threshold search consumes them.

## Known limitations

- **Smallest-arm variance:** HUMAN_REVIEW has 52 randomized test rows
  (bootstrap AUC CI width ≈ 0.34); all per-arm CIs are wide; single-run
  orderings can flip under reseeding.
- **Calibration fragility at tiny slices:** sigmoid calibration on 99
  randomized CONTROL validation rows inverted that arm's test ranking
  (documented above); a minimum-rows guard or cross-fitted calibrator is a
  Day 6 candidate.
- **Pooled-bootstrap indexing (found and fixed during verification):** the
  Task 6 canonical capture exposed that the micro-average block's pooled
  strata were built without pool offsets, so every resample drew from the
  leading (CONTROL) segment and the originally reported calibrated micro AUC
  point 0.603136 sat outside its CI95 [0.428128, 0.526357]. Fixed minimally
  in `ml/action_evaluation.py` with running-offset strata and regression-
  pinned by
  `test_micro_average_bootstrap_ci_contains_point_estimate_under_heterogeneous_arms`;
  corrected micro CIs bracket their point estimates for both bundle kinds,
  per-arm CIs are byte-identical pre/post fix. See DAY5_RESULTS.md.
- **Raw-vs-calibrated band difference:** raw bundles are gated at ±0.25
  logit, calibrated at ±0.40; canonical raw contrasts missed the band on
  RETRY_LATER (−0.5095) and HUMAN_REVIEW (+0.3867) while all calibrated
  contrasts landed inside ±0.40 — band choice materially changes verdicts
  and is always reported alongside `bundle_kind`.
- **Finite-sample noise:** effect-contrast gaps and interaction cells carry
  sampling error the bands do not formally propagate; the fatigue cell
  (n_cell=24) attenuated to ~0 after calibration, consistent with its
  `attenuation_expected` annotation.
- **No pooled model, no optimizer:** sample-efficient pooling and any
  threshold/action optimization remain future work by design.
- All numbers measure the synthetic simulator, not any production system.

## Reproducibility

One action-model seed family (20260826) drives XGBoost `random_state`; the
evaluator derives its bootstrap generator exactly once from the named seed;
`ml/incremental.py` draws no randomness at all. Canonical rerun check: two
fresh `evaluate_action_models` calls on identical inputs produce identical
sorted-JSON SHA-256 digests
(`984541b156fe959e731f34b52cf0fcace98423c8fc8ed88db55f7a382a858b67`), and the
digest reproduced identically across two separate script executions.

## Test evidence

Day 5 suites (focused counts, fresh):
`tests/test_observations.py` 28 · `tests/test_action_model.py` 37 ·
`tests/test_action_evaluation.py` 34 · `tests/test_incremental.py` 24 —
123 new; plus 462 pre-existing = **585 passed** locally
(`585 passed in 41.13s`) and identically inside Docker
(`585 passed in 43.65s`). Full tails in DAY5_RESULTS.md.

## Day 6 options

1. Deepen the pooled view now that its bootstrap is fixed and pinned:
   per-arm contribution diagnostics for the micro block (which arm drives
   the pooled statistic, with leave-one-arm-out contrasts).
2. Sample efficiency: prototype the pooled action-feature model against the
   per-arm form on identical splits, judged on the same ground-truth battery.
3. Calibration robustness for small slices: minimum-rows guard, cross-fitted
   sigmoid, or shrinkage toward the raw pipeline.
4. Deferred plumbing from Day 4: `data/treatment_outcomes.csv` writer +
   `python -m simulation.generate` CLI wrapper (module APIs still return
   frames only).
5. Seed-sensitivity sweep (multiple master seeds) to quantify finite-sample
   ordering noise in the incremental tables.
6. Out-of-sample stage-1 probability refits per segment to measure — not just
   note — eligibility drift.
7. Threshold/action optimization remains out of scope unless explicitly
   re-scoped with its own plan.
