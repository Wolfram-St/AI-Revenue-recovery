# Day 5 Results — Action-Aware Recovery Model

All numbers below are fresh outputs from the canonical run captured verbatim:
dataset regenerated with `generate_dataset(5000, seed=42)`, chronological
70/15/15 split, baseline retrained with seed 42 and sigmoid-calibrated on
validation only, calibrated probabilities predicted over the FULL 5,000-row
frame, observation frame assembled under policy `master_seed 20260826`,
per-arm action models trained/calibrated with seed 20260826, evaluation
bootstrap B=500 seed 20260826.

## Run configuration

| Item | Value |
| --- | --- |
| Dataset seed | 42 |
| Rows | 5,000 × 19 context columns → observation frame 5,000 × 30 |
| Split | chronological 70/15/15 over observations (train 3,500 / validation 750 / test 750) |
| Day 2 baseline | XGBoost (300×d4, lr 0.1), sigmoid calibration fit on validation; untouched otherwise |
| Probabilities fed to stage 1 | min 0.237922 / max 0.722157 / mean 0.479985 over 5,000 rows |
| Policy | v1.0 (`config/treatment_policy.yaml`), master_seed 20260826 |
| Action-model seed | 20260826 (`train_action_models(..., seed=20260826)`) |
| Calibration | per-arm sigmoid `CalibratedClassifierCV(FrozenEstimator(...))`, fit on validation∩randomized∩arm |
| Evaluation bootstrap | stratified within-arm row resampling, B=500, seed 20260826, level 0.95 |
| Bundle kinds evaluated | raw (gate band ±0.25 logit) AND calibrated (gate band ±0.40 logit) |
| Strata | randomized 3,812 · safety_censored 1,188 (full frame) |

## Captured output — splits and strata

| Segment | rows | randomized | safety_censored | first_ts → last_ts |
| --- | --- | --- | --- | --- |
| TRAIN | 3,500 | 2,670 | 830 | 2026-01-01 00:00 → 2026-02-06 10:45 |
| VALIDATION | 750 | 584 | 166 | 2026-02-06 11:00 → 2026-02-14 06:15 |
| TEST | 750 | 558 | 192 | 2026-02-14 06:30 → 2026-02-22 01:45 |

SAFETY-CENSORED OUT-OF-SCOPE (never fitted, never action-conditionally
evaluated): full_frame **1,188** · train 830 · validation 166 · test 192.
Arm sources full frame: CONTROL/randomized 743 · CONTROL/safety_censored
1,188 · HUMAN_REVIEW/randomized 383 · REQUEST_UPDATE/randomized 581 ·
RETRY_LATER/randomized 973 · RETRY_NOW/randomized 1,132.

## Captured output — per-arm row counts

| Arm | train | calibration (val) | test | small_segments flagged (<100) |
| --- | --- | --- | --- | --- |
| CONTROL | 528 | 99 | 116 | validation:CONTROL |
| RETRY_NOW | 785 | 178 | 169 | none |
| RETRY_LATER | 686 | 154 | 133 | none |
| REQUEST_UPDATE | 400 | 93 | 88 | validation + test |
| HUMAN_REVIEW | 271 | 60 | 52 | validation + test |

Metadata `small_segments` = [('validation','CONTROL'),
('validation','REQUEST_UPDATE'), ('validation','HUMAN_REVIEW')]; test-side
flags recorded by the evaluator. Calibration method=sigmoid,
fit_on=validation_randomized_only.

## Captured output — predictive metrics, bundle_kind=raw (band ±0.25)

OBSERVED SIMULATED OUTCOME. CIs are seeded stratified bootstrap 95%
(B=500). `brier_day2`/`auc_day2` = Day 2 baseline on the SAME rows.

| Arm | n | AUC model [CI95] | Brier model [CI95] | Brier day2 | AUC day2 |
| --- | --- | --- | --- | --- | --- |
| CONTROL | 116 | 0.607750 [0.504715, 0.712078] | 0.280993 [0.226842, 0.336929] | 0.241738 | 0.583607 |
| RETRY_NOW | 169 | 0.575986 [0.472537, 0.687725] | 0.215274 [0.174124, 0.258368] | 0.247613 | 0.521505 |
| RETRY_LATER | 133 | 0.565977 [0.476467, 0.658892] | 0.287546 [0.242552, 0.334771] | 0.244136 | 0.596552 |
| REQUEST_UPDATE | 88 | 0.543412 [0.404106, 0.673706] | 0.257443 [0.197916, 0.318362] | 0.239075 | 0.587735 |
| HUMAN_REVIEW | 52 | 0.563725 [0.398850, 0.734348] | 0.274803 [0.191348, 0.360701] | 0.234493 | 0.604575 |
| MICRO (all) | 558 | 0.592562 [0.542630, 0.641043] | 0.258360 [0.233410, 0.281238] | 0.242994 | 0.566280 |

Per-arm CIs use the correct single-stratum path; the MICRO row reflects the
corrected pooled-stratum indexing (defect fix below) and contains its point
estimate for both AUC and Brier.

## Captured output — predictive metrics, bundle_kind=calibrated (band ±0.40)

| Arm | n | AUC model [CI95] | Brier model [CI95] | Brier day2 | AUC day2 |
| --- | --- | --- | --- | --- | --- |
| CONTROL | 116 | 0.392250 [0.287922, 0.495285] ⚠ | 0.254817 [0.246318, 0.262768] | 0.241738 | 0.583607 |
| RETRY_NOW | 169 | 0.575986 [0.472537, 0.687725] | 0.190238 [0.162566, 0.219678] | 0.247613 | 0.521505 |
| RETRY_LATER | 133 | 0.565977 [0.476467, 0.658892] | 0.252379 [0.225628, 0.282068] | 0.244136 | 0.596552 |
| REQUEST_UPDATE | 88 | 0.543412 [0.404106, 0.673706] | 0.211347 [0.181159, 0.245417] | 0.239075 | 0.587735 |
| HUMAN_REVIEW | 52 | 0.563725 [0.398850, 0.734348] | 0.226339 [0.187883, 0.270963] | 0.234493 | 0.604575 |
| MICRO (all) | 558 | 0.603136 [0.558902, 0.653195] | 0.225168 [0.211186, 0.236742] | 0.242994 | 0.566280 |

⚠ Recorded honest finding: sigmoid calibration on 99 randomized CONTROL
validation rows inverted that arm's test ranking (raw 0.607750 → calibrated
0.392250; Pearson r flipped +0.513728 → −0.513613). Reported plainly; small-
slice calibration robustness is a Day 6 candidate.

Defect found and FIXED during Task 6 verification: the first canonical
capture reported the calibrated micro row as 0.603136 with CI95
[0.428128, 0.526357] — point estimate outside its own interval, impossible
under a correctly indexed percentile bootstrap. Root cause: the pooled
strata passed to the bootstrap were built WITHOUT pool offsets
(`np.arange(n_arm)` per arm), so applied against the concatenated arrays
every resample drew from the leading (CONTROL) segment. Fix (minimal, in
`ml/action_evaluation.py`): running-offset index blocks per arm, preserving
seeded determinism and stratified-within-arm semantics; the per-arm path is
untouched and its numbers are byte-identical pre/post fix (verified by
diffing the two captures). Regression-pinned by
`test_micro_average_bootstrap_ci_contains_point_estimate_under_heterogeneous_arms`.
Corrected micro rows above contain their point estimates for BOTH bundle
kinds; point estimates themselves never changed.

## Captured output — ground-truth agreement (SIMULATED GROUND TRUTH replay)

Secondary probability-scale comparison with Jensen annotation ("secondary
comparison; carries the documented Jensen floor (models converge to the
noise-integrated propensity, not the pre-noise sigmoid)"):

| Arm | mean\|P̂−T\| raw | Pearson r / Spearman ρ raw | mean\|P̂−T\| calib | Pearson r / Spearman ρ calib |
| --- | --- | --- | --- | --- |
| CONTROL | 0.207757 | +0.513728 / +0.501242 | 0.125058 | −0.513613 / −0.501242 ⚠ |
| RETRY_NOW | 0.159677 | +0.569899 / +0.571473 | 0.102158 | +0.567640 / +0.571473 |
| RETRY_LATER | 0.206420 | +0.510854 / +0.485386 | 0.088636 | +0.516684 / +0.485386 |
| REQUEST_UPDATE | 0.184020 | +0.474850 / +0.445088 | 0.079998 | +0.473143 / +0.445088 |
| HUMAN_REVIEW | 0.244253 | +0.421087 / +0.439512 | 0.088310 | +0.421077 / +0.439512 |

Calibration shrinks predictions toward the integrated-truth scale, cutting
mean |P̂−T| by roughly half on every arm — while simultaneously inverting the
CONTROL ranking (⚠ above).

PRIMARY check — logit-scale effect contrasts vs configured effects:

| Arm | configured | estimated raw | gap raw (band ±0.25) | estimated calib | gap calib (band ±0.40) |
| --- | --- | --- | --- | --- | --- |
| CONTROL | 0.00 | 0.0000 | 0.0000 yes | 0.0000 | 0.0000 yes |
| RETRY_NOW | +0.60 | 0.6385 | +0.0385 yes | 0.8324 | +0.2324 yes |
| RETRY_LATER | +0.35 | −0.1595 | −0.5095 **NO** | 0.5693 | +0.2193 yes |
| REQUEST_UPDATE | +0.45 | 0.6017 | +0.1517 yes | 0.5479 | +0.0979 yes |
| HUMAN_REVIEW | +0.25 | 0.6367 | +0.3867 **NO** | 0.5424 | +0.2924 yes |

Raw verdicts are honest misses (RETRY_LATER's fatigue interaction drags its
aggregate contrast negative at finite n; HUMAN_REVIEW overshoots its weak
+0.25 effect with only 116 control rows anchoring the contrast); after
calibration every arm lands inside its ±0.40 band, though gaps remain
positive — models spread arms apart more than the truth does.

Interaction cells:

| Cell | configured | estimated raw | gap raw | estimated calib | gap calib | n_cell | attenuation_expected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RETRY_NOW \| failure_category==temporary_decline (GATED) | +0.40 | 0.1992 | −0.2008 (inside ±0.25) | 0.3901 | −0.0099 | 79 | False |
| RETRY_LATER \| attempt_number>=3 (annotated) | −0.25 | −0.3408 | −0.0908 | +0.0436 | +0.2936 | 24 | True |

The gated RETRY_NOW interaction is recovered inside both bands. The fatigue
cell carries `attenuation_expected=True`: at n_cell=24 with gradient-boosting
shrinkage the weak −0.25 effect attenuates toward zero after calibration
(+0.0436), exactly the annotated behavior rather than a gate.

## Determinism

Two fresh `evaluate_action_models` calls on identical inputs:

```
run A sha256(sorted-json): 984541b156fe959e731f34b52cf0fcace98423c8fc8ed88db55f7a382a858b67
run B sha256(sorted-json): 984541b156fe959e731f34b52cf0fcace98423c8fc8ed88db55f7a382a858b67
match: True   (digest also reproduced identically across two separate script executions)
```

The digest differs from the pre-fix capture (cba0d81d…) exactly because the
micro CI values changed under the corrected indexing; every point estimate,
per-arm number, contrast, and incremental quantity is identical between the
two captures.

bundle_kind recorded: `calibrated`, gate_band_logit ±0.40 (raw run records
`raw`, ±0.25).

## Incremental recovery — MODEL ESTIMATE vs SIMULATED GROUND TRUTH

n_randomized_test_rows = 558; identical row set for both tables
(fingerprint match 1202702195678624065). Paired per-row differences computed
on identical rows.

| Arm | model mean Δ | model paired median | truth mean Δ | truth paired median | model − truth |
| --- | --- | --- | --- | --- | --- |
| RETRY_NOW | 0.1934 | 0.2066 | 0.1536 | 0.1406 | +0.0398 |
| RETRY_LATER | 0.1440 | 0.1579 | 0.0671 | 0.0734 | +0.0769 |
| REQUEST_UPDATE | 0.1317 | 0.1434 | 0.0937 | 0.0968 | +0.0380 |
| HUMAN_REVIEW | 0.1295 | 0.1339 | 0.0531 | 0.0551 | +0.0764 |

Honest reading: direction is positive for every treated arm in both worlds,
but model estimates OVERSHOOT truth by +0.038 to +0.077 absolute (consistent
with the positive contrast gaps above), and the model orders RETRY_LATER
above REQUEST_UPDATE while truth orders the reverse.

## Incremental revenue — MODEL ESTIMATE vs SIMULATED GROUND TRUTH

Constants imported from `recovery.scoring`: intervention_cost_inr = 10.0 ·
unknown_category_risk_fraction = 0.05 (risk-penalty applied to 74 unknown-
category rows in each table).

| Arm | model ₹/case | model total ₹ | truth ₹/case | truth total ₹ |
| --- | --- | --- | --- | --- |
| RETRY_NOW | 532.54 | 297,157.11 | 445.48 | 248,576.36 |
| RETRY_LATER | 404.29 | 225,594.01 | 174.05 | 97,118.74 |
| REQUEST_UPDATE | 350.26 | 195,446.06 | 256.53 | 143,141.51 |
| HUMAN_REVIEW | 361.17 | 201,535.10 | 130.44 | 72,787.87 |

`cost_simplification_note` (embedded verbatim in every revenue table): "A
single retry-cost constant is applied uniformly to all treated arms including
REQUEST_UPDATE and HUMAN_REVIEW, whose true economics differ." All figures
are labeled MODEL ESTIMATE or SIMULATED GROUND TRUTH at source, are non-causal,
and are NOT an optimization target in Day 5.

## Verification evidence

Focused counts per suite file (fresh runs):

| Suite file | Tests |
| --- | --- |
| tests/test_observations.py | 28 |
| tests/test_action_model.py | 37 |
| tests/test_action_evaluation.py | 34 |
| tests/test_incremental.py | 24 |
| Day 5 subtotal | 123 |
| Pre-existing (Day 1–4 suites) | 462 |
| **Total** | **585** |

Per-commit history (reconstructed fresh from detached-worktree reruns;
the micro-bootstrap indexing fix landed after `9cf0df1` and adds one
evaluation test):

| Commit | Focused suite at commit | Full suite at commit |
| --- | --- | --- |
| `bebcc5a` Task 1 | test_observations.py 28 | 490 |
| `9d97220` Task 2 | test_action_model.py 29 | 519 |
| `92dd16e` Task 3 | test_action_model.py 37 | 527 |
| `b47603b` Task 4 | test_action_evaluation.py 33 | 560 |
| `9cf0df1` Task 5 | test_incremental.py 24 | 584 |
| micro-bootstrap fix (this working tree) | test_action_evaluation.py 34 | 585 |

Full-suite tails (fresh runs, both environments, with the fix applied):

- Local: `.venv\Scripts\python -m pytest -q` → `585 passed in 41.13s`
- Docker: `docker build -t recoverai-app .` → OK;
  `docker run --rm recoverai-app python -m pytest -q` → `585 passed in 43.65s`

RED/GREEN evidence for the regression pin: the new containment test FAILED
against unfixed code (buggy micro AUC CI95 lower bound 1.0 vs pooled point
0.598994 — every resample drew from the near-separable CONTROL segment) and
passes after the offset fix; all 33 pre-existing evaluation tests stayed
green unchanged before and after, and per-arm captured rows are identical
across the two captures.

Artifact check: `data/treatment_outcomes.csv` still absent (CLI writer
deferred to Day 6 option); git tree clean apart from the three new Day 5
documentation files.

## Day 5 gate

| Gate check | Evidence | Result |
| --- | --- | --- |
| Canonical pipeline executed end-to-end with real captured outputs | dataset → baseline → probabilities → assemble → split → fit → calibrate → evaluate (raw+calibrated) → incremental tables; every table above verbatim | PASS |
| Training population randomized-only (D-M1) | metadata train rows == randomized∩arm counts (528/785/686/400/271); safety-censored excluded everywhere; out-of-scope count 1,188 reported | PASS |
| Ground-truth replay methodology (D-M5) | noise-integrated Gauss–Hermite k=20 public replay sharing one logit helper; behavior pinned by existing outcomes tests; primary checks logit-scale, secondary annotated | PASS |
| Primary effect-contrast recovery vs kind-aware bands | calibrated: all five arms inside ±0.40 (gaps +0.0000…+0.2924); gated RETRY_NOW×temporary_decline cell recovered (gap −0.0099); raw band misses on RETRY_LATER/HUMAN_REVIEW reported plainly; fatigue cell annotated attenuation_expected | PASS |
| Predictive metrics with bootstrap CIs vs Day 2 baseline on same slices (D-M7) | per-arm AUC/Brier ± CI95 for model and baseline side by side; DGP-transfer-mismatch caveat documented | PASS |
| Calibration discipline | per-arm sigmoid on validation∩randomized∩arm via FrozenEstimator; small_segments recorded; CONTROL tiny-slice inversion honestly reported as limitation | PASS (with recorded caveat) |
| Incremental reporting labels + uniform-cost disclosure (D-M6) | MODEL ESTIMATE / SIMULATED GROUND TRUTH embedded; row-set fingerprint match; cost_simplification_note verbatim; model-vs-truth overshoot stated | PASS |
| Label discipline | no causal estimators; "causal" only inside disclaimers; OBSERVED SIMULATED OUTCOME / SIMULATED GROUND TRUTH vocabulary throughout | PASS |
| Baseline boundary held | Day 2 baseline untouched (additive-only change in simulation/outcomes.py); purity tests pin `simulated_recovered` target; ground-truth columns never features | PASS |
| Determinism | two-run sorted-json SHA-256 identical (984541b1…); reproduced across script executions | PASS |
| Full test suite green | 123 new + 462 pre-existing = 585 passed locally AND in Docker | PASS |
| Pooled (micro-average) bootstrap CI reliability | offset defect found by the Task 6 capture → RED regression pin (buggy CI lower bound 1.0 vs point 0.598994) → minimal running-offset fix in `ml/action_evaluation.py` → corrected micro CIs contain their point estimates for BOTH bundle kinds; per-arm rows byte-identical pre/post fix | **PASS** (fixed within Day 5) |
| Documentation complete | DAY5.md covers build, D-M1 rationale, tradeoffs, calibration, replay methodology, baseline caveats, eligibility drift, limitations, Day 6 options; both mandated sentences verbatim | PASS |

## GO/NO-GO for Day 6: **GO**

GO for Day 6 action-aware iteration, inside the unchanged synthetic-world
boundary. The one defect surfaced by the Task 6 verification gate — the
micro-average block's pooled-bootstrap strata lacked pool offsets, letting
every resample draw from the leading CONTROL segment — was diagnosed,
fixed minimally, and regression-pinned WITHIN Day 5: corrected micro CIs
bracket their point estimates for both raw and calibrated bundles, all
pre-existing tests pass unchanged, and per-arm evidence is untouched. The
honest weaknesses that remain (tiny-slice calibration fragility on CONTROL,
wide smallest-arm CIs, model-over-truth overshoot in incremental contrasts)
are documented as limitations and mapped to concrete Day 6 options rather
than smoothed over. Nothing in Day 5 supports any real-world causal or
production claim, and the Day 2 baseline remains exactly what it was.
