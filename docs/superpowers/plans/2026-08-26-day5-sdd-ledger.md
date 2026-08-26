# SDD Progress Ledger — Day 5 Action-Aware Recovery Model

Plan: `docs/superpowers/plans/2026-08-26-day5-action-aware-model.md`
Workflow: subagent-driven development (fresh implementer + independent reviewer per task, fix/re-review loops)
Branch/worktree: `feature/day5-action-aware-model` @ `D:/airev-day5` (base: `2848d57` merged Day 4 main; plan commits `4e2dd2c`→`1c10bf9` incl. three plan-review fix rounds)
Boundary: action-aware estimates are synthetic-world-only (`simulated_recovered` target); Day 2 baseline `P(recovered | context)` untouched; randomized-stratum training only (D-M1).

Provenance note: this ledger was written retroactively at Task 6 from
verifiable artifacts — the five task commits, fresh detached-worktree pytest
reruns per commit, and the reviewers' fix annotations embedded in the shipped
code/tests (`review F1…F9` markers). No verdict below claims more provenance
than that.

| Task | Scope | Commit | Focused tests | Full suite (local; Docker at HEAD) | Implementer review | Re-review |
|---|---|---|---|---|---|---|
| 1 | Observation assembly (attempts ⋈ assignments ⋈ outcomes ⋈ timeline), randomized/safety_censored strata, chronological split wrappers | `bebcc5a` | 28 | 490 / — | APPROVE (no fix-loop markers in shipped code) | n/a (approved) |
| 2 | Per-arm pipeline fitting on randomized∩arm, explicit `simulated_recovered` target + purity guards | `9d97220` | 29 | 519 / — | APPROVE after in-task review loop (F1: paired-difference gate statistic + tolerance restated, marker in test_action_model.py) | n/a (approved) |
| 3 | Per-arm sigmoid calibration on validation via FrozenEstimator, small_segments flags, validation-remainder Brier sanity check | `92dd16e` | 37 | 527 / — | APPROVE after in-task review loop (tolerance derivation restated honestly per review F1) | n/a (approved) |
| 4 | Ground-truth replay (`ground_truth_propensity`, Gauss–Hermite k=20, shared logit helper refactor) + evaluation battery vs Day 2 baseline on same slices | `b47603b` | 33 | 560 / — | APPROVE after heaviest review loop (F1 tie-corrected AP equivalence probe; F2 kind-aware gate band; F3 gated-vs-annotated cells; F4 baseline AUC beside Brier; F5 consolidated finiteness guard; F6 noise-sigma discipline; F7 sklearn-equivalence all arms; F8 thin-arm branch; F9 loud feature-column guard — markers across ml/action_evaluation.py, simulation/outcomes.py, tests) | n/a (approved) |
| 5 | Incremental recovery/revenue tables, MODEL ESTIMATE vs SIMULATED GROUND TRUTH twins, row-set fingerprints, uniform-cost disclosure | `9cf0df1` | 24 | 584 / — | APPROVE after in-task review loop (F1 cross-check: attempt_id fingerprint equality required when both sides computable, marker in ml/incremental.py) | n/a (approved) |
| 6 | Documentation + Day 5 gate (DAY5.md, DAY5_RESULTS.md, this ledger; canonical-run capture incl. two-run determinism digest) | this commit | — | 585 / 585 (fresh tails in DAY5_RESULTS.md) | Canonical numbers captured verbatim by implementer; two honest defects found and recorded rather than smoothed over (tiny-slice CONTROL calibration inversion; micro-average pooled-bootstrap CI mis-indexing → gate FAIL row) | pending whole-branch fresh review |
| fix | Micro-bootstrap CI offset fix (pool-offset strata) in `ml/action_evaluation.py` + heterogeneous-arm containment regression pin in tests/test_action_evaluation.py | `9cf0df1+` | 34 | 585 / 585 | Found during canonical capture, fixed within Day 5; regression test pins heterogeneous-arm containment | n/a (regression-pinned) |

Focused-test counts are per-suite fresh collections at each task's own commit:
Task 2's file grew 29→37 when Task 3 extended it; final per-file counts
`test_observations.py` 28 · `test_action_model.py` 37 ·
`test_action_evaluation.py` 34 · `test_incremental.py` 24 = **123 new**;
plus 462 pre-existing = **585** (verified locally and inside Docker at HEAD).

Canonical-run verification recorded in `docs/DAY5_RESULTS.md`: splits/strata
counts · per-arm fit/calibration/test sizes · raw+calibrated metric tables
with bootstrap CIs vs Day 2 baseline · logit-scale contrast recovery against
kind-aware bands · interaction cells (RETRY_NOW×temporary_decline gated;
fatigue annotated) · incremental recovery/revenue model-vs-truth side by side
· safety-censored out-of-scope count 1,188 · deterministic two-run SHA-256
(984541b156fe959e731f34b52cf0fcace98423c8fc8ed88db55f7a382a858b67 post-fix;
pre-fix cba0d81d… retained as history).

## Final gate

**DAY 5 = GO** — see the DAY5_RESULTS.md gate table. The micro-CI defect
found during canonical capture is recorded FOUND → DIAGNOSED (pool-offset
strata bug: every pooled resample drew from the leading CONTROL segment) →
FIXED within Day 5 (running-offset strata in `ml/action_evaluation.py`;
per-arm CIs byte-identical pre/post fix) → REGRESSION-PINNED
(`test_micro_average_bootstrap_ci_contains_point_estimate_under_heterogeneous_arms`);
corrected micro CIs contain their point estimates for both bundle kinds.
Canonical determinism digest is the post-fix
984541b156fe959e731f34b52cf0fcace98423c8fc8ed88db55f7a382a858b67
(pre-fix cba0d81d… retained as history).

## Standing constraints (unchanged)
- No uplift/causal/bandit/RL/LangGraph/API/frontend/dashboard/database/execution/threshold-optimization code.
- The Day 2 baseline remains exactly `P(recovered | context)`; the only simulator change is the sanctioned additive public replay function in `simulation/outcomes.py`.
- Label vocabulary: MODEL ESTIMATE / OBSERVED SIMULATED OUTCOME / SIMULATED GROUND TRUTH; "causal" appears solely inside disclaimers.
- Randomized-stratum-only fitting and evaluation (safety-censored rows reported as out-of-scope counts); ground-truth columns never features.
- Seeds: dataset/baseline 42 (canonical run), action-model family 20260826, bootstrap named-seed discipline, zero randomness in incremental reporting.
