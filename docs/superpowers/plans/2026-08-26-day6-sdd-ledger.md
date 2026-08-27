# SDD Progress Ledger — Day 6 Decision Engine Evidence

Plan: `docs/superpowers/plans/2026-08-26-day6-decision-engine-evidence.md`
Workflow: subagent-driven development (fresh implementer + independent reviewer per task, fix/re-review loops)
Branch/worktree: `feature/day6-decision-engine` @ `D:/airev-day6` (base: Day 5 main incl. micro-bootstrap fix; plan commits `208e321`→`c97f687` incl. two plan-review rounds)
Boundary: everything is synthetic-world-only over the frozen Day 4 simulator; Day 2 baseline and Day 5 per-arm reference untouched; randomized-stratum-only fitting/evaluation; candidate recommendation → deterministic policy engine → authorized_action invariant enforced by tests.

Provenance note: this ledger was written at Task 6 from verifiable artifacts —
the five task commits, fresh detached-worktree pytest reruns per commit, the
canonical capture under `%TEMP%\opencode`, and reviewers' fix annotations
embedded in shipped code/tests (`review F…` markers). No verdict below claims
more provenance than that.

| Task | Scope | Commit | Focused tests | Full suite (local; Docker at HEAD) | Implementer review | Re-review |
|---|---|---|---|---|---|---|
| 1 | Treatment/outcome CLI (`generate`/`validate`/`summary`), byte-reproducible sha256 artifact, dual seed semantics, purity-pinned baseline-chain imports | `5267fba` | 14 | 599 / — | APPROVE after review loop (`.gitignore`-presence guard added; environment-skip behavior visible as the single Docker skip) | n/a (approved) |
| 2 | Pooled action-feature model: one pipeline, whitelist+one-hot arm features, counterfactual copy discipline, explicit `simulated_recovered` target | `18d96ac` | 34 | 633 / — | APPROVE (input-frame immutability + calibration immutability pinned) | n/a (approved) |
| 3 | Comparison harness: predictive/agreement/contrast sections for both families on identical segments, corrected-offset pooled bootstrap, D-E2 rule engine, curves/stability protocols, complexity proxies | `7e8fc5c` | 26 | 659 / — | APPROVE with documented DEVIATION NOTE: contracted fit-seconds require stdlib `time` (`from time import perf_counter`) inside the otherwise Day-5 import whitelist (marker in ml/model_comparison.py + test) | n/a (approved) |
| 4 | Decision-quality evidence + uncertainty: match rate/regret/quantiles, bootstrap CIs + overlaps, inventory, native policy-safety probe, provenance digest | `2c9c9b0` | 37 | 696 / — | APPROVE after fix loop (DEVIATION NOTES recorded: stdlib hashlib/json + recovery root required by digest/cost contracts; plan-sketch pooled twin replaced by one-bundle-per-call contract — markers in ml/decision_evidence.py docstring) | n/a (approved) |
| 5 | Evidence classifier + bounded recommender behind D-E5 gate; OptimizerNotJustifiedError; candidate always through decide_action | `d2ad0cb` | 73 (this commit also grew test_decision_evidence.py 37→43) | 775 / — | APPROVE after cross-module fix loop (contract-defect regression pins `review F1`: before the fix the GENUINE decision_evidence output failed the classifier's structure validation — no canonical bundle could ever pass; now pinned on BOTH sides: tests/test_decision_evidence.py:914, tests/test_decision_policy.py:929) | n/a (approved) |
| 6 | Documentation + canonical capture + Day 6 gate (DAY6.md, DAY6_RESULTS.md, this ledger) | this commit | — | 775 / 774 passed + 1 env-skip (fresh tails in DAY6_RESULTS.md) | Canonical outputs captured verbatim by implementer (Gate A verdict per_arm; Gate B sha256 proof; Gate C classification OPTIMIZER_JUSTIFIED); honest disclosures recorded rather than smoothed over | pending whole-branch fresh review |

Focused-test counts are fresh collections at each task's own commit via
detached-worktree reruns (`git worktree add --detach`): 14 → 34 → 26 → 37 →
73, full suites 599 → 633 → 659 → 696 → 775. Final per-file counts at HEAD:
`test_cli.py` 14 · `test_pooled_model.py` 34 · `test_model_comparison.py` 26 ·
`test_decision_evidence.py` 43 · `test_decision_policy.py` 73 = **190 new**
(Task 5's commit carried six additional evidence-suite regression pins, hence
43 ≠ the 37 collected at Task 4's own commit); plus 585 pre-existing =
**775** verified locally and inside Docker (the Docker run reports
774 passed + 1 skipped: `tests/test_cli.py:335` requires a `.gitignore`
absent from the image context).

Canonical-run verification recorded in `docs/DAY6_RESULTS.md`:

- Gate A (D-E2, calibrated bundles only): preferred_model = **per_arm** —
  pooled passes only ground-truth agreement (MAE 0.079932 vs 0.096832) and
  fails strict micro-Brier CI non-overlap ([0.211744, 0.237970] vs
  [0.211186, 0.236742]), smallest-arm Brier (0.230161 vs 0.226339), and
  interaction-band recovery (gated cell gap −0.784001 vs ±0.40);
- learning curves {0.25, 0.50, 1.00} both families (prefix-confound disclosed)
  and four-seed stability (micro-Brier sd 0.000935 / 0.000250; pooled lift
  compression ≈0.01–0.04 vs per-arm ≈0.13–0.19);
- Gate C: match rate 427/558 = 0.765233 [0.730065, 0.800401]; relative regret
  0.138288 (p50/p90/p99 ₹0/₹174.76/₹871.43); RETRY_NOW disjoint from all three
  other arms → 3 ≥ 2 disjoint CI pairs; probe 3/3 STOP overrides;
- classifier: **OPTIMIZER_JUSTIFIED**, reasons [], provenance_digest
  `0471c5e047ef9ad152e5bd86994e4028533215d29b5e541d82137c52b224caa0`;
- determinism: evidence dict sorted-json SHA-256 identical across two fresh
  calls (`03b40316a1b41f3d24c346d90dccec0b27c0b707ba33f050721428f1b053f889`);
- CLI: same-seed sha256 identical twice
  (`576632a9c8d505297031dff0455e1bfc3ba9e7e777e6a338891ef8f10be1f40f`);
  seed 43 → different file, still valid;
  `data/treatment_outcomes.csv` gitignored, never committed.

## Final gate

**DAY 6 = GO (implementer side), final classification OPTIMIZER JUSTIFIED** —
pending whole-branch fresh review APPROVE, which the plan requires to convert
GO into the day's formal exit. All four D-E5 criteria pass on the canonical
run with sensitivities reported plainly (regret margin thin: 0.1383 vs 0.15,
per-seed span 0.1176–0.1435; match rate robust; probe flips only with business-
rule changes; CI-pair count hinges on RETRY_NOW's separation). The Gate A
verdict keeps per-arm as the reference production family; nothing was deleted.
Disclosures embedded in DAY6_RESULTS.md: single-canonical-run verdict with
stability REPORTED-not-certified; learning-curve prefix = early-time-window
confound; chosen ≥2-disjoint-CI-PAIRS semantics; injected-probability upward
bias vs baseline-scale R006/R007/R008 gating; validate certifies schema, not
attempt-context integrity. The recommender remains constructible-only-behind-
the-gate advisory scaffolding whose candidates never bypass
`decide_action`.

## Standing constraints (unchanged)

- No contextual bandit / RL / autonomous experimentation / uplift estimator /
  causal machinery / LangGraph / API / frontend / dashboard / database writes /
  payment execution / Razorpay integration anywhere in Day 6.
- The Day 2 baseline stays exactly `P(recovered | context)`; the Day 5 per-arm
  family stays the reference production form; pooled is a measured candidate.
- Label vocabulary: MODEL ESTIMATE / OBSERVED SIMULATED OUTCOME /
  SIMULATED GROUND TRUTH; "causal" appears solely inside disclaimers.
- Randomized-stratum-only fitting/evaluation; safety-censored rows appear only
  as out-of-scope counts; ground-truth columns never features.
- Seeds: dataset/baseline 42 (canonical world), action-model family 20260826,
  stability seeds {20260826, 1, 2, 3}, bootstrap B=500 named-seed discipline,
  `--seed` (dataset) distinct from `policy master_seed` in every CLI output.
- Pre-registered thresholds (D-E2/D-E5) fixed before the canonical run;
  documented as conventions, never statistical tests.
