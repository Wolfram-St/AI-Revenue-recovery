# Day 6 Decision Engine Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine — on pre-registered evidence criteria — whether RecoverAI can safely progress from action-aware estimation to incremental-revenue-based action selection; build the reproducible treatment/outcome CLI artifact; and run the pooled-vs-per-arm model investigation. The per-arm models remain the reference baseline throughout. A scientifically honest OPTIMIZER NOT YET JUSTIFIED is an acceptable, first-class outcome.

**Architecture:** New `simulation/cli.py` (argparse-based `python -m simulation.cli` with `generate` / `validate` / `summary` subcommands) wrapping the frozen Day 4 chain and writing the gitignored `data/treatment_outcomes.csv`. New `ml/pooled_model.py`: a single XGBoost pipeline whose features are the Day 2 decision-time whitelist PLUS one-hot assigned arm — trained/calibrated on the identical randomized strata as `ml/action_model.py` for apples-to-apples comparison. New `ml/model_comparison.py`: comparison harness computing predictive, calibration, ground-truth-recovery, interaction-recovery, sample-efficiency (learning curves at 25%/50%/100% train fractions), stability (multi-seed), and complexity measurements for both model families. New `ml/decision_evidence.py`: uncertainty quantification (bootstrap CIs on IncrementalRevenue(a), pairwise CI overlaps, revenue-estimation variance across seeds) plus decision-quality measurement (decision-match rate and revenue regret vs the noise-integrated ground-truth argmax). New `ml/decision_policy.py`: the pre-registered evidence classifier returning `OPTIMIZER_JUSTIFIED` / `OPTIMIZER_NOT_YET_JUSTIFIED` with reason list — and a bounded recommender class that is constructible ONLY when the classifier says justified; it produces candidate recommendations that still flow through the deterministic policy engine (`decide_action`) which remains the sole authorization boundary.

**Tech Stack:** Python 3.12+, pandas/numpy, scikit-learn, XGBoost (pinned); argparse stdlib; pytest; Docker. No new dependencies.

**Spec:** Master loop contract · Day 4 plan D1/D1b/D2/D5/D6 · Day 5 plan D-M1..M7 · `docs/DAY4*.md`, `docs/DAY5*.md`.

## Global Constraints

- The Day 2 baseline (`P(recovered | context)`) and Day 5 per-arm models remain untouched except where explicitly listed; per-arm stays the REFERENCE.
- No contextual bandits, RL, autonomous experimentation, LangGraph, API, frontend, dashboard, database writes, payment execution, Razorpay integration.
- Safety architecture invariant (enforced by tests whenever a recommender exists): candidate recommendation → deterministic policy engine → authorized_action; STOP dominance preserved; recommender never bypasses policy.
- Synthetic-world-only language discipline everywhere (MODEL ESTIMATE / SIMULATED GROUND TRUTH / OBSERVED SIMULATED OUTCOME).
- `data/treatment_outcomes.csv` is generated + gitignored; never committed.
- Pre-registered thresholds in this plan are fixed BEFORE the canonical comparison run; they are conventions (documented as such), not statistical tests.

---

## Binding Design Decisions

### D-E1. Comparison protocol (Gate A)
Both model families train on the IDENTICAL randomized-stratum train/validation segments of the same assembled observation frame (seed 20260826 family); evaluate once on the identical test segment.
Pooled model: features = `build_feature_matrix(X)` + one-hot `assigned_action` via ColumnTransformer categorical block appended; single sigmoid calibration on pooled validation rows; per-arm slices evaluated by filtering pooled predictions.
Measurements, both families: micro + per-arm AUC/Brier (bootstrap CIs B=500 seeded), mean |P̂−integrated TRUE|, logit-scale effect contrasts vs config (+ interaction cells), learning curves at train fractions {0.25, 0.50, 1.00} (micro-Brier + smallest-arm Brier at each point), multi-seed stability (seeds {20260826, 1, 2, 3} → sd of micro-Brier and of arm-mean incrementals), wall-clock fit seconds, parameter counts.

### D-E2. Pre-registered pooled-preference rule
Both bundle kinds are computed and REPORTED; the rule is applied to the **calibrated bundles only** (the shipped form per Day 5), with raw results retained as history. CIs use the Day-5 stratified-bootstrap scheme with corrected pool offsets. Pooled becomes the preferred production model ONLY if ALL hold on the canonical 100%-fraction calibrated run:
1. **Strict CI non-overlap on micro-Brier**: pooled CI95 upper < per-arm CI95 lower (point-ordering alone is insufficient — noted as context, not evidence);
2. pooled mean |P̂−integrated TRUE| ≤ per-arm mean |P̂−TRUE| (ground-truth agreement no worse);
3. smallest-arm (HUMAN_REVIEW) test Brier(pooled) ≤ per-arm HUMAN_REVIEW Brier;
4. interaction-cell recovery within band for pooled as for per-arm, carrying the `attenuation_expected` annotation semantics (fatigue cell reported-not-gated, mirroring Day 5).
Otherwise per-arm remains preferred. Either way both stay available; nothing is deleted.

### D-E3. CLI contract (Gate B)
`python -m simulation.cli generate [--rows 5000] [--seed 42] [--out data/treatment_outcomes.csv]` → writes CSV via the frozen chain (assemble_observations internal path) + prints summary JSON line; exit code non-zero on validation failure. **Seed semantics: `--seed` is the DATASET-generation seed (default 42 = canonical world); the policy master_seed always comes from `config/treatment_policy.yaml` — the two are distinct and both recorded in CLI output.** DAY6 tables state which world each comes from.
**Purity scope (resolves the baseline-dependency):** `generate` must reproduce the frozen chain INCLUDING calibrated Day-2 baseline probabilities; therefore simulation/cli.py may import exactly `ml.train.train_baseline`, `ml.evaluate.calibrate_model`, and `ml.train.predict_recovery_probability` (documented, seed-pinned) — importing any action-model/comparison module (`ml.action_model`, `ml.action_evaluation`, `ml.incremental`, `ml.pooled_model`, `ml.model_comparison`) remains forbidden and purity-tested.
`validate [--csv PATH]` → runs `validate_treatment_dataset` report. `summary [--csv PATH]` → reporting.summarize_arms + observed_diagnostics JSON. Same seed+rows ⇒ byte-identical file (sha256 pinned by test). `.gitignore` gains `data/treatment_outcomes.csv`.

### D-E4. Decision-quality metrics (Gate C measurement)
On randomized test contexts. **Candidate set = the 4 treated arms; CONTROL excluded** (uniform retry-cost makes CONTROL revenue undefined/negative — argmax over treated arms only). Ties broken deterministically by ARM_ORDER precedence on both model and truth sides. Because cost/risk terms are arm-independent constants per row, argmax decisions reduce to argmax incremental-recovery × amount — stated in docs so cost sensitivity is not misread.
For each treated arm a:
- `model_incremental_revenue(a)` from the CALIBRATED bundle (shipped form, documented);
- `truth_incremental_revenue(a)` from noise-integrated `ground_truth_propensity`.
Metrics:
1. **Decision-match rate**: fraction of contexts where argmax_a MODEL revenue == argmax_a TRUTH revenue (ARM_ORDER tie-break both sides); binomial CI reported.
2. **Relative regret**: E[truth_revenue(truth_argmax)] − E[truth_revenue(model_argmax)], divided by E[truth_revenue(truth_argmax)]; guard: denominator ≤ 0 → reported undefined; **per-row regret quantiles (p50/p90/p99) also reported** (mean-relative-regret hides heavy tails).
3. **Bootstrap CI** (B=500 seeded) around each treated arm's mean model incremental revenue; pairwise CI-overlap matrix.
4. Uncertainty inventory: per-arm n, calibration status, propensity-range overlap, seed-variance of arm-mean incrementals AND of match rate/regret (from D-E1 stability runs).

### D-E5. Pre-registered optimizer classification rule
The classifier returns `OPTIMIZER_JUSTIFIED` iff ALL hold on the canonical calibrated-bundle run:
1. decision-match rate ≥ 0.60;
2. relative regret ≤ 0.15;
3. policy-safety probe passes (crafted STOP cases never receive a positive recommendation after policy gating);
4. at least two treated arms have mutually non-overlapping bootstrap CIs (i.e., distinctions the evidence CAN support exist).
Otherwise `OPTIMIZER_NOT_YET_JUSTIFIED` with machine-readable reasons. Thresholds are transparent conventions chosen before measurement; sensitivity to them is reported (metrics printed alongside).

If NOT justified: no recommender is enabled; `ml/decision_policy.py` still ships the classifier + the bounded recommender class whose constructor raises `OptimizerNotJustifiedError` unless constructed with an evidence bundle whose classification is JUSTIFIED. The gate is ADVISORY scaffolding (an in-process caller could fabricate an evidence dict) — docstrings state this, and evidence bundles carry a `provenance_digest` field (sorted-json SHA pattern from Day 5) so non-canonical bundles are visibly so. Docs identify the exact evidence needed next.

### D-E6. Learning-curve protocol
Fractions {0.25, 0.50, 1.00} apply to the TRAIN segment only; calibration at every fraction uses the FULL validation segment unchanged (comparability across fractions); per-arm family refits all five arms at each fraction (intended and counted); stability seeds {20260826, 1, 2, 3}. DAY6.md states explicitly that the optimizer verdict is single-canonical-run with stability REPORTED, not stability-certified.

---

### Task 1: Treatment/outcome CLI artifact

**Files:** Create `simulation/cli.py`; Modify `.gitignore`; Test `tests/test_cli.py`

Subcommands per D-E3. Tests: generate→file exists+header exact+row count; same-seed sha256 byte-identical twice; different seed → different file but validator valid; validate subcommand returns report valid:true on good CSV, false on tampered CSV; summary JSON parses with labels; --help works; unknown subcommand exits non-zero; purity (imports simulation+stdlib argparse/json only; no ml/recovery imports).

- [ ] RED → GREEN → FULL → Docker → review → commit `feat: add reproducible treatment outcome CLI`

### Task 2: Pooled action-feature model

**Files:** Create `ml/pooled_model.py`; Test `tests/test_pooled_model.py`

Interfaces mirroring Day 5 discipline: `train_pooled_model(train_frame, validation_frame, seed=...) -> (PooledModelBundle, metadata)` (features = build_feature_matrix X + assigned_action categorical; y = simulated_recovered EXPLICIT; zero-randomized-row ValueError impossible but guarded); `calibrate_pooled_model(bundle, validation_frame)`; `predict_pooled_probability(bundle, context_frame, action)` (sets action column to requested arm — counterfactual query semantics, documented). Metadata: pooled row counts, small_segments, seed. Tests: target discipline (wrong-label fixture), stratum exclusion, reproducibility, bounds, counterfactual-query semantics (same context different action → different probability generally), calibration immutability, purity.

- [ ] RED → GREEN → FULL → Docker → review → commit `feat: add pooled action-feature model as comparison candidate`

### Task 3: Pooled vs per-arm comparison harness

**Files:** Create `ml/model_comparison.py`; Test `tests/test_model_comparison.py`

`compare_models(per_arm_bundle, pooled_bundle, baseline_model, test_frame, policy, train_frames_by_fraction=None, seeds=(...)) -> dict` producing sections: predictive (micro/per-arm tables ×2 families ×CIs), ground_truth_agreement, effect_contrasts (+interaction cells), learning_curves (fractions × families × micro/smallest-arm Brier), stability (seed variance), complexity (fit seconds, param counts). Applies D-E2 rule → `preferred_model` field + reasons. Tests: hand-checkable small fixture structure; determinism; rule application unit tests with synthetic metric dicts (both outcomes forced); fraction curve shape sanity; purity.

- [ ] RED → GREEN → FULL → Docker → review → commit `feat: compare pooled and per-arm models on pre-registered criteria`

### Task 4: Decision-quality evidence + uncertainty

**Files:** Create `ml/decision_evidence.py`; Test `tests/test_decision_evidence.py`

Implements D-E4 metrics over the CALIBRATED per-arm bundle + pooled twin (both reported): decision-match rate, relative regret, bootstrap CIs per treated arm incremental revenue, CI-overlap matrix, uncertainty inventory. Tests: hand-computed constant-probability fixtures (perfect match rate 1.0; adversarial mis-ordering fixture → match rate low); regret math hand-checked; CI overlap counting; determinism; purity.

- [ ] RED → GREEN → FULL → Docker → review → commit `feat: quantify decision quality and uncertainty against synthetic ground truth`

### Task 5: Evidence classifier + bounded recommender

**Files:** Create `ml/decision_policy.py`; Test `tests/test_decision_policy.py`

Implements D-E5 classifier consuming Task 4 evidence (+policy safety probe using real decide_action on crafted STOP frames with recommender candidates injected). `OptimizerNotJustifiedError` exception. `BoundedRecommender(evidence)` constructs only when classification==JUSTIFIED; method `recommend(context_row) -> CandidateRecommendation(top_action, incremental_revenue_estimate, policy_authorized_action, policy_reason)` — ALWAYS passes candidate through `decide_action`; STOP dominance asserted by construction; recommendation fields carry MODEL ESTIMATE label. Tests: classifier unit tests with synthetic evidence dicts (all four criteria pass/fail combinations); recommender-construction gating (raises when NOT justified); policy-safety probes (opted-out/fraud/hard-decline contexts → authorized STOP regardless of candidate); purity.

- [ ] RED → GREEN → FULL → Docker → review → commit `feat: classify optimizer justification behind pre-registered evidence gates`

### Task 6: Canonical run + documentation + Day 6 gate

**Files:** Create `docs/DAY6.md`, `docs/DAY6_RESULTS.md`, `docs/superpowers/plans/2026-08-26-day6-sdd-ledger.md`; update ledger references

Canonical run captures REAL outputs: full comparison table (D-E2 verdict + which rule criteria passed), learning-curve numbers, CLI sha256 reproducibility proof, decision-quality metrics (match rate, regret, CIs, overlaps), classifier output with reasons, policy-safety probe results. DAY6.md documents all owner points incl. uncertainty examination and the safety architecture diagram. DAY6_RESULTS.md gate table → final classification **OPTIMIZER JUSTIFIED** or **NOT YET JUSTIFIED** (whichever the canonical evidence yields — reported plainly either way). Commit `docs: close day 6 decision engine evidence verification`.

---

## Final verification sweep

Local suite · Docker suite · CLI sha256 reproducibility · comparison determinism · policy-safety tests · leakage guards · label-language audit · docs-vs-numbers check · git diff/status · whole-branch fresh reviewer APPROVE required for DAY 6 = GO (with explicit optimizer classification inside the GO).

## Non-goals

contextual bandits · RL · autonomous experimentation · uplift estimators · causal machinery · LangGraph · API · frontend · dashboard · database writes · payment execution · Razorpay integration · threshold optimization beyond the measured evidence.
