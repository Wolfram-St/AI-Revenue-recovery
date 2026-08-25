# Day 4 Simulated Treatment Outcomes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a transparent synthetic treatment/outcome environment that attaches action-assigned observations and simulated recovery outcomes to the existing payment-attempt timeline, unlocking future action-aware modeling — without touching the Day 2 baseline contract.

**Architecture:** New declarative configuration `config/treatment_policy.yaml` loaded by a typed validated loader (`simulation/config.py`). A staged simulator (`simulation/treatment.py`) assigns treatments to the canonical 5,000-row dataset: stage 1 applies the frozen safety policy as an eligibility gate (STOP ⇒ forced control), stage 2 randomizes eligible rows across `{CONTROL, RETRY_NOW, RETRY_LATER, REQUEST_UPDATE, HUMAN_REVIEW}` with fixed documented probabilities. Outcome generation (`simulation/outcomes.py`) computes a documented base recovery propensity from decision-time context, adds action-specific logit effects plus one named interaction, adds Gaussian logit noise, and samples Bernoulli outcomes; recovered revenue is full-or-zero. Temporal wiring (`simulation/temporal.py` inside outcomes module if size allows) stamps `treatment_timestamp`/`outcome_timestamp` strictly after failure. Ground-truth columns (base propensity, effect, pre-noise propensity, assignment probability) are stored per row and classified evaluation-only. Dataset contract + validator (`simulation/dataset.py`) classify every field; tests prove the Day 2 feature builder cannot consume any new field. Reporting utilities (`simulation/reporting.py`) emit arm counts/rates/overlap diagnostics labeled SIMULATED GROUND TRUTH vs OBSERVED SIMULATED OUTCOME. No causal estimators of any kind.

**Tech Stack:** Python 3.12+, numpy (Generator), pandas, PyYAML (already pinned), pytest, Docker. No new dependencies.

**Spec:** Master loop contract · `docs/DAY1.md`, `docs/DAY2.md`, `docs/DAY3.md`, evaluation protocol · `config/business_rules.yaml` v1.1 (reused read-only as the eligibility gate) · `db/schema.sql`.

## Global Constraints

- The Day 2 baseline remains exactly `P(recovered | context)`. No baseline file is modified; no treatment/outcome field may enter its feature matrix (enforced by test).
- The existing `recovered` label stays untouched and separate; the simulator produces NEW action-aware outcome fields (`simulated_recovered`, `simulated_recovered_amount_inr`).
- Every stochastic draw comes from explicit seeds derived from one master seed via `numpy.random.Generator.spawn`; same command + same seed ⇒ byte-identical output.
- No uplift/causal/bandit/RL/LangGraph/API/frontend/dashboard/database/execution code.
- Causal language ban: artifacts may say SIMULATED GROUND TRUTH, OBSERVED SIMULATED OUTCOME, CONTROL, TREATMENT; the phrase "causal estimate" may appear only in disclaimers.
- Confounding is allowed but must be documented where it exists; overlap/positivity must be measured and reported, never assumed.

---

## Design Decisions (binding for all tasks)

### D1. Treatment assignment = two-stage hybrid (documented confounding)
- **Probability provenance (binding):** `assign_treatments(df, probabilities, policy, policy_config=None)` receives caller-supplied recovery probabilities aligned to `df` rows — produced by the frozen Day 2 pipeline (`predict_recovery_probability(model, df)`, sigmoid-calibrated), never refit and never persisted inside the simulator. The stage-1 gate injects them into each row's context as `recovery_probability` before calling `decide_action` directly (per-row direct calls — the engine's ERV/no-op path is NOT part of assignment). *Documented consequence:* because R006/R007/R008 consume model probability, eligibility depends on model quality as well as true context; this model-dependence is part of the documented selection confounding.
- **Stage 1 — eligibility gate (deterministic):** run the shipped `decide_action` policy over each row's decision-time context. Any row whose authorized action is `STOP` is **forced to CONTROL** ("safety-censored controls"). Rationale: treating safety-blocked customers would contradict the product's own rules.
  - *Documented selection confounding:* eligibility depends on context (fraud, opt-out, hard decline, retry count, low probability), so the probability of being assignable varies with context. Within the eligible pool, assignment is randomized. Cross-arm comparisons are therefore adjusted-for-nothing naive differences; only within-eligible-pool contrasts approach unconfoundedness. This is reported verbatim in docs/results.
- **Stage 2 — randomized arms among eligible rows** with fixed probabilities (config): `CONTROL 0.20`, `RETRY_NOW 0.30`, `RETRY_LATER 0.25`, `REQUEST_UPDATE 0.15`, `HUMAN_REVIEW 0.10`. Chosen to guarantee **overlap**: every eligible row has positive probability for every arm (positivity holds by construction within the eligible stratum).
- `CONTROL` definition (explicit): a row that receives **no intervention**. Two sources: (a) randomized control arm among eligible rows ("randomized controls" — clean natural-recovery signal), (b) safety-stopped rows ("safety-censored controls"). Both are labeled; reporting utilities separate them. Safety-censored rows carry `assignment_probability = 0.0` (no stage-2 draw occurred; pins semantics for validators and future weighted estimators).

### D1b. Seed-stream allocation convention (binding, prevents cross-stage stream reuse)
All stochastic components derive from ONE master seed (`policy.master_seed`) via `numpy.random.default_rng(master_seed).spawn(k)` in a FIXED order: **child 0 = stage-2 assignment multinomial · child 1 = outcome Bernoulli draws · child 2 = temporal resolution windows**. Modules must accept their child generator (or the master policy plus a named child index constant exported from `simulation/config.py`: `SEED_STREAM_ASSIGNMENT=0`, `SEED_STREAM_OUTCOMES=1`, `SEED_STREAM_TEMPORAL=2`). Bare `default_rng(seed)` re-derivation inside any module is forbidden — this prevents row-aligned correlation between outcome draws and resolution windows that determinism tests cannot catch.

### D2. Synthetic ground truth (the simulator IS the world)
For row i assigned action a:
```
base_logit(i)      = b0 + Σ bk·feature_k(i)          # documented coefficients
effect_logit(i,a)  = m(a) + interaction(i,a)         # closed-form, config numbers
propensity(i,a)    = sigmoid(base_logit + effect_logit)     # PRE-noise probability
noise(i)           = Normal(0, sigma)                # logit-scale noise (child stream 1)
recovered_sim(i)   ~ Bernoulli(sigmoid(logit(propensity(i, assigned_arm)) + noise))
```
- `base_logit` coefficients mirror the Day 1 generator family (temporary_decline +0.95 … hard_decline −1.45, history/attempt/method terms) so propensities live in a realistic band; exact values are data in the YAML, not hidden code.
- Main effects `m(a)` (logit shifts vs control): `RETRY_NOW +0.60`, `RETRY_LATER +0.35`, `REQUEST_UPDATE +0.45`, `HUMAN_REVIEW +0.25`, `CONTROL 0`. Chosen to be meaningful but not deterministic.
- Exactly TWO interaction terms, named and bounded (closed enum — no user-supplied expressions): `RETRY_NOW × temporary_decline = +0.40` (immediate retry works best on transient failures); `RETRY_LATER × attempt_number≥3 = −0.25` (late-stage fatigue).
- Noise: `sigma = 0.50` logit-scale Gaussian.
- Stored ground truth per row: `base_recovery_propensity`, `action_effect_logit`, `propensity_under_assignment`, `assignment_probability` (stage-2 probability of the realized arm; 0.0 for safety-censored rows). These are **GROUND TRUTH / evaluation-only**, never features.

### D3. Outcome & revenue generation
- `simulated_recovered ~ Bernoulli(propensity_under_assignment ⊕ noise)` (single draw per row).
- Revenue: full-or-zero — `simulated_recovered_amount_inr = amount_inr` if recovered else `0.00`. Partial recovery explicitly NOT modeled (documented limitation).

### D4. Temporal semantics
- `failure_ts = event_timestamp` (existing).
- Treatment delay by arm (fixed, documented): RETRY_NOW `+15min`, REQUEST_UPDATE `+2h`, HUMAN_REVIEW `+4h`, RETRY_LATER `+24h`.
- `outcome_ts = treatment_ts + resolution window` drawn Uniform[1h, 48h] (seeded).
- CONTROL rows: `treatment_timestamp = null` (no intervention occurred); `outcome_ts = failure_ts + Uniform[1h,48h]` observation horizon.
- Invariant tested: `failure_ts < treatment_ts < outcome_ts` for treated; `failure_ts < outcome_ts` for controls; nulls exactly as specified.

### D5. Field classification (additive schema, nothing removed)
| field | class |
|---|---|
| attempt_id | join key / identifier |
| assigned_action, arm_source (randomized|safety_censored), assignment_probability | TREATMENT METADATA |
| treatment_timestamp, outcome_timestamp | OUTCOME-TIMING metadata |
| simulated_recovered, simulated_recovered_amount_inr | OUTCOME |
| base_recovery_propensity, action_effect_logit, propensity_under_assignment | GROUND TRUTH |
| event_timestamp | existing evaluation-only metadata |
New dataset artifact: `data/treatment_outcomes.csv` (generated, gitignored like the base CSV) produced by `python -m simulation.generate` CLI wrapper; the module API returns frames.

### D6. Evaluation methodology boundary
Reporting utilities output: arm counts (split randomized vs safety-censored), recovery rate by arm, recovered INR by arm, OBSERVED SIMULATED OUTCOME differences (treatment−control), SIMULATED GROUND TRUTH table (effects from config), overlap diagnostics (per-arm assignment-probability min/max, propensity ranges). Nothing computes a causal estimate; docs state plainly why naive differences are biased for cross-stratum comparison and unbiased-ish only within eligible strata.

### D7. Future interface (Day 5+, not implemented)
The joined frame `(payment_attempts ⋈ treatment_outcomes) on attempt_id` provides `(context, assigned_action, simulated_recovered)` triplets — sufficient inputs for future action-aware models. Day 4 builds no estimator.

---

### Task 1: Treatment policy specification/configuration

**Files:** Create `config/treatment_policy.yaml`, `simulation/config.py`; Test `tests/test_treatment_config.py`

**Interfaces:** `load_treatment_policy(path="config/treatment_policy.yaml") -> TreatmentPolicy` (frozen dataclass: version, master_seed, arm names+probabilities, main effects, interactions, noise sigma, delays, resolution window bounds); `validate` rejects: non-canonical actions, probabilities not summing to 1 (tol 1e-9), effects outside [−3, +3], sigma ≤ 0 or > 5, negative delays, unknown keys for effects, missing sections. Deterministic loading (two loads equal).

- [ ] RED: schema/probability/effect-range/rejection/determinism tests fail (module missing)
- [ ] GREEN: implement loader mirroring Day 3 config-loader discipline (typed, loud ValueErrors, yaml.safe_load, frozen dataclasses, no executable config)
- [ ] Full suite + Docker + review → commit `feat: add synthetic treatment policy configuration`

### Task 2: Treatment assignment generator

**Files:** Create `simulation/treatment.py`; Test `tests/test_treatment_assignment.py`

**Interfaces:** `assign_treatments(df, probabilities, policy, policy_config=None) -> pd.DataFrame` indexed like df: `assigned_action`, `arm_source`, `assignment_probability`. Stage 1 calls real `decide_action` per row with `recovery_probability=probabilities[i]` injected into the context (direct calls; the engine ERV path is not used). Stage 2 uses seed-stream child 0 (D1b). Deterministic under fixed master seed; different master seeds differ stochastically (statistical test on large n, not flaky equality). Tests: STOP rows always CONTROL+safety_censored with `assignment_probability==0.0`; eligible arms match configured set; empirical frequencies within tolerance on n=20000 synthetic frame (loose bounds, e.g., ±0.03); positivity: every eligible row has positive probability for every arm; no impossible actions; input df not mutated; missing/NaN probabilities rejected loudly.

- [ ] RED → GREEN → full suite → Docker → review → commit `feat: assign treatments with eligibility gate and randomized arms`

### Task 3: Outcome simulation

**Files:** Create `simulation/outcomes.py`; Test `tests/test_outcomes.py`

**Interfaces:** `simulate_outcomes(df_with_assignments, policy) -> pd.DataFrame`: `simulated_recovered`, plus ground-truth columns per D2. Uses seed-stream child 1 (D1b). Base-propensity coefficients read from policy config (not hardcoded). Tests: control mean ≈ sigmoid(base) within MC tolerance on large n; each arm's observed rate moves in the direction/magnitude of its known effect (large-n sanity bands); boundaries impossible (probabilities strictly in (0,1) given finite logits + noise — verify no 0/1 emitted); different seeds → different draws, same seed → identical; interaction visible (temporary_decline RETRY_NOW rate > unknown-category RETRY_NOW rate at matched amounts); no future info (function consumes only decision-time columns — enforced by column whitelist internally).

- [ ] RED → GREEN → full suite → Docker → review → commit `feat: simulate action-aware recovery outcomes with stored ground truth`

### Task 4: Recovered revenue generation

**Files:** Extend `simulation/outcomes.py`; extend `tests/test_outcomes.py`

**Rules:** full-or-zero per D3; rounding to 2dp; never exceeds amount; never negative. Tests: unrecovered ⇒ 0.00 exactly; recovered ⇒ == amount_inr rounded; bounds hold across 5000-row run; determinism.

- [ ] RED → GREEN → full suite → Docker → review → commit `feat: generate bounded simulated recovered revenue`

### Task 5: Data contract + validation (+ leakage guard)

**Files:** Create `simulation/dataset.py`; Test `tests/test_treatment_dataset.py`

**Interfaces:** `build_treatment_dataset(df) -> pd.DataFrame` (join key + all D5 columns, classified order); `dataset_contract() -> dict`; `validate_treatment_dataset(df) -> dict` (row alignment with attempts frame, classification completeness, temporal null rules, bounds, dtype checks). **Leakage-guard tests:** `ml.features.build_feature_matrix` applied to the merged frame must (a) succeed, (b) contain zero treatment/outcome columns (explicit whitelist design), and `FORBIDDEN_FEATURES` extended in TEST assertions only (no production change needed — assert the builder selects none of the new fields even when present in input).

- [ ] RED → GREEN → full suite → Docker → review → commit `feat: define additive treatment/outcome dataset contract with leakage guards`

### Task 6: Temporal integration

**Files:** Create `simulation/temporal.py`; Test `tests/test_temporal_treatment.py`

**Interfaces:** `stamp_treatment_timeline(df_with_outcomes, policy) -> DataFrame` adding `treatment_timestamp`/`outcome_timestamp` per D4 (delays from config; resolution windows from seed-stream child 2 per D1b). Tests: strict ordering invariant incl. arm-specific delay correctness; CONTROL null-treatment rule; impossible timestamps rejected (treatment before failure); reproducible under seed; timezone-consistent UTC.

- [ ] RED → GREEN → full suite → Docker → review → commit `feat: integrate treatment and outcome timing after failure events`

### Task 7: Ground-truth / evaluation utilities

**Files:** Create `simulation/reporting.py`; Test `tests/test_reporting.py`

**Interfaces:** `summarize_arms(df) -> dict`, `observed_differences(df) -> dict` (labeled OBSERVED SIMULATED OUTCOME), `ground_truth_table(policy) -> dict` (SIMULATED GROUND TRUTH), `overlap_diagnostics(df) -> dict` (assignment-probability ranges, propensity ranges, per-arm counts split randomized/safety-censored). Labels embedded as literal `"label"` fields in outputs. Tests: hand-computed fixtures for rates/revenue/differences; labels present and correct; no function named or documented as causal.

- [ ] RED → GREEN → full suite → Docker → review → commit `feat: report simulated ground truth and observed arm diagnostics`

### Task 8: Documentation + Day 4 gate

**Files:** Create `docs/DAY4.md`, `docs/DAY4_RESULTS.md`; update ledger

Run canonical pipeline (5000 rows, seed from config), capture REAL outputs: arm counts (randomized vs safety-censored), recovery rates by arm, revenue by arm, observed differences, overlap table, ground-truth table, example treated/control records, temporal-ordering proof stats. Document all owner-mandated points including the three verbatim sentences:
- "The treatment environment is synthetic and does not establish real-world causal treatment effects."
- "The Day 2 baseline remains P(recovered | context)."
- "Action-aware observations are now available for future modeling, but no action-aware predictive model is implemented in Day 4."
Gate table → GO/NO-GO for Day 5 action-aware modeling. Document the known in-sample caveat: stage-1 probabilities come from Day 2 model predictions over all 5,000 rows, so train-row probabilities are mildly in-sample (slightly optimistic feeding R006/R007/R008); note also that treated rows resolve on `treatment_ts + window` vs controls on `failure_ts + window` — a systematic horizon asymmetry irrelevant to rate reporting but relevant to any future time-to-recovery analysis. Commit `docs: close day 4 treatment simulation verification`.

---

## Final verification sweep (after Task 8)

Local suite · Docker suite · regenerate base CSV + treatment CSV (byte-identical rerun) · action counts · control/treatment coverage · temporal ordering · amount bounds · leakage guards · git diff/status · secrets scan · forbidden-tech scan · causal-language audit · docs-vs-numbers check · final Docker run. Whole-branch fresh reviewer must APPROVE before DAY 4 = GO.

## Non-goals (this phase)

uplift modeling · causal inference/treatment-effect estimation · contextual bandits · RL · LangGraph · API · frontend · dashboard · database writes · payment execution · Razorpay integration · threshold optimization · LLM assignment · modifying Day 2 baseline files.
