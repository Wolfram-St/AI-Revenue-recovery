# Day 6 — Decision Engine Evidence

Status: implemented on branch `feature/day6-decision-engine`.

## What was built

Day 6 asks the pre-registered question: can RecoverAI safely progress from
action-aware estimation (Day 5) to incremental-revenue-based action
selection? Five modules were added on top of the untouched Day 2 baseline and
Day 5 per-arm reference:

1. `simulation/cli.py` — argparse CLI (`python -m simulation.cli` with
   `generate` / `validate` / `summary`) wrapping the frozen Day 4 chain and
   writing the gitignored `data/treatment_outcomes.csv`. `generate` prints one
   JSON line recording BOTH seeds distinctly (`--seed` is the DATASET-generation
   seed, default 42 = canonical world; the policy `master_seed` always comes
   from `config/treatment_policy.yaml`). Same seed+rows ⇒ byte-identical file,
   pinned by an in-suite sha256 test. Purity-tested: exactly the three
   documented ml baseline imports, never any action-model/comparison module.
2. `ml/pooled_model.py` — the Gate A comparison candidate: ONE XGBoost
   pipeline whose features are the Day 2 whitelist plus one-hot
   `assigned_action`, trained/calibrated on the identical randomized strata as
   `ml/action_model.py`. Prediction is counterfactual by construction (the
   query frame's arm column is overwritten on a copy; caller frames are never
   mutated). The per-arm family remains the REFERENCE production form.
3. `ml/model_comparison.py` — the Gate A harness: predictive tables (micro +
   per-arm AUC/PR-AUC/Brier with seeded stratified-bootstrap CIs, corrected
   pool offsets) for both families on the identical randomized test segment,
   ground-truth agreement, logit-scale effect contrasts (+ interaction cells),
   learning curves, multi-seed stability, complexity proxies, and the pure
   D-E2 rule engine producing `preferred_model`.
4. `ml/decision_evidence.py` — Gate C measurement (D-E4): decision-match rate
   vs the noise-integrated truth argmax (binomial CI), relative regret with a
   denominator guard plus per-row regret quantiles, per-treated-arm bootstrap
   CIs around mean model revenue with pairwise overlap lists, uncertainty
   inventory (incl. caller-supplied seed-variance replicates), the native
   `policy_safety_probe` over three crafted STOP contexts, and a
   `provenance_digest` (sorted-json SHA) that makes non-canonical bundles
   visibly so. Reuses the Day 5 incremental/revenue twin modules unchanged.
5. `ml/decision_policy.py` — the pre-registered D-E5 classifier
   (`OPTIMIZER_JUSTIFIED` / `OPTIMIZER_NOT_YET_JUSTIFIED` with machine-readable
   reasons) and the bounded recommender that is constructible ONLY from
   JUSTIFIED-classified evidence; every recommendation still flows through the
   deterministic `decide_action` engine, which remains the sole authorization
   boundary.

All numbers quoted here and in DAY6_RESULTS.md come from one fresh canonical
run: dataset seed 42, chronological 70/15/15, baseline seed 42 calibrated on
validation, probabilities predicted over the FULL 5,000-row frame, policy
`master_seed 20260826`, action-model/bootstrap seed 20260826.

## Gate structure

| Gate | Question | Module | Pre-registered rule |
| --- | --- | --- | --- |
| A | Is the pooled family better than the per-arm reference? | `ml/model_comparison.py` | D-E2 preference rule (below) |
| B | Is the synthetic treatment/outcome artifact reproducible? | `simulation/cli.py` | byte-identical sha256 per seed+rows; validator green |
| C | Does revenue-based action selection beat truth-argmax often enough, safely enough? | `ml/decision_evidence.py` + `ml/decision_policy.py` | D-E5 classification rule (below) |

## Comparison protocol (D-E1)

Both families train on the IDENTICAL randomized-stratum train/validation
segments of one assembled observation frame (seed 20260826 family) and are
evaluated ONCE on the identical randomized test segment (558 rows). The rule
is applied to the CALIBRATED bundles only — the shipped Day 5 form — raw
bundles are refused loudly. Measurements per family: micro/per-arm
AUC/PR-AUC/Brier with bootstrap CIs (B=500 seeded, stratified within arm with
corrected pool offsets), mean |P̂−integrated TRUE| (secondary, Jensen-floor
annotated), logit-scale effect contrasts vs config (+ interaction cells),
learning curves, stability across seeds {20260826, 1, 2, 3}, wall-clock fit
seconds, parameter-count capacity proxies (n_estimators × max_depth — coarse,
never an exact node count).

Learning-curve protocol (D-E6): fractions {0.25, 0.50, 1.00} apply to the
TRAIN segment only, taken as a deterministic PREFIX of its randomized rows
(frame order preserved); calibration at every fraction uses the FULL
validation segment unchanged; the per-arm family refits all five arms at every
fraction (intended and counted). Stability seeds refit BOTH families on the
full segments; population sds across seeds are reported.

## Pre-registered rules (restated before the verdict)

These thresholds and semantics were fixed BEFORE the canonical run (plan
§Binding Design Decisions). They are transparent conventions, NOT statistical
tests, and sensitivity to them is reported alongside the metrics.

### D-E2 — pooled-preference rule (Gate A)

Pooled becomes the preferred production candidate ONLY if ALL FOUR hold on
the canonical 100%-fraction calibrated run; otherwise per-arm stays preferred
and both families remain available:

1. **Strict CI non-overlap on micro-Brier**: pooled CI95 upper STRICTLY below
   the per-arm CI95 lower (point ordering alone is insufficient — noted as
   context, never as evidence);
2. **Ground-truth agreement no worse**: pooled mean |P̂−integrated TRUE|
   (arm-mean) ≤ per-arm;
3. **Smallest randomized test arm (HUMAN_REVIEW-shaped) Brier no worse**:
   pooled test Brier ≤ per-arm on those exact rows;
4. **Interaction-cell recovery within band** — POOLED-only gating
   interpretation: every pooled interaction cell WITHOUT the
   `attenuation_expected` annotation must land inside the kind-aware ±0.40
   calibrated logit band. Annotated weak-negative cells (the RETRY_LATER ×
   attempt_number≥3 fatigue cell) are REPORTED, never gated, mirroring Day 5;
   per-arm interaction recovery continues to be reported under the Day 5
   discipline but does not feed this criterion. Comparisons with non-finite
   inputs fail CLOSED toward per-arm.

### D-E5 — optimizer justification rule (Gate C)

`OPTIMIZER_JUSTIFIED` iff ALL hold on the canonical calibrated-bundle run:

1. decision-match rate ≥ 0.60;
2. relative regret ≤ 0.15 (undefined regret fails with the literal reason);
3. policy-safety probe passes (crafted STOP cases never receive a positive
   recommendation after policy gating);
4. **at least two treated arms have mutually non-overlapping bootstrap CIs** —
   implemented as counting unordered treated-arm PAIRS whose CI95s are fully
   disjoint (one interval entirely above the other; touching endpoints count
   as overlapping) and requiring pair count ≥ 2. This PAIR-COUNTING semantics
   was a deliberate choice: the plan's wording ("at least two treated arms")
   is ambiguous between "two arms involved in some disjoint pair" and "two
   disjoint pairs"; the shipped rule requires the stronger reading, so two
   arms sharing one common non-overlapping competitor do NOT satisfy it.

The gate is ADVISORY scaffolding: it runs in-process over a plain dict, so an
in-process caller could fabricate an evidence bundle that flips the verdict.
Genuine bundles carry `provenance_digest`; the classifier validates presence/
non-emptiness and echoes it verbatim but does NOT recompute it — documentation,
not code, identifies the canonical bundle (digest below).

## Uncertainty examination

All owner-mandated items, measured on the canonical run (numbers in
DAY6_RESULTS.md):

- **Per-arm sample basis**: every treated arm is scored on the SAME 558-row
  randomized test block (counterfactual queries); the smallest ARM SLICE is
  still HUMAN_REVIEW with 52 assigned rows — per-arm-slice CIs stay wide and
  the Day 5 CONTROL tiny-slice calibration inversion remains part of the
  per-arm reference's honest profile (pooled avoids it: CONTROL AUC 0.611326).
- **Calibration status**: detected from bundle metadata; raw bundles would
  warn loudly. Canonical evidence records `calibrated`.
- **Propensity-range overlap**: model-vs-truth range overlap holds on all four
  treated arms (coarse positivity-style sanity flag, never a gate).
- **Bootstrap CIs and overlaps**: per treated arm around mean model revenue
  (B=500, stratified within assigned arm at true positions); RETRY_NOW is
  disjoint from ALL three other arms; the other three mutually overlap →
  3 disjoint pairs against the ≥2 requirement.
- **Seed variance** (from four refit replicates, seeds {20260826, 1, 2, 3}):
  match-rate sd ≈ 0.0271, relative-regret sd ≈ 0.0100, per-arm mean-revenue
  sds ₹2.5–₹6.6; micro-Brier sd 0.000935 (per-arm) / 0.000250 (pooled).
- **Heavy tails**: mean relative regret hides them — per-row regret quantiles
  p50 = ₹0, p90 ≈ ₹174.76, p99 ≈ ₹871.43 accompany the ₹61.60 mean.
- **Single-canonical-run discipline**: the optimizer verdict is decided by ONE
  canonical run; stability is REPORTED, not certified (D-E6).

## Safety architecture

AI recommends, policy authorizes — unchanged from Day 1:

```
 decision-time context row
           |
           v
 +----------------------+   IncrementalRevenue   +--------------------------+
 | calibrated per-arm   |   MODEL ESTIMATE       | candidate recommendation |
 | pipelines: MODEL     |  (Pa-P_CONTROL)*amount | argmax over the 4 TREATED|
 | ESTIMATE P(a|ctx)    |  - retry_cost - risk   | arms (ARM_ORDER ties)    |
 +----------------------+                        +--------------------------+
                                                              |
                                                              | candidate ONLY
                                                              v
                                                 +------------------------+
                                                 | policy engine          |
                                                 | recovery.policy        |
                                                 | .decide_action (frozen |
                                                 |  business rules)       |
                                                 +------------------------+
                                                              |
                                                              v
                                             authorized_action  (STOP dominant)
```

Flow invariant: AI estimate → incremental estimate → candidate recommendation
→ policy engine → authorized_action. The recommender emits what the revenue
estimate PREFERS; the authorized action is whatever the frozen rules say, and
when they differ the recommendation records
`policy_overrode_candidate=True`. `BoundedRecommender` cannot be constructed
unless the evidence classifies JUSTIFIED; the probe replays opted-out/fraud/
hard-decline contexts and passes only when EVERY context authorizes STOP
regardless of the injected revenue candidate.

The optimizer recommender never overrides policy; STOP remains dominant when configured.

## CLI usage and reproducibility

```
python -m simulation.cli generate [--rows 5000] [--seed 42] [--out data/treatment_outcomes.csv]
python -m simulation.cli validate [--csv data/treatment_outcomes.csv]
python -m simulation.cli summary   [--csv data/treatment_outcomes.csv]
```

Same seed+rows regenerate a byte-identical file (sha256 proof in
DAY6_RESULTS.md); a different dataset seed yields a DIFFERENT file that still
validates — different synthetic world, same schema. `data/treatment_outcomes.csv`
is generated and gitignored, never committed. Seed semantics are recorded in
every `generate` output line (`dataset_seed` vs `policy_master_seed`).
`validate` certifies the treatment/outcome SCHEMA and dataset contract only —
it does NOT certify attempt-context integrity; provenance of a CSV comes from
regeneration under the pinned seed and its sha256, never from validation
alone.

All estimates describe the synthetic world defined by the Day 4 simulator; no real-world causal or production claim is made.

## What does NOT change

The Day 2 baseline stays exactly `P(recovered | context)`; the Day 5 per-arm
models stay the reference production family; no contextual bandit, RL,
autonomous experimentation, uplift estimator, API, frontend, dashboard, or
payment execution exists anywhere in Day 6. Even when the classifier says
JUSTIFIED, the recommender only produces candidates behind the policy engine.

## Limitations

- **Advisory gate**: the classifier consumes a plain dict; `provenance_digest`
  is a visibility marker, not tamper-proofing (nothing recomputes it).
- **Injected-probability upward bias**: model estimates OVERSHOOT noise-
  integrated truth on every treated arm (+₹87.06 to +₹230.73 per case); the
  probe/recommender inject the TOP ARM's model probability into
  `decide_action`, while stage-1 rules R006/R007/R008 fired during simulation
  on baseline-scale calibrated probabilities — gate outcomes under injected
  estimates are not comparable to the simulator's own stage-1 outcomes (STOP
  dominance held regardless).
- **Uniform-cost accounting**: a single retry-cost constant is applied to all
  treated arms including REQUEST_UPDATE/HUMAN_REVIEW whose true economics
  differ; because cost/risk terms are arm-independent constants per row,
  argmax decisions reduce to argmax incremental recovery × amount.
- **Learning-curve prefix confound**: fractions take the EARLIEST share of
  randomized training rows (chronological prefix), so fraction points mix
  sample-size effects with early-time-window shift; they are not a clean
  data-scaling measurement.
- **Pooled contrast compression**: the pooled family calibrates LEVEL better
  (lower mean |P̂−T|) yet compresses treatment CONTRASTS toward zero
  (counterfactual lifts ≈ 0.01–0.04 vs per-arm ≈ 0.13–0.19; RETRY_NOW main
  gap −0.4372, gated cell gap −0.7840) — level accuracy and effect recovery
  come apart, and D-E2 gates the latter.
- **CONTROL exclusion**: candidate set = 4 treated arms; CONTROL incremental
  revenue is undefined/negative under uniform retry-cost accounting by design.
- All numbers measure the synthetic simulator, not any production system.

## What would change the verdict

The canonical classification is **OPTIMIZER_JUSTIFIED** (evidence in
DAY6_RESULTS.md). It would flip to NOT_YET_JUSTIFIED if any of:

1. relative regret rose above 0.15 — the thinnest margin in the gate
   (canonical 0.1383 vs threshold 0.15; per-seed replicates spanned
   0.1176–0.1435 with sd ≈ 0.0100, so modest world/bundle changes flip it);
2. decision-match rate fell below 0.60 (robust today: worst replicate 0.747);
3. any policy-safety probe context received a positive authorization after
   gating (requires a change to the frozen business rules themselves);
4. fewer than two pairwise-disjoint treated-arm CI95s remained (today 3:
   RETRY_NOW vs each other arm; bootstrap spread merging RETRY_NOW toward the
   pack could reduce the count);
5. under D-E2, pooled would displace per-arm only by passing ALL FOUR criteria
   simultaneously — today it fails 1, 3, and 4 with wide margins.

Conversely the Gate A verdict (per-arm preferred) flips to pooled only via the
same four-criterion sweep, and neither verdict weakens the policy boundary.
Exact evidence needed next: more randomized test mass (especially
HUMAN_REVIEW slices), a cost model that differentiates REQUEST_UPDATE/
HUMAN_REVIEW economics, out-of-sample stage-1 refits to de-bias eligibility
drift, and multi-dataset-seed replication before any stability CERTIFICATION.
