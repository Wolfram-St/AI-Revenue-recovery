# Day 4 — Simulated Treatment Outcomes

Status: implemented on branch `feature/day4-treatment-simulation`.

## What was implemented

1. `simulation/config.py` — loads the declarative
   `config/treatment_policy.yaml` (v1.0, `master_seed 20260826`) into typed
   frozen dataclasses via `yaml.safe_load` with a closed key vocabulary and
   loud `ValueError`s; exports the seed-stream child constants
   (`SEED_STREAM_ASSIGNMENT=0`, `SEED_STREAM_OUTCOMES=1`,
   `SEED_STREAM_TEMPORAL=2`).
2. `simulation/treatment.py` — two-stage treatment assignment over the
   canonical 5,000-row dataset: a deterministic safety-eligibility gate
   followed by randomized arm draws.
3. `simulation/outcomes.py` — simulated action-aware outcomes plus stored
   SYNTHETIC GROUND TRUTH columns, and full-or-zero recovered revenue.
4. `simulation/temporal.py` — treatment/outcome timestamp stamping strictly
   after each recorded payment failure.
5. `simulation/dataset.py` — additive D5 dataset contract, one-to-one join on
   `attempt_id`, structured validator, and leakage-guard surface.
6. `simulation/reporting.py` — labeled arm summaries: SIMULATED GROUND TRUTH
   vs OBSERVED SIMULATED OUTCOME. No causal estimators of any kind exist in
   Day 4.

## Treatment policy

Everything stochastic or effect-shaped about the synthetic world is data in
`config/treatment_policy.yaml`, validated by the loader — never hidden code:

| Section | Values |
| --- | --- |
| version | `1.0` |
| master_seed | `20260826` |
| arm_probabilities | CONTROL `0.20` · RETRY_NOW `0.30` · RETRY_LATER `0.25` · REQUEST_UPDATE `0.15` · HUMAN_REVIEW `0.10` |
| main_effects_logit | CONTROL `0.0` · RETRY_NOW `+0.60` · RETRY_LATER `+0.35` · REQUEST_UPDATE `+0.45` · HUMAN_REVIEW `+0.25` |
| interactions_logit | exactly two: `RETRY_NOW × failure_category == temporary_decline → +0.40`; `RETRY_LATER × attempt_number >= 3 → −0.25` |
| noise_sigma_logit | `0.5` |
| treatment_delay_hours | RETRY_NOW `0.25` · REQUEST_UPDATE `2.0` · HUMAN_REVIEW `4.0` · RETRY_LATER `24.0` |
| resolution_window_hours | uniform draw bounds `[1.0, 48.0]` |
| base_propensity_terms | intercept `−0.35`, category effects (`temporary_decline +0.95` … `hard_decline −1.45`), history/attempt/fraud/amount/method/device terms mirroring the Day 1 generator family |

Every number is an ILLUSTRATIVE SYNTHETIC choice describing the simulated
world, not an estimate from any real provider data.

## CONTROL definition

A row that receives **no intervention**. CONTROL has two labeled sources
(D1), separated everywhere in reporting:

1. **Randomized controls** — eligible rows drawn to the CONTROL arm by the
   stage-2 multinomial (`arm_source="randomized"`,
   `assignment_probability=0.20`). Clean natural-recovery signal.
2. **Safety-censored controls** — rows whose stage-1 policy authorization is
   STOP are forced to CONTROL because treating safety-blocked customers would
   contradict the product's own rules (`arm_source="safety_censored"`,
   `assignment_probability=0.0` — no stage-2 draw occurred).

In the canonical run: 743 randomized controls + 1,188 safety-censored = 1,931
CONTROL rows of 5,000.

## Assignment mechanism (two-stage hybrid)

Stage 1 calls the shipped `decide_action` **directly per row** over
decision-time context — the engine's ERV/no-op candidate path is explicitly
NOT part of assignment — with caller-supplied recovery probabilities injected
as `recovery_probability`. Those probabilities come from the frozen Day 2
pipeline (`predict_recovery_probability(model, df)`, sigmoid-calibrated fit on
validation only), computed once over the FULL 5,000-row frame; they are never
refit and never persisted inside the simulator. Any STOP authorization forces
CONTROL/safety-censored. Stage 2 randomizes every remaining eligible row
across the five configured arms with one vectorized multinomial draw from
seed-stream child 0. NaN values in any policy-referenced context column are
rejected loudly before the gate (NaN compares False inside rule conditions and
would silently bypass STOP rules). Canonical-run stage-1 gate outcome:
STOP 1,188 · RETRY_LATER 2,953 · REQUEST_UPDATE 808 · RETRY_NOW 41 ·
HUMAN_REVIEW 10.

Seed-stream discipline (D1b): all three stochastic stages derive from ONE
master seed via `numpy.random.default_rng(master_seed).spawn(k)` in fixed
order — child 0 stage-2 assignment multinomial, child 1 outcome Bernoulli +
logit noise, child 2 temporal resolution windows. Bare `default_rng(seed)`
re-derivation inside modules is forbidden, so stages can never reuse or
reorder each other's streams.

## Outcome generation

For row i assigned arm a (D2):

```
base_logit(i)      = documented coefficient family over decision-time context
effect_logit(i,a)  = main_effects_logit[a] + matching interaction terms
propensity(i,a)    = sigmoid(base_logit + effect_logit)   # PRE-noise probability
noise(i)           ~ Normal(0, sigma = 0.5)               # logit-scale, child stream 1
simulated_recovered(i) ~ Bernoulli(sigmoid(logit(propensity(i,a)) + noise(i)))
```

The base-propensity coefficients mirror the Day 1 generator family so
propensities land in a realistic band; measured canonical-run propensity range
under assignment is [0.064042, 0.931203]. Boundaries 0/1 are impossible given
finite logits plus noise (unit-tested). Stored ground truth per row:
`base_recovery_propensity`, `action_effect_logit`,
`propensity_under_assignment`, `assignment_probability` — GROUND TRUTH /
evaluation-only, never features.

## Recovered revenue generation

Full-or-zero (D3): a recovered payment returns exactly
`round(amount_inr, 2)`; an unrecovered payment returns exactly `0.00`.
Partial recovery is explicitly NOT modeled. Payouts can never be negative or
exceed the settled payable (canonical run: 0 violations across 5,000 rows).

## Temporal semantics

`failure_ts = event_timestamp`. Treated rows:
`treatment_ts = failure_ts + arm delay` (RETRY_NOW +15 min, REQUEST_UPDATE
+2 h, HUMAN_REVIEW +4 h, RETRY_LATER +24 h), then
`outcome_ts = treatment_ts + Uniform[1h, 48h]` (seeded, child stream 2).
CONTROL rows carry the documented null semantic
`treatment_timestamp = pd.NaT` — no intervention occurred — and their
observation horizon anchors at the failure itself:
`outcome_ts = failure_ts + Uniform[1h, 48h]`. Strict ordering is invariant-
tested and holds in the canonical run: `failure_ts < treatment_ts <
outcome_ts` for all treated rows, `failure_ts < outcome_ts` for all controls,
nulls exactly as specified — **0 temporal-ordering violations across 5,000
rows**, observed per-arm delays exactly equal to config, resolution windows
inside [1.019433, 47.970612] h.

## Confounding (documented verbatim from plan D1)

> *Documented selection confounding:* eligibility depends on context (fraud,
> opt-out, hard decline, retry count, low probability), so the probability of
> being assignable varies with context. Within the eligible pool, assignment
> is randomized. Cross-arm comparisons are therefore adjusted-for-nothing
> naive differences; only within-eligible-pool contrasts approach
> unconfoundedness.

Model-dependence is part of this same confounding: because rules R006/R007/
R008 consume model probability, eligibility depends on model quality as well
as true context; this model-dependence is part of the documented selection
confounding. The canonical run makes the strata gap concrete: aggregate
CONTROL recovery rate 0.368721 (includes safety-censored rows) vs randomized-
only CONTROL rate 0.585464 — a 0.22 gap produced purely by eligibility
selection, not by any action.

## Overlap and positivity

Positivity holds **within the eligible stratum by construction**: every
eligible row kept positive stage-2 probability for every arm
(0.20/0.30/0.25/0.15/0.10). This is a scoped claim — full-population
estimands are NOT supported because eligibility was context-dependent.
Measured overlap (canonical run): assignment-probability ranges are constant
per source as configured (safety-censored rows exactly 0.0);
propensity-under-assignment ranges per arm — CONTROL [0.064042, 0.820066],
RETRY_NOW [0.336145, 0.931203], RETRY_LATER [0.238400, 0.898626],
REQUEST_UPDATE [0.346334, 0.899257], HUMAN_REVIEW [0.327557, 0.842119] —
overlap heavily across arms within the eligible pool.

## Label discipline

Reporting outputs embed literal labels. **SIMULATED GROUND TRUTH** marks
quantities known by construction from the declarative policy (effects,
probabilities, noise). **OBSERVED SIMULATED OUTCOME** marks quantities
measured from this synthetic run (counts, rates, revenue, naive differences).
The phrase "causal estimate" may appear only inside disclaimers; no function
is named or documented as causal. Observed-difference outputs carry the fixed
non-causal note verbatim.

The treatment environment is synthetic and does not establish real-world causal treatment effects.

## What this does NOT prove

Nothing here estimates uplift, incremental causal revenue, or real-world
treatment effects.

The Day 2 baseline remains P(recovered | context).

No baseline file changed and no treatment/outcome field can enter its feature
matrix (whitelist design proven by leakage-guard tests against a merged
hostile frame). Expected Recovery Value played no role in assignment (the
gate calls `decide_action` directly), so the illustrative ₹10-cost artifact
from Day 3 scoring is absent here by construction and remains unrelated to
treatment assignment. Naive cross-arm differences mix strata and must never
be read as effects; only randomized-vs-randomized contrasts within the
eligible pool approach unconfoundedness, and even those estimate the
SIMULATED world's parameters, not the real one's.

Action-aware observations are now available for future modeling, but no action-aware predictive model is implemented in Day 4.

## Known limitations

- **In-sample probability caveat:** stage-1 probabilities come from Day 2
  model predictions over ALL 5,000 rows, so train-row probabilities are
  mildly in-sample (slightly optimistic feeding R006/R007/R008). Test-split
  rows remain clean; validation rows influenced only the calibrator shape.
- **Horizon asymmetry:** treated rows resolve on `treatment_ts + window`
  while controls resolve on `failure_ts + window` — a systematic horizon
  difference irrelevant to rate reporting but relevant to any future
  time-to-recovery analysis.
- **Full-or-zero revenue:** partial recovery is not modeled.
- **Finite-n ordering noise:** with ~600–1,100 cases per treated arm,
  observed rate orderings can deviate from ground-truth effect ordering (see
  DAY4_RESULTS.md for the honest comparison).
- **No CSV artifact yet:** `data/treatment_outcomes.csv` was NOT created;
  the `python -m simulation.generate` CLI wrapper that produces it lands
  with Day 5 or as immediate follow-up work. All module APIs return frames.
- All numbers measure the synthetic simulator, not any production system.

## Reproducibility

One master seed (`20260826`) feeds `Generator.spawn` children in the fixed
order 0/1/2 (assignment / outcomes / temporal). Within each child the draw
order is fixed (single vectorized batches in row order). Identical inputs and
policy reproduce byte-identical output: the canonical rerun matched SHA-256
digests for the assignment frame, outcome frame, and timeline columns. Same
command + same seed ⇒ byte-identical results.

## How Day 5 could use the observations

The joined frame `(payment_attempts ⋈ treatment_outcomes) on attempt_id`
provides `(context, assigned_action, simulated_recovered)` triplets for all
5,000 rows — sufficient inputs for future action-aware models (e.g., an
action-conditional probability estimator trained on the triplets, evaluated
against the stored ground-truth propensities). Day 4 builds no such
estimator; modeling is Day 5 work behind the same labeled-synthetic boundary.

Action-aware observations are now available for future modeling, but no action-aware predictive model is implemented in Day 4.

## Test evidence

Day 4 suites (focused counts, fresh): `tests/test_treatment_config.py` 48 ·
`tests/test_treatment_assignment.py` 29 · `tests/test_outcomes.py` 32 ·
`tests/test_treatment_dataset.py` 43 · `tests/test_temporal_treatment.py` 26 ·
`tests/test_reporting.py` 21 — 199 new; plus 263 pre-existing = **462
passed** locally and identically inside Docker (tails in DAY4_RESULTS.md).
