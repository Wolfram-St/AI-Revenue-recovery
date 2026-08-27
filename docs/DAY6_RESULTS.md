# Day 6 Results — Decision Engine Evidence

All numbers below are fresh outputs from one canonical run captured verbatim:
dataset regenerated with `generate_dataset(5000, seed=42)`, chronological
70/15/15 split, baseline retrained with seed 42 and sigmoid-calibrated on
validation only, calibrated probabilities predicted over the FULL 5,000-row
frame, observation frame assembled under policy `master_seed 20260826`,
per-arm AND pooled models trained/calibrated with seed 20260826 on identical
randomized segments, evaluation bootstrap B=500 seed 20260826. Gate A numbers
come from `compare_models(...)`, Gate C numbers from
`decision_evidence(calibrated_per_arm_bundle, test_frame, policy,
stability_runs=<4 seed replicates>)` followed by
`classify_optimizer_justification(evidence)`.

## Run configuration

| Item | Value |
| --- | --- |
| Dataset seed | 42 (CLI `dataset_seed`; distinct from policy master_seed) |
| Rows | 5,000 × 19 context columns → observation frame 5,000 × 30 |
| Split | chronological 70/15/15 over observations (train 3,500 / validation 750 / test 750) |
| Day 2 baseline | XGBoost (300×d4, lr 0.1), sigmoid calibration fit on validation; probabilities min 0.237922 / max 0.722157 / mean 0.479985 over 5,000 rows |
| Policy | v1.0 (`config/treatment_policy.yaml`), master_seed 20260826 (`policy_master_seed`) |
| Model-family seed | 20260826 for BOTH families (`train_action_models(...)` and `train_pooled_model(...)`) |
| Calibration | per-family sigmoid on validation∩randomized(∩arm); pooled trains ONE pipeline on all 2,670 randomized train rows |
| Evaluation bootstrap | stratified within arm, corrected pool offsets, B=500, seed 20260826, level 0.95 |
| Bundle kind gated | calibrated only (D-E2 refuses raw bundles loudly) |
| Strata | randomized 3,812 · safety_censored 1,188 (full frame; out-of-scope everywhere) |

Captured splits identical to Day 5's canonical world: TRAIN 3,500
(2,670 randomized / 830 censored), VALIDATION 750 (584/166), TEST 750
(558/192); safety-censored out-of-scope counts 1,188 / 830 / 166 / 192.

Per-arm row counts (randomized-only train/cal(val)/test): CONTROL 528/99/116 ·
RETRY_NOW 785/178/169 · RETRY_LATER 686/154/133 · REQUEST_UPDATE 400/93/88 ·
HUMAN_REVIEW 271/60/52; small segments flagged: validation CONTROL (99),
validation+test REQUEST_UPDATE (93/88), validation+test HUMAN_REVIEW (60/52).
Pooled metadata: `train_rows 2670 · validation_rows 584`, features = 14
whitelist columns + `assigned_action`.

## Gate A — D-E2 rule application (captured verbatim)

```
[RULE APPLICATION] rule_id=D-E2 applies_to=calibrated_bundles_only
[RULE VERDICT] preferred_model=per_arm
  criterion=strict_micro_brier_ci_non_overlap passed=False
    evidence={"comparison": "pooled_upper_strictly_below_per_arm_lower", "per_arm_lower": 0.21118636203621186, "per_arm_upper": 0.2367422253465264, "pooled_lower": 0.21174431471469599, "pooled_upper": 0.2379697378120509}
  criterion=ground_truth_agreement_no_worse passed=True
    evidence={"comparison": "pooled_arm_mean_no_worse_than_per_arm", "per_arm_mean_abs_error_vs_integrated_true": 0.09683222147777804, "pooled_mean_abs_error_vs_integrated_true": 0.07993192953821567}
  criterion=smallest_arm_brier_no_worse passed=False
    evidence={"comparison": "pooled_smallest_arm_brier_no_worse", "per_arm_brier": 0.226339008163736, "pooled_brier": 0.2301607566784865, "smallest_test_arm": "HUMAN_REVIEW"}
  criterion=interaction_recovery_within_band passed=False
    evidence={"annotated_not_gated": ["RETRY_LATER|attempt_number>=3"], "band": 0.4, "gated_cells": {"RETRY_NOW|failure_category==temporary_decline": -0.784001497686478}}
```

n_randomized_test_rows=558 · smallest_test_arm=HUMAN_REVIEW · bundle_kind=
calibrated · scheme=stratified_within_arm_row_resampling_with_corrected_pool_
offsets · B=500 · level 0.95. Pooled fails criteria 1, 3, and 4 → per-arm
remains the reference production family; both families stay available.

Criterion-by-criterion reading:

1. **Strict micro-Brier CI non-overlap — FAIL**: intervals overlap heavily
   (per-arm [0.211186, 0.236742] vs pooled [0.211744, 0.237970]); the point
   estimates are near-identical (0.225168 vs 0.225119) which is context, not
   evidence.
2. **Ground-truth agreement no worse — PASS**: pooled arm-mean |P̂−integrated
   TRUE| 0.079932 beats per-arm 0.096832.
3. **Smallest-arm Brier no worse — FAIL**: HUMAN_REVIEW pooled 0.230161 vs
   per-arm 0.226339.
4. **Interaction recovery within band (POOLED cells gated) — FAIL**: the
   gated RETRY_NOW×temporary_decline cell gap −0.784001 sits far outside
   ±0.40; the fatigue cell carries `attenuation_expected=True` and is
   reported, never gated.

## Captured output — predictive metrics (OBSERVED SIMULATED OUTCOME)

CIs: seeded stratified bootstrap 95% (B=500). Both families scored on the
identical 558-row randomized test segment.

family=per_arm (the Day 5 calibrated reference):

| Slice | n | AUC [CI95] | Brier [CI95] | PR-AUC |
| --- | --- | --- | --- | --- |
| CONTROL | 116 | 0.392250 [0.287922, 0.495285] ⚠ | 0.254817 [0.246318, 0.262768] | 0.453647 |
| RETRY_NOW | 169 | 0.575986 [0.472537, 0.687725] | 0.190238 [0.162566, 0.219678] | 0.769696 |
| RETRY_LATER | 133 | 0.565977 [0.476467, 0.658892] | 0.252379 [0.225628, 0.282068] | 0.612528 |
| REQUEST_UPDATE | 88 | 0.543412 [0.404106, 0.673706] | 0.211347 [0.181159, 0.245417] | 0.717712 |
| HUMAN_REVIEW | 52 | 0.563725 [0.398850, 0.734348] | 0.226339 [0.187883, 0.270963] | 0.688000 |
| MICRO (all) | 558 | 0.603136 [0.558902, 0.653195] | 0.225168 [0.211186, 0.236742] | 0.712428 |

⚠ carried honestly from Day 5: sigmoid calibration on 99 randomized CONTROL
validation rows inverts that arm's ranking.

family=pooled:

| Slice | n | AUC [CI95] | Brier [CI95] | PR-AUC |
| --- | --- | --- | --- | --- |
| CONTROL | 116 | 0.611326 [0.505310, 0.717294] | 0.254708 [0.225297, 0.285295] | 0.645793 |
| RETRY_NOW | 169 | 0.600538 [0.494100, 0.702516] | 0.189629 [0.165677, 0.216202] | 0.778565 |
| RETRY_LATER | 133 | 0.591494 [0.494065, 0.679343] | 0.246894 [0.220151, 0.275416] | 0.650477 |
| REQUEST_UPDATE | 88 | 0.488160 [0.366434, 0.610539] | 0.218381 [0.182334, 0.255525] | 0.717694 |
| HUMAN_REVIEW | 52 | 0.504902 [0.314422, 0.665359] | 0.230161 [0.186314, 0.279801] | 0.656146 |
| MICRO (all) | 558 | 0.596975 [0.545326, 0.645376] | 0.225119 [0.211744, 0.237970] | 0.704862 |

Notable honest finding: the pooled family does NOT reproduce the per-arm
CONTROL tiny-slice inversion (AUC 0.611326 where per-arm shows 0.392250 ⚠),
yet its REQUEST_UPDATE/HUMAN_REVIEW slices rank at chance — sharing strength
helps where rows are plentiful and blunts arm-specific signal where they are
not.

## Captured output — ground-truth agreement (SIMULATED GROUND TRUTH replay)

Secondary probability-scale comparison; Jensen-floor annotated ("models
converge to the noise-integrated propensity, not the pre-noise sigmoid"):

| Arm | per-arm mean\|P̂−T\| (r / ρ) | pooled mean\|P̂−T\| (r / ρ) |
| --- | --- | --- |
| CONTROL | 0.125058 (−0.513613 / −0.501242 ⚠) | 0.087171 (+0.654259 / +0.638658) |
| RETRY_NOW | 0.102158 (+0.567640 / +0.571473) | 0.088575 (+0.721183 / +0.741572) |
| RETRY_LATER | 0.088636 (+0.516684 / +0.485386) | 0.076644 (+0.683778 / +0.647351) |
| REQUEST_UPDATE | 0.079998 (+0.473143 / +0.445088) | 0.065542 (+0.608448 / +0.598640) |
| HUMAN_REVIEW | 0.088310 (+0.421077 / +0.439512) | 0.081728 (+0.560359 / +0.553488) |
| **arm mean** | **0.096832** | **0.079932** |

Pooled wins absolute-calibration agreement on every arm — and still loses the
rule, because D-E2 gates effect RECOVERY too:

Logit-scale main-effect contrasts (band ±0.40, calibrated):

| Arm | configured | per-arm est / gap | pooled est / gap |
| --- | --- | --- | --- |
| CONTROL | 0.00 | 0.0000 / 0.0000 yes | 0.0000 / 0.0000 yes |
| RETRY_NOW | +0.60 | 0.8324 / +0.2324 yes | 0.1628 / −0.4372 **NO** |
| RETRY_LATER | +0.35 | 0.5693 / +0.2193 yes | 0.0003 / −0.3497 yes |
| REQUEST_UPDATE | +0.45 | 0.5479 / +0.0979 yes | 0.1475 / −0.3025 yes |
| HUMAN_REVIEW | +0.25 | 0.5424 / +0.2924 yes | 0.1462 / −0.1038 yes |

Interaction cells:

| Cell | configured | per-arm est / gap | pooled est / gap | n_cell | attenuation_expected |
| --- | --- | --- | --- | --- | --- |
| RETRY_NOW \| failure_category==temporary_decline (GATED) | +0.40 | 0.3901 / −0.0099 | −0.3840 / **−0.7840** | 79 | False |
| RETRY_LATER \| attempt_number>=3 (annotated) | −0.25 | +0.0436 / +0.2936 | −0.3252 / −0.0752 | 24 | True |

The pooled family compresses treatment contrasts toward zero (its sign even
flips on the gated cell) — level accuracy without effect recovery.

Complexity (capacity proxy = n_estimators × max_depth per fitted pipeline;
fit seconds measured during a timed refit, machine-dependent by nature):

| Family | fits | fit_seconds | capacity proxy total |
| --- | --- | --- | --- |
| per_arm | 5 | 0.906976 | 6,000 |
| pooled | 1 | 0.382 | 1,200 |

## Learning curves (captured)

```
[LEARNING CURVES PROTOCOL] fractions apply to the TRAIN segment only (deterministic
prefix of its randomized rows); calibration at every fraction uses the FULL validation
segment unchanged; the per-arm family refits all five arms at each fraction; points
score the identical randomized test segment
 fraction  n_train | pa_micro_brier pa_small_brier | po_micro_brier po_small_brier
     0.25      668 |       0.225537       0.218842 (52) |       0.231049       0.226933 (52)
     0.50     1335 |       0.225792       0.221485 (52) |       0.225057       0.216369 (52)
     1.00     2670 |       0.225168       0.226339 (52) |       0.225119       0.230161 (52)
```

Reading (with the mandatory confound): fractions take a chronological PREFIX,
so lower fractions are EARLIER time windows, not random subsamples — these
points mix sample size with time-window shift and cannot isolate a pure
learning-scaling law. Micro-Brier is nearly flat for both families; the
pooled family improves from 0.231049 (quarter data) to parity at full data,
while smallest-arm (HUMAN_REVIEW) Brier DEGRADES with more data for both
families (prefix-window shift suspected, unproven here).

## Stability across seeds {20260826, 1, 2, 3} (captured)

```
     seed | pa_micro_brier po_micro_brier | per-arm mean incrementals (RN, RL, RU, HR) | pooled mean incrementals (RN, RL, RU, HR)
        1 |       0.224649       0.225210 | 0.1964, 0.1450, 0.1380, 0.1304 | 0.0415, 0.0144, 0.0282, 0.0171
        2 |       0.224746       0.224574 | 0.1961, 0.1449, 0.1363, 0.1310 | 0.0408, 0.0134, 0.0284, 0.0167
        3 |       0.226966       0.225106 | 0.1949, 0.1433, 0.1380, 0.1292 | 0.0360, 0.0121, 0.0245, 0.0138
 20260826 |       0.225168       0.225119 | 0.1934, 0.1440, 0.1317, 0.1295 | 0.0441, 0.0143, 0.0300, 0.0191
[STABILITY SD] micro_brier_sd: per_arm=0.000935 pooled=0.000250
[STABILITY SD] mean_incremental_sd_by_arm[per_arm]: RETRY_NOW=0.0012, RETRY_LATER=0.0007, REQUEST_UPDATE=0.0026, HUMAN_REVIEW=0.0007
[STABILITY SD] mean_incremental_sd_by_arm[pooled]: RETRY_NOW=0.0030, RETRY_LATER=0.0009, REQUEST_UPDATE=0.0020, HUMAN_REVIEW=0.0019
```

Mean incrementals are MODEL ESTIMATE counterfactual lifts P̂(a) − P̂(CONTROL)
averaged over ALL randomized test rows (uniform view; nothing causal). Honest
finding: per-arm lifts hold ≈ 0.13–0.20 across seeds while pooled lifts
collapse to ≈ 0.01–0.04 — independent confirmation of the pooled contrast
compression above. The optimizer verdict itself remains SINGLE-CANONICAL-RUN:
stability is reported, not certified.

## Gate C — decision-quality evidence (captured)

Candidate set = the 4 treated arms; CONTROL excluded (uniform retry-cost
accounting makes its revenue undefined/negative by construction). Ties break
to earlier ARM_ORDER precedence on BOTH model and truth sides; cost/risk
terms are arm-independent constants per row, so argmax decisions reduce to
argmax incremental recovery × amount.

```
[DECISION MATCH RATE] matches=427/558 rate=0.765233 binomial_ci95=[0.730065, 0.800401]
[RELATIVE REGRET] absolute_regret_inr=61.6043 relative_regret=0.138288 expected_best_truth_revenue_inr=445.4774
[REGRET QUANTILES] {"p50": 0.0, "p90": 174.76409250161802, "p99": 871.4322784876683}
```

Per-treated-arm revenue evidence (MODEL ESTIMATE vs SIMULATED GROUND TRUTH;
bootstrap CI95 around mean MODEL revenue, B=500 stratified at true assigned-
arm positions):

| Arm | model ₹/case | truth ₹/case | model−truth | CI95 mean model revenue | ci_overlap_with |
| --- | --- | --- | --- | --- | --- |
| RETRY_NOW | 532.54 | 445.48 | +87.06 | [489.778355, 581.142045] | [] |
| RETRY_LATER | 404.29 | 174.05 | +230.24 | [368.615962, 442.265746] | REQUEST_UPDATE, HUMAN_REVIEW |
| REQUEST_UPDATE | 350.26 | 256.53 | +93.74 | [311.805619, 386.972149] | RETRY_LATER, HUMAN_REVIEW |
| HUMAN_REVIEW | 361.17 | 130.44 | +230.73 | [327.304441, 396.741387] | RETRY_LATER, REQUEST_UPDATE |

Overlap matrix (touching endpoints count as overlapping):

| | RETRY_NOW | RETRY_LATER | REQUEST_UPDATE | HUMAN_REVIEW |
| --- | --- | --- | --- | --- |
| RETRY_NOW | — | DISJOINT | DISJOINT | DISJOINT |
| RETRY_LATER | DISJOINT | — | OVERLAP | OVERLAP |
| REQUEST_UPDATE | DISJOINT | OVERLAP | — | OVERLAP |
| HUMAN_REVIEW | DISJOINT | OVERLAP | OVERLAP | — |

→ 3 pairwise-disjoint CI pairs (criterion 4 requires ≥ 2).

Uncertainty inventory (canonical):

- per-arm n: 558 for every treated arm (shared counterfactual block);
  calibration_status `calibrated`;
- propensity-range overlap model-vs-truth: TRUE on all four treated arms
  (coarse positivity-style sanity flag, never a gate);
- seed variance (from four refitted replicates, seeds {20260826, 1, 2, 3}):

| Seed | match rate | relative regret | RN ₹ | RL ₹ | RU ₹ | HR ₹ |
| --- | --- | --- | --- | --- | --- | --- |
| 20260826 | 0.7652 | 0.1383 | 532.54 | 404.29 | 350.26 | 361.17 |
| 1 | 0.7473 | 0.1386 | 547.35 | 406.00 | 363.13 | 363.79 |
| 2 | 0.7563 | 0.1435 | 537.98 | 404.36 | 364.67 | 369.72 |
| 3 | 0.8172 | 0.1176 | 548.24 | 399.33 | 365.97 | 355.87 |
| **population sd** | **0.027134** | **0.009956** | 6.5630 | 2.4983 | 6.2855 | 4.9870 |

Policy-safety probe (three crafted STOP contexts through the frozen business
rules; passes only when EVERY context authorizes STOP regardless of candidate):

```
[POLICY SAFETY PROBE] passed=True
  context=customer_opted_out candidate=RETRY_LATER    authorized=STOP  overrode=True
  context=fraud_risk         candidate=RETRY_LATER    authorized=STOP  overrode=True
  context=hard_decline       candidate=RETRY_LATER    authorized=STOP  overrode=True
```

The recommender moved every candidate to RETRY_LATER; the policy engine
overrode ALL three to STOP. Disclosure: the probe injects the TOP ARM's model
probability into `decide_action`; those injected estimates are upward-biased
vs truth and sit on a different scale than the baseline-calibrated
probabilities that drove R006/R007/R008 during simulation, so probe gate
outcomes are not comparable to the simulator's stage-1 outcomes — STOP
dominance held regardless of the injection.

## Classifier output (captured verbatim)

```
[CLASSIFIER] classify_optimizer_justification(evidence)
{
  "classification": "OPTIMIZER_JUSTIFIED",
  "criteria": [
    {
      "criterion": "decision_match_rate_at_or_above_threshold",
      "passed": true,
      "evidence": {
        "observed": 0.7652329749103942,
        "threshold": 0.6
      }
    },
    {
      "criterion": "relative_regret_at_or_below_threshold",
      "passed": true,
      "evidence": {
        "observed": 0.13828818025724943,
        "threshold": 0.15
      }
    },
    {
      "criterion": "policy_safety_probe_passed",
      "passed": true,
      "evidence": {
        "observed": true
      }
    },
    {
      "criterion": "minimum_non_overlapping_treated_arm_ci_pairs",
      "passed": true,
      "evidence": {
        "non_overlapping_pairs": 3,
        "required": 2
      }
    }
  ],
  "reasons": [],
  "provenance_digest": "0471c5e047ef9ad152e5bd86994e4028533215d29b5e541d82137c52b224caa0"
}
```

## Determinism

Two fresh `decision_evidence` calls on identical inputs (same bundle, frames,
and stability replicates):

```
run A sha256(sorted-json): 03b40316a1b41f3d24c346d90dccec0b27c0b707ba33f050721428f1b053f889
run B sha256(sorted-json): 03b40316a1b41f3d24c346d90dccec0b27c0b707ba33f050721428f1b053f889
match: True
module provenance_digest (embedded): 0471c5e047ef9ad152e5bd86994e4028533215d29b5e541d82137c52b224caa0
provenance digests equal across runs: True
```

## Incremental twins (Day 5 modules reused unchanged)

n_randomized_test_rows=558; identical row set for both tables (fingerprint
match 1202702195678624065); constants intervention_cost_inr=10.0 ·
unknown_category_risk_fraction=0.05 (74 unknown-category rows risk-penalized).
Numbers byte-match the Day 5 post-fix canonical capture:

| Arm | model mean Δ | truth mean Δ | model ₹/case | truth ₹/case |
| --- | --- | --- | --- | --- |
| RETRY_NOW | 0.1934 | 0.1536 | 532.54 | 445.48 |
| RETRY_LATER | 0.1440 | 0.0671 | 404.29 | 174.05 |
| REQUEST_UPDATE | 0.1317 | 0.0937 | 350.26 | 256.53 |
| HUMAN_REVIEW | 0.1295 | 0.0531 | 361.17 | 130.44 |

Model estimates overshoot truth on every treated arm (+₹87.06 to +₹230.73 per
case) — the upward bias that drives both the regret tails and the injected-
probability disclosure. `cost_simplification_note` embedded verbatim in every
revenue table: "A single retry-cost constant is applied uniformly to all
treated arms including REQUEST_UPDATE and HUMAN_REVIEW, whose true economics
differ."

## CLI reproducibility proof (Gate B, captured)

```
=== RUN A (seed 42) ===
{"dataset_seed": 42, "policy_master_seed": 20260826, "rows": 5000, "path": "data\\treatment_outcomes.csv", "sha256": "576632a9c8d505297031dff0455e1bfc3ba9e7e777e6a338891ef8f10be1f40f"}
exit=0
=== RUN B (seed 42) ===
{"dataset_seed": 42, "policy_master_seed": 20260826, "rows": 5000, "path": "data\\treatment_outcomes.csv", "sha256": "576632a9c8d505297031dff0455e1bfc3ba9e7e777e6a338891ef8f10be1f40f"}
exit=0
=== VALIDATE ===
{"valid": true, "violations": [], "row_count": 5000, "column_count": 30, "classification_complete": true, "column_contract_valid": true}
exit=0
=== RUN C (seed 43, different world) ===
{"dataset_seed": 43, "policy_master_seed": 20260826, "rows": 5000, "path": "...\\day6_seed43.csv", "sha256": "f73de89c1beaf3df83f5233b478096a5e4a29efc8b6f42eca3cadd84b1514d81"}
{"valid": true, "violations": [], "row_count": 5000, "column_count": 30, "classification_complete": true, "column_contract_valid": true}
```

Same seed ⇒ byte-identical sha256; different dataset seed ⇒ different file,
still schema-valid. `summary` prints per-arm OBSERVED SIMULATED OUTCOME rates
(CONTROL 0.3687 overall / 0.5855 randomized-only · RETRY_NOW 0.7182 ·
RETRY_LATER 0.6300 · REQUEST_UPDATE 0.6850 · HUMAN_REVIEW 0.6501) plus
overlap diagnostics with positivity holding within the eligible stratum by
construction. The CSV stays gitignored and was never committed.

## Required disclosures

1. **Single-canonical-run verdict**: OPTIMIZER_JUSTIFIED is decided by ONE
   canonical run at dataset seed 42 / action seed 20260826. Stability across
   estimator seeds is REPORTED (regret sd ≈ 0.0100, match-rate sd ≈ 0.0271,
   all replicates still passing their thresholds) — NOT certified; no
   multi-DATASET-seed replication exists yet.
2. **Learning-curve prefix confound**: curve fractions take the earliest
   chronological prefix of randomized training rows, so points conflate
   sample size with early-time-window shift; no clean scaling claim is made.
3. **CI-pair semantics chosen**: D-E5 criterion 4 counts unordered treated-
   arm PAIRS with fully disjoint CI95s (touching endpoints overlap) and
   requires ≥ 2 — the stronger of two readings of "at least two treated arms
   have mutually non-overlapping CIs"; chosen before measurement.
4. **Injected-probability upward bias**: model revenue overshoots truth on
   every arm (+₹87.06…+₹230.73); the probe/recommender inject top-arm model
   probabilities into `decide_action` whereas stage-1 rules R006/R007/R008
   fired on baseline-scale calibrated probabilities during simulation —
   argmax-over-4-arms decisions and probe gates therefore live on a different
   probability scale than the simulator's own gating.
5. **Validate certifies schema, not context integrity**: the CLI validator
   checks column contract + treatment/outcome dataset rules only; a tampered
   CSV with valid schema and corrupted contexts would still "validate".
   Provenance comes from pinned-seed regeneration + sha256, never from
   validation alone.

## Verification evidence

Focused counts per suite file (fresh collections at HEAD d2ad0cb):
test_cli.py 14 · test_pooled_model.py 34 · test_model_comparison.py 26 ·
test_decision_evidence.py 43 · test_decision_policy.py 73 = **190 new**; plus
585 pre-existing = **775 total locally**.

Full-suite tails (fresh runs, both environments):

- Local: `.venv\Scripts\python -m pytest -q` → `775 passed in 104.74s (0:01:44)`
- Docker: `docker build -t recoverai-app .` → OK;
  `docker run --rm recoverai-app python -m pytest -q` →
  `774 passed, 1 skipped in 96.80s (0:01:36)` — the one skip is
  `tests/test_cli.py:335` (".gitignore absent from this context (e.g. Docker
  image)"), an environment guard that cannot run inside an image built via
  `.dockerignore`.

Artifact check: `data/treatment_outcomes.csv` exists locally (CLI proof) and
is gitignored; git tree contains ONLY the three new Day 6 documentation files.

## Day 6 gate

| Gate check | Evidence | Result |
| --- | --- | --- |
| Canonical chain executed end-to-end with real captured outputs | dataset → baseline → calibrated probabilities → assemble → split → BOTH families trained/calibrated on identical segments → compare_models → learning_curves → stability_check → decision_evidence → classifier; every table above verbatim | PASS |
| Comparison protocol (D-E1) | identical randomized train/validation/test segments both families; calibrated-bundles-only rule (raw refused); corrected-offset bootstrap B=500; curves at {0.25,0.50,1.00}; four-seed stability called explicitly with (20260826, 1, 2, 3) | PASS |
| Pre-registered D-E2 rule applied as written | verdict per_arm; criteria 1/3/4 fail with wide margins, criterion 2 passes; pooled-only interaction-gating interpretation documented; annotated fatigue cell reported-not-gated | PASS (verdict: per-arm remains preferred) |
| CLI artifact reproducibility (D-E3/Gate B) | same-seed sha256 identical twice (576632a9…); different seed → different valid file; validate/summary green; dataset_seed vs policy_master_seed recorded distinctly; purity-tested imports | PASS |
| Decision-quality measurement (D-E4/Gate C) | match rate 427/558 = 0.765233 [0.730065, 0.800401]; relative regret 0.138288 with p50/p90/p99 = ₹0/₹174.76/₹871.43; per-arm bootstrap CIs + overlap matrix (3 disjoint pairs); uncertainty inventory incl. 4-run seed variance | PASS |
| Policy safety (D-E5 criterion 3) | three STOP contexts (opted-out/fraud/hard-decline) all overridden to STOP despite positive candidates; recommender constructible only behind JUSTIFIED classification; candidate→decide_action invariant tested | PASS |
| Classifier discipline (D-E5) | thresholds fixed pre-measurement; ≥2-disjoint-CI-PAIRS semantics documented; provenance_digest echoed; advisory-gate honesty stated in docstrings and docs | PASS |
| Safety architecture invariant | AI estimate → incremental estimate → candidate recommendation → policy engine → authorized_action; STOP dominance preserved by construction and tests; no bypass path | PASS |
| Label language audit | MODEL ESTIMATE / OBSERVED SIMULATED OUTCOME / SIMULATED GROUND TRUTH throughout; "causal" only inside disclaimers; synthetic-world-only scope notes in every module | PASS |
| Baseline boundary held | Day 2 baseline untouched; per-arm reference untouched; purity tests pin `simulated_recovered` target and import whitelists | PASS |
| Determinism | evidence dict sorted-json SHA-256 identical across two fresh calls (03b40316…); module provenance_digest stable (0471c5e0…) | PASS |
| Full test suite green | 190 new + 585 pre-existing = 775 local / 774+1 env-skip Docker | PASS |
| Documentation complete | DAY6.md covers build, gates, protocol, restated rules (incl. pooled-gating + CI-pair semantics choices), uncertainty examination, safety diagram, CLI usage, limitations, verdict-flippers; both mandated sentences verbatim | PASS |

## Final classification: **OPTIMIZER JUSTIFIED**

All four D-E5 criteria pass on the canonical run: match rate 0.765233 ≥ 0.60;
relative regret 0.138288 ≤ 0.15; policy-safety probe 3/3 STOP; 3 ≥ 2 disjoint
CI pairs. Reported plainly, WITH its sensitivities — what flips it:

- **Regret margin is thin**: 0.1383 against a 0.15 threshold; per-seed
  replicates spanned 0.1176–0.1435 (sd 0.0100). A slightly harder world or a
  modestly worse bundle clears the line. This is the fragile criterion.
- Match rate is robust (worst replicate 0.747 vs 0.60).
- CI-pair count would drop below 2 only if RETRY_NOW stopped separating from
  all three other arms simultaneously.
- The probe can only flip if the frozen business rules change — and any such
  change re-opens the safety question wholesale.
- The gate is advisory scaffolding: an in-process caller could fabricate a
  bundle; the canonical bundle is identified by its provenance digest
  0471c5e047ef9ad152e5bd86994e4028533215d29b5e541d82137c52b224caa0 and by
  this document, not by code.

JUSTIFIED enables constructing the bounded recommender — nothing more. It
authorizes NO deployment, NO real-world action, and no bypass of
`decide_action`. Before any stability CERTIFICATION: multi-dataset-seed
replication, differentiated cost models for REQUEST_UPDATE/HUMAN_REVIEW, and
out-of-sample stage-1 refits remain the identified next evidence.

All estimates describe the synthetic world defined by the Day 4 simulator; no real-world causal or production claim is made.
