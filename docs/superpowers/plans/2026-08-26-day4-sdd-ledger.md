# SDD Progress Ledger — Day 4 Simulated Treatment Outcomes

Plan: `docs/superpowers/plans/2026-08-26-day4-treatment-outcomes.md`
Workflow: subagent-driven development (fresh implementer + independent reviewer per task, fix/re-review loops)
Branch/worktree: `feature/day4-treatment-simulation` @ `D:/airev-day4` (base: `4fdd889` = post-plan Day 3 main)
Boundary: model estimates `P(recovered | context)` only — never action-conditional; all treatment/outcome work is labeled synthetic.

| Task | Scope | Commit | Focused tests | Full suite (local = Docker) | Implementer review | Re-review |
|---|---|---|---|---|---|---|
| 1 | Treatment policy YAML + validated frozen-dataclass loader + seed-stream constants | `9f872fb` | 48 | 311 / 311 | APPROVE after fix loop | APPROVE |
| 2 | Two-stage assignment (STOP safety gate → randomized arms), spawn-child 0 discipline | `23f0908` | 29 | 340 / 340 | APPROVE (+polish) | n/a (approved) |
| 3 | Outcome simulation with stored ground truth, spawn-child 1, fixed draw order | `d701757` | 32 | 369 / 369 | REQUEST_CHANGES → fixed | APPROVE |
| 4 | Full-or-zero recovered revenue (payout-side rounding, bounds) | `111473d` | 32 | 372 / 372 | REQUEST_CHANGES → fixed | APPROVE |
| 5 | Additive D5 dataset contract + validator + leakage-guard tests | `2ba47a2` | 43 | 415 / 415 | REQUEST_CHANGES → fixed | APPROVE |
| 6 | Temporal timeline stamping (arm delays, control-null rule, spawn-child 2) | `30c1d72` | 26 | 441 / 441 | APPROVE (+polish) | n/a (approved) |
| 7 | Labeled reporting utilities (SIMULATED GROUND TRUTH vs OBSERVED SIMULATED OUTCOME) | `8ec00b2` | 21 | 462 / 462 | APPROVE (+polish) | n/a (approved) |
| 8 | Documentation + Day 4 gate (DAY4.md, DAY4_RESULTS.md, this ledger; canonical-run capture) | `ef86b46` | — | 462 / 462 (fresh tails in DAY4_RESULTS.md) | APPROVE — all headline numbers independently reproduced by reviewer's own rerun | n/a (approved) |

Focused-test counts are per-suite fresh collections: `test_treatment_config.py` 48 ·
`test_treatment_assignment.py` 29 · `test_outcomes.py` 32 ·
`test_treatment_dataset.py` 43 · `test_temporal_treatment.py` 26 ·
`test_reporting.py` 21 = **199 new**; plus 263 pre-existing = **462**.

Canonical-run verification recorded in `docs/DAY4_RESULTS.md`: 0 temporal-ordering
violations · 0 amount-bounds violations · validator valid · leakage guard clean ·
byte-identical determinism rerun under master_seed 20260826.

## Standing constraints (unchanged)
- No uplift/causal/bandit/RL/LangGraph/API/frontend/dashboard/database/execution code.
- The Day 2 baseline remains exactly `P(recovered | context)`; no new field can enter its feature matrix.
- Causal-language ban: SIMULATED GROUND TRUTH / OBSERVED SIMULATED OUTCOME / CONTROL / TREATMENT only; "causal estimate" appears solely inside disclaimers.
- Every stochastic draw derives from master_seed via `Generator.spawn` children 0/1/2 in fixed order; no bare `default_rng` re-derivation.
