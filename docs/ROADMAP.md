# ROADMAP.md

## Objective
Move from current Razorpay-centered baseline to a strong **PayPal + AI** hackathon submission with deterministic controls, auditable actions, and clear demo evidence.

> PayPal integration is a planned roadmap item; it is not already implemented.

## Milestones
- **M0:** Baseline hardening complete
- **M1:** Provider abstraction live with Razorpay compatibility
- **M2:** PayPal sandbox Orders/Payments + Webhooks operational
- **M3:** AI + policy + approval UX integrated
- **M4:** End-to-end demo + security + submission readiness

## Phase Plan

### Phase 0 — Baseline Hardening
- Goals: stabilize current behavior and surface seams.
- Tasks:
  - inventory Razorpay coupling points
  - codify simulation/sandbox/production modes
  - tighten audit evidence consistency
- Likely files: `recovery/recovery_agent.py`, `recovery/recovery_executor.py`, `recovery/audit_trail.py`, `tests/`
- Dependencies: none
- Acceptance criteria:
  - current tests pass in touched areas
  - documented coupling map exists
- Demo evidence: baseline run + audit output snapshot
- Rollback: revert to pre-hardening commit

### Phase 1 — Provider Abstraction
- Goals: provider-neutral interfaces + normalized models.
- Tasks:
  - introduce provider interface
  - adapt current Razorpay path behind adapter
  - normalize money/currency representation (`amount_minor`, ISO currency)
  - establish true recovered-state semantics independent of provider-specific fields
- Likely files: `recovery/` new provider modules, policy/executor touchpoints, tests
- Dependencies: Phase 0
- Acceptance criteria:
  - existing Razorpay behavior preserved through adapter path
  - provider-neutral tests pass
- Rollback: feature-flag/adapter fallback to direct Razorpay path

### Phase 2 — PayPal Sandbox Integration
- Goals: first-class PayPal sandbox path.
- Tasks:
  - OAuth client credentials flow
  - Orders/Payments create+capture/read
  - Invoicing support (create/send/status)
  - webhook verify/normalize/idempotent ingestion
- Likely files: `recovery/providers/paypal_*`, webhook modules, API wiring, config
- Dependencies: Phase 1
- Acceptance criteria:
  - sandbox transaction lifecycle demonstrable
  - webhook handling deterministic + replay-safe
- Demo evidence: sandbox API traces + normalized case progression
- Rollback: disable PayPal adapter via config toggle

### Phase 3 — AI Agent UX + Policy/Approval Controls
- Goals: meaningful AI usage without unbounded autonomy.
- Tasks:
  - AI recommendation outputs (ranked actions + rationale)
  - deterministic policy authorization gate
  - explicit human approval controls for consequential actions
  - bounded execution path and audit linkage
- Likely files: `ml/`, `recovery/policy.py`, `recovery/recovery_executor.py`, frontend views
- Dependencies: Phases 1-2
- Acceptance criteria:
  - unauthorized actions are blocked with evidence
  - UI clearly shows recommendation vs authorized action
- Rollback: disable AI recommendation layer and keep deterministic rules only

### Phase 4 — Observability, Security, and Release Readiness
- Goals: hackathon-grade reliability, security, and narrative.
- Tasks:
  - observability/audit dashboards
  - secret-scanning and dependency review
  - end-to-end tests (simulation + sandbox)
  - docs + pitch/demo script completion
- Likely files: docs, tests, audit/telemetry surfaces, release checklist
- Dependencies: all previous phases
- Acceptance criteria:
  - rehearsed end-to-end demo
  - complete evidence bundle for judges
- Rollback: lock to last green milestone tag

## Critical Path (Dependency-Aware)
1. Baseline hardening
2. Provider abstraction
3. PayPal OAuth + Orders/Payments
4. Webhook normalization/idempotency
5. Approval-gated execution
6. Frontend demo + E2E validation
7. Final submission package

## Parallelizable Workstreams
- Frontend demo scaffolding can run in parallel with backend provider abstraction once API contracts stabilize.
- Documentation and release prep can run parallel to testing once core flows are functionally complete.
- Red-team/security review can run parallel to observability finishing steps.

## Prioritized Backlog
1. Provider interface contract and Razorpay adapter extraction
2. Canonical payment state model + recovered-state semantics
3. PayPal sandbox OAuth + Orders/Payments operations
4. Webhook verification + normalization + idempotency store
5. Policy/approval gating updates
6. Frontend provider-aware timeline and approval UX
7. E2E simulation/sandbox test packs
8. Final docs/demo/release checklist
