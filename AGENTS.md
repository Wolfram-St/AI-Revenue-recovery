# AGENTS.md

## 1) Mission
Build a production-credible, hackathon-ready **AI + PayPal revenue recovery platform** by evolving the current Razorpay-first baseline without breaking deterministic safety controls.

## 2) Current Baseline (must be preserved while migrating)
- API: `app/main.py` (FastAPI app factory + routes)
- Recovery orchestration: `recovery/recovery_agent.py`
- Current payment provider integration: `recovery/razorpay_client.py`
- Deterministic policy authorization: `recovery/policy.py`
- Action execution boundary: `recovery/recovery_executor.py`
- Audit trail/persistence mapping: `recovery/audit_trail.py`
- Action-aware ML: `ml/action_model.py`
- Local stack support: `docker-compose.yml`, `db/`

## 3) Target Direction
Introduce PayPal sandbox-first capabilities (Orders/Payments/Invoicing/Webhooks) through provider abstraction, normalized payment models, policy-bounded AI recommendations, and auditable execution flows.

## 4) Non-Negotiable Invariants
1. **Policy before action:** no autonomous action may bypass deterministic authorization rules.
2. **Auditability:** every consequential decision/action must produce durable evidence.
3. **Provider neutrality:** new provider logic must avoid hard-coding Razorpay assumptions in shared domains.
4. **No secret leakage:** no credentials, tokens, keys, or live payment identifiers in source or logs.
5. **Backward safety:** existing Razorpay behavior must remain testable while PayPal path is introduced.

## 5) Money-Movement Safety Rules
- Distinguish and label behavior as one of:
  - **Simulation:** synthetic/local, no real gateway calls.
  - **Sandbox:** PayPal/Razorpay test environments only.
  - **Production-like:** guarded code paths that could move real money if real creds exist.
- **Human approval is required** before any consequential money movement path is enabled/executed.
- Default to simulation/sandbox behavior in development and demos.
- Never commit real credentials, webhook secrets, or customer PII to repository files.

## 6) Repository Reconnaissance Requirements (before edits)
Agents must inspect relevant code and docs first:
- `app/`, `recovery/`, `ml/`, `frontend/`, `db/`, `tests/`, `simulation/`, `config/`, `docs/`
- At minimum review symbols listed in Section 2 when touching payment/recovery flows.
- Confirm existing tests and execute targeted tests for changed areas.

## 7) Standard Orchestration Lifecycle
Every task follows:
1. **Inspect** repository + current behavior
2. **Plan** minimal, reversible change set
3. **Implement** surgical edits
4. **Test** targeted checks first, then broader checks as needed
5. **Review** for correctness, safety, and regression risk
6. **Document** behavior/design updates
7. **Update `log.md`** with evidence, status, risks, and next action

## 8) Branch / Commit / PR Expectations
- Keep changes focused and traceable.
- Use small commits aligned to one meaningful unit each.
- PR descriptions must list:
  - scope
  - files touched
  - validation evidence
  - unresolved risks/blockers
- Do not include unrelated refactors in payment-sensitive tasks.

## 9) Testing Expectations
- Run targeted tests for modified domains (e.g., `tests/test_recovery_agent.py`, `tests/test_razorpay_webhook.py`, policy/audit tests).
- For doc-only tasks: perform lightweight validation (file existence, link/path sanity, markdown structure).
- Record exact commands + outcomes in `log.md`.

## 10) Environment and Secrets Rules
- Use `.env.example` as schema only; never commit filled secret values.
- Use sandbox credentials outside repository (environment variables or secret manager).
- Prohibit hard-coded keys in Python, frontend, Docker, CI config, or docs.

## 11) `log.md` Usage Contract
`/log.md` is the living orchestration ledger.
- Every agent session must update:
  - current phase/milestone
  - completion estimate
  - completed/in-progress/blocked work
  - validation evidence
  - next concrete action
- No task is “done” unless `log.md` reflects evidence and pending risks.

## 12) Gemini 3.1 Pro Guidance (Coordinator Role)
Gemini 3.1 Pro should operate as **coordinator/reviewer**, not as a policy bypass:
- Decompose work, assign specialized agents, and enforce deterministic guardrails.
- Require proof artifacts from workers (tests, diffs, risk notes).
- Reject changes that bypass policy authorization, audit logging, sandbox boundaries, or human approval requirements.
- Prioritize minimal safe integration over speculative rewrites.
