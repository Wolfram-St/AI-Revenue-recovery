# DEFINITION_OF_DONE.md

## Purpose
A change can be marked complete only when objective evidence exists and is logged in `log.md`.

## Completion Criteria by Area

### 1) Provider Integrations (Razorpay/PayPal)
- [ ] provider adapter follows common interface contract
- [ ] simulation and sandbox behavior explicitly documented
- [ ] webhook verification and normalized mapping covered by tests
- **Evidence required:** adapter tests + sample normalized payloads + mode labels

### 2) AI Features
- [ ] AI output is recommendation, not direct execution
- [ ] deterministic policy gate authorizes final action
- [ ] fallback behavior exists for model/unavailable states
- **Evidence required:** tests showing recommendation/policy separation

### 3) Agent Actions / Orchestration
- [ ] action lifecycle follows inspect->plan->implement->test->review->document->update log
- [ ] handoffs include files, tests, risks, next owner
- **Evidence required:** updated `log.md` entries and handoff artifacts

### 4) Payment State Transitions
- [ ] normalized states are used across providers
- [ ] true recovered-state semantics are explicit and tested
- **Evidence required:** transition table/tests and edge-case assertions

### 5) Webhooks
- [ ] signature verification implemented and tested
- [ ] idempotency/replay handling validated
- [ ] malformed payload behavior is deterministic
- **Evidence required:** webhook test suite outputs + replay test proof

### 6) Database Changes
- [ ] schema migration and rollback plan included
- [ ] audit data compatibility maintained
- **Evidence required:** migration scripts + migration test or dry-run output

### 7) Frontend Features
- [ ] provider/mode status is visible
- [ ] recommendation vs authorized action is clearly separated
- [ ] approval requirements are visible in UX
- **Evidence required:** screenshots/video + UI test checks (if present)

### 8) Tests
- [ ] targeted tests pass for all touched modules
- [ ] regressions in critical paths checked
- **Evidence required:** command list + pass/fail outputs

### 9) Security
- [ ] secret scan clean for changed files
- [ ] no real credentials/payment identifiers committed
- [ ] new dependencies reviewed for risk
- **Evidence required:** secret scan output + security review notes

### 10) Documentation and Demo Readiness
- [ ] docs reflect actual implementation status (no false completion claims)
- [ ] roadmap/log updated with pending work and risks
- [ ] demo flow reproducible in simulation/sandbox
- **Evidence required:** updated docs + runnable demo/test steps

## Rule for Marking Complete in `log.md`
Do **not** mark an item complete unless corresponding evidence above is present and referenced in the same `log.md` session entry.
