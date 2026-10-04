# AGENT_WORKFLOW.md

## 1) Purpose
Define the execution workflow for Gemini 3.1 Pro orchestrating specialized agency agents in this repository.

## 2) Orchestration Steps

1. **Task intake**
   - Parse objective, constraints, and acceptance criteria.
   - Label request type: docs-only / code-change / migration / incident.

2. **Repository grounding**
   - Inspect `app/`, `recovery/`, `ml/`, `frontend/`, `db/`, `tests/`, `simulation/`, `config/`, `docs/`.
   - Confirm baseline symbols: `app/main.py`, `recovery/recovery_agent.py`, `recovery/razorpay_client.py`, `recovery/policy.py`, `recovery/recovery_executor.py`, `recovery/audit_trail.py`, `ml/action_model.py`, `docker-compose.yml`.

3. **Decomposition**
   - Split into minimal independent units.
   - Mark dependencies and critical path.

4. **Parallel work**
   - Run independent roles in parallel only when file ownership does not overlap.
   - Sequence dependent edits (policy -> executor -> API wiring -> tests).

5. **Integration**
   - Merge outputs with smallest safe diff.
   - Resolve conflicts using policy/compliance + architect arbitration.

6. **Test gates**
   - Run targeted tests for touched modules.
   - Record exact command/output evidence.

7. **Review gates**
   - Security/secrets review
   - observability/audit review
   - red-team sanity checks for abuse paths

8. **Documentation updates**
   - Update relevant docs and `log.md`.

9. **Stop conditions**
   - Stop if requirements are ambiguous, risky, or require real-money execution without approval.
   - Stop if credentials/secrets are missing for required sandbox calls and no simulation fallback exists.

## 3) Handoff Format (Mandatory)
```markdown
### Handoff: <role>
- Scope completed:
- Files inspected:
- Files changed:
- Tests run (command + result):
- Security/Policy concerns:
- Remaining risks/blockers:
- Next expected consumer role:
```

## 4) Evidence Requirements
A work unit is not accepted without:
- reproducible commands
- pass/fail output summary
- changed file list
- explicit note on simulation vs sandbox vs production behavior

## 5) Example Task Prompts per Role

### Repository Analyst
"Map current Razorpay coupling and list exact seams for provider abstraction using `recovery/recovery_agent.py`, `recovery/razorpay_client.py`, `recovery/recovery_executor.py`, and related tests."

### PayPal Integration Specialist
"Implement PayPal sandbox client flow (OAuth token + Orders create/capture + webhook verification) behind provider interface without modifying existing Razorpay behavior."

### Payment-Domain Architect
"Define canonical provider-neutral payment model and status taxonomy; produce migration notes and compatibility constraints for current audit flows."

### AI/ML Specialist
"Constrain AI recommendations to produce ranked options only; ensure final action remains policy-authorized and auditable."

### Policy/Compliance Specialist
"Update deterministic rules for approval-gated actions and verify STOP precedence remains enforceable for high-risk contexts."

### Backend/API Engineer
"Add provider-neutral endpoints/services while preserving existing route compatibility from `app/main.py`."

### Frontend/Demo Engineer
"Add demo-visible provider badges, recovery timeline, and approval-state surfaces using existing frontend architecture."

### QA/Test Engineer
"Add/adjust targeted tests for provider adapters, webhook normalization, policy authorization, and audit evidence integrity."

### Security/Secrets Reviewer
"Scan changed files for secrets and verify no real credentials, tokens, or payment identifiers are committed."

### Observability/Audit Reviewer
"Ensure all action paths emit structured audit entries and traceable decision metadata."

### Documentation/Release Agent
"Update architecture, roadmap, change management, and definition-of-done docs based on merged implementation evidence."

### Red-Team Reviewer
"Simulate misuse attempts (replay webhooks, unauthorized action triggers, prompt-injected recommendations) and report mitigations."
