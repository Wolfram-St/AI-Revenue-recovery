# CHANGE_MANAGEMENT.md

## 1) Purpose
Define how changes are proposed, implemented, validated, rolled out, and recorded for this repository.

## 2) Change Lifecycle
1. Propose scope and risks
2. Define affected modules/files
3. Implement minimal reversible changes
4. Run targeted validation
5. Security/secrets review
6. Record decisions, evidence, rollback notes
7. Update `log.md`

## 3) Change-Type Rules

### API Changes
- Must maintain backward compatibility where possible.
- Document contract diffs and client impact.
- Add/update API tests for changed endpoints.

### Schema Changes
- Require migration and rollback plan.
- Avoid destructive changes without compatibility layer.
- Validate against `db/schema.sql` and audit data requirements.

### Provider Changes
- Must go through provider abstraction boundary.
- New provider behavior must not regress current Razorpay path.
- Require simulation/sandbox tests and mode-label evidence.

### ML Changes
- AI may recommend; policy authorizes.
- Keep deterministic fallbacks for model unavailability.
- Include evaluation evidence and drift/risk notes.

### Policy Changes
- Must preserve deterministic decision semantics.
- High-risk actions need explicit approval-gated controls.
- Add tests proving deny/stop behavior.

### Frontend Changes
- Must label simulation/sandbox behavior clearly.
- Do not imply real-money execution unless approved and explicit.
- Add/update UI tests where available.

### Dependency Changes
- Justify why dependency is needed.
- Evaluate security impact and pin versions appropriately.
- Run impacted tests and note runtime/deploy impacts.

### Documentation-Only Changes
- Verify accuracy against actual code paths.
- Never claim unimplemented features as complete.

## 4) Compatibility & Rollback Strategy
- Prefer additive changes and feature flags.
- Keep old behavior reachable until replacement is validated.
- Rollback package must include:
  - commit reference
  - toggles/flags to disable new path
  - schema rollback or forward-fix plan

## 5) Migration Guidance
- Migrate provider-specific fields to canonical normalized models incrementally.
- Use backfills/dual-write only with explicit validation checkpoints.
- Maintain audit continuity during migration.

## 6) Release Checklist
- [ ] scope reviewed and approved
- [ ] targeted tests pass
- [ ] security/secret scan passed
- [ ] docs updated (`ARCHITECTURE`, `ROADMAP`, `log.md`)
- [ ] rollback instructions documented
- [ ] unresolved risks captured explicitly
