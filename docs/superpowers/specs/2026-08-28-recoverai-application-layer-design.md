# RecoverAI Application Layer Design

## Purpose

Add a thin, publicly accessible HTTP and interactive UI layer over the existing Day 1-7 RecoverAI system for a Razorpay AI Buildathon demonstration. The application must expose the existing revenue-recovery intelligence without duplicating or bypassing domain logic.

The product demonstration should make the real-life workflow legible:

payment failure -> revenue at risk detection -> AI action analysis -> constrained portfolio optimization -> deterministic policy authorization -> explainable proposed action -> audit evidence.

This is an anonymous synthetic-data demo. It is not a production payment execution system.

## Scope

### In scope

- FastAPI HTTP backend.
- Typed request and response schemas.
- Thin application services/adapters.
- Dashboard read model.
- Recovery case listing and detail APIs.
- Bounded case analysis API.
- Portfolio optimization API using the existing Day 7 optimizer.
- Audit/explanation API surface where existing data supports it.
- React interactive frontend.
- Docker Compose integration.
- Structured HTTP error handling.
- Backend API tests and high-value frontend flow tests.
- Demo-safe configuration and CORS.

### Out of scope

- Real Razorpay API integration.
- Payment execution or retries against real accounts.
- Customer communication execution.
- Authentication, accounts, or multi-tenancy.
- Model retraining from the UI.
- Reimplementation of Day 1-7 scoring, policy, ML, evidence, or optimization logic.
- Production job queues, streaming, or real-time infrastructure.

## Architectural principle

The new layer is an adapter layer, not a second domain engine:

React UI
  -> HTTP/JSON
FastAPI routes
  -> application services/adapters
Existing recovery, ML, policy, optimizer modules
  -> PostgreSQL and existing artifacts

Routes perform validation, dependency acquisition, service invocation, and HTTP response translation. Services orchestrate existing public interfaces. Existing domain modules remain the source of truth for recommendations, policy decisions, candidate construction, optimization, evidence, and accounting.

The dependency direction must not reverse: domain modules must not import FastAPI, React concepts, HTTP schemas, or frontend concerns.

## Backend structure

Add a top-level application package:

app/
  main.py
  dependencies.py
  api/
    routes/
      health.py
      dashboard.py
      cases.py
      portfolio.py
      audit.py
    schemas/
      dashboard.py
      cases.py
      portfolio.py
  services/
    dashboard_service.py
    recovery_service.py
    portfolio_service.py

Names may be adjusted to match the repository's established conventions, but the responsibilities remain separate.

## API surface

### GET /health

Public health endpoint for Docker and deployment checks.

Response contains service status and a stable service identifier. It must not disclose secrets, database URLs, stack traces, or private environment values.

### GET /api/dashboard

Returns a product-level read model built from existing demo data and persisted or reproducible results. Intended fields include revenue at risk, estimated recoverable value where supported by existing semantics, case counts/statuses, action distribution, policy-blocked counts, and portfolio summary.

The endpoint aggregates existing data. It must not create a duplicate scoring engine.

### GET /api/cases

Returns a paginated recovery-case list. Initial filters are limited to fields actually consumed by the UI, such as status, failure category, recommendation/authorization state, policy override state, and amount threshold.

### GET /api/cases/{case_id}

Returns the case context and available decision explanation, including payment/failure context, recommendation information, expected value where produced by the existing core, optimizer recommendation when applicable, final policy authorization, rule/reason metadata, and audit information where available.

### POST /api/cases/{case_id}/analyze

A bounded analysis operation. It loads the decision-time context, invokes existing prediction/scoring/action logic, passes the resulting candidate through existing policy authorization, and returns a structured explanation.

It does not execute a payment, contact a customer, mutate model artifacts, or bypass policy.

The response must visibly separate AI recommendation from policy authorization.

### POST /api/portfolio/optimize

Accepts bounded optimization constraints, initially including monetary budget and human-review capacity.

The service flow is:

eligible decision-time rows
-> existing candidate universe construction
-> existing pre-allocation policy screen
-> existing model predictions/candidate values
-> existing exact 2D DP optimizer
-> existing post-allocation authorization
-> structured response

The HTTP layer must preserve Day 7 semantics:

- allocation and authorization are separate stages;
- optimizer_recommendation and authorized_action are not conflated;
- STOP rules remain dominant;
- budget and HR accounting remain based on frozen allocation, not retroactively freed by authorization overrides;
- outcomes are not joined before allocation is frozen;
- evaluation is not silently mixed into optimization.

The exact response schema must be mapped from actual Day 7 public interfaces rather than independently inventing alternate business fields.

### GET /api/audit

Expose available audit/read-only decision history through bounded query parameters. If the existing implementation does not yet persist sufficient audit data, the route should either expose the supported subset or be deferred rather than fabricating history.

## Frontend

Add a React frontend focused on four screens.

### Dashboard

The landing screen answers: how much revenue is at risk, and what is RecoverAI doing about it?

Show high-value summary metrics, recovery opportunity visualization, priority cases, and policy protection/override summaries.

### Recovery Cases

Show failed-payment opportunities in a browsable table with meaningful status and decision columns. Selecting a row opens its detail view.

### Decision Explorer

This is the primary Buildathon demonstration screen. It shows:

failed payment
-> AI recovery estimate
-> action comparison
-> best candidate
-> policy gate
-> authorized action

The policy gate must be visually prominent. The UI should demonstrate that AI recommendation is not equivalent to final authorization.

### Portfolio Optimizer

Provide interactive controls for budget and human-review capacity. On submission, call the portfolio API and display selected count, expected/net value using existing semantics, budget allocation, remaining budget, HR consumption, solver metadata, action distribution, selected/unallocated opportunities, policy overrides, and existing exact-DP versus greedy comparison evidence where available.

The frontend must display the actual solver result. It must not independently calculate an alternative optimization result.

## Public-demo safety

The demo uses synthetic data only and clearly identifies itself as synthetic/demo mode where appropriate.

The public layer must not:

- execute real payments;
- send customer communications;
- expose database credentials, environment secrets, or stack traces;
- accept arbitrary SQL or unrestricted command execution;
- bypass deterministic policy authorization;
- mutate model artifacts through public endpoints.

No authentication is included in this first subsystem. The public surface is therefore deliberately limited to safe, bounded demo operations.

## Error handling

Expected HTTP categories:

- 400 for malformed requests where the framework/service cannot interpret input;
- 404 for missing public resources such as a recovery case;
- 422 for validly structured requests that violate bounded input/business constraints;
- 503 for unavailable required demo dependencies;
- 500 for unexpected failures, without leaking stack traces.

Domain exceptions remain domain exceptions until translated at the API boundary. Safety failures must not be swallowed and converted into fabricated successful results.

Existing optimizer size/feasibility errors should map to a structured client-visible error when appropriate, preserving fail-closed behavior.

## Configuration and CORS

Use environment-based configuration. Initial settings may include:

- DATABASE_URL
- API_HOST
- API_PORT
- CORS_ORIGINS
- DEMO_MODE

The frontend receives only public configuration needed to call the API. Secrets never enter the frontend bundle.

CORS must be explicitly configured for local development and deployment rather than opened indiscriminately by default.

## Docker and deployment shape

Extend the existing container architecture without replacing the Day 1-7 verification path:

- db: PostgreSQL
- backend: FastAPI plus existing Python modules
- frontend: React application

For development, frontend and backend may run on separate ports with explicit CORS. Production hosting may use a reverse proxy or platform routing, but this is a deployment concern and must not change domain imports.

## Data and determinism

The demo operates against seeded synthetic data and existing deterministic/reproducible contracts where available.

Public interactions may change bounded request parameters such as portfolio budget and HR capacity, but must not mutate the canonical model artifacts or silently alter the underlying dataset.

If an operation depends on persisted demo state, its initialization and reset behavior must be explicit and reproducible.

## Testing strategy

Existing Day 1-7 tests remain the primary regression safety net and must continue passing.

Add backend tests for:

- health;
- dashboard response contract;
- case listing;
- missing case handling;
- case analysis integration;
- invalid portfolio input;
- successful portfolio optimization;
- domain exception to HTTP error translation;
- policy recommendation/authorization separation through the public layer.

Add frontend tests for the highest-value user journey:

Dashboard load
-> case list
-> case detail
-> recommendation visibility
-> policy decision visibility
-> portfolio constraint change
-> optimization request
-> allocation result rendering.

At least one integration path must exercise real existing core logic rather than mocking every domain dependency. Unit tests may mock service dependencies at the HTTP/UI boundary where useful.

## Acceptance criteria

The subsystem is complete only when:

1. The existing Day 1-7 core remains importable and its relevant tests continue to pass.
2. The backend starts and exposes documented bounded endpoints.
3. The frontend can drive the primary recovery-case and portfolio flows.
4. AI recommendation and policy authorization are visibly and structurally separate.
5. The portfolio endpoint uses the existing Day 7 optimization path and does not duplicate its solver.
6. STOP dominance and existing policy behavior cannot be bypassed through the HTTP layer.
7. The public demo performs no real payment or customer communication action.
8. Errors are structured and do not expose stack traces or secrets.
9. Docker integration supports the existing system plus the new application layer.
10. The primary Buildathon journey can be demonstrated end-to-end with synthetic data.

## Implementation order

1. Inspect actual current branch interfaces and compose/runtime files.
2. Freeze the application service contracts around real existing APIs.
3. Add backend configuration, dependencies, schemas, error translation, and health endpoint.
4. Implement dashboard and case read paths.
5. Implement bounded case analysis using existing recovery/ML/policy interfaces.
6. Implement portfolio orchestration using actual Day 7 public interfaces.
7. Add API integration tests.
8. Add React shell and API client.
9. Implement Dashboard, Cases, Decision Explorer, and Portfolio Optimizer screens.
10. Add high-value frontend tests.
11. Extend Docker Compose and verify full stack startup.
12. Run relevant existing regression suites and new application tests.

## Non-goals and future extensions

Real Razorpay integration, asynchronous workflow execution, authenticated users, multi-tenancy, live webhooks, customer communication delivery, and production-scale deployment are intentionally deferred. They can later consume the same application-service boundary without requiring the current UI to reach into domain internals.
