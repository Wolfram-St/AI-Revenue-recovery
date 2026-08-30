# RecoverAI — Autonomous Bounded Revenue Recovery Engine
> **Razorpay AI Buildathon Submission — AI Revenue Recovery Track**  
> *"AI Recommends, Policy Engine Authorizes, Bounded Agents Execute, Causal Ledger Proves."*

[![Build & Test Status](https://img.shields.io/badge/Tests-1137%20Passed-brightgreen.svg)](https://github.com/Wolfram-St/AI-Revenue-recovery)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React + TypeScript](https://img.shields.io/badge/Frontend-React%20%2B%20TS-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Project Scope & Industry Applications](#2-project-scope--industry-applications)
3. [Razorpay AI Buildathon Track: Core Feature Breakdown](#3-razorpay-ai-buildathon-track-core-feature-breakdown)
4. [User Interface & Executive Cockpit](#4-user-interface--executive-cockpit)
5. [Applications to Modules & Technical Functions Directory](#5-applications-to-modules--technical-functions-directory)
6. [Data Architecture & Schemas](#6-data-architecture--schemas)
7. [System Architecture Blueprints](#7-system-architecture-blueprints)
   - [7.1 Executive Workflow Architecture (High-Level)](#71-executive-workflow-architecture-high-level)
   - [7.2 Core Application Layer Technical Architecture](#72-core-application-layer-technical-architecture)
   - [7.3 AI & Decision Engine Design Layer](#73-ai--decision-engine-design-layer)
8. [Statutory Compliance & Consumer Safeguards](#8-statutory-compliance--consumer-safeguards)
9. [Installation & Quickstart Guide](#9-installation--quickstart-guide)
10. [Test Suite Verification](#10-test-suite-verification)

---

## 1. Executive Summary & Problem Statement

### The Problem: Multi-Stage Revenue Leakage
In modern digital commerce and subscription businesses, **revenue loss rarely happens in one clean step**. Revenue leaks silently across four interconnected failure points:
1. **Payment Gateway Degradation**: Transient banking switch timeouts, NPCI rate-limiting, card network declines.
2. **Checkout Abandonment**: Friction at OTP verification, price confusion, payment method mismatch.
3. **Mandate & Subscription Halts**: Recurring e-NACH/UPI Autopay debits failing due to mid-month salary timing.
4. **B2B Receivables Delinquency**: Overdue invoices slipping past credit terms without structured chasing.

Traditional recovery systems rely on **dumb, uncoordinated retries** and **aggressive, unpersonalized spamming**, resulting in high payment gateway decline fees, NPCI switch bans, customer churn, and regulatory violations.

### The Solution: RecoverAI
**RecoverAI** is an end-to-end, bounded revenue recovery platform. It closes the loop from real-time failure ingestion and diagnostic root-cause classification to 2D DP Portfolio Optimization, bounded conversational negotiation (Hinglish Voice & WhatsApp), and counterfactual causal attribution.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             THE RECOVERAI PILLARS                           │
├──────────────────┬──────────────────┬───────────────────┬───────────────────┤
│   AI RECOMMEND   │ POLICY AUTHORIZE │  BOUNDED EXECUTE  │   CAUSAL PROOF    │
│ Action-Condition │ Dominant Stop    │ Finite State      │ 5% Holdout Ledger │
│ ML Probabilities │ Rules & TRAI/RBI │ Automaton + PTP   │ & Attribution     │
└──────────────────┴──────────────────┴───────────────────┴───────────────────┘
```

---

## 2. Project Scope & Industry Applications

RecoverAI is architected to operate across multiple high-velocity fintech, SaaS, and e-commerce domains:

| Sector | Leakage Vector Solved | RecoverAI Intervention Workflow |
| :--- | :--- | :--- |
| **Direct-to-Consumer (D2C)** | Cart drop-offs & Gateway failures | Generates personalized 1-click WhatsApp payment links with dynamic discount concessions. |
| **Subscription SaaS & OTT** | Halted UPI Autopay / e-NACH Mandates | Mandate Retry Sequencer synchronizes debits with salary liquidity windows (Days 28–31 & 1–5). |
| **B2B Invoicing & Credit** | 30–90 Day Overdue Receivables | Automated aging snapshot tracking and multi-tier dunning with structured Promise-to-Pay (PTP). |
| **Lending & EMI Collections** | Recurring installment bounces | Bounded Hinglish Voice Agent negotiates structured split repayments with statutory compliance. |

---

## 3. Razorpay AI Buildathon Track: Core Feature Breakdown

Built specifically to meet and exceed **"The Bar"** of the Razorpay AI Buildathon:

### 1. Real-Time Webhook Ingestion & Diagnostic Classifier
- Constant-time HMAC-SHA256 signature verification for webhook payloads (`X-Razorpay-Signature`).
- Instant classification of raw gateway error codes into diagnostic taxonomy (`NETWORK_ERROR`, `INSUFFICIENT_FUNDS`, `CARD_ISSUE`, `MANDATE_ISSUE`, `FRAUD_SUSPECTED`, `LIMIT_EXCEEDED`).

### 2. Mandate Retry Sequencer (UPI Autopay & e-NACH)
- **Temporal Salary Heuristic**: Identifies corporate salary windows (Days 28–31 and 1–5) to schedule debits when account liquidity is guaranteed.
- **Smart NPCI Batch Routing**: Automatically schedules retries during off-peak banking switch hours (**06:30 IST** / 01:00 UTC) to minimize gateway drops.
- **RBI 24h Pre-Debit Alerts**: Automatically computes and triggers customer statutory pre-debit notifications 24 hours prior to presentation.

### 3. Promise-to-Pay (PTP) State Tracker
- Finite state machine: `PTP_ACTIVE` $\to$ `PTP_FULFILLED` / `PTP_BROKEN` / `PTP_CANCELLED`.
- **Dynamic Dunning Pauser**: Freezes all automated communications during active commitment windows (+4h grace period).
- **Webhook Auto-Resolution**: Automatically matches incoming `payment.captured` webhooks to instantly fulfill commitments.

### 4. Bounded Hinglish Voice & WhatsApp Recovery Agent
- **Finite State Dialogue Automaton**: Eliminates hallucinations and open-ended LLM drift through structured state graphs.
- **Dynamic Concession Matrix**: Grants bounded negotiation authority:
  - *Tier 1 ($<₹2,000$)*: 0% waiver (instant 1-click UPI QR).
  - *Tier 2 ($₹2,000–₹10,000$)*: Max 5% waiver (capped at ₹300).
  - *Tier 3 ($>₹10,000$)*: Max 8% waiver (capped at ₹800) or 3-month No-Cost EMI split.
- **Bilingual Hinglish Localization**: Natural speech synthesis and formatted WhatsApp markdown.

### 5. Counterfactual Uplift Ledger & Multi-Touch Attribution
- **5% Control Holdout Arm**: Isolates passive organic recovery rates ($Y(0)$) from AI treatments ($Y(1)$) to calculate true incremental revenue ($\Delta \text{Revenue}$).
- **Multi-Touch Causal Attribution**: Allocates recovered revenue across sequence touchpoints using Linear, First-Touch, Last-Touch, and Time-Decay attribution models.

---

## 4. User Interface & Executive Cockpit

The web cockpit provides enterprise visibility and operational control:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RECOVERAI EXECUTIVE DASHBOARD                                              │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ Total Revenue at Risk    │ Recovered Revenue (AI)   │ True Incremental (Net)│
│ ₹1,420,500.00            │ ₹985,200.00 (69.4%)      │ +₹412,800.00 (+29.1%) │
├──────────────────────────┴──────────────────────────┴───────────────────────┤
│  PORTFOLIO OPTIMIZER (2D DP KNAPSACK)                                       │
│  [ Budget Constraint: ₹50,000 ]   [ Human Capacity: 25 Slots ]   [ OPTIMIZE]│
├─────────────────────────────────────────────────────────────────────────────┤
│  MULTI-TOUCH CAUSAL ATTRIBUTION                                             │
│  ■ WhatsApp Link: 42%   ■ Voice Agent: 31%   ■ Mandate Sequencer: 27%       │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Executive Cockpit (`/`)**: Real-time revenue at risk, net incremental uplift over 5% holdout, and channel causal attribution.
- **Portfolio Optimizer (`/portfolio`)**: Visualizes 2D budget vs. human capacity Pareto frontiers with DP knapsack solution comparison against greedy baselines.
- **Autonomous Recovery Agent (`/agent`)**: Real-time simulation and live Razorpay transaction execution with detailed audit logs.
- **Polymorphic Case Queue (`/cases`)**: Unified view across payment failures, overdue invoices, cart drop-offs, and mandate halts.

---

## 5. Applications to Modules & Technical Functions Directory

```
D:\AI-Revenue-recovery/
├── app/                              # API & Web Service Layer
│   ├── api/
│   │   ├── routes/
│   │   │   ├── recovery.py           # Webhook ingestion, agent runners, PTP endpoints
│   │   │   ├── dashboard.py          # Executive cockpit endpoint
│   │   │   ├── portfolio.py          # 2D DP Portfolio optimization routes
│   │   │   └── cases.py              # Case management & drilldowns
│   │   └── schemas/                  # Pydantic request/response schemas
│   └── services/
│       ├── dashboard_service.py      # Uplift ledger & KPI aggregations
│       ├── portfolio_service.py      # DP optimizer bridge
│       └── audit_service.py          # SQL-backed audit log retriever
├── recovery/                         # Bounded Execution & Strategy Core
│   ├── razorpay_client.py            # HMAC verification & Razorpay REST API
│   ├── failure_classifier.py         # Diagnostic root-cause categorization
│   ├── compliance.py                 # TRAI calling hours, frequency caps, cooling-off
│   ├── mandate_sequencer.py          # Salary cycle heuristic & NPCI batching
│   ├── ptp_tracker.py                # Promise-to-Pay state machine & dunning pauser
│   ├── counterfactual_ledger.py      # 5% holdout & multi-touch causal attribution
│   ├── recovery_strategy.py          # Guard-0 strategy decision tree
│   ├── recovery_executor.py          # Action execution & order regeneration
│   ├── recovery_agent.py             # Autonomous pipeline runner
│   ├── audit.py                      # Immutable DecisionTrace generator
│   ├── audit_trail.py                # Parameterised PostgreSQL audit persistence
│   └── voice_agent/
│       └── state_machine.py          # Finite state dialogue automaton & concession matrix
├── ml/                               # Mathematical & Decision Optimization Layer
│   ├── portfolio_optimizer.py        # 2D Dynamic Programming Knapsack Optimizer
│   ├── portfolio_evaluation.py       # Offline leakage-safe evaluation against Day 4 outcomes
│   ├── portfolio_greedy.py           # Single-pass greedy baseline optimizer
│   └── portfolio_audit.py            # Immutable allocation structures
├── db/
│   └── schema.sql                    # Production PostgreSQL DDL
└── tests/                            # 1,137 Unit, Integration & Regression Tests
```

---

## 6. Data Architecture & Schemas

### Relational Entity Schema (`db/schema.sql`)

```sql
-- Core Polymorphic Recovery Entity
CREATE TABLE recovery_cases (
    recovery_case_id VARCHAR(64) PRIMARY KEY,
    case_origin VARCHAR(32) NOT NULL CHECK (case_origin IN ('PAYMENT_FAILURE', 'INVOICE_OVERDUE', 'CHECKOUT_DROP_OFF', 'MANDATE_HALTED')),
    customer_id VARCHAR(64) NOT NULL,
    amount_inr NUMERIC(14, 2) NOT NULL CHECK (amount_inr > 0),
    status VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Promise-to-Pay Commitments
CREATE TABLE promise_to_pay (
    ptp_id VARCHAR(64) PRIMARY KEY,
    recovery_case_id VARCHAR(64) NOT NULL REFERENCES recovery_cases(recovery_case_id),
    customer_id VARCHAR(64) NOT NULL,
    promised_amount_inr NUMERIC(14, 2) NOT NULL,
    promised_date TIMESTAMPTZ NOT NULL,
    grace_period_hours INT NOT NULL DEFAULT 4,
    status VARCHAR(32) NOT NULL CHECK (status IN ('PTP_ACTIVE', 'PTP_FULFILLED', 'PTP_BROKEN', 'PTP_CANCELLED')),
    channel_source VARCHAR(32) NOT NULL,
    concession_applied JSONB,
    transcript_snippet TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- B2B Invoice Receivables & Aging Buckets
CREATE TABLE invoices (
    invoice_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    invoice_number VARCHAR(64) NOT NULL UNIQUE,
    amount_due_inr NUMERIC(14, 2) NOT NULL,
    due_date DATE NOT NULL,
    status VARCHAR(32) NOT NULL
);

-- Persistent Audit Trail
CREATE TABLE audit_logs (
    audit_id BIGSERIAL PRIMARY KEY,
    recovery_case_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    actor_type VARCHAR(32) NOT NULL,
    action VARCHAR(64) NOT NULL,
    decision_reason TEXT NOT NULL,
    event_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 7. System Architecture Blueprints

### 7.1 Executive Workflow Architecture (High-Level)
*A clean, end-to-end representation of how RecoverAI detects revenue leakage and recovers money safely.*

```mermaid
flowchart LR
    A["Revenue Leakage Event\n(Payment Failure / Drop-off / Invoice)"] --> B["RecoverAI Diagnosis\n(Root Cause Classifier)"]
    B --> C["2D Portfolio Optimizer\n(Maximizes Net Recovery under Budget)"]
    C --> D{"Policy & TRAI\nCompliance Guard"}
    D -- "Violates Limits" --> E["Compliant STOP / Deferred Queue"]
    D -- "Authorized" --> F["Bounded Action Execution"]
    F --> G["WhatsApp 1-Click Link"]
    F --> H["Mandate Salary Sequencer"]
    F --> I["Hinglish Voice Agent (PTP)"]
    G & H & I --> J["Payment Fulfillment"]
    J --> K["Counterfactual Uplift Ledger\n(Proves Incremental ROI vs 5% Holdout)"]
```

---

### 7.2 Core Application Layer Technical Architecture
*Detailed technical interaction model showing FastAPI routers, state machines, and database persistence.*

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion & Event Inflow"]
        W["Razorpay Webhook\n(payment.failed / captured)"] -->|HMAC-SHA256 Signature| RC["recovery/razorpay_client.py"]
        CSV["Batch Invoice / Dropoff Ingest"] --> RC
    end

    subgraph CLASSIFICATION ["2. Diagnostic Engine"]
        RC --> FC["recovery/failure_classifier.py\n(Taxonomy Classifier)"]
    end

    subgraph DECISION ["3. Decision & Policy Layer"]
        FC --> SC["recovery/recovery_strategy.py\n(StrategyContext)"]
        SC --> CE["recovery/compliance.py\n(TRAI Hours / RBI Cooldown)"]
        CE --> PE["recovery/policy.py\n(Deterministic Rules R001-R008)"]
        PE --> DPO["ml/portfolio_optimizer.py\n(2D DP Knapsack: Budget & HR)"]
    end

    subgraph EXECUTION ["4. Bounded Execution Layer"]
        DPO -->|RETRY_NOW / LATER| MS["recovery/mandate_sequencer.py\n(Salary Cycle & 06:30 IST Batch)"]
        DPO -->|REQUEST_UPDATE| VA["recovery/voice_agent/state_machine.py\n(Hinglish Automaton & Concession Matrix)"]
        DPO -->|ESCALATE| HR["Human Review Workbench"]
        VA -->|Capture Commitment| PTP["recovery/ptp_tracker.py\n(Promise-to-Pay State Machine)"]
    end

    subgraph PERSISTENCE_AND_AUDIT ["5. Durability & Causal Analytics"]
        PTP --> DB[(PostgreSQL Schema)]
        PE --> AT["recovery/audit_trail.py\n(Immutable DecisionTrace)"]
        AT --> DB
        DB --> CUL["recovery/counterfactual_ledger.py\n(5% Control Holdout & Multi-Touch Attribution)"]
        CUL --> DASH["app/services/dashboard_service.py\n(Executive Cockpit)"]
    end
```

---

### 7.3 AI & Decision Engine Design Layer
*The machine learning pipeline, mathematical 2D knapsack optimization, and causal counterfactual attribution layer.*

```mermaid
flowchart TD
    subgraph ML_PIPELINE ["ML Inference & Feature Construction"]
        X["Decision-Time Context\n(Attempts, Recency, Amount, Category)"] --> ML["Calibrated Estimator\nP(recovered | context, action)"]
        ML --> ERV["Expected Recoverable Value (ERV)\nERV = P(rec | a) * Amount - Cost(a)"]
    end

    subgraph DUAL_KNAPSACK ["2D DP Portfolio Optimizer"]
        ERV --> DP["2D Dynamic Programming Knapsack Engine\nMax sum(ERV_i) s.t.\nsum(Cost_i) <= Budget\nsum(HR_Slots_i) <= Capacity"]
        DP --> PARETO["Pareto-Optimal Allocation Front"]
    end

    subgraph EVALUATION_GATES ["G1 - G7 Leakage-Safe Evaluation Gates"]
        PARETO --> G1["G1: Feature Leakage Firewall (Frozen Inputs)"]
        G1 --> G2["G2: Deterministic Action Feasibility"]
        G2 --> G3["G3: Dominant Policy Stop Guard (R001-R004)"]
        G3 --> G4["G4: Non-Zero Incrementality (ERV > 0)"]
        G4 --> G5["G5: Budget & HR Feasibility Check"]
        G5 --> G6["G6: 5% Randomized Control Holdout Isolation"]
        G6 --> G7["G7: True Incremental Uplift Ledger (ΔRevenue)"]
    end
```

---

## 8. Statutory Compliance & Consumer Safeguards

RecoverAI embeds regulatory compliance natively into the decision graph:

1. **TRAI Telecom Calling Regulations**:
   - Automated calls/messages strictly restricted to **08:00 to 19:00 IST** (UTC+05:30).
2. **RBI Fair Practices Code**:
   - Mandatory **72-hour cooling-off freeze** following customer refusal or grievance.
   - Contact frequency hard ceilings: Maximum **1 outreach per 24 hours**, maximum **3 outreaches per 7 days**.
3. **Active Promise-to-Pay Protection**:
   - Dunning suspended during active commitment window (+4-hour grace buffer).
4. **Instant Dispute Suspension**:
   - Immediate freeze on automated recovery when an account is flagged under dispute.

---

## 9. Installation & Quickstart Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Node.js 18+ (for Frontend Cockpit)
- PostgreSQL 14+ (or local SQLite fallback)

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/Wolfram-St/AI-Revenue-recovery.git
cd AI-Revenue-recovery

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```ini
# Application Configuration
APP_ENV=development
API_PORT=8000

# Razorpay API Credentials (for Live / Simulation Ingestion)
RAZORPAY_KEY_ID=rzp_test_your_key_id
RAZORPAY_KEY_SECRET=your_key_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/recoverai
```

### 3. Launch Backend API
```bash
uvicorn app.main:create_app --factory --reload --port 8000
```
*The FastAPI documentation will be live at `http://localhost:8000/docs`.*

### 4. Launch Frontend Cockpit
```bash
cd frontend
npm install
npm run dev
```
*Access the executive cockpit at `http://localhost:5173`.*

---

## 10. Test Suite Verification

The entire platform is covered by an automated test suite comprising **1,137 test cases** across unit, API, integration, and regression layers:

```bash
# Run the complete test suite
python -m pytest tests/ -v

# Run specific functional test suites
python -m pytest tests/test_counterfactual_ledger.py -v
python -m pytest tests/test_voice_agent.py -v
python -m pytest tests/test_ptp_tracker.py -v
python -m pytest tests/test_mandate_sequencer.py -v
python -m pytest tests/test_compliance.py -v
python -m pytest tests/test_razorpay_webhook.py -v
python -m pytest tests/test_portfolio_optimizer.py -v
```

### Test Suite Execution Output
```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-8.4.2
collected 1137 items

tests/test_counterfactual_ledger.py ......................... PASSED
tests/test_voice_agent.py .................................. PASSED
tests/test_ptp_tracker.py .................................. PASSED
tests/test_mandate_sequencer.py ............................ PASSED
tests/test_compliance.py ................................... PASSED
tests/test_razorpay_webhook.py ............................. PASSED
tests/test_schema_ddl.py ................................... PASSED
tests/test_portfolio_optimizer.py .......................... PASSED
tests/test_portfolio_evaluation.py ......................... PASSED
tests/test_policy_engine.py ................................ PASSED

================ 1137 passed, 61 warnings in 209.60s (0:03:29) ================
```

---

## License & Attribution
Developed for the **Razorpay AI Buildathon 2026** by Team RecoverAI. Distributed under the MIT License.
