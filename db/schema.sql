-- RecoverAI Production Relational Schema
-- AI recommends; deterministic policy rules authorize actions.

-- ============================================================================
-- 1. Core Customers & Accounts
-- ============================================================================

CREATE TABLE customers (
    customer_id VARCHAR(64) PRIMARY KEY,
    customer_tenure_days INTEGER NOT NULL CHECK (customer_tenure_days >= 0),
    successful_payment_count INTEGER NOT NULL DEFAULT 0 CHECK (successful_payment_count >= 0),
    failed_payment_count INTEGER NOT NULL DEFAULT 0 CHECK (failed_payment_count >= 0),
    historical_recovery_count INTEGER NOT NULL DEFAULT 0 CHECK (historical_recovery_count >= 0),
    customer_opted_out BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 2. Transactional Payments & Attempts
-- ============================================================================

CREATE TABLE payments (
    payment_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id),
    amount_inr NUMERIC(14,2) NOT NULL CHECK (amount_inr > 0),
    currency VARCHAR(3) NOT NULL DEFAULT 'INR',
    payment_method VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE payment_attempts (
    attempt_id VARCHAR(64) PRIMARY KEY,
    payment_id VARCHAR(64) NOT NULL REFERENCES payments(payment_id),
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    attempted_at TIMESTAMPTZ NOT NULL,
    failure_code VARCHAR(64),
    failure_category VARCHAR(32) NOT NULL,
    issuer_response VARCHAR(64),
    device_type VARCHAR(32),
    country VARCHAR(2),
    is_hard_decline BOOLEAN NOT NULL DEFAULT FALSE,
    fraud_risk BOOLEAN NOT NULL DEFAULT FALSE,
    recovered BOOLEAN,
    recovery_time_hours NUMERIC(10,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 3. B2B Invoices & Invoice Aging
-- ============================================================================

CREATE TABLE invoices (
    invoice_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id),
    invoice_number VARCHAR(64) NOT NULL UNIQUE,
    amount_inr NUMERIC(14,2) NOT NULL CHECK (amount_inr > 0),
    currency VARCHAR(3) NOT NULL DEFAULT 'INR',
    status VARCHAR(32) NOT NULL DEFAULT 'ISSUED' 
        CHECK (status IN ('DRAFT', 'ISSUED', 'OVERDUE', 'PAID', 'PARTIALLY_PAID', 'CANCELLED', 'DISPUTED')),
    issue_date DATE NOT NULL,
    due_date DATE NOT NULL,
    paid_amount_inr NUMERIC(14,2) NOT NULL DEFAULT 0.00 CHECK (paid_amount_inr >= 0),
    paid_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE invoice_aging_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    invoice_id VARCHAR(64) NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    days_overdue INTEGER NOT NULL CHECK (days_overdue >= 0),
    aging_bucket VARCHAR(32) NOT NULL 
        CHECK (aging_bucket IN ('CURRENT', '1_30_DAYS', '31_60_DAYS', '61_90_DAYS', '90_PLUS_DAYS')),
    outstanding_amount_inr NUMERIC(14,2) NOT NULL CHECK (outstanding_amount_inr >= 0),
    dunning_stage VARCHAR(32) NOT NULL DEFAULT 'REMINDER_1',
    dispute_status VARCHAR(32) NOT NULL DEFAULT 'NONE' 
        CHECK (dispute_status IN ('NONE', 'UNDER_REVIEW', 'CONFIRMED', 'RESOLVED')),
    last_reminder_sent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 4. Checkout Sessions & Cart Abandonment
-- ============================================================================

CREATE TABLE checkout_sessions (
    checkout_session_id VARCHAR(64) PRIMARY KEY,
    customer_id VARCHAR(64) REFERENCES customers(customer_id),
    cart_token VARCHAR(128) NOT NULL UNIQUE,
    cart_value_inr NUMERIC(14,2) NOT NULL CHECK (cart_value_inr > 0),
    currency VARCHAR(3) NOT NULL DEFAULT 'INR',
    abandonment_stage VARCHAR(32) NOT NULL 
        CHECK (abandonment_stage IN ('CART_VIEWED', 'ADDRESS_ENTERED', 'PAYMENT_METHOD_SELECTED', 'OTP_SUBMIT_FAILED', 'GATEWAY_ERROR')),
    recovery_link_token VARCHAR(128) UNIQUE,
    recovery_link_url TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'ABANDONED' 
        CHECK (status IN ('ACTIVE', 'ABANDONED', 'ENGAGED', 'RECOVERED', 'EXPIRED')),
    expires_at TIMESTAMPTZ NOT NULL,
    recovered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 5. Recovery Cases & Polymorphic Origins
-- ============================================================================

CREATE TABLE recovery_cases (
    recovery_case_id VARCHAR(64) PRIMARY KEY,
    case_origin VARCHAR(32) NOT NULL DEFAULT 'PAYMENT_FAILURE'
        CHECK (case_origin IN ('PAYMENT_FAILURE', 'INVOICE_OVERDUE', 'CHECKOUT_DROP_OFF', 'MANDATE_HALTED')),
    payment_id VARCHAR(64) REFERENCES payments(payment_id),
    failed_attempt_id VARCHAR(64) REFERENCES payment_attempts(attempt_id),
    invoice_id VARCHAR(64) REFERENCES invoices(invoice_id),
    checkout_session_id VARCHAR(64) REFERENCES checkout_sessions(checkout_session_id),
    recovery_probability NUMERIC(6,5),
    expected_recovery_value_inr NUMERIC(14,2),
    recommended_action VARCHAR(32),
    recommended_at TIMESTAMPTZ,
    status VARCHAR(32) NOT NULL DEFAULT 'OPEN',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_case_origin_reference CHECK (
        payment_id IS NOT NULL OR invoice_id IS NOT NULL OR checkout_session_id IS NOT NULL
    )
);

-- ============================================================================
-- 6. Promise-to-Pay (PTP) State Tracker
-- ============================================================================

CREATE TABLE promise_to_pay (
    ptp_id VARCHAR(64) PRIMARY KEY,
    recovery_case_id VARCHAR(64) NOT NULL REFERENCES recovery_cases(recovery_case_id) ON DELETE CASCADE,
    customer_id VARCHAR(64) NOT NULL REFERENCES customers(customer_id),
    promised_amount_inr NUMERIC(14,2) NOT NULL CHECK (promised_amount_inr > 0),
    promised_date TIMESTAMPTZ NOT NULL,
    grace_period_hours INTEGER NOT NULL DEFAULT 4 CHECK (grace_period_hours >= 0),
    status VARCHAR(32) NOT NULL DEFAULT 'PTP_ACTIVE' 
        CHECK (status IN ('PTP_ACTIVE', 'PTP_FULFILLED', 'PTP_BROKEN', 'PTP_CANCELLED')),
    channel_source VARCHAR(32) NOT NULL 
        CHECK (channel_source IN ('VOICE_AGENT', 'WHATSAPP', 'HUMAN_AGENT', 'SELF_SERVE', 'EMAIL')),
    concession_applied JSONB NOT NULL DEFAULT '{}'::jsonb,
    transcript_snippet TEXT,
    fulfilled_at TIMESTAMPTZ,
    broken_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 7. Recovery Actions, Outcomes & Audit Trails
-- ============================================================================

CREATE TABLE recovery_actions (
    recovery_action_id VARCHAR(64) PRIMARY KEY,
    recovery_case_id VARCHAR(64) NOT NULL REFERENCES recovery_cases(recovery_case_id) ON DELETE CASCADE,
    action_type VARCHAR(32) NOT NULL,
    authorized BOOLEAN NOT NULL DEFAULT FALSE,
    authorization_reason TEXT,
    scheduled_for TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE recovery_outcomes (
    outcome_id VARCHAR(64) PRIMARY KEY,
    recovery_action_id VARCHAR(64) NOT NULL REFERENCES recovery_actions(recovery_action_id) ON DELETE CASCADE,
    outcome VARCHAR(32) NOT NULL,
    recovered_amount_inr NUMERIC(14,2) NOT NULL DEFAULT 0 CHECK (recovered_amount_inr >= 0),
    observed_at TIMESTAMPTZ NOT NULL,
    notes TEXT
);

CREATE TABLE audit_logs (
    audit_id BIGSERIAL PRIMARY KEY,
    recovery_case_id VARCHAR(64) REFERENCES recovery_cases(recovery_case_id) ON DELETE SET NULL,
    event_type VARCHAR(64) NOT NULL,
    actor_type VARCHAR(32) NOT NULL,
    action VARCHAR(64),
    decision_reason TEXT,
    event_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 8. Performance Indexes
-- ============================================================================

CREATE INDEX idx_payments_customer ON payments(customer_id);
CREATE INDEX idx_attempts_payment ON payment_attempts(payment_id);
CREATE INDEX idx_attempts_failure_category ON payment_attempts(failure_category);
CREATE INDEX idx_invoices_customer ON invoices(customer_id);
CREATE INDEX idx_invoices_status_due ON invoices(status, due_date);
CREATE INDEX idx_invoice_aging_invoice ON invoice_aging_snapshots(invoice_id);
CREATE INDEX idx_invoice_aging_bucket ON invoice_aging_snapshots(aging_bucket);
CREATE INDEX idx_checkout_sessions_status ON checkout_sessions(status);
CREATE INDEX idx_checkout_sessions_token ON checkout_sessions(recovery_link_token);
CREATE INDEX idx_recovery_cases_status ON recovery_cases(status);
CREATE INDEX idx_recovery_cases_origin ON recovery_cases(case_origin);
CREATE INDEX idx_recovery_cases_invoice ON recovery_cases(invoice_id);
CREATE INDEX idx_recovery_cases_checkout ON recovery_cases(checkout_session_id);
CREATE INDEX idx_ptp_case ON promise_to_pay(recovery_case_id);
CREATE INDEX idx_ptp_customer ON promise_to_pay(customer_id);
CREATE INDEX idx_ptp_status_date ON promise_to_pay(status, promised_date);
CREATE INDEX idx_recovery_actions_scheduled ON recovery_actions(scheduled_for);
CREATE INDEX idx_audit_logs_case ON audit_logs(recovery_case_id);
