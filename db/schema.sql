-- RecoverAI Day 1 relational schema
-- AI recommends; deterministic policy rules authorize actions.

CREATE TABLE customers (
    customer_id VARCHAR(64) PRIMARY KEY,
    customer_tenure_days INTEGER NOT NULL CHECK (customer_tenure_days >= 0),
    successful_payment_count INTEGER NOT NULL DEFAULT 0 CHECK (successful_payment_count >= 0),
    failed_payment_count INTEGER NOT NULL DEFAULT 0 CHECK (failed_payment_count >= 0),
    historical_recovery_count INTEGER NOT NULL DEFAULT 0 CHECK (historical_recovery_count >= 0),
    customer_opted_out BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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

CREATE TABLE recovery_cases (
    recovery_case_id VARCHAR(64) PRIMARY KEY,
    payment_id VARCHAR(64) NOT NULL REFERENCES payments(payment_id),
    failed_attempt_id VARCHAR(64) NOT NULL REFERENCES payment_attempts(attempt_id),
    recovery_probability NUMERIC(6,5),
    expected_recovery_value_inr NUMERIC(14,2),
    recommended_action VARCHAR(32),
    recommended_at TIMESTAMPTZ,
    status VARCHAR(32) NOT NULL DEFAULT 'OPEN',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE recovery_actions (
    recovery_action_id VARCHAR(64) PRIMARY KEY,
    recovery_case_id VARCHAR(64) NOT NULL REFERENCES recovery_cases(recovery_case_id),
    action_type VARCHAR(32) NOT NULL,
    authorized BOOLEAN NOT NULL DEFAULT FALSE,
    authorization_reason TEXT,
    scheduled_for TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE recovery_outcomes (
    outcome_id VARCHAR(64) PRIMARY KEY,
    recovery_action_id VARCHAR(64) NOT NULL REFERENCES recovery_actions(recovery_action_id),
    outcome VARCHAR(32) NOT NULL,
    recovered_amount_inr NUMERIC(14,2) NOT NULL DEFAULT 0 CHECK (recovered_amount_inr >= 0),
    observed_at TIMESTAMPTZ NOT NULL,
    notes TEXT
);

CREATE TABLE audit_logs (
    audit_id BIGSERIAL PRIMARY KEY,
    recovery_case_id VARCHAR(64) REFERENCES recovery_cases(recovery_case_id),
    event_type VARCHAR(64) NOT NULL,
    actor_type VARCHAR(32) NOT NULL,
    action VARCHAR(64),
    decision_reason TEXT,
    event_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_payments_customer ON payments(customer_id);
CREATE INDEX idx_attempts_payment ON payment_attempts(payment_id);
CREATE INDEX idx_attempts_failure_category ON payment_attempts(failure_category);
CREATE INDEX idx_recovery_cases_status ON recovery_cases(status);
CREATE INDEX idx_recovery_actions_scheduled ON recovery_actions(scheduled_for);
CREATE INDEX idx_audit_logs_case ON audit_logs(recovery_case_id);
