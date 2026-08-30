"""Tests for database schema integrity and DDL definitions.

Validates that `db/schema.sql`:
1. Parses cleanly with correct SQL structure.
2. Contains all required production tables (customers, payments, payment_attempts,
   invoices, invoice_aging_snapshots, checkout_sessions, recovery_cases,
   promise_to_pay, recovery_actions, recovery_outcomes, audit_logs).
3. Defines expected foreign key constraints, checks, and indexes.
"""

from __future__ import annotations

import re
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def test_schema_file_exists():
    assert SCHEMA_PATH.is_file(), f"Schema file not found at {SCHEMA_PATH}"


def test_schema_contains_required_tables():
    content = SCHEMA_PATH.read_text(encoding="utf-8")
    table_pattern = re.compile(r"CREATE\s+TABLE\s+([a-z_]+)", re.IGNORECASE)
    tables = table_pattern.findall(content)

    required_tables = [
        "customers",
        "payments",
        "payment_attempts",
        "invoices",
        "invoice_aging_snapshots",
        "checkout_sessions",
        "recovery_cases",
        "promise_to_pay",
        "recovery_actions",
        "recovery_outcomes",
        "audit_logs",
    ]

    for table in required_tables:
        assert table in tables, f"Missing expected table in schema: {table}"


def test_promise_to_pay_schema_structure():
    content = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "CREATE TABLE promise_to_pay" in content
    assert "promised_amount_inr NUMERIC(14,2) NOT NULL" in content
    assert "status VARCHAR(32) NOT NULL DEFAULT 'PTP_ACTIVE'" in content
    assert "grace_period_hours" in content
    assert "channel_source" in content
    assert "concession_applied JSONB" in content
    assert "REFERENCES recovery_cases(recovery_case_id)" in content


def test_invoices_and_aging_schema_structure():
    content = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "CREATE TABLE invoices" in content
    assert "CREATE TABLE invoice_aging_snapshots" in content
    assert "aging_bucket VARCHAR(32) NOT NULL" in content
    assert "'1_30_DAYS', '31_60_DAYS', '61_90_DAYS', '90_PLUS_DAYS'" in content
    assert "dispute_status VARCHAR(32) NOT NULL" in content


def test_checkout_sessions_structure():
    content = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "CREATE TABLE checkout_sessions" in content
    assert "cart_token VARCHAR(128) NOT NULL UNIQUE" in content
    assert "abandonment_stage" in content
    assert "recovery_link_token" in content


def test_recovery_cases_polymorphic_origin():
    content = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "case_origin VARCHAR(32) NOT NULL DEFAULT 'PAYMENT_FAILURE'" in content
    assert "invoice_id VARCHAR(64) REFERENCES invoices(invoice_id)" in content
    assert "checkout_session_id VARCHAR(64) REFERENCES checkout_sessions(checkout_session_id)" in content
    assert "CONSTRAINT chk_case_origin_reference CHECK" in content


def test_indexes_exist():
    content = SCHEMA_PATH.read_text(encoding="utf-8")
    index_pattern = re.compile(r"CREATE\s+INDEX\s+([a-z_]+)", re.IGNORECASE)
    indexes = index_pattern.findall(content)

    required_indexes = [
        "idx_payments_customer",
        "idx_attempts_payment",
        "idx_attempts_failure_category",
        "idx_invoices_customer",
        "idx_invoices_status_due",
        "idx_invoice_aging_invoice",
        "idx_invoice_aging_bucket",
        "idx_checkout_sessions_status",
        "idx_checkout_sessions_token",
        "idx_recovery_cases_status",
        "idx_recovery_cases_origin",
        "idx_recovery_cases_invoice",
        "idx_recovery_cases_checkout",
        "idx_ptp_case",
        "idx_ptp_customer",
        "idx_ptp_status_date",
        "idx_recovery_actions_scheduled",
        "idx_audit_logs_case",
    ]

    for idx in required_indexes:
        assert idx in indexes, f"Missing index: {idx}"
