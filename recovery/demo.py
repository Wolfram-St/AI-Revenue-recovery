"""End-to-end demo for the Recovery Agent.

Run this script to see the full pipeline in action::

    python -m recovery.demo

It will:
1. Generate 30 synthetic payment failures
2. Run the full detect > diagnose > decide > execute pipeline
3. Print a formatted summary showing money recovered
4. Print the full audit trail
"""

from __future__ import annotations

import sys
import time

from recovery.recovery_agent import RecoveryAgent, AgentConfig, generate_synthetic_failures


def _print_header(text: str) -> None:
    width = 70
    print()
    print("=" * width)
    print("  " + text)
    print("=" * width)


def _print_row(label: str, value: str, width: int = 35) -> None:
    print("  " + label.ljust(width) + value)


def run_demo() -> None:
    """Run the full recovery agent demo."""

    _print_header("RECOVERAI - Revenue Recovery Agent Demo")
    print()
    print("  This demo runs the full autonomous recovery pipeline:")
    print("  DETECT > DIAGNOSE > DECIDE > EXECUTE > AUDIT")
    print()

    # Step 1: Generate synthetic failures
    _print_header("Step 1: Generating Synthetic Payment Failures")
    failures = generate_synthetic_failures(count=30)
    total_at_risk = sum(f["amount"] for f in failures)
    _print_row("Failures generated:", str(len(failures)))
    _print_row("Total revenue at risk:", "Rs.{:.2f}".format(total_at_risk / 100))
    print()

    # Show sample failures
    print("  Sample failures:")
    for f in failures[:5]:
        print("    - {}: Rs.{:.2f} via {} ({})".format(
            f['id'], f['amount']/100, f['method'],
            f.get('error_reason', 'unknown')))
    if len(failures) > 5:
        print("    ... and {} more".format(len(failures) - 5))
    print()

    # Step 2: Configure the agent
    _print_header("Step 2: Configuring Recovery Agent")
    config = AgentConfig(
        budget_limit_paise=100000,  # Rs.1000 budget
        max_retries_per_payment=3,
        human_review_capacity=5,
        simulation=True,
    )
    _print_row("Budget limit:", "Rs.{:.2f}".format(config.budget_limit_paise / 100))
    _print_row("Max retries per payment:", str(config.max_retries_per_payment))
    _print_row("HR review capacity:", str(config.human_review_capacity))
    _print_row("Mode:", "SIMULATION (no real API calls)")
    print()

    # Step 3: Run the agent
    _print_header("Step 3: Running Recovery Agent Pipeline")
    agent = RecoveryAgent.simulation(config=config)

    start_time = time.time()
    result = agent.run_batch(failures)
    elapsed = (time.time() - start_time) * 1000

    _print_row("Pipeline completed in:", "{:.1f}ms".format(elapsed))
    print()

    # Step 4: Results
    _print_header("Step 4: Recovery Results")

    print()
    print("  METRICS")
    print("  " + "-" * 40)
    _print_row("Total payments processed:", str(result.total_processed))
    _print_row("Total amount at risk:", "Rs.{:.2f}".format(result.total_amount_paise / 100))
    _print_row("Successfully retried:", str(result.recovered_count))
    _print_row("Amount retried:", "Rs.{:.2f}".format(result.recovered_paise / 100))
    _print_row("Recovery rate:", "{:.1f}%".format(result.recovery_rate * 100))
    _print_row("Escalated to human:", str(result.escalated_count))
    _print_row("Stopped (no retry):", str(result.stopped_count))
    _print_row("Budget used:", "Rs.{:.2f}".format(result.budget_used_paise / 100))
    _print_row("Budget remaining:", "Rs.{:.2f}".format(result.budget_remaining_paise / 100))
    print()

    # Step 5: Action breakdown
    _print_header("Step 5: Action Breakdown")

    action_dist: dict[str, int] = {}
    cat_dist: dict[str, int] = {}
    for entry in result.audit_entries:
        action_dist[entry.action_type] = action_dist.get(entry.action_type, 0) + 1
        cat_dist[entry.failure_category] = cat_dist.get(entry.failure_category, 0) + 1

    for action, count in sorted(action_dist.items()):
        bar = "#" * count
        _print_row("{}:".format(action), "{} {}".format(count, bar))
    print()

    # Step 6: Root cause breakdown
    _print_header("Step 6: Root Cause Breakdown")

    for cat, count in sorted(cat_dist.items()):
        bar = "#" * count
        _print_row("{}:".format(cat), "{} {}".format(count, bar))
    print()

    # Step 7: Audit trail
    _print_header("Step 7: Audit Trail (first 10 entries)")

    print()
    print("  {:<20} {:>10} {:<25} {:<20} {}".format(
        "Payment ID", "Amount", "Root Cause", "Action", "OK"))
    print("  " + "-" * 80)

    for entry in result.audit_entries[:10]:
        amount = "Rs.{:.2f}".format(entry.amount_paise / 100)
        ok = "Y" if entry.execution_success else "N"
        print("  {:<20} {:>10} {:<25} {:<20} {}".format(
            entry.payment_id, amount, entry.failure_category,
            entry.action_type, ok))

    if len(result.audit_entries) > 10:
        print("  ... and {} more entries".format(len(result.audit_entries) - 10))
    print()

    # Final verdict
    _print_header("DEMO COMPLETE")
    print()
    print("  Money recovered (simulated): Rs.{:.2f}".format(result.recovered_paise / 100))
    print("  Recovery rate: {:.1f}%".format(result.recovery_rate * 100))
    print("  Audit trail: {} entries".format(len(result.audit_entries)))
    print()
    print("  To run with real Razorpay data:")
    print("    1. Sign up at https://razorpay.com (free test mode)")
    print("    2. Generate test API keys")
    print("    3. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in .env")
    print("    4. Start the server: python -m uvicorn app.main:app --reload")
    print("    5. Open http://localhost:8000/recovery-agent")
    print("    6. Or call POST /api/recovery/run for live data")
    print()


if __name__ == "__main__":
    run_demo()
