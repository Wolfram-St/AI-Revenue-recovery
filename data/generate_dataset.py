"""Generate the synthetic Day 1 payment-attempt dataset.

This script creates 5,000 deterministic records when run with the default seed.
It intentionally keeps post-intervention outcomes separate from decision-time
features so Day 2 model training can enforce leakage controls.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 42
ROWS = 5000
OUTPUT = Path(__file__).with_name("payment_attempts.csv")

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
DEVICE_TYPES = ["android", "ios", "web"]
COUNTRIES = ["IN", "AE", "SG", "US"]
FAILURES = [
    ("T001", "temporary_decline", "issuer_temporary_decline"),
    ("H001", "hard_decline", "issuer_hard_decline"),
    ("M001", "payment_method_issue", "invalid_or_expired_method"),
    ("A001", "authentication_required", "authentication_required"),
    ("U001", "unknown", "unknown_issuer_error"),
]

FIELDS = [
    "attempt_id", "payment_id", "customer_id", "amount_inr", "payment_method",
    "attempt_number", "customer_tenure_days", "successful_payment_count",
    "failed_payment_count", "historical_recovery_count", "customer_opted_out",
    "failure_code", "failure_category", "issuer_response", "device_type",
    "country", "fraud_risk", "recovered",
]


def build_row(rng: random.Random, index: int) -> dict[str, object]:
    customer_id = f"CUS-{rng.randint(1, 1000):04d}"
    payment_id = f"PAY-{index:06d}"
    attempt_id = f"ATT-{index:06d}"
    method = rng.choice(PAYMENT_METHODS)
    amount = round(rng.lognormvariate(7.4, 0.75), 2)
    amount = min(max(amount, 99.0), 100000.0)
    attempt_number = rng.choices([1, 2, 3], weights=[70, 22, 8])[0]
    tenure = rng.randint(30, 1500)
    successful = rng.randint(0, 35)
    failed = rng.randint(0, 10)
    historical_recovery = rng.randint(0, min(successful, 8))
    opted_out = rng.random() < 0.025
    failure_code, category, issuer = rng.choice(FAILURES)
    device = rng.choice(DEVICE_TYPES)
    country = rng.choices(COUNTRIES, weights=[88, 4, 4, 4])[0]
    fraud_risk = rng.random() < 0.018

    # Synthetic outcome mechanism only. These outcome rules are deliberately
    # kept out of model-feature construction in Day 2.
    base = {
        "temporary_decline": 0.68,
        "hard_decline": 0.05,
        "payment_method_issue": 0.18,
        "authentication_required": 0.46,
        "unknown": 0.28,
    }[category]
    probability = base
    probability += min(successful, 10) * 0.012
    probability += min(historical_recovery, 5) * 0.025
    probability -= max(attempt_number - 1, 0) * 0.10
    probability -= 0.55 if opted_out else 0
    probability -= 0.60 if fraud_risk else 0
    probability = max(0.01, min(0.97, probability))
    recovered = rng.random() < probability

    return {
        "attempt_id": attempt_id,
        "payment_id": payment_id,
        "customer_id": customer_id,
        "amount_inr": amount,
        "payment_method": method,
        "attempt_number": attempt_number,
        "customer_tenure_days": tenure,
        "successful_payment_count": successful,
        "failed_payment_count": failed,
        "historical_recovery_count": historical_recovery,
        "customer_opted_out": opted_out,
        "failure_code": failure_code,
        "failure_category": category,
        "issuer_response": issuer,
        "device_type": device,
        "country": country,
        "fraud_risk": fraud_risk,
        "recovered": recovered,
    }


def generate(path: Path = OUTPUT, rows: int = ROWS, seed: int = SEED) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(1, rows + 1):
            writer.writerow(build_row(rng, index))


if __name__ == "__main__":
    generate()
    print(f"Generated {ROWS} rows at {OUTPUT}")
