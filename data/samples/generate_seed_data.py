#!/usr/bin/env python3
# data/samples/generate_seed_data.py
"""Generate realistic synthetic datasets for the Monte Carlo DQ POC.

Produces three CSV files:
  - orders.csv        — e-commerce order transactions
  - customers.csv     — customer master data with intentional quality issues
  - events.csv        — time-series clickstream events
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Orders dataset
# ---------------------------------------------------------------------------


def generate_orders(n: int = 5_000) -> pd.DataFrame:
    """Generate a synthetic orders DataFrame.

    Args:
        n: Number of order rows to generate.

    Returns:
        DataFrame with order transaction data including injected quality issues.
    """
    now = datetime.utcnow()
    order_ids = [f"ORD-{i:07d}" for i in range(1, n + 1)]
    customer_ids = rng.integers(1, 500, size=n)
    amounts = rng.lognormal(mean=4.5, sigma=1.2, size=n).round(2)
    statuses = rng.choice(
        ["pending", "confirmed", "shipped", "delivered", "cancelled"],
        p=[0.05, 0.15, 0.20, 0.55, 0.05],
        size=n,
    )
    created_at = [now - timedelta(hours=int(rng.integers(0, 720))) for _ in range(n)]

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "amount_usd": amounts,
            "status": statuses,
            "created_at": created_at,
            "updated_at": created_at,
        }
    )

    # Inject quality issues
    null_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[null_idx, "amount_usd"] = np.nan

    dup_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # Inject outlier amounts
    outlier_idx = rng.choice(len(df), size=5, replace=False)
    df.loc[outlier_idx, "amount_usd"] = rng.choice([0.01, 99999.99, -5.00], size=5)

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Customers dataset
# ---------------------------------------------------------------------------


def generate_customers(n: int = 1_000) -> pd.DataFrame:
    """Generate a synthetic customer master DataFrame.

    Args:
        n: Number of customer records.

    Returns:
        DataFrame with customer data and injected quality issues.
    """
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.io", "example.com"]
    first_names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
    last_names = ["Smith", "Jones", "Williams", "Brown", "Taylor", "Wilson", "Moore"]

    customer_ids = list(range(1, n + 1))
    first = [random.choice(first_names) for _ in range(n)]
    last = [random.choice(last_names) for _ in range(n)]
    emails = [
        f"{f.lower()}.{l.lower()}{rng.integers(1, 999)}@{random.choice(domains)}"
        for f, l in zip(first, last)
    ]
    ages = rng.integers(18, 80, size=n).astype(float)
    ltv = rng.lognormal(mean=6, sigma=1.5, size=n).round(2)
    country = rng.choice(["US", "GB", "DE", "FR", "JP", "CA", "AU"], size=n)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "first_name": first,
            "last_name": last,
            "email": emails,
            "age": ages,
            "lifetime_value_usd": ltv,
            "country": country,
            "created_at": [
                datetime.utcnow() - timedelta(days=int(rng.integers(0, 1825))) for _ in range(n)
            ],
        }
    )

    # Inject nulls in email (~5%)
    null_email = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[null_email, "email"] = None

    # Inject invalid ages (negative / > 120)
    df.loc[rng.choice(n, 10, replace=False), "age"] = rng.choice([-1, 150, 200], size=10)

    # Inject duplicate emails
    dup_customers = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[dup_customers, "email"] = df["email"].iloc[0]

    return df


# ---------------------------------------------------------------------------
# Events (clickstream) dataset
# ---------------------------------------------------------------------------


def generate_events(n: int = 20_000) -> pd.DataFrame:
    """Generate a synthetic clickstream events DataFrame.

    Args:
        n: Number of event rows.

    Returns:
        DataFrame with time-series clickstream data.
    """
    now = datetime.utcnow()
    event_types = rng.choice(
        ["page_view", "click", "add_to_cart", "purchase", "search"],
        p=[0.50, 0.25, 0.10, 0.05, 0.10],
        size=n,
    )
    session_ids = [f"sess-{rng.integers(1, n // 5):06d}" for _ in range(n)]
    user_ids = rng.integers(1, 500, size=n)
    page_urls = rng.choice(
        ["/home", "/products", "/cart", "/checkout", "/about", "/search"],
        size=n,
    )
    durations = rng.exponential(scale=45, size=n).round(1)
    timestamps = [now - timedelta(seconds=int(rng.integers(0, 3600 * 48))) for _ in range(n)]

    df = pd.DataFrame(
        {
            "event_id": range(1, n + 1),
            "session_id": session_ids,
            "user_id": user_ids,
            "event_type": event_types,
            "page_url": page_urls,
            "duration_seconds": durations,
            "timestamp": timestamps,
        }
    )

    # Inject nulls
    null_dur = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[null_dur, "duration_seconds"] = np.nan

    # Inject extreme duration outliers
    outlier_ev = rng.choice(n, size=20, replace=False)
    df.loc[outlier_ev, "duration_seconds"] = rng.choice([0.001, 9999.9], size=20)

    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Baseline snapshot (slightly different distribution for drift testing)
# ---------------------------------------------------------------------------


def generate_orders_baseline(n: int = 4_000) -> pd.DataFrame:
    """Generate a historical baseline orders snapshot with different stats.

    Args:
        n: Number of rows.

    Returns:
        Baseline DataFrame with slightly shifted distributions.
    """
    rng2 = np.random.default_rng(SEED + 1)
    now = datetime.utcnow()
    order_ids = [f"ORD-B{i:07d}" for i in range(1, n + 1)]
    customer_ids = rng2.integers(1, 500, size=n)
    # Lower mean amount to simulate drift
    amounts = rng2.lognormal(mean=3.8, sigma=1.0, size=n).round(2)
    statuses = rng2.choice(
        ["pending", "confirmed", "shipped", "delivered", "cancelled"],
        p=[0.10, 0.20, 0.25, 0.40, 0.05],
        size=n,
    )
    created_at = [now - timedelta(hours=int(rng2.integers(720, 1440))) for _ in range(n)]
    return pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "amount_usd": amounts,
            "status": statuses,
            "created_at": created_at,
            "updated_at": created_at,
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: generate and write all seed datasets."""
    parser = argparse.ArgumentParser(description="Generate Monte Carlo DQ seed data")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--orders", type=int, default=5_000)
    parser.add_argument("--customers", type=int, default=1_000)
    parser.add_argument("--events", type=int, default=20_000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating orders.csv …")
    generate_orders(args.orders).to_csv(out / "orders.csv", index=False)

    print("Generating orders_baseline.csv …")
    generate_orders_baseline(4_000).to_csv(out / "orders_baseline.csv", index=False)

    print("Generating customers.csv …")
    generate_customers(args.customers).to_csv(out / "customers.csv", index=False)

    print("Generating events.csv …")
    generate_events(args.events).to_csv(out / "events.csv", index=False)

    print(f"\n✅  Seed data written to {out.resolve()}")


if __name__ == "__main__":
    main()
