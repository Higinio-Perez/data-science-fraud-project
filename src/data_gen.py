"""
data_gen.py

Synthetic transaction + customer data generator for an AML/Fraud Data Science project.

Goal (simple):
- Create a realistic-looking dataset of customer transactions with a rare fraud label (e.g., ~1–2%).
- Avoid using real data (privacy), but still mimic real patterns:
  * Heavy-tailed transaction amounts
  * Ecommerce vs card-present
  * Country mismatch vs home country
  * Night/weekend effects
  * "Velocity" (bursts of activity)
  * "Device risk" (proxy signal)
  * Mild concept drift over time (fraud patterns shift slightly)

Outputs:
- data/customers.csv
- data/transactions.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_transactions(
    n_customers: int = 4000,
    n_tx: int = 120_000,
    start_date: str = "2025-01-01",
    days: int = 120,
    fraud_rate: float = 0.015,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic customers and transactions.

    Parameters
    ----------
    n_customers : int
        Number of unique customers to create.
    n_tx : int
        Number of transactions to generate.
    start_date : str
        Start date for timestamps (YYYY-MM-DD).
    days : int
        Time span (in days) over which transactions occur.
    fraud_rate : float
        Target overall fraud rate (approximate). Example: 0.015 -> ~1.5%
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    transactions_df : pd.DataFrame
        Transaction-level dataset with features and fraud label.
    customers_df : pd.DataFrame
        Customer-level dataset with base attributes (home_country, base_risk).
    """
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # 1) Create customers
    # -------------------------------------------------------------------------
    # base_risk: "latent risk propensity" for each customer.
    # Using a Beta distribution skewed toward low values to mimic most customers being low-risk.
    customers_df = pd.DataFrame(
        {
            "customer_id": np.arange(n_customers),
            "base_risk": rng.beta(2, 15, size=n_customers),
            "home_country": rng.choice(
                ["ES", "FR", "DE", "IT", "GB", "NL", "BE"],
                size=n_customers,
                p=[0.18, 0.12, 0.14, 0.12, 0.20, 0.12, 0.12],
            ),
        }
    )

    # -------------------------------------------------------------------------
    # 2) Create transaction skeleton: assign customer_id + timestamp
    # -------------------------------------------------------------------------
    customer_id = rng.integers(0, n_customers, size=n_tx)

    start = pd.Timestamp(start_date)
    timestamps = start + pd.to_timedelta(
        rng.integers(0, days * 24 * 60, size=n_tx), unit="m"
    )
    hour = timestamps.hour
    day_index = (timestamps - start).days.values  # day offset since start

    # -------------------------------------------------------------------------
    # 3) Transaction attributes: category, channel, amount
    # -------------------------------------------------------------------------
    merchant_category = rng.choice(
        ["grocery", "electronics", "travel", "crypto", "gaming", "fashion", "fuel", "restaurants"],
        size=n_tx,
        p=[0.18, 0.14, 0.10, 0.05, 0.06, 0.14, 0.10, 0.23],
    )

    channel = rng.choice(
        ["card_present", "ecommerce"],
        size=n_tx,
        p=[0.55, 0.45],
    )

    # Heavy-tailed amounts: lognormal is a classic for payments (many small, few large)
    amount = rng.lognormal(mean=3.1, sigma=0.9, size=n_tx)

    # Category/channel multipliers (simple realism):
    amount *= np.where(merchant_category == "travel", 1.8, 1.0)
    amount *= np.where(merchant_category == "crypto", 2.2, 1.0)
    amount *= np.where(channel == "ecommerce", 1.15, 1.0)

    # -------------------------------------------------------------------------
    # 4) Derived signals/features used later for labeling (and modeling)
    # -------------------------------------------------------------------------
    # "Night" transactions: 00:00–05:59
    night = ((hour >= 0) & (hour <= 5)).astype(int)

    # Weekend: Saturday/Sunday
    weekend = (pd.Series(timestamps).dt.dayofweek >= 5).astype(int).values

    # High amount indicator: top 5% by amount
    high_amount = (amount > np.quantile(amount, 0.95)).astype(int)

    # Home country per transaction (lookup from customers)
    home_country = customers_df.loc[customer_id, "home_country"].values

    # Transaction country: usually home, sometimes different (like travel or online anomalies)
    country = np.where(
        rng.random(n_tx) < 0.88,
        home_country,
        rng.choice(["ES", "FR", "DE", "IT", "GB", "NL", "BE", "US", "AE", "TR"], size=n_tx),
    )
    country_change = (country != home_country).astype(int)

    # Device risk proxy:
    # - random normal baseline
    # - higher if ecommerce
    # - higher if country differs from home
    device_risk = rng.normal(0, 1, size=n_tx)
    device_risk += 0.7 * country_change + 0.4 * (channel == "ecommerce").astype(int)

    # Velocity proxy:
    # In real life you'd compute this from time windows per customer.
    # Here we simulate it: ecommerce tends to have more "bursty" patterns.
    velocity = rng.poisson(lam=1.2 + 1.0 * (channel == "ecommerce"), size=n_tx)
    velocity += rng.binomial(3, 0.12 + 0.10 * night, size=n_tx)

    # Customer-level latent risk, broadcasted to each transaction
    base_risk = customers_df.loc[customer_id, "base_risk"].values

    # -------------------------------------------------------------------------
    # 5) Fraud "score" and label generation
    # -------------------------------------------------------------------------
    # Concept drift:
    # Fraud patterns shift slightly over time. We model this as a smooth periodic term.
    drift = 0.25 * np.sin(2 * np.pi * day_index / days)

    # Linear-ish score (log-odds style) combining signals.
    # This is a *simulated* underlying risk function.
    score = (
        -4.6
        + 1.2 * high_amount
        + 0.9 * country_change
        + 0.6 * night
        + 0.5 * (merchant_category == "crypto").astype(int)
        + 0.4 * (merchant_category == "electronics").astype(int)
        + 0.5 * (channel == "ecommerce").astype(int)
        + 0.35 * velocity
        + 0.55 * device_risk
        + 2.2 * base_risk
        # Drift affects certain patterns more (e.g., cross-border + crypto)
        + drift * (0.8 * country_change + 0.6 * (merchant_category == "crypto").astype(int))
    )

    # Convert score to probability with logistic function
    p = 1 / (1 + np.exp(-score))

    # Adjust probabilities so the *average* is close to the desired fraud_rate.
    # This is a simple scaling trick (not perfect, but works well for synthetic data).
    p = p * (fraud_rate / p.mean())
    p = np.clip(p, 0, 0.95)

    # Sample fraud label from Bernoulli(p)
    is_fraud = (rng.random(n_tx) < p).astype(int)

    # -------------------------------------------------------------------------
    # 6) Build final transactions DataFrame
    # -------------------------------------------------------------------------
    transactions_df = pd.DataFrame(
        {
            "transaction_id": np.arange(n_tx),
            "customer_id": customer_id,
            "timestamp": timestamps,
            "amount": amount,
            "merchant_category": merchant_category,
            "channel": channel,
            "country": country,
            "home_country": home_country,
            "hour": hour,
            "night": night,
            "weekend": weekend,
            "country_change": country_change,
            "device_risk": device_risk,
            "velocity": velocity,
            "is_fraud": is_fraud,
        }
    )

    return transactions_df, customers_df


if __name__ == "__main__":
    # Generate data with default parameters
    transactions, customers = generate_transactions()

    # Save to CSV files inside the project
    transactions.to_csv("data/transactions.csv", index=False)
    customers.to_csv("data/customers.csv", index=False)

    # Print confirmation + realized fraud rate
    print("Saved data/transactions.csv and data/customers.csv")
    print("Fraud rate:", transactions["is_fraud"].mean())
