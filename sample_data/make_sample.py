"""Generate a deliberately messy customer-churn CSV for demos and tests."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 2000
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
tenure = np.where(contract == "Month-to-month", rng.integers(1, 30, n), rng.integers(6, 72, n))
monthly = np.round(rng.normal(70, 25, n).clip(18, 140), 2)
support_calls = rng.poisson(1.5, n) + (contract == "Month-to-month") * rng.poisson(1, n)
internet = rng.choice(["Fiber", "DSL", "No internet"], n, p=[0.45, 0.4, 0.15])
logit = (-1.2 + 1.4 * (contract == "Month-to-month") - 0.04 * tenure + 0.015 * (monthly - 70)
         + 0.35 * support_calls + 0.5 * (internet == "Fiber"))
churn = rng.random(n) < 1 / (1 + np.exp(-logit))

df = pd.DataFrame({
    "Customer ID": [f"C{100000 + i}" for i in range(n)],
    "Signup Date": pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
    "Region": rng.choice(["North", "South", "East", "West"], n),
    "Contract Type": contract,
    "Internet Service": internet,
    "Tenure (months)": tenure,
    "Monthly Charges": monthly,
    "Total Charges": np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2),
    "Support Calls": support_calls,
    "Paperless Billing": rng.choice(["Yes", "No"], n),
    "Notes": [None] * n,
    "Churn": np.where(churn, "Yes", "No"),
})
df["Signup Date"] = df["Signup Date"].dt.strftime("%Y-%m-%d")
df["Monthly Charges"] = df["Monthly Charges"].map(lambda v: f"${v:,.2f}")
for col, frac in [("Region", 0.05), ("Monthly Charges", 0.03), ("Support Calls", 0.04)]:
    idx = rng.choice(n, int(n * frac), replace=False)
    df[col] = df[col].astype(object)
    df.loc[idx, col] = rng.choice(["", "N/A", None], len(idx))
df = pd.concat([df, df.sample(25, random_state=1)])  # duplicates

out = Path(__file__).with_name("customer_churn.csv")
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows)")
