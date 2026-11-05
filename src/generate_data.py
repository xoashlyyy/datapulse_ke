"""Generate a synthetic, fully anonymized student cost/academic dataset.

No real student data is used. Deliberate flaws (missing values, duplicates,
outliers) are injected so the cleaning pipeline has real work to do.
"""
import numpy as np
import pandas as pd

PROGRAMMES = ["Computer Science", "Business", "Nursing", "Engineering", "Education"]
COUNTIES = ["Nairobi", "Kiambu", "Murang'a", "Machakos", "Nakuru", "Kisumu"]


def generate_students(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "student_id": [f"S{i:05d}" for i in range(n)],
            "programme": rng.choice(PROGRAMMES, n),
            "year": rng.integers(1, 5, n),
            "county": rng.choice(COUNTIES, n),
            "monthly_rent": rng.normal(8000, 2500, n).clip(2000, 20000).round(),
            "monthly_food": rng.normal(6000, 1500, n).clip(1500, 12000).round(),
            "monthly_transport": rng.normal(3000, 900, n).clip(500, 8000).round(),
            "fees_per_semester": rng.normal(45000, 9000, n).clip(20000, 80000).round(),
            "attendance_pct": rng.normal(80, 12, n).clip(30, 100).round(1),
            "study_hours_week": rng.normal(15, 6, n).clip(0, 50).round(1),
            "part_time_job": rng.integers(0, 2, n),
        }
    )
    monthly = df.monthly_rent + df.monthly_food + df.monthly_transport
    stress = (monthly - monthly.mean()) / monthly.std()
    gpa = (
        2.2
        + 0.025 * df.study_hours_week
        + 0.012 * df.attendance_pct
        - 0.15 * stress
        - 0.10 * df.part_time_job
        + rng.normal(0, 0.25, n)
    )
    df["gpa"] = gpa.clip(0, 4).round(2)

    # inject realistic mess
    df.loc[rng.choice(n, 25, replace=False), "attendance_pct"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "monthly_rent"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "monthly_food"] = 90000  # outliers
    df = pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)  # duplicates
    return df


def generate_monthly_costs(months: int = 36, seed: int = 7) -> pd.DataFrame:
    """Monthly average cost-of-living index with trend + seasonality."""
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    value = 17000 + 90 * t + 600 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 250, months)
    dates = pd.date_range("2023-01-01", periods=months, freq="MS")
    return pd.DataFrame({"month": dates, "avg_monthly_cost": value.round(0)})
