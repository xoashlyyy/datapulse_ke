"""Cleaning + feature engineering."""
import pandas as pd

NUMERIC_FILL = ["monthly_rent", "monthly_food", "monthly_transport",
                "attendance_pct", "study_hours_week", "fees_per_semester"]


def clip_outliers_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(q1 - k * iqr, q3 + k * iqr)


def clean_students(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset="student_id").copy()
    for col in NUMERIC_FILL:
        out[col] = out[col].fillna(out[col].median())
        out[col] = clip_outliers_iqr(out[col])
    out["monthly_living_cost"] = out.monthly_rent + out.monthly_food + out.monthly_transport
    out["semester_total_cost"] = out.fees_per_semester + out.monthly_living_cost * 4
    out["cost_burden"] = out.monthly_living_cost / out.monthly_living_cost.median()
    return out.reset_index(drop=True)
