"""Clustering, GPA prediction and cost forecasting."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["monthly_living_cost", "study_hours_week", "attendance_pct", "gpa"]
GPA_FEATURES = ["monthly_living_cost", "study_hours_week", "attendance_pct",
                "part_time_job", "year"]


def cluster_students(df: pd.DataFrame, k: int = 4, seed: int = 42):
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return km.labels_, float(silhouette_score(X, km.labels_))


def label_clusters(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Per-cluster means so each segment can be described in plain words."""
    tmp = df.assign(cluster=labels)
    summary = tmp.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
    summary["size"] = tmp.groupby("cluster").size()
    return summary


@dataclass
class GpaModel:
    model: RandomForestRegressor
    mae: float
    r2: float

    def predict(self, **features) -> float:
        row = pd.DataFrame([features])[GPA_FEATURES]
        return float(np.clip(self.model.predict(row)[0], 0, 4))


def train_gpa_model(df: pd.DataFrame, seed: int = 42) -> GpaModel:
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[GPA_FEATURES], df["gpa"], test_size=0.2, random_state=seed)
    rf = RandomForestRegressor(n_estimators=200, random_state=seed).fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    return GpaModel(rf, float(mean_absolute_error(y_te, pred)), float(r2_score(y_te, pred)))


def _design(frame: pd.DataFrame) -> pd.DataFrame:
    months = pd.get_dummies(frame["moy"], prefix="m", dtype=float)
    months = months.reindex(columns=[f"m_{i}" for i in range(1, 13)], fill_value=0.0)
    return pd.concat([frame[["t"]].reset_index(drop=True), months.reset_index(drop=True)], axis=1)


def forecast_costs(monthly: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """Linear trend + month-of-year seasonality via one-hot regression."""
    m = monthly.copy()
    m["t"] = np.arange(len(m))
    m["moy"] = m["month"].dt.month
    lr = LinearRegression().fit(_design(m), m["avg_monthly_cost"])
    dates = pd.date_range(m["month"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    fut = pd.DataFrame({"month": dates, "t": np.arange(len(m), len(m) + horizon)})
    fut["moy"] = fut["month"].dt.month
    fut["forecast"] = lr.predict(_design(fut)).round(0)
    return fut[["month", "forecast"]]
