import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.db import load_table
from src.models import (cluster_students, forecast_costs, label_clusters,
                        train_gpa_model)
from src.pipeline import run

st.set_page_config(page_title="DataPulse KE", layout="wide")
DATA = Path("data")


@st.cache_data
def load():
    students, monthly = load_table("students_clean"), load_table("monthly_costs")
    if students is None or monthly is None:
        if not (DATA / "students_clean.csv").exists():
            run()
        students = pd.read_csv(DATA / "students_clean.csv")
        monthly = pd.read_csv(DATA / "monthly_costs.csv", parse_dates=["month"])
    return students, monthly


students, monthly = load()
st.title("DataPulse KE")
st.caption("Anonymized student cost and academic patterns (synthetic demo data).")

with st.sidebar:
    options = sorted(students.programme.unique())
    prog = st.multiselect("Programme", options, default=options)
view = students[students.programme.isin(prog)]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segments", "GPA predictor", "Cost forecast"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(view))
    c2.metric("Avg monthly living cost", f"KES {view.monthly_living_cost.mean():,.0f}")
    c3.metric("Avg GPA", f"{view.gpa.mean():.2f}")
    st.plotly_chart(px.scatter(view, x="monthly_living_cost", y="gpa", color="programme",
                               title="Living cost vs GPA"), width="stretch")
    st.plotly_chart(px.box(view, x="programme", y="semester_total_cost",
                           title="Semester total cost by programme"), width="stretch")

with tab2:
    k = st.slider("Number of segments", 2, 8, 4)
    labels, sil = cluster_students(view, k)
    st.write(f"Silhouette score: **{sil:.2f}** (closer to 1 means better separated)")
    st.dataframe(label_clusters(view, labels))
    st.plotly_chart(px.scatter(view.assign(cluster=labels.astype(str)), x="monthly_living_cost",
                               y="gpa", color="cluster"), width="stretch")

with tab3:
    model = train_gpa_model(students)
    st.write(f"Holdout MAE: **{model.mae:.2f}** GPA points, R²: **{model.r2:.2f}**")
    a, b = st.columns(2)
    cost = a.slider("Monthly living cost (KES)", 5000, 40000, 17000, 500)
    hours = a.slider("Study hours / week", 0, 50, 15)
    att = b.slider("Attendance %", 30, 100, 80)
    job = b.selectbox("Part-time job", [0, 1], format_func=lambda v: "Yes" if v else "No")
    yr = b.selectbox("Year", [1, 2, 3, 4])
    pred = model.predict(monthly_living_cost=cost, study_hours_week=hours,
                         attendance_pct=att, part_time_job=job, year=yr)
    st.metric("Predicted GPA", f"{pred:.2f}")

with tab4:
    horizon = st.slider("Months ahead", 3, 12, 6)
    fc = forecast_costs(monthly, horizon)
    hist = monthly.rename(columns={"avg_monthly_cost": "value"}).assign(kind="actual")
    fut = fc.rename(columns={"forecast": "value"}).assign(kind="forecast")
    st.plotly_chart(px.line(pd.concat([hist, fut]), x="month", y="value", color="kind"),
                    width="stretch")
