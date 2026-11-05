from src.clean import clean_students
from src.generate_data import generate_monthly_costs, generate_students
from src.models import cluster_students, forecast_costs, train_gpa_model


def test_cleaning_removes_duplicates_and_nulls():
    clean = clean_students(generate_students())
    assert clean.student_id.is_unique
    assert clean.isna().sum().sum() == 0
    assert clean.monthly_food.max() < 90000


def test_clustering_returns_label_per_row():
    clean = clean_students(generate_students())
    labels, sil = cluster_students(clean, k=3)
    assert len(labels) == len(clean) and -1 <= sil <= 1


def test_gpa_model_learns_signal():
    clean = clean_students(generate_students())
    model = train_gpa_model(clean)
    assert model.r2 > 0.2
    p = model.predict(monthly_living_cost=15000, study_hours_week=20,
                      attendance_pct=90, part_time_job=0, year=2)
    assert 0 <= p <= 4


def test_forecast_horizon_and_growth():
    monthly = generate_monthly_costs()
    fc = forecast_costs(monthly, horizon=6)
    assert len(fc) == 6
    assert fc.forecast.iloc[-1] > monthly.avg_monthly_cost.iloc[0]
