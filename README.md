# DataPulse KE: AI and Data Intelligence Platform

A data platform that cleans anonymized student cost and academic data, finds patterns with clustering,
predicts GPA from cost and study habits, forecasts cost-of-living trends, and presents everything in an
interactive Streamlit dashboard.

> **Data note:** the project ships with a *synthetic* data generator so it runs anywhere and never involves
> real student records. To use real data, replace the generator with your own CSV, keep it anonymized
> (no names, ID numbers or contact details), and get written permission from the data owner first.

## What it does

| Stage | What happens | Code |
|---|---|---|
| Generate | Builds 600 synthetic students (with deliberate missing values, duplicates, outliers) and a 36-month cost series | `src/generate_data.py` |
| Clean | Removes duplicates, fills gaps with medians, clips outliers (IQR), engineers `monthly_living_cost`, `semester_total_cost`, `cost_burden` | `src/clean.py` |
| Segment | K-Means clustering on standardized features, reports silhouette score and per-segment profile | `src/models.py` |
| Predict | Random Forest predicts GPA, evaluated on a held-out 20% (MAE, R²) | `src/models.py` |
| Forecast | Linear trend plus month-of-year seasonality, projects 3 to 12 months ahead | `src/models.py` |
| Serve | Streamlit dashboard with four tabs; reads PostgreSQL if available, CSV otherwise | `app/streamlit_app.py` |

## Architecture

```
generate_data.py --> clean.py --> pipeline.py --+--> data/*.csv  (always)
                                                +--> PostgreSQL  (if DATABASE_URL is set)
                                                          |
                              models.py <-----------------+
                                  |
                          streamlit_app.py  (Overview | Segments | GPA predictor | Cost forecast)
```

## Tech stack

Python 3.12, pandas, NumPy, scikit-learn, SQLAlchemy, PostgreSQL 16, Streamlit, Plotly, pytest, Docker.

## Quick start

### Option 1: Docker (includes PostgreSQL)

```bash
docker compose up --build
# open http://localhost:8501
```

### Option 2: local Python

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline             # creates data/*.csv
streamlit run app/streamlit_app.py
```

To use PostgreSQL locally: `export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/dbname`
before running the pipeline and the app.

## Run the tests

```bash
python -m pytest -q
```

The 4 tests check that cleaning removes duplicates and nulls, clustering returns one label per student,
the GPA model beats a trivial baseline (R² above 0.2) and returns values in 0 to 4, and the forecast
returns the right horizon and follows the upward trend.

## Verified

Run in a clean virtualenv from `requirements.txt` only: 4 tests pass. The pipeline was run against a real
PostgreSQL 16 server (tables `students_clean` and `monthly_costs` created, 600 rows), and the Streamlit app was
then started with the CSV files deleted to prove it reads from the database. All four tabs were exercised
(segment slider, programme filter, predictor, forecast) with no exceptions. Docker was not available in the
build environment, so `docker compose up` itself was not run (the Dockerfile passes hadolint, the compose file parses).

## Design decisions worth knowing

- **Median imputation plus IQR clipping** is robust to the outliers that the generator injects. Mean imputation would be dragged by them.
- **Scaling before K-Means:** without `StandardScaler`, the cost column (thousands) would dominate GPA (0 to 4).
- **Silhouette score** is shown so a user can see that choosing k is a judgment call, not a fact.
- **The GPA model deliberately excludes `gpa`-derived features** to avoid target leakage.
- **CSV fallback** keeps the dashboard usable without a database, which makes demos easy.

## Limitations and next steps

- Synthetic data means the relationships are ones *I coded in*; the model recovers them, which proves the pipeline works, not that the real world behaves this way.
- Forecasting uses a simple regression. Next step: compare with Prophet or SARIMAX using a rolling-origin backtest.
- Add model explanations (feature importance, SHAP) to the predictor tab.
- Add data-quality checks (Great Expectations or pandera) at the cleaning boundary.
- Schedule the pipeline (cron or Airflow) instead of running it on demand.

## Be ready to explain (interview prep)

1. Why did you scale features before clustering?
2. How do you know your GPA model is not just memorizing? (Train/test split, holdout metrics.)
3. What would you change if the real data had 2 million rows?
4. What privacy risks exist in student data, and how does anonymization fall short on its own?
