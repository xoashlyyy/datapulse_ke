"""Run the full pipeline: generate -> clean -> persist."""
from pathlib import Path

from src.clean import clean_students
from src.db import save_table
from src.generate_data import generate_monthly_costs, generate_students


def run(data_dir: str = "data"):
    Path(data_dir).mkdir(exist_ok=True)
    raw = generate_students()
    clean = clean_students(raw)
    monthly = generate_monthly_costs()
    raw.to_csv(f"{data_dir}/students_raw.csv", index=False)
    clean.to_csv(f"{data_dir}/students_clean.csv", index=False)
    monthly.to_csv(f"{data_dir}/monthly_costs.csv", index=False)
    saved = save_table(clean, "students_clean") and save_table(monthly, "monthly_costs")
    print(f"raw={len(raw)} clean={len(clean)} saved_to_postgres={saved}")


if __name__ == "__main__":
    run()
