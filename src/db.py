"""Optional PostgreSQL persistence. Falls back to CSV when DATABASE_URL is unset."""
import os

import pandas as pd
from sqlalchemy import create_engine


def get_engine():
    url = os.getenv("DATABASE_URL")
    return create_engine(url) if url else None


def save_table(df: pd.DataFrame, name: str) -> bool:
    engine = get_engine()
    if engine is None:
        return False
    df.to_sql(name, engine, if_exists="replace", index=False)
    return True


def load_table(name: str):
    engine = get_engine()
    if engine is None:
        return None
    try:
        return pd.read_sql_table(name, engine)
    except Exception:
        return None
