"""Загрузка и очистка данных Credit Risk Dataset."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "credit_risk_dataset.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "credit_risk_clean.csv"

# Пороги аномалий (обоснование в notebooks/01_eda.ipynb)
MAX_AGE = 80
MAX_EMP_LENGTH = 40
MIN_WORK_START_AGE = 14
MAX_INCOME = 1_000_000


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Читает сырой CSV."""
    return pd.read_csv(path)


def anomaly_mask(df: pd.DataFrame) -> pd.Series:
    """True для строк с аномальными значениями (пропуски сюда не попадают)."""
    return (
        (df["person_age"] > MAX_AGE)
        | (df["person_emp_length"] > MAX_EMP_LENGTH)
        | (df["person_emp_length"] > df["person_age"] - MIN_WORK_START_AGE)
        | (df["person_income"] > MAX_INCOME)
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет аномалии, затем дубликаты. Исходный датафрейм не меняется."""
    out = df.loc[~anomaly_mask(df)]
    out = out.drop_duplicates(keep="first")
    return out.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    n_anomalies = int(anomaly_mask(raw).sum())
    n_duplicates = int(raw.loc[~anomaly_mask(raw)].duplicated().sum())
    cleaned = clean(raw)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False)

    print(f"Было строк:         {len(raw)}")
    print(f"Удалено аномалий:   {n_anomalies}")
    print(f"Удалено дубликатов: {n_duplicates}")
    print(f"Осталось строк:     {len(cleaned)}")
    print(f"Доля дефолтов:      {cleaned['loan_status'].mean():.4f}")