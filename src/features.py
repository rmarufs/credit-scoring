"""Подготовка признаков для модели кредитного скоринга."""

import pandas as pd

DROP_COLS: list[str] = ["loan_int_rate"]

CAT_FEATURES: list[str] = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

INPUT_FEATURES: list[str] = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

FEATURE_ORDER: list[str] = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

MODEL_FEATURES: list[str] = FEATURE_ORDER


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Формирует матрицу признаков для CatBoost без изменения исходного датафрейма."""
    cols_to_drop = [c for c in [*DROP_COLS, "loan_status"] if c in df.columns]
    out = df.drop(columns=cols_to_drop).copy()
    out["loan_percent_income"] = out["loan_amnt"] / out["person_income"]
    for col in CAT_FEATURES:
        out[col] = out[col].astype(str)
    return out[FEATURE_ORDER]
