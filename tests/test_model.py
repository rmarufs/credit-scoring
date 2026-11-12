"""Быстрые тесты подготовки признаков и сохранённой модели."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_ORDER, prepare_features
from src.train import MODEL_PATH


def test_prepare_features_three_rows() -> None:
    """Проверяет порядок колонок, удаление loan_int_rate и пересчёт loan_percent_income на 3 строках."""
    raw_df = pd.DataFrame(
        [
            {
                "person_age": 25,
                "person_income": 50_000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3.0,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 10_000,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.99,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4,
            },
            {
                "person_age": 32,
                "person_income": 80_000,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": np.nan,
                "loan_intent": "MEDICAL",
                "loan_grade": "A",
                "loan_amnt": 16_000,
                "loan_int_rate": 7.9,
                "loan_percent_income": 0.0,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 8,
            },
            {
                "person_age": 40,
                "person_income": 60_000,
                "person_home_ownership": "OWN",
                "person_emp_length": 12.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "D",
                "loan_amnt": 15_000,
                "loan_int_rate": 15.2,
                "loan_percent_income": 0.10,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 14,
            },
        ]
    )

    out = prepare_features(raw_df)

    assert list(out.columns) == FEATURE_ORDER
    assert "loan_int_rate" not in out.columns
    np.testing.assert_allclose(
        out["loan_percent_income"].to_numpy(),
        np.array([0.20, 0.20, 0.25]),
    )
    assert np.isnan(out.loc[1, "person_emp_length"])


def test_model_predict_single_row() -> None:
    """Проверяет загрузку models/model.pkl и предсказание вероятности дефолта на одной строке."""
    model = joblib.load(MODEL_PATH)
    single_row = pd.DataFrame(
        [
            {
                "person_age": 28,
                "person_income": 55_000,
                "person_home_ownership": "RENT",
                "person_emp_length": 4.0,
                "loan_intent": "VENTURE",
                "loan_grade": "B",
                "loan_amnt": 11_000,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6,
            }
        ]
    )

    X_one = prepare_features(single_row)
    proba = float(model.predict_proba(X_one)[0, 1])

    assert 0.0 <= proba <= 1.0
