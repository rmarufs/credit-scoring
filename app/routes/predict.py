"""Роут расчёта вероятности дефолта (/predict)."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends
import pandas as pd

from app.model_loader import get_loaded_model, get_threshold
from app.schemas import PredictRequest, PredictResponse
from src.features import prepare_features

router = APIRouter(tags=["scoring"])

PREDICT_OPENAPI_EXAMPLES: dict[str, Any] = {
    "low_risk_approve": {
        "summary": "Надёжный заёмщик (approve)",
        "description": "Стабильный доход, ипотека, грейд B и умеренная сумма кредита.",
        "value": {
            "person_age": 28,
            "person_income": 65000,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 5,
            "loan_intent": "EDUCATION",
            "loan_grade": "B",
            "loan_amnt": 8000,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 6,
        },
    },
    "high_risk_decline": {
        "summary": "Рискованная заявка (decline)",
        "description": "Высокая доля кредита от дохода, аренда жилья, грейд E и прошлый дефолт в кредитной истории.",
        "value": {
            "person_age": 24,
            "person_income": 28000,
            "person_home_ownership": "RENT",
            "person_emp_length": 1,
            "loan_intent": "DEBTCONSOLIDATION",
            "loan_grade": "E",
            "loan_amnt": 16000,
            "cb_person_default_on_file": "Y",
            "cb_person_cred_hist_length": 3,
        },
    },
}


@router.post("/predict", response_model=PredictResponse, summary="Оценка вероятности дефолта")
def predict(
    payload: Annotated[PredictRequest, Body(openapi_examples=PREDICT_OPENAPI_EXAMPLES)],
    model: Any = Depends(get_loaded_model),
    threshold: float = Depends(get_threshold),
) -> PredictResponse:
    """Считает вероятность дефолта по одной заявке и возвращает решение approve/decline."""
    raw_df = pd.DataFrame([payload.model_dump()])
    features_df = prepare_features(raw_df)
    proba = float(model.predict_proba(features_df)[0, 1])
    decision = "decline" if proba >= threshold else "approve"
    return PredictResponse(
        probability_of_default=proba,
        decision=decision,
        threshold=threshold,
    )
