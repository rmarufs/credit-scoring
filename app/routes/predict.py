"""Роут расчёта вероятности дефолта (/predict)."""

from typing import Any

from fastapi import APIRouter, Depends
import pandas as pd

from app.model_loader import get_loaded_model, get_threshold
from app.schemas import PredictRequest, PredictResponse
from src.features import prepare_features

router = APIRouter(tags=["scoring"])


@router.post("/predict", response_model=PredictResponse, summary="Оценка вероятности дефолта")
def predict(
    payload: PredictRequest,
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
