"""Роут проверки состояния сервиса (/health)."""

from fastapi import APIRouter, Request

from app.config import settings
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Проверка состояния сервиса")
def health_check(request: Request) -> HealthResponse:
    """Возвращает статус сервиса, флаг загрузки модели, порог и версию CatBoost."""
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    threshold = float(getattr(request.app.state, "threshold", settings.threshold))
    catboost_version = getattr(request.app.state, "catboost_version", None)
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        threshold=threshold,
        catboost_version=catboost_version,
    )
