"""Загрузка модели и метаданных при старте приложения (lifespan)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
import joblib

from app.config import Settings, settings as default_settings

logger = logging.getLogger("credit_scoring.api")


def load_artifacts_into_state(app: FastAPI, cfg: Settings) -> None:
    """Загружает модель и метаданные с диска и сохраняет их в app.state."""
    app.state.settings = cfg
    app.state.model = None
    app.state.metadata = {}
    app.state.model_loaded = False
    app.state.threshold = float(cfg.threshold)
    app.state.catboost_version = None

    if cfg.metadata_path.is_file():
        try:
            metadata: dict[str, Any] = json.loads(
                cfg.metadata_path.read_text(encoding="utf-8")
            )
            app.state.metadata = metadata
            versions = metadata.get("versions", {})
            if isinstance(versions, dict):
                app.state.catboost_version = versions.get("catboost")
        except Exception as exc:
            logger.warning("Не удалось прочитать метаданные %s: %s", cfg.metadata_path, exc)

    if cfg.model_path.is_file():
        try:
            app.state.model = joblib.load(cfg.model_path)
            app.state.model_loaded = True
            logger.info("Модель успешно загружена из %s", cfg.model_path)
        except Exception as exc:
            logger.error("Ошибка загрузки модели из %s: %s", cfg.model_path, exc)
    else:
        logger.warning("Файл модели не найден по пути %s", cfg.model_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan-контекст FastAPI: однократная загрузка артефактов при старте."""
    cfg: Settings = getattr(app.state, "settings", None) or default_settings
    load_artifacts_into_state(app, cfg)
    yield
    app.state.model = None
    app.state.model_loaded = False


def get_loaded_model(request: Request) -> Any:
    """FastAPI-зависимость: возвращает модель из app.state или выбрасывает 503."""
    if not getattr(request.app.state, "model_loaded", False) or request.app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель кредитного скоринга не загружена.",
        )
    return request.app.state.model


def get_threshold(request: Request) -> float:
    """Возвращает рабочий порог отсечения из состояния приложения."""
    return float(getattr(request.app.state, "threshold", default_settings.threshold))
