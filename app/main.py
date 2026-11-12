"""Точка входа FastAPI-сервиса кредитного скоринга."""

from collections.abc import Awaitable, Callable
import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse

from app.config import Settings, settings as default_settings
from app.model_loader import lifespan
from app.routes.health import router as health_router
from app.routes.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("credit_scoring.api")


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    """Создаёт и настраивает экземпляр FastAPI-приложения."""
    application = FastAPI(
        title="Credit Scoring API",
        description=(
            "Сервис оценки вероятности дефолта заёмщика на базе модели CatBoost. "
            "Принимает анкетные и кредитные параметры заявки и возвращает вероятность дефолта и решение. "
            "Сервис демонстрационный, реальные персональные данные вводить не нужно. "
            "Бесплатный сервер засыпает после простоя, первый запрос может занять около минуты."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = custom_settings or default_settings

    @application.get("/", include_in_schema=False)
    def redirect_to_docs() -> RedirectResponse:
        """Перенаправляет корневой запрос на интерактивную документацию Swagger UI."""
        return RedirectResponse(url="/docs")

    @application.middleware("http")
    async def log_request_timing(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Логирует только метод, путь, код ответа и время обработки без персональных данных."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "%s %s -> %d (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    application.include_router(health_router)
    application.include_router(predict_router)
    return application


app = create_app()
