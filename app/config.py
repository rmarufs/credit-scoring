"""Конфигурация API через переменные окружения."""

import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "model.pkl"
DEFAULT_METADATA_PATH = ROOT / "models" / "metadata.json"


def _load_default_threshold(metadata_path: Path = DEFAULT_METADATA_PATH) -> float:
    """Читает порог по умолчанию из models/metadata.json."""
    if metadata_path.is_file():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return float(data.get("threshold", 0.16))
        except (OSError, ValueError, TypeError):
            return 0.16
    return 0.16


class Settings(BaseSettings):
    """Настройки сервиса кредитного скоринга."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = DEFAULT_MODEL_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    threshold: float = _load_default_threshold()


settings = Settings()
