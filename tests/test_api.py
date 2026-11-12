"""Тесты FastAPI-эндпоинтов /health и /predict."""

from pathlib import Path
import sys
from typing import Any

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.main import app, create_app


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    """Возвращает базовую валидную заявку с возможностью переопределить отдельные поля."""
    base: dict[str, Any] = {
        "person_age": 28,
        "person_income": 65_000.0,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 5.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 8_000.0,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6,
    }
    base.update(overrides)
    return base


def test_health_returns_200_and_model_loaded() -> None:
    """Проверяет, что /health возвращает 200 и model_loaded=True."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["threshold"] == pytest.approx(0.16)
        assert data["catboost_version"] == "1.2.10"


def test_predict_valid_request() -> None:
    """Проверяет ответ 200, диапазон вероятности [0, 1] и согласованность decision с порогом."""
    with TestClient(app) as client:
        response = client.post("/predict", json=_valid_payload())
        assert response.status_code == 200
        data = response.json()
        proba = data["probability_of_default"]
        threshold = data["threshold"]
        decision = data["decision"]

        assert 0.0 <= proba <= 1.0
        assert threshold == pytest.approx(0.16)
        expected_decision = "decline" if proba >= threshold else "approve"
        assert decision == expected_decision


def test_predict_null_emp_length_allowed() -> None:
    """Проверяет, что person_emp_length = null допустим."""
    with TestClient(app) as client:
        response = client.post("/predict", json=_valid_payload(person_emp_length=None))
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["probability_of_default"] <= 1.0


@pytest.mark.parametrize(
    "invalid_payload",
    [
        _valid_payload(person_age=150),
        _valid_payload(loan_grade="Z"),
        _valid_payload(person_income=0),
        _valid_payload(person_age=22, person_emp_length=10.0),
        {k: v for k, v in _valid_payload().items() if k != "loan_amnt"},
    ],
    ids=[
        "age_150",
        "grade_Z",
        "income_0",
        "emp_length_gt_age_minus_14",
        "missing_field",
    ],
)
def test_predict_invalid_payloads_return_422(invalid_payload: dict[str, Any]) -> None:
    """Проверяет, что невалидные входные данные отклоняются с кодом 422."""
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422


def test_predict_high_risk_vs_low_risk_ordering() -> None:
    """Смысловая проверка: высокий loan_percent_income и плохой грейд дают большую вероятность дефолта."""
    low_risk = _valid_payload(
        person_income=120_000.0,
        person_home_ownership="OWN",
        loan_grade="A",
        loan_amnt=5_000.0,
        cb_person_default_on_file="N",
    )
    high_risk = _valid_payload(
        person_income=25_000.0,
        person_home_ownership="RENT",
        loan_grade="F",
        loan_amnt=15_000.0,
        cb_person_default_on_file="Y",
    )

    with TestClient(app) as client:
        resp_low = client.post("/predict", json=low_risk)
        resp_high = client.post("/predict", json=high_risk)
        assert resp_low.status_code == 200
        assert resp_high.status_code == 200

        proba_low = resp_low.json()["probability_of_default"]
        proba_high = resp_high.json()["probability_of_default"]
        assert proba_high > proba_low
        assert resp_low.json()["decision"] == "approve"
        assert resp_high.json()["decision"] == "decline"


def test_predict_without_model_returns_503(tmp_path: Path) -> None:
    """Проверяет, что при отсутствии файла модели /health показывает model_loaded=false, а /predict возвращает 503."""
    missing_settings = Settings(
        model_path=tmp_path / "nonexistent_model.pkl",
        metadata_path=tmp_path / "nonexistent_metadata.json",
    )
    app_without_model = create_app(custom_settings=missing_settings)

    with TestClient(app_without_model) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["model_loaded"] is False

        predict_resp = client.post("/predict", json=_valid_payload())
        assert predict_resp.status_code == 503
