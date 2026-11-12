"""Pydantic-схемы запросов и ответов API кредитного скоринга."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HomeOwnershipLiteral = Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
LoanIntentLiteral = Literal[
    "DEBTCONSOLIDATION",
    "EDUCATION",
    "HOMEIMPROVEMENT",
    "MEDICAL",
    "PERSONAL",
    "VENTURE",
]
LoanGradeLiteral = Literal["A", "B", "C", "D", "E", "F", "G"]
DefaultOnFileLiteral = Literal["Y", "N"]
DecisionLiteral = Literal["approve", "decline"]


class PredictRequest(BaseModel):
    """Входные данные кредитной заявки для расчёта вероятности дефолта."""

    model_config = ConfigDict(extra="forbid")

    person_age: int = Field(
        ...,
        ge=20,
        le=80,
        description="Возраст заёмщика в годах (от 20 до 80).",
    )
    person_income: float = Field(
        ...,
        gt=0,
        le=1_000_000,
        description="Годовой доход заёмщика (> 0 и <= 1 000 000).",
    )
    person_home_ownership: HomeOwnershipLiteral = Field(
        ...,
        description="Статус владения жильём: RENT, OWN, MORTGAGE или OTHER.",
    )
    person_emp_length: float | None = Field(
        default=None,
        ge=0,
        le=40,
        description="Стаж работы в годах (от 0 до 40 или null).",
    )
    loan_intent: LoanIntentLiteral = Field(
        ...,
        description="Цель кредита.",
    )
    loan_grade: LoanGradeLiteral = Field(
        ...,
        description="Кредитный грейд (от A до G).",
    )
    loan_amnt: float = Field(
        ...,
        gt=0,
        le=35_000,
        description="Запрашиваемая сумма кредита (> 0 и <= 35 000).",
    )
    cb_person_default_on_file: DefaultOnFileLiteral = Field(
        ...,
        description="Наличие исторического дефолта в кредитном бюро: Y или N.",
    )
    cb_person_cred_hist_length: int = Field(
        ...,
        ge=2,
        le=30,
        description="Длина кредитной истории в годах (от 2 до 30).",
    )

    @model_validator(mode="after")
    def validate_emp_length_vs_age(self) -> "PredictRequest":
        """Проверяет, что стаж работы не превышает возраст минус 14 лет."""
        if self.person_emp_length is not None:
            max_allowed_emp = self.person_age - 14
            if self.person_emp_length > max_allowed_emp:
                raise ValueError(
                    f"Стаж работы (person_emp_length={self.person_emp_length}) "
                    f"не может превышать person_age - 14 ({max_allowed_emp})."
                )
        return self


class PredictResponse(BaseModel):
    """Ответ модели со скоринговой оценкой и решением по заявке."""

    probability_of_default: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Оценка вероятности дефолта в диапазоне [0, 1].",
    )
    decision: DecisionLiteral = Field(
        ...,
        description="Решение по заявке: decline при probability_of_default >= threshold, иначе approve.",
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Применённый порог принятия решения.",
    )


class HealthResponse(BaseModel):
    """Ответ эндпоинта проверки состояния сервиса."""

    status: str = Field(..., description="Статус работы сервиса.")
    model_loaded: bool = Field(..., description="Флаг успешной загрузки модели.")
    threshold: float = Field(..., description="Текущий порог отсечения.")
    catboost_version: str | None = Field(
        default=None,
        description="Версия библиотеки CatBoost из metadata.json.",
    )
