"""Обучение финальной модели CatBoost и сохранение артефактов."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import catboost
from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import clean, load_raw
from src.evaluate import compute_metrics, compute_threshold_metrics
from src.features import (
    CAT_FEATURES,
    DROP_COLS,
    FEATURE_ORDER,
    INPUT_FEATURES,
    prepare_features,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.16
USE_CALIBRATION = False
COST_C = 5

BEST_PARAMS: dict[str, int | float] = {
    "depth": 4,
    "learning_rate": 0.15248263275183993,
    "l2_leaf_reg": 4.743226441772372,
    "random_strength": 2.1115768380392548,
    "bagging_temperature": 2.6397323786149323,
    "border_count": 254,
    "iterations": 662,
}


def train() -> dict[str, Any]:
    """Обучает финальную модель, оценивает на test и сохраняет артефакты в models/."""
    print("1/4 Загрузка и очистка данных...")
    df = clean(load_raw())
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train_feat = prepare_features(X_train)
    X_test_feat = prepare_features(X_test)
    print(f"    Train: {len(X_train_feat)} строк | Test: {len(X_test_feat)} строк")

    print("2/4 Обучение CatBoostClassifier...")
    model = CatBoostClassifier(
        cat_features=CAT_FEATURES,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
        **BEST_PARAMS,
    )
    model.fit(X_train_feat, y_train)

    print("3/4 Оценка на отложенной выборке test...")
    test_proba = model.predict_proba(X_test_feat)[:, 1]
    metrics = compute_metrics(y_test, test_proba)
    threshold_metrics = compute_threshold_metrics(y_test, test_proba, THRESHOLD)

    metadata: dict[str, Any] = {
        "model_name": "CatBoostClassifier",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_size": int(len(X_train_feat)),
        "test_size": int(len(X_test_feat)),
        "input_features": INPUT_FEATURES,
        "dropped_columns": DROP_COLS,
        "features": FEATURE_ORDER,
        "cat_features": CAT_FEATURES,
        "best_params": BEST_PARAMS,
        "random_seed": RANDOM_STATE,
        "use_calibration": USE_CALIBRATION,
        "threshold": THRESHOLD,
        "cost_assumption": {
            "c": COST_C,
            "fp_cost": 1.0,
            "fn_cost": float(COST_C),
        },
        "test_metrics": {k: round(v, 4) for k, v in metrics.items()},
        "test_threshold_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in threshold_metrics.items()
        },
        "versions": {
            "python": platform.python_version(),
            "catboost": catboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    print("4/4 Сохранение артефактов...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nФинальные метрики на test:")
    for name, val in metrics.items():
        print(f"  {name:<10}: {val:.4f}")
    print(
        f"\nОперационные метрики при пороге {THRESHOLD:.2f} (c={COST_C}):\n"
        f"  TN={threshold_metrics['tn']}, FP={threshold_metrics['fp']}, "
        f"FN={threshold_metrics['fn']}, TP={threshold_metrics['tp']}\n"
        f"  approval_rate : {threshold_metrics['approval_rate']:.4f}\n"
        f"  recall        : {threshold_metrics['recall']:.4f}\n"
        f"  precision     : {threshold_metrics['precision']:.4f}\n"
        f"  bad_rate      : {threshold_metrics['bad_rate']:.4f}"
    )
    print(f"\nСохранено: {MODEL_PATH.relative_to(ROOT)}, {METADATA_PATH.relative_to(ROOT)}")
    return metadata


if __name__ == "__main__":
    train()
