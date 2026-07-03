from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

LASSO_PATH = MODELS_DIR / "lasso_regression.pkl"
TREE_PATH = MODELS_DIR / "decision_tree_regressor.pkl"
METRICS_PATH = REPORTS_DIR / "metricas_modelos.json"

FEATURE_COLUMNS = [
    "PM 10",
    "SO2",
    "NO2",
    "O3",
    "CO",
    "ANO",
    "MES",
    "DIA",
    "HORA",
    "ESTACION",
]

DEFAULT_VALUES = {
    "pm10": "80",
    "so2": "8",
    "no2": "30",
    "o3": "15",
    "co": "1.2",
    "ano": "2020",
    "mes": "6",
    "dia": "15",
    "hora": "8",
    "estacion": "ATE",
}


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {"metrics": [], "stations": ["ATE"]}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def load_models() -> dict:
    if not LASSO_PATH.exists() or not TREE_PATH.exists():
        return {}
    return {
        "lasso": joblib.load(LASSO_PATH),
        "tree": joblib.load(TREE_PATH),
    }


def build_input(form_data: dict) -> pd.DataFrame:
    row = {
        "PM 10": float(form_data.get("pm10", DEFAULT_VALUES["pm10"])),
        "SO2": float(form_data.get("so2", DEFAULT_VALUES["so2"])),
        "NO2": float(form_data.get("no2", DEFAULT_VALUES["no2"])),
        "O3": float(form_data.get("o3", DEFAULT_VALUES["o3"])),
        "CO": float(form_data.get("co", DEFAULT_VALUES["co"])),
        "ANO": int(form_data.get("ano", DEFAULT_VALUES["ano"])),
        "MES": int(form_data.get("mes", DEFAULT_VALUES["mes"])),
        "DIA": int(form_data.get("dia", DEFAULT_VALUES["dia"])),
        "HORA": int(form_data.get("hora", DEFAULT_VALUES["hora"])),
        "ESTACION": form_data.get("estacion", DEFAULT_VALUES["estacion"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict(model_key: str, form_data: dict) -> float:
    models = load_models()
    if model_key not in models:
        raise FileNotFoundError("Primero entrene los modelos con scripts/entrenar_modelos.py")
    value = models[model_key].predict(build_input(form_data))[0]
    return round(float(value), 2)
