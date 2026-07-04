from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

# Rutas donde se guardan los modelos entrenados y las metricas.
LASSO_PATH = MODELS_DIR / "lasso_regression.pkl"
TREE_PATH = MODELS_DIR / "decision_tree_regressor.pkl"
METRICS_PATH = REPORTS_DIR / "metricas_modelos.json"
HISTORY_PATH = REPORTS_DIR / "historial_predicciones.csv"

# Orden exacto de columnas que el modelo espera recibir.
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

# Nombres legibles de cada modelo, reutilizados en resultados, historial y la API.
MODEL_LABELS = {
    "lasso": "Lasso Regression",
    "tree": "Decision Tree Regressor",
}

# Columnas del historial de predicciones (reports/historial_predicciones.csv).
HISTORY_COLUMNS = [
    "fecha_hora",
    "pm10",
    "so2",
    "no2",
    "o3",
    "co",
    "ano",
    "mes",
    "dia",
    "hora",
    "estacion",
    "prediccion_lasso",
    "prediccion_tree",
    "categoria_lasso",
    "categoria_tree",
]

# Valores iniciales que aparecen en el formulario de prediccion.
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

# Categorias de calidad de aire para PM2.5 (24h), basadas en los cortes del AQI de la EPA,
# el mismo estandar internacional que suelen citar los reportes de calidad de aire en Lima.
AIR_QUALITY_BREAKPOINTS = [
    (0.0, 12.0, "Buena", "#28684d"),
    (12.1, 35.4, "Moderada", "#c9a227"),
    (35.5, 55.4, "Danina a grupos sensibles", "#d9822b"),
    (55.5, 150.4, "Danina a la salud", "#b44b35"),
    (150.5, 250.4, "Muy danina", "#7a3b8c"),
    (250.5, float("inf"), "Peligrosa", "#5a1f1f"),
]


def load_metrics() -> dict:
    # Carga metricas, variables y reglas del arbol para mostrarlas en el dashboard.
    if not METRICS_PATH.exists():
        return {"metrics": [], "stations": ["ATE"]}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def load_models() -> dict:
    # Carga los .pkl generados por scripts/entrenar_modelos.py.
    if not LASSO_PATH.exists() or not TREE_PATH.exists():
        return {}
    return {
        "lasso": joblib.load(LASSO_PATH),
        "tree": joblib.load(TREE_PATH),
    }


def build_input(form_data: dict) -> pd.DataFrame:
    # Convierte los datos escritos en el formulario HTML a una fila de pandas.
    # Los nombres deben coincidir con las columnas usadas durante el entrenamiento.
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


def classify_air_quality(pm25_value: float) -> dict:
    # Ubica el valor predicho dentro de las categorias de calidad de aire para PM2.5.
    for low, high, categoria, color in AIR_QUALITY_BREAKPOINTS:
        if low <= pm25_value <= high:
            return {"categoria": categoria, "color": color}
    # No deberia llegar aqui porque el ultimo rango no tiene techo, pero por seguridad:
    return {"categoria": "Peligrosa", "color": "#5a1f1f"}


def predict(model_key: str, form_data: dict) -> float:
    # Selecciona el modelo pedido por el usuario y devuelve la prediccion de PM 2.5.
    models = load_models()
    if model_key not in models:
        raise FileNotFoundError("Primero entrene los modelos con scripts/entrenar_modelos.py")
    value = models[model_key].predict(build_input(form_data))[0]
    return round(float(value), 2)


def predict_both(form_data: dict) -> dict:
    # Calcula la prediccion de Lasso y Decision Tree para el mismo input, con su
    # categoria de calidad de aire, para mostrarlas lado a lado en el dashboard.
    models = load_models()
    if not models:
        raise FileNotFoundError("Primero entrene los modelos con scripts/entrenar_modelos.py")

    row = build_input(form_data)
    resultado = {}
    for model_key, model in models.items():
        valor = round(float(model.predict(row)[0]), 2)
        resultado[model_key] = {
            "modelo": MODEL_LABELS[model_key],
            "valor": valor,
            **classify_air_quality(valor),
        }
    return resultado


def append_history(form_data: dict, predictions: dict) -> None:
    # Guarda la prediccion en reports/historial_predicciones.csv para llevar un registro
    # de las pruebas hechas desde el formulario.
    row = {
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pm10": form_data.get("pm10"),
        "so2": form_data.get("so2"),
        "no2": form_data.get("no2"),
        "o3": form_data.get("o3"),
        "co": form_data.get("co"),
        "ano": form_data.get("ano"),
        "mes": form_data.get("mes"),
        "dia": form_data.get("dia"),
        "hora": form_data.get("hora"),
        "estacion": form_data.get("estacion"),
        "prediccion_lasso": predictions.get("lasso", {}).get("valor"),
        "prediccion_tree": predictions.get("tree", {}).get("valor"),
        "categoria_lasso": predictions.get("lasso", {}).get("categoria"),
        "categoria_tree": predictions.get("tree", {}).get("categoria"),
    }
    row_df = pd.DataFrame([row], columns=HISTORY_COLUMNS)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if HISTORY_PATH.exists():
        row_df.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(HISTORY_PATH, mode="w", header=True, index=False)


def load_history(limit: int = 15) -> list[dict]:
    # Devuelve las ultimas predicciones guardadas, la mas reciente primero.
    if not HISTORY_PATH.exists():
        return []
    history_df = pd.read_csv(HISTORY_PATH)
    return history_df.tail(limit).iloc[::-1].to_dict(orient="records")
