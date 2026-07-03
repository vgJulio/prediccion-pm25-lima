"""Aplicacion web Flask para predecir PM2.5 en Lima.

La aplicacion usa los dos modelos solicitados en el proyecto:
- Lasso Regression
- Decision Tree Regressor
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
GRAPHICS_DIR = BASE_DIR / "graphics"
REPORTS_DIR = BASE_DIR / "reports"

SUMMARY_PATH = REPORTS_DIR / "resumen_ejecucion.json"
LASSO_MODEL_PATH = MODELS_DIR / "modelo_lasso_pm25.pkl"
TREE_MODEL_PATH = MODELS_DIR / "modelo_arbol_pm25.pkl"

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

STATIONS = [
    "ATE",
    "CAMPO DE MARTE",
    "CARABAYLLO",
    "HUACHIPA",
    "PUENTE PIEDRA",
    "SAN BORJA ",
    "SAN JUAN DE LURIGANCHO",
    "SANTA ANITA",
    "VILLA MARIA DEL TRIUNFO",
    "VILLA MARIA DEL TRIUNFO 2",
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


app = Flask(__name__)


def load_summary() -> dict:
    """Carga metricas, coeficientes e importancias guardadas."""
    if not SUMMARY_PATH.exists():
        return {"metrics": [], "top_lasso_coefficients": [], "top_tree_importances": []}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def load_models() -> dict:
    """Carga los pipelines entrenados con Joblib."""
    return {
        "lasso": joblib.load(LASSO_MODEL_PATH),
        "tree": joblib.load(TREE_MODEL_PATH),
    }


SUMMARY = load_summary()
MODELS = load_models()


def build_input_dataframe(form_data: dict) -> pd.DataFrame:
    """Convierte el formulario HTML en una fila compatible con Scikit-Learn."""
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


def predict_pm25(model_key: str, form_data: dict) -> float:
    """Realiza una prediccion de PM2.5 con el modelo seleccionado."""
    input_data = build_input_dataframe(form_data)
    prediction = MODELS[model_key].predict(input_data)[0]
    return round(float(prediction), 2)


def get_metrics_by_model() -> dict:
    """Organiza las metricas por nombre de modelo para mostrarlas en HTML."""
    metrics = {}
    for row in SUMMARY.get("metrics", []):
        metrics[row["Modelo"]] = row
    return metrics


@app.route("/")
def index():
    """Pagina principal con resumen del proyecto y graficos."""
    return render_template(
        "index.html",
        summary=SUMMARY,
        metrics=get_metrics_by_model(),
    )


@app.route("/comparacion")
def comparacion():
    """Pagina comparativa entre Lasso y Decision Tree Regressor."""
    return render_template(
        "comparacion.html",
        summary=SUMMARY,
        metrics=SUMMARY.get("metrics", []),
    )


@app.route("/prediccion-lasso", methods=["GET", "POST"])
def prediccion_lasso():
    """Formulario de prediccion usando Lasso Regression."""
    form_values = DEFAULT_VALUES.copy()
    prediction = None
    error = None

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        try:
            prediction = predict_pm25("lasso", form_values)
        except ValueError:
            error = "Revise los datos ingresados. Todos los campos numericos son obligatorios."

    return render_template(
        "prediccion_lasso.html",
        form_values=form_values,
        prediction=prediction,
        error=error,
        stations=STATIONS,
    )


@app.route("/prediccion-decision-tree", methods=["GET", "POST"])
def prediccion_decision_tree():
    """Formulario de prediccion usando Decision Tree Regressor."""
    form_values = DEFAULT_VALUES.copy()
    prediction = None
    error = None

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        try:
            prediction = predict_pm25("tree", form_values)
        except ValueError:
            error = "Revise los datos ingresados. Todos los campos numericos son obligatorios."

    return render_template(
        "prediccion_desicion_tree.html",
        form_values=form_values,
        prediction=prediction,
        error=error,
        stations=STATIONS,
    )


@app.route("/graphics/<path:filename>")
def graphics(filename: str):
    """Permite mostrar los graficos generados por Matplotlib."""
    return send_from_directory(GRAPHICS_DIR, filename)


@app.route("/reports/<path:filename>")
def reports(filename: str):
    """Permite descargar reportes del proyecto."""
    return send_from_directory(REPORTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
