from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Backend sin ventana: necesario para generar imagenes dentro de Flask.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mismo patron dual de import que usan app.py y predictor.py.
if __package__:
    from .predictor import classify_air_quality
else:
    from predictor import classify_air_quality


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "datos_limpios.csv"
TEST_DATA_PATH = DATA_DIR / "test" / "datos_prueba.csv"
MODELS_DIR = PROJECT_DIR / "models"
LASSO_PATH = MODELS_DIR / "lasso_regression.pkl"
TREE_PATH = MODELS_DIR / "decision_tree_regressor.pkl"

TARGET = "PM 2.5"
POLLUTANTS = ["PM 10", "PM 2.5", "SO2", "NO2", "O3", "CO"]

# Mismos colores que usa scripts/entrenar_modelos.py para que los graficos en vivo combinen
# con las imagenes pre-generadas del dashboard.
PRIMARY_COLOR = "#28684d"
SECONDARY_COLOR = "#4f8f7a"
ACCENT_COLOR = "#b44b35"
GRID_COLOR = "#dce4df"

# Cache simple en memoria: el CSV limpio pesa ~27 MB y no cambia entre requests, asi que
# se lee una sola vez por proceso en vez de recargarlo en cada visita a /exploracion.
_clean_data_cache: pd.DataFrame | None = None


def load_clean_data() -> pd.DataFrame:
    # Carga (o reutiliza) el dataset limpio generado por scripts/limpiar_datos.py.
    global _clean_data_cache
    if _clean_data_cache is None:
        _clean_data_cache = pd.read_csv(CLEAN_DATA_PATH)
    return _clean_data_cache


def style_axis(axis) -> None:
    # Mismo estilo visual que los graficos guardados en reports/imagenes.
    axis.set_facecolor("#ffffff")
    axis.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#b8c8bf")
    axis.spines["bottom"].set_color("#b8c8bf")
    axis.tick_params(colors="#405148")
    axis.xaxis.label.set_color("#405148")
    axis.yaxis.label.set_color("#405148")


def build_eda_summary(estacion: str | None = None) -> dict:
    # Calcula estadisticas descriptivas con pandas/numpy al momento del request, filtrando
    # opcionalmente por estacion de monitoreo.
    df = load_clean_data()
    if estacion:
        df = df[df["ESTACION"] == estacion]

    # describe() de los contaminantes: cuenta, media, desviacion, cuartiles, etc.
    describe_df = df[POLLUTANTS].describe().round(2)
    describe_table = {
        "columns": describe_df.columns.tolist(),
        "rows": [
            {"estadistico": index, "valores": row.tolist()}
            for index, row in describe_df.iterrows()
        ],
    }

    # Matriz de correlacion entre contaminantes: ayuda a ver que tan relacionado esta
    # cada gas con PM 2.5.
    correlation_df = df[POLLUTANTS].corr().round(2)
    correlation_table = {
        "columns": correlation_df.columns.tolist(),
        "rows": [
            {"variable": index, "valores": row.tolist()}
            for index, row in correlation_df.iterrows()
        ],
    }

    # Promedio de PM 2.5 por estacion (groupby + mean), redondeado y ordenado de mayor a menor.
    promedio_estacion = (
        df.groupby("ESTACION")[TARGET].mean().round(2).sort_values(ascending=False)
    )

    # Patron horario: promedio de PM 2.5 por hora del dia.
    promedio_hora = df.groupby("HORA")[TARGET].mean().round(2)

    return {
        "total_filas": int(df.shape[0]),
        "describe": describe_table,
        "correlacion": correlation_table,
        "promedio_por_estacion": list(
            zip(promedio_estacion.index.tolist(), promedio_estacion.tolist())
        ),
        "promedio_por_hora": list(
            zip(promedio_hora.index.tolist(), promedio_hora.tolist())
        ),
    }


def build_histogram_png(estacion: str | None = None) -> bytes:
    # Genera un histograma de PM 2.5 en memoria (sin guardar archivo) usando matplotlib.
    df = load_clean_data()
    titulo = "Distribucion de PM2.5 - todas las estaciones"
    if estacion:
        df = df[df["ESTACION"] == estacion]
        titulo = f"Distribucion de PM2.5 - {estacion}"

    fig, axis = plt.subplots(figsize=(7.6, 5.0), facecolor="#f7faf8")
    axis.hist(
        df[TARGET].dropna(),
        bins=50,
        color=SECONDARY_COLOR,
        edgecolor="#ffffff",
        linewidth=0.5,
    )
    media = float(np.mean(df[TARGET]))
    axis.axvline(media, color=ACCENT_COLOR, linewidth=2, label=f"Media: {media:.2f}")
    axis.set_title(titulo, fontsize=13, fontweight="bold", pad=10)
    axis.set_xlabel("PM2.5")
    axis.set_ylabel("Frecuencia")
    axis.legend(frameon=False)
    style_axis(axis)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


# Coordenadas aproximadas (centro del distrito donde opera cada estacion). SENAMHI no
# publica la coordenada exacta del sensor en los datos abiertos, asi que esto es una
# aproximacion solo para ubicar la estacion en el mapa, no la posicion exacta del equipo.
STATION_COORDINATES = {
    "ATE": (-12.0431, -76.9147),
    "CAMPO DE MARTE": (-12.0764, -77.0428),
    "CARABAYLLO": (-11.8582, -77.0483),
    "HUACHIPA": (-11.9767, -76.9078),
    "PUENTE PIEDRA": (-11.8672, -77.0764),
    "SAN BORJA": (-12.1083, -77.0044),
    "SAN JUAN DE LURIGANCHO": (-11.9967, -77.0089),
    "SAN MARTIN DE PORRES": (-11.9847, -77.0844),
    "SANTA ANITA": (-12.0464, -76.9689),
    "VILLA MARIA DEL TRIUNFO": (-12.1572, -76.9358),
}


def residuales_por_estacion() -> list[dict]:
    # Calcula el error de cada modelo por estacion sobre el set de prueba: carga los .pkl,
    # predice con sklearn y agrupa los residuales (valor real - prediccion) con pandas.
    if not TEST_DATA_PATH.exists() or not LASSO_PATH.exists() or not TREE_PATH.exists():
        return []

    test_df = pd.read_csv(TEST_DATA_PATH).copy()
    lasso = joblib.load(LASSO_PATH)
    tree = joblib.load(TREE_PATH)

    test_df["pred_lasso"] = lasso.predict(test_df)
    test_df["pred_tree"] = tree.predict(test_df)
    # Residual = valor real menos prediccion. Positivo => el modelo subestimo PM2.5.
    test_df["residual_lasso"] = test_df[TARGET] - test_df["pred_lasso"]
    test_df["residual_tree"] = test_df[TARGET] - test_df["pred_tree"]
    test_df["abs_residual_lasso"] = test_df["residual_lasso"].abs()
    test_df["abs_residual_tree"] = test_df["residual_tree"].abs()

    agrupado = (
        test_df.groupby("ESTACION")
        .agg(
            filas=("residual_tree", "count"),
            mae_lasso=("abs_residual_lasso", "mean"),
            mae_tree=("abs_residual_tree", "mean"),
            sesgo_lasso=("residual_lasso", "mean"),
            sesgo_tree=("residual_tree", "mean"),
        )
        .round(2)
        .sort_values("mae_tree", ascending=False)
    )

    return [
        {"estacion": estacion.strip(), **fila.to_dict()}
        for estacion, fila in agrupado.iterrows()
    ]


def build_residuales_png() -> bytes:
    # Grafico de barras con el error absoluto promedio (MAE) del Decision Tree por estacion,
    # generado en memoria a partir de residuales_por_estacion().
    datos = residuales_por_estacion()
    fig, axis = plt.subplots(figsize=(7.8, 5.4), facecolor="#f7faf8")

    if datos:
        datos_ordenados = sorted(datos, key=lambda item: item["mae_tree"])
        estaciones = [item["estacion"] for item in datos_ordenados]
        valores = [item["mae_tree"] for item in datos_ordenados]
        axis.barh(estaciones, valores, color=PRIMARY_COLOR)
        axis.set_xlabel("MAE (Decision Tree)")
    else:
        axis.text(0.5, 0.5, "Sin datos de prueba disponibles", ha="center", va="center")

    axis.set_title(
        "Error absoluto promedio por estacion (set de prueba)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    style_axis(axis)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def build_station_map_data() -> list[dict]:
    # Promedio historico de PM2.5 por estacion (pandas groupby), combinado con su
    # clasificacion de calidad de aire y coordenadas aproximadas para el mapa.
    df = load_clean_data()
    promedio_por_estacion = df.groupby("ESTACION")[TARGET].mean()

    estaciones = []
    for estacion, promedio in promedio_por_estacion.items():
        nombre = estacion.strip()
        coords = STATION_COORDINATES.get(nombre)
        if coords is None:
            continue
        valor = round(float(promedio), 2)
        estaciones.append(
            {
                "estacion": nombre,
                "lat": coords[0],
                "lon": coords[1],
                "promedio_pm25": valor,
                **classify_air_quality(valor),
            }
        )
    return sorted(estaciones, key=lambda item: item["promedio_pm25"], reverse=True)


def recalcular_metricas_en_vivo() -> list[dict]:
    # Vuelve a calcular MAE/RMSE/R2 sobre data/test/datos_prueba.csv, cargando los .pkl y
    # prediciendo con sklearn en el momento del request (no lee metricas_modelos.json).
    if not TEST_DATA_PATH.exists() or not LASSO_PATH.exists() or not TREE_PATH.exists():
        return []

    test_df = pd.read_csv(TEST_DATA_PATH)
    y_test = test_df[TARGET]

    lasso = joblib.load(LASSO_PATH)
    tree = joblib.load(TREE_PATH)

    lasso_predictions = lasso.predict(test_df)
    tree_predictions = tree.predict(test_df)

    resultados = []
    for nombre, predictions in (
        ("Lasso Regression", lasso_predictions),
        ("Decision Tree Regressor", tree_predictions),
    ):
        mse = mean_squared_error(y_test, predictions)
        resultados.append(
            {
                "modelo": nombre,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_test, predictions)),
                "filas_evaluadas": int(len(y_test)),
            }
        )
    return resultados
