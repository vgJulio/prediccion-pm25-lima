"""Proyecto de regresion para predecir PM2.5 en Lima.

Este script reproduce la logica metodologica del notebook de Regresion
Logistica usado en clase: carga de datos, inspeccion, limpieza, separacion
de variables, division train/test, entrenamiento, prediccion y evaluacion.

Modelos permitidos:
- Lasso Regression
- Decision Tree Regressor
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree


DATA_PATH = Path(
    r"C:\Users\JULIO\Downloads\datos_horarios_contaminacion_lima.xlsx"
)
OUTPUT_DIR = Path(__file__).resolve().parent
MODELS_DIR = OUTPUT_DIR / "modelos"
FIGURES_DIR = OUTPUT_DIR / "graficos"
TABLES_DIR = OUTPUT_DIR / "tablas"

RANDOM_STATE = 42
TEST_SIZE = 0.30
TARGET = "PM 2.5"
CONTAMINANTS = ["PM 10", "PM 2.5", "SO2", "NO2", "O3", "CO"]
PREDICTOR_POLLUTANTS = ["PM 10", "SO2", "NO2", "O3", "CO"]
NUMERIC_FEATURES = [
    "PM 10",
    "SO2",
    "NO2",
    "O3",
    "CO",
    "ANO",
    "MES",
    "DIA",
    "HORA",
]
CATEGORICAL_FEATURES = ["ESTACION"]
DUPLICATE_KEYS = ["CODIGO ESTACION", "ANO", "MES", "DIA", "HORA"]


def make_output_dirs() -> None:
    """Crea las carpetas donde se guardan modelos, tablas y graficos."""
    for directory in [MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lee el archivo Excel original con Pandas."""
    return pd.read_excel(path)


def convert_pollutants_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte contaminantes a numeros y corrige comas decimales."""
    clean_df = df.copy()
    for column in CONTAMINANTS:
        clean_df[column] = clean_df[column].astype(str).str.replace(",", ".")
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")
    return clean_df


def remove_duplicate_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina registros repetidos para la misma estacion y hora."""
    duplicate_mask = df.duplicated(subset=DUPLICATE_KEYS)
    duplicate_count = int(duplicate_mask.sum())
    clean_df = df.drop_duplicates(subset=DUPLICATE_KEYS).copy()
    return clean_df, duplicate_count


def impute_predictor_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa contaminantes predictores sin modificar la variable objetivo.

    La imputacion sigue la logica del notebook del proyecto: primero una
    interpolacion temporal por estacion; luego medianas por estacion y hora.
    Si queda algun valor faltante, se completa con la mediana global.
    """
    clean_df = df.sort_values(DUPLICATE_KEYS).copy()

    for column in PREDICTOR_POLLUTANTS:
        clean_df[column] = clean_df.groupby("CODIGO ESTACION")[column].transform(
            lambda values: values.interpolate(
                method="linear",
                limit=6,
                limit_direction="both",
            )
        )
        clean_df[column] = clean_df[column].fillna(
            clean_df.groupby(["CODIGO ESTACION", "HORA"])[column].transform(
                "median"
            )
        )
        clean_df[column] = clean_df[column].fillna(
            clean_df.groupby("CODIGO ESTACION")[column].transform("median")
        )
        clean_df[column] = clean_df[column].fillna(clean_df[column].median())

    return clean_df


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica limpieza y devuelve el dataset listo para modelado."""
    original_shape = df.shape
    missing_before = df.isna().sum().to_dict()

    clean_df = convert_pollutants_to_numeric(df)
    clean_df, duplicate_count = remove_duplicate_records(clean_df)
    clean_df = impute_predictor_pollutants(clean_df)

    rows_before_target_drop = len(clean_df)
    clean_df = clean_df.dropna(subset=[TARGET]).copy()
    clean_df = clean_df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    metadata = {
        "original_shape": list(original_shape),
        "shape_after_cleaning": list(clean_df.shape),
        "duplicates_removed": duplicate_count,
        "rows_removed_missing_target": int(rows_before_target_drop - len(clean_df)),
        "missing_before": missing_before,
        "missing_after": clean_df.isna().sum().to_dict(),
        "station_count": int(clean_df["ESTACION"].nunique()),
        "year_min": int(clean_df["ANO"].min()),
        "year_max": int(clean_df["ANO"].max()),
    }
    return clean_df, metadata


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables independientes X y variable dependiente y."""
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return x, y


def build_one_hot_encoder() -> OneHotEncoder:
    """Crea un codificador compatible con versiones recientes y antiguas."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_lasso_pipeline() -> Pipeline:
    """Construye el flujo de preprocesamiento y modelo Lasso."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    model = Lasso(alpha=0.05, max_iter=10000, random_state=RANDOM_STATE)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_tree_pipeline() -> Pipeline:
    """Construye el flujo de preprocesamiento y arbol de decision."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    model = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=8,
        min_samples_leaf=100,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray]:
    """Entrena, predice y calcula metricas de regresion."""
    start_train = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    predictions = pipeline.predict(x_test)
    predict_time = time.perf_counter() - start_predict

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "Modelo": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_test, predictions),
        "Tiempo entrenamiento (s)": train_time,
        "Tiempo prediccion (s)": predict_time,
    }
    return metrics, predictions


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Obtiene los nombres de variables despues del preprocesamiento."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def extract_lasso_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Extrae e interpreta los coeficientes aprendidos por Lasso."""
    feature_names = get_feature_names(pipeline)
    coefficients = pipeline.named_steps["model"].coef_
    coef_df = pd.DataFrame(
        {
            "Variable": feature_names,
            "Coeficiente": coefficients,
            "Valor absoluto": np.abs(coefficients),
        }
    )
    return coef_df.sort_values("Valor absoluto", ascending=False)


def extract_tree_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extrae la importancia de variables del arbol de decision."""
    feature_names = get_feature_names(pipeline)
    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {"Variable": feature_names, "Importancia": importances}
    )
    return importance_df.sort_values("Importancia", ascending=False)


def save_actual_vs_predicted_plot(
    y_test: pd.Series,
    predictions: np.ndarray,
    model_name: str,
    file_name: str,
) -> None:
    """Grafica valores reales contra valores predichos."""
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.25, s=8)
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="red")
    plt.title(f"PM2.5 real vs predicho - {model_name}")
    plt.xlabel("PM2.5 real")
    plt.ylabel("PM2.5 predicho")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=160)
    plt.close()


def save_residual_plot(
    y_test: pd.Series,
    predictions: np.ndarray,
    model_name: str,
    file_name: str,
) -> None:
    """Grafica residuos para revisar errores del modelo."""
    residuals = y_test - predictions
    plt.figure(figsize=(7, 5))
    plt.scatter(predictions, residuals, alpha=0.25, s=8)
    plt.axhline(0, color="red", linewidth=1)
    plt.title(f"Residuos - {model_name}")
    plt.xlabel("PM2.5 predicho")
    plt.ylabel("Residuo")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=160)
    plt.close()


def save_bar_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    file_name: str,
    top_n: int = 15,
) -> None:
    """Grafica barras horizontales para coeficientes o importancias."""
    plot_df = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df[x_column], plot_df[y_column])
    plt.title(title)
    plt.xlabel(y_column)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=160)
    plt.close()


def save_metric_comparison_plot(metrics_df: pd.DataFrame) -> None:
    """Grafica la comparacion de metricas principales."""
    metric_columns = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, metric_columns):
        axis.bar(metrics_df["Modelo"], metrics_df[metric])
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Comparacion de modelos")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparacion_metricas.png", dpi=160)
    plt.close()


def save_exploratory_plots(df: pd.DataFrame) -> None:
    """Genera graficos exploratorios del problema ambiental."""
    plt.figure(figsize=(8, 5))
    df[TARGET].hist(bins=50)
    plt.title("Distribucion de PM2.5")
    plt.xlabel("PM2.5")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distribucion_pm25.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    df.groupby("HORA")[TARGET].mean().plot(marker="o")
    plt.title("PM2.5 promedio por hora")
    plt.xlabel("Hora")
    plt.ylabel("PM2.5 promedio")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pm25_promedio_hora.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    df.groupby("ESTACION")[TARGET].mean().sort_values().plot(kind="bar")
    plt.title("PM2.5 promedio por estacion")
    plt.xlabel("Estacion")
    plt.ylabel("PM2.5 promedio")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pm25_promedio_estacion.png", dpi=160)
    plt.close()


def save_tree_visualization(pipeline: Pipeline) -> None:
    """Guarda una visualizacion compacta del arbol de decision."""
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Decision Tree Regressor para PM2.5 (primeros niveles)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "arbol_decision_pm25.png", dpi=160)
    plt.close()

    tree_rules = export_text(model, feature_names=feature_names, max_depth=4)
    (TABLES_DIR / "reglas_arbol_pm25.txt").write_text(
        tree_rules,
        encoding="utf-8",
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convierte un DataFrame pequeno a tabla Markdown sin dependencias extra."""
    table_df = df.copy()
    headers = list(table_df.columns)
    rows = table_df.astype(str).values.tolist()
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_experiment() -> dict:
    """Ejecuta el experimento completo y guarda artefactos."""
    make_output_dirs()
    plt.style.use("seaborn-v0_8-whitegrid")

    raw_df = load_dataset()
    clean_df, metadata = prepare_dataset(raw_df)
    x, y = split_features_target(clean_df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    lasso_pipeline = build_lasso_pipeline()
    tree_pipeline = build_tree_pipeline()

    lasso_metrics, lasso_predictions = evaluate_model(
        "Lasso Regression",
        lasso_pipeline,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    tree_metrics, tree_predictions = evaluate_model(
        "Decision Tree Regressor",
        tree_pipeline,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    metrics_df = pd.DataFrame([lasso_metrics, tree_metrics])
    metrics_df.to_csv(TABLES_DIR / "metricas_modelos.csv", index=False)
    (TABLES_DIR / "metricas_modelos.md").write_text(
        dataframe_to_markdown(metrics_df.round(4)),
        encoding="utf-8",
    )

    lasso_coef_df = extract_lasso_coefficients(lasso_pipeline)
    tree_importance_df = extract_tree_importance(tree_pipeline)
    lasso_coef_df.to_csv(TABLES_DIR / "coeficientes_lasso.csv", index=False)
    tree_importance_df.to_csv(
        TABLES_DIR / "importancia_variables_arbol.csv",
        index=False,
    )

    save_exploratory_plots(clean_df)
    save_actual_vs_predicted_plot(
        y_test,
        lasso_predictions,
        "Lasso Regression",
        "lasso_real_vs_predicho.png",
    )
    save_actual_vs_predicted_plot(
        y_test,
        tree_predictions,
        "Decision Tree Regressor",
        "arbol_real_vs_predicho.png",
    )
    save_residual_plot(
        y_test,
        lasso_predictions,
        "Lasso Regression",
        "lasso_residuos.png",
    )
    save_residual_plot(
        y_test,
        tree_predictions,
        "Decision Tree Regressor",
        "arbol_residuos.png",
    )
    save_bar_plot(
        lasso_coef_df,
        "Variable",
        "Coeficiente",
        "Coeficientes principales de Lasso",
        "coeficientes_lasso.png",
    )
    save_bar_plot(
        tree_importance_df,
        "Variable",
        "Importancia",
        "Importancia de variables del arbol",
        "importancia_variables_arbol.png",
    )
    save_metric_comparison_plot(metrics_df)
    save_tree_visualization(tree_pipeline)

    joblib.dump(lasso_pipeline, MODELS_DIR / "modelo_lasso_pm25.pkl")
    joblib.dump(tree_pipeline, MODELS_DIR / "modelo_arbol_pm25.pkl")

    tree_model = tree_pipeline.named_steps["model"]
    tree_features = get_feature_names(tree_pipeline)
    root_feature_index = int(tree_model.tree_.feature[0])
    root_feature = tree_features[root_feature_index]
    root_threshold = float(tree_model.tree_.threshold[0])

    summary = {
        "metadata": metadata,
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "metrics": metrics_df.to_dict(orient="records"),
        "top_lasso_coefficients": lasso_coef_df.head(10).to_dict(
            orient="records"
        ),
        "top_tree_importances": tree_importance_df.head(10).to_dict(
            orient="records"
        ),
        "tree_root_feature": root_feature,
        "tree_root_threshold": root_threshold,
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "target_std": float(y.std()),
    }
    (OUTPUT_DIR / "resumen_ejecucion.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
