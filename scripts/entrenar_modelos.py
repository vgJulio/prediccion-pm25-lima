from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from limpiar_datos import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    TARGET,
    clean_data,
    load_raw_data,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
IMAGES_DIR = REPORTS_DIR / "imagenes"
TEST_DIR = PROJECT_DIR / "data" / "test"

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Lasso se entrena solo con variables numericas.
# La columna ESTACION es texto y se deja fuera para mantener el modelo lineal simple.
LASSO_FEATURES = NUMERIC_FEATURES

# Decision Tree usa numericas + ESTACION, porque la estacion se codifica a numeros.
TREE_FEATURES = FEATURE_COLUMNS

# Colores reutilizados para que todos los graficos tengan el mismo estilo visual.
PRIMARY_COLOR = "#28684d"
SECONDARY_COLOR = "#4f8f7a"
ACCENT_COLOR = "#b44b35"
GRID_COLOR = "#dce4df"
TEXT_COLOR = "#18211d"


def style_axis(axis) -> None:
    # Aplica el estilo visual comun a los graficos exportados al dashboard.
    axis.set_facecolor("#ffffff")
    axis.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#b8c8bf")
    axis.spines["bottom"].set_color("#b8c8bf")
    axis.tick_params(colors="#405148")
    axis.title.set_color(TEXT_COLOR)
    axis.xaxis.label.set_color("#405148")
    axis.yaxis.label.set_color("#405148")


def one_hot_encoder() -> OneHotEncoder:
    # OneHotEncoder convierte texto como ESTACION en columnas numericas de 0 y 1.
    # El try permite que funcione con versiones nuevas y antiguas de scikit-learn.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    # Prepara las columnas antes de entrenar:
    # - Numericas: rellena faltantes con la mediana.
    # - Categoricas: rellena faltantes y convierte texto con OneHotEncoder.
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        # Lasso necesita escalado porque sus coeficientes dependen de la magnitud.
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", one_hot_encoder()),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_lasso_model() -> Pipeline:
    # Pipeline de Lasso: primero limpia/escala datos numericos y luego entrena.
    # No se incluye ESTACION porque Lasso trabaja mejor aqui con variables numericas directas.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                LASSO_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Lasso(alpha=0.05, max_iter=10000, random_state=RANDOM_STATE)),
        ]
    )


def build_tree_model() -> Pipeline:
    # Pipeline del arbol: usa numericas y ESTACION codificada.
    # max_depth y min_samples_leaf evitan que el arbol memorice demasiado el dataset.
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=8,
                    min_samples_leaf=100,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    # Calcula las metricas con datos de prueba que el modelo no vio al entrenar.
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "modelo": name,
        # MAE: error absoluto promedio. Menor es mejor.
        "mae": float(mean_absolute_error(y_test, predictions)),
        # MSE: error cuadratico medio. Penaliza mas los errores grandes.
        "mse": float(mse),
        # RMSE: raiz del MSE. Esta en la misma unidad de PM 2.5. Menor es mejor.
        "rmse": float(np.sqrt(mse)),
        # R2: proporcion de variacion explicada por el modelo. Mayor es mejor.
        "r2": float(r2_score(y_test, predictions)),
    }


def save_prediction_plot(y_test: pd.Series, predictions: np.ndarray, title: str, file_name: str) -> None:
    # Grafico de puntos: compara PM 2.5 real contra PM 2.5 predicho.
    fig, axis = plt.subplots(figsize=(7.6, 5.4), facecolor="#f7faf8")
    axis.scatter(
        y_test,
        predictions,
        alpha=0.22,
        s=10,
        color=PRIMARY_COLOR,
        edgecolors="none",
    )
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    axis.plot(
        [min_value, max_value],
        [min_value, max_value],
        color=ACCENT_COLOR,
        linewidth=2.2,
        label="Referencia ideal",
    )
    axis.set_title(title, fontsize=14, fontweight="bold", pad=12)
    axis.set_xlabel("PM2.5 real")
    axis.set_ylabel("PM2.5 predicho")
    axis.legend(frameon=False, loc="upper left")
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / file_name, dpi=170, bbox_inches="tight")
    plt.close()


def save_lasso_equation(lasso_model: Pipeline) -> dict:
    # Arma la ecuacion lineal de Lasso: PM2.5 = intercepto + suma(coeficiente * variable escalada).
    # Los coeficientes salen escalados (StandardScaler), por eso no se pueden usar directo con los
    # valores originales del formulario; el dashboard debe aclarar que son "variables escaladas".
    feature_names = lasso_model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = lasso_model.named_steps["model"].coef_
    intercept = float(lasso_model.named_steps["model"].intercept_)
    alpha = float(lasso_model.named_steps["model"].alpha)

    terms = [
        {"variable": name, "coeficiente": float(value)}
        for name, value in zip(feature_names, coefficients)
    ]
    # Ordena por peso absoluto para mostrar primero las variables mas influyentes.
    terms.sort(key=lambda term: abs(term["coeficiente"]), reverse=True)

    return {
        "intercepto": intercept,
        "alpha": alpha,
        "terminos": terms,
        # Cuenta cuantas variables Lasso "apago" (coeficiente en cero), efecto tipico de la
        # penalizacion L1.
        "variables_en_cero": sum(1 for term in terms if term["coeficiente"] == 0.0),
    }


def save_feature_importance(tree_model: Pipeline) -> list[dict]:
    # Extrae cuales variables influyeron mas en las decisiones del arbol.
    preprocessor = tree_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = tree_model.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"variable": feature_names, "importancia": importances})
    importance_df = importance_df.sort_values("importancia", ascending=False).head(12)

    fig, axis = plt.subplots(figsize=(8.4, 5.8), facecolor="#f7faf8")
    values = importance_df["importancia"].iloc[::-1]
    variables = importance_df["variable"].iloc[::-1]
    bars = axis.barh(variables, values, color=PRIMARY_COLOR)
    axis.set_title("Importancia de variables - Decision Tree", fontsize=14, fontweight="bold", pad=12)
    axis.set_xlabel("Importancia")
    for bar in bars:
        width = bar.get_width()
        axis.text(
            width + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            color="#405148",
            fontsize=9,
        )
    style_axis(axis)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "feature_importance.png", dpi=170, bbox_inches="tight")
    plt.close()
    return importance_df.to_dict(orient="records")


def save_distribution(clean_df: pd.DataFrame) -> None:
    # Histograma para mostrar como se distribuyen los valores reales de PM 2.5.
    fig, axis = plt.subplots(figsize=(8.2, 5.4), facecolor="#f7faf8")
    axis.hist(
        clean_df[TARGET],
        bins=55,
        color=SECONDARY_COLOR,
        edgecolor="#ffffff",
        linewidth=0.5,
    )
    median_value = clean_df[TARGET].median()
    axis.axvline(
        median_value,
        color=ACCENT_COLOR,
        linewidth=2,
        label=f"Mediana: {median_value:.2f}",
    )
    axis.set_title("Distribucion de PM2.5", fontsize=14, fontweight="bold", pad=12)
    axis.set_xlabel("PM2.5")
    axis.set_ylabel("Frecuencia")
    axis.legend(frameon=False)
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "distribucion_pm25.png", dpi=170, bbox_inches="tight")
    plt.close()


def save_metrics_plot(metrics_df: pd.DataFrame) -> None:
    # Grafico de barras para comparar el RMSE de ambos modelos.
    fig, axis = plt.subplots(figsize=(7.8, 5.2), facecolor="#f7faf8")
    colors = [SECONDARY_COLOR, PRIMARY_COLOR]
    bars = axis.bar(metrics_df["modelo"], metrics_df["rmse"], color=colors[: len(metrics_df)])
    axis.set_title("Comparacion de modelos por RMSE", fontsize=14, fontweight="bold", pad=12)
    axis.set_ylabel("RMSE")
    axis.tick_params(axis="x", rotation=10)
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.25,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            color="#405148",
            fontweight="bold",
        )
    style_axis(axis)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "comparacion_modelos.png", dpi=170, bbox_inches="tight")
    plt.close()


def save_tree_decision_view(tree_model: Pipeline) -> dict:
    # Genera una imagen con los primeros niveles del arbol para explicar su decision.
    estimator = tree_model.named_steps["model"]
    feature_names = list(tree_model.named_steps["preprocessor"].get_feature_names_out())

    fig, axis = plt.subplots(figsize=(15, 7), facecolor="#f7faf8")
    plot_tree(
        estimator,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        max_depth=2,
        fontsize=9,
        ax=axis,
    )
    axis.set_title(
        "Primeras decisiones del Decision Tree",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "decision_tree_decisiones.png", dpi=170, bbox_inches="tight")
    plt.close()

    rules = export_text(estimator, feature_names=feature_names, max_depth=3)
    rule_lines = [line for line in rules.splitlines() if line.strip()]

    # Guarda las reglas en texto para mostrarlas o revisarlas fuera del dashboard.
    (REPORTS_DIR / "reglas_decision_tree.txt").write_text(rules, encoding="utf-8")

    # La raiz es la primera pregunta que hace el Decision Tree.
    root_feature_index = int(estimator.tree_.feature[0])
    root_feature = feature_names[root_feature_index]
    root_threshold = float(estimator.tree_.threshold[0])

    # Datos del modelo matematico del arbol: criterio de division y prediccion por hoja.
    # sklearn calcula cada corte minimizando el MSE de los grupos resultantes; la prediccion
    # de una hoja es el promedio de PM 2.5 de las muestras que caen en ella.
    tree_math = {
        "criterion": estimator.criterion,
        "n_leaves": int(estimator.get_n_leaves()),
        "max_depth_configurado": estimator.max_depth,
        "min_samples_leaf": estimator.min_samples_leaf,
        "root_samples": int(estimator.tree_.n_node_samples[0]),
        "root_prediction": float(estimator.tree_.value[0][0][0]),
    }
    return {
        "tree_root_feature": root_feature,
        "tree_root_threshold": root_threshold,
        "tree_math": tree_math,
        "tree_rules": rule_lines[:18],
    }


def main() -> None:
    # Crea carpetas necesarias si aun no existen.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = PROCESSED_DIR / "datos_modelo.csv"
    if clean_path.exists():
        # Si ya existe el dataset limpio para modelar, se reutiliza.
        model_df = pd.read_csv(clean_path)
    else:
        # Si no existe, se limpia desde el Excel original.
        raw_df = load_raw_data()
        clean_df = clean_data(raw_df)
        clean_df.to_csv(PROCESSED_DIR / "datos_limpios.csv", index=False)
        model_df = clean_df[FEATURE_COLUMNS + [TARGET]].copy()
        model_df.to_csv(clean_path, index=False)

    x = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET]

    # Divide los datos: una parte entrena y otra evalua el desempeno real.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    pd.concat([x_test, y_test], axis=1).to_csv(TEST_DIR / "datos_prueba.csv", index=False)

    lasso = build_lasso_model()
    tree = build_tree_model()

    # Lasso solo recibe las variables numericas definidas en LASSO_FEATURES.
    lasso.fit(x_train[LASSO_FEATURES], y_train)

    # Decision Tree recibe todas las variables, incluida ESTACION.
    tree.fit(x_train, y_train)

    # Guarda los modelos entrenados en .pkl para que el dashboard prediga sin reentrenar.
    joblib.dump(lasso, MODELS_DIR / "lasso_regression.pkl")
    joblib.dump(tree, MODELS_DIR / "decision_tree_regressor.pkl")

    # Predicciones sobre datos de prueba para calcular metricas y graficos.
    lasso_predictions = lasso.predict(x_test[LASSO_FEATURES])
    tree_predictions = tree.predict(x_test)
    metrics = [
        evaluate("Lasso Regression", lasso, x_test[LASSO_FEATURES], y_test),
        evaluate("Decision Tree Regressor", tree, x_test, y_test),
    ]
    metrics_df = pd.DataFrame(metrics)

    # Guarda metricas en CSV/Excel para sustentarlas en el informe o exposicion.
    metrics_df.to_csv(REPORTS_DIR / "metricas_modelos.csv", index=False)
    metrics_df.to_excel(REPORTS_DIR / "resultados.xlsx", index=False)

    save_distribution(model_df)
    save_prediction_plot(y_test, lasso_predictions, "PM2.5 real vs predicho - Lasso", "lasso_real_vs_predicho.png")
    save_prediction_plot(y_test, tree_predictions, "PM2.5 real vs predicho - Decision Tree", "tree_real_vs_predicho.png")
    top_importance = save_feature_importance(tree)
    save_metrics_plot(metrics_df)
    tree_decision = save_tree_decision_view(tree)
    lasso_equation = save_lasso_equation(lasso)

    # JSON central: el dashboard lee este archivo para mostrar metricas, reglas y variables.
    summary = {
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "features_by_model": {
            "Lasso Regression": LASSO_FEATURES,
            "Decision Tree Regressor": TREE_FEATURES,
        },
        "categorical_handling": {
            "Lasso Regression": "No usa ESTACION porque es texto; se entrena solo con variables numericas.",
            "Decision Tree Regressor": "Usa ESTACION transformada con OneHotEncoder dentro del pipeline.",
        },
        "rows": int(model_df.shape[0]),
        "columns": int(model_df.shape[1]),
        "stations": sorted(model_df["ESTACION"].dropna().unique().tolist()),
        "metrics": metrics,
        "top_importance": top_importance,
        "lasso_equation": lasso_equation,
        **tree_decision,
    }
    (REPORTS_DIR / "metricas_modelos.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
