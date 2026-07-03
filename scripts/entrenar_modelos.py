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
LASSO_FEATURES = NUMERIC_FEATURES
TREE_FEATURES = FEATURE_COLUMNS
PRIMARY_COLOR = "#28684d"
SECONDARY_COLOR = "#4f8f7a"
ACCENT_COLOR = "#b44b35"
GRID_COLOR = "#dce4df"
TEXT_COLOR = "#18211d"


def style_axis(axis) -> None:
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
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
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
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "modelo": name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
    }


def save_prediction_plot(y_test: pd.Series, predictions: np.ndarray, title: str, file_name: str) -> None:
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


def save_feature_importance(tree_model: Pipeline) -> list[dict]:
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
    (REPORTS_DIR / "reglas_decision_tree.txt").write_text(rules, encoding="utf-8")

    root_feature_index = int(estimator.tree_.feature[0])
    root_feature = feature_names[root_feature_index]
    root_threshold = float(estimator.tree_.threshold[0])
    return {
        "tree_root_feature": root_feature,
        "tree_root_threshold": root_threshold,
        "tree_rules": rule_lines[:18],
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = PROCESSED_DIR / "datos_modelo.csv"
    if clean_path.exists():
        model_df = pd.read_csv(clean_path)
    else:
        raw_df = load_raw_data()
        clean_df = clean_data(raw_df)
        clean_df.to_csv(PROCESSED_DIR / "datos_limpios.csv", index=False)
        model_df = clean_df[FEATURE_COLUMNS + [TARGET]].copy()
        model_df.to_csv(clean_path, index=False)

    x = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    pd.concat([x_test, y_test], axis=1).to_csv(TEST_DIR / "datos_prueba.csv", index=False)

    lasso = build_lasso_model()
    tree = build_tree_model()
    lasso.fit(x_train[LASSO_FEATURES], y_train)
    tree.fit(x_train, y_train)

    joblib.dump(lasso, MODELS_DIR / "lasso_regression.pkl")
    joblib.dump(tree, MODELS_DIR / "decision_tree_regressor.pkl")

    lasso_predictions = lasso.predict(x_test[LASSO_FEATURES])
    tree_predictions = tree.predict(x_test)
    metrics = [
        evaluate("Lasso Regression", lasso, x_test[LASSO_FEATURES], y_test),
        evaluate("Decision Tree Regressor", tree, x_test, y_test),
    ]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(REPORTS_DIR / "metricas_modelos.csv", index=False)
    metrics_df.to_excel(REPORTS_DIR / "resultados.xlsx", index=False)

    save_distribution(model_df)
    save_prediction_plot(y_test, lasso_predictions, "PM2.5 real vs predicho - Lasso", "lasso_real_vs_predicho.png")
    save_prediction_plot(y_test, tree_predictions, "PM2.5 real vs predicho - Decision Tree", "tree_real_vs_predicho.png")
    top_importance = save_feature_importance(tree)
    save_metrics_plot(metrics_df)
    tree_decision = save_tree_decision_view(tree)

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
        **tree_decision,
    }
    (REPORTS_DIR / "metricas_modelos.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
