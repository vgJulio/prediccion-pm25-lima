from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_DIR / "data" / "raw" / "datos_horarios_contaminacion_lima.xlsx"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# Variable que queremos predecir con los modelos.
TARGET = "PM 2.5"

# Columnas de contaminantes que llegan desde el Excel original.
POLLUTANTS = ["PM 10", "PM 2.5", "SO2", "NO2", "O3", "CO"]

# Contaminantes que se usan como entrada para estimar PM 2.5.
PREDICTOR_POLLUTANTS = ["PM 10", "SO2", "NO2", "O3", "CO"]

# Variables numericas usadas por los modelos.
NUMERIC_FEATURES = ["PM 10", "SO2", "NO2", "O3", "CO", "ANO", "MES", "DIA", "HORA"]

# Variable de texto: representa la estacion de monitoreo.
CATEGORICAL_FEATURES = ["ESTACION"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Estas columnas identifican una medicion horaria unica por estacion.
DUPLICATE_KEYS = ["CODIGO ESTACION", "ANO", "MES", "DIA", "HORA"]


def load_raw_data(path: Path = RAW_DATA) -> pd.DataFrame:
    # Carga el dataset sucio original conservando el nombre del archivo.
    return pd.read_excel(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Se trabaja sobre una copia para no modificar el DataFrame original.
    clean_df = df.copy()

    # Convierte valores con coma decimal a numeros reales.
    for column in POLLUTANTS:
        clean_df[column] = clean_df[column].astype(str).str.replace(",", ".", regex=False)
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    # Elimina registros repetidos de la misma estacion, fecha y hora.
    clean_df = clean_df.drop_duplicates(subset=DUPLICATE_KEYS).copy()
    clean_df = clean_df.sort_values(DUPLICATE_KEYS)

    # Rellena datos faltantes de predictores sin inventar PM 2.5.
    for column in PREDICTOR_POLLUTANTS:
        # Primero interpola dentro de cada estacion para respetar su comportamiento.
        clean_df[column] = clean_df.groupby("CODIGO ESTACION")[column].transform(
            lambda values: values.interpolate(limit_direction="both")
        )
        # Si aun faltan datos, usa la mediana de la misma estacion y hora.
        clean_df[column] = clean_df[column].fillna(
            clean_df.groupby(["CODIGO ESTACION", "HORA"])[column].transform("median")
        )
        # Ultimo respaldo: mediana general de la columna.
        clean_df[column] = clean_df[column].fillna(clean_df[column].median())

    # PM 2.5 es la variable objetivo; si falta, esa fila no sirve para entrenar.
    clean_df = clean_df.dropna(subset=[TARGET]).copy()

    # Quita filas que aun no tengan todas las variables de entrada necesarias.
    clean_df = clean_df.dropna(subset=FEATURE_COLUMNS)
    return clean_df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)

    # Este archivo queda listo para entrenar: entradas del modelo + PM 2.5.
    model_df = clean_df[FEATURE_COLUMNS + [TARGET]].copy()

    clean_df.to_csv(PROCESSED_DIR / "datos_limpios.csv", index=False)
    model_df.to_csv(PROCESSED_DIR / "datos_modelo.csv", index=False)

    print(f"Datos originales: {raw_df.shape}")
    print(f"Datos limpios: {clean_df.shape}")
    print(f"Archivo generado: {PROCESSED_DIR / 'datos_limpios.csv'}")


if __name__ == "__main__":
    main()
