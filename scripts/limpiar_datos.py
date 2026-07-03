from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_DIR / "data" / "raw" / "datos_horarios_contaminacion_lima.xlsx"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TARGET = "PM 2.5"
POLLUTANTS = ["PM 10", "PM 2.5", "SO2", "NO2", "O3", "CO"]
PREDICTOR_POLLUTANTS = ["PM 10", "SO2", "NO2", "O3", "CO"]
NUMERIC_FEATURES = ["PM 10", "SO2", "NO2", "O3", "CO", "ANO", "MES", "DIA", "HORA"]
CATEGORICAL_FEATURES = ["ESTACION"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DUPLICATE_KEYS = ["CODIGO ESTACION", "ANO", "MES", "DIA", "HORA"]


def load_raw_data(path: Path = RAW_DATA) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    for column in POLLUTANTS:
        clean_df[column] = clean_df[column].astype(str).str.replace(",", ".", regex=False)
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df.drop_duplicates(subset=DUPLICATE_KEYS).copy()
    clean_df = clean_df.sort_values(DUPLICATE_KEYS)

    for column in PREDICTOR_POLLUTANTS:
        clean_df[column] = clean_df.groupby("CODIGO ESTACION")[column].transform(
            lambda values: values.interpolate(limit_direction="both")
        )
        clean_df[column] = clean_df[column].fillna(
            clean_df.groupby(["CODIGO ESTACION", "HORA"])[column].transform("median")
        )
        clean_df[column] = clean_df[column].fillna(clean_df[column].median())

    clean_df = clean_df.dropna(subset=[TARGET]).copy()
    clean_df = clean_df.dropna(subset=FEATURE_COLUMNS)
    return clean_df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    model_df = clean_df[FEATURE_COLUMNS + [TARGET]].copy()

    clean_df.to_csv(PROCESSED_DIR / "datos_limpios.csv", index=False)
    model_df.to_csv(PROCESSED_DIR / "datos_modelo.csv", index=False)

    print(f"Datos originales: {raw_df.shape}")
    print(f"Datos limpios: {clean_df.shape}")
    print(f"Archivo generado: {PROCESSED_DIR / 'datos_limpios.csv'}")


if __name__ == "__main__":
    main()
