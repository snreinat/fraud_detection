"""Carga y limpieza del dataset."""

import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "dataset.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Carga del CSV crudo parseando la fecha."""
    return pd.read_csv(path, parse_dates=["fecha"])

# Se descarta la coluna k para el entrenamineto del modelo, ya que se identifico que es uniforme, único por fila y sin correlación.
DROP_COLS = ["k"]   


def clean(df):
    """Limpieza base: descarta columnas sin señal y trata faltantes informativos."""
    df = df.copy()
    df = df.drop(columns=DROP_COLS)          # descarta 'k' (ruido)
    df["o"] = df["o"].fillna("VACIO")        # el vacío de 'o' como categoría propia

    df["fecha"] = pd.to_datetime(df["fecha"])  # asegura tipo fecha
    df["hora"] = df["fecha"].dt.hour           # señal temporal (nueva feature)

    return df