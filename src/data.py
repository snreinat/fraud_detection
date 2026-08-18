"""Carga y limpieza del dataset."""

import pandas as pd

RAW_PATH = "data/dataset.csv"

#Carga del CSV crudo parseando la fecha.
def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["fecha"])

# Se descarta la coluna k para el entrenamineto del modelo, ya que se identifico que es uniforme, único por fila y sin correlación.
DROP_COLS = ["k"]   


#Limpieza estructural sin aprender del train.
#Descarta ruido y trata el vacío de 'o' como categoría propia.
def clean(df):
    df = df.copy()
    df = df.drop(columns=DROP_COLS)
    df["o"] = df["o"].fillna("VACIO")  
    return df