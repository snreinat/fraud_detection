


"""Split temporal (out-of-time): ordena por fecha y corta en el tiempo.
    Entrena con lo viejo, evalúa con lo nuevo (simula producción).
    Devuelve (train, val, test)."""
def temporal_split(df, date_col="fecha", train_frac=0.7, val_frac=0.15):
    df_ordenado = df.sort_values(date_col)
    n = len(df_ordenado)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df_ordenado.iloc[:n_train]
    val = df_ordenado.iloc[n_train:n_train + n_val]
    test = df_ordenado.iloc[n_train + n_val:]
    return train, val, test


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

NUM_FEATURES = ["b", "c", "d", "e", "f", "h", "l", "m", "monto", "score"]
CAT_LOW  = ["a", "g", "n", "o", "p"]   # baja cardinalidad → one-hot
CAT_HIGH = ["j"]                        # alta cardinalidad → target encoding


"""Preprocesador que se ajusta SOLO en train (evita fugas).
    - Numéricas: imputación por mediana.
    - Cat. baja cardinalidad: imputación + one-hot (agrupa las raras).
    - Cat. alta cardinalidad (j): target encoding con validación cruzada.
    """
def build_preprocessor():
    num = SimpleImputer(strategy="median")

    cat_low = Pipeline([
        ("imput", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=0.01, sparse_output=False)),
    ])

    cat_high = TargetEncoder(target_type="binary")

    return ColumnTransformer([
        ("num", num, NUM_FEATURES),
        ("cat_low", cat_low, CAT_LOW),
        ("cat_high", cat_high, CAT_HIGH),
    ])