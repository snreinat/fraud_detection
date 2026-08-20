
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

NUM_FEATURES = ["b", "c", "d", "e", "f", "h", "l", "m", "monto", "score", "hora"]
CAT_LOW  = ["a", "g", "n", "o", "p"]   # baja cardinalidad → one-hot
CAT_HIGH = ["j"]                        # alta cardinalidad → target encoding


def temporal_split(df, date_col="fecha", train_frac=0.7, val_frac=0.15):
    """Split temporal (out-of-time): ordena por fecha y corta en el tiempo.
    Entrena con lo viejo, evalúa con lo nuevo (simula producción).
    Devuelve (train, val, test)."""
    df_ordenado = df.sort_values(date_col)
    n = len(df_ordenado)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df_ordenado.iloc[:n_train]
    val = df_ordenado.iloc[n_train:n_train + n_val]
    test = df_ordenado.iloc[n_train + n_val:]
    return train, val, test




def build_preprocessor(num=None, low=None, high=None):
    """Preprocesador que se ajusta SOLO en train (evita fugas).
    - Numéricas: imputación por mediana.
    - Cat. baja cardinalidad: imputación + one-hot (agrupa las raras).
    - Cat. alta cardinalidad (j): target encoding con validación cruzada.
    Preprocesador ajustado solo en train. num/low/high permiten variar las
    columnas (p. ej. para  ablation)."""
    num  = NUM_FEATURES if num  is None else num
    low  = CAT_LOW      if low  is None else low
    high = CAT_HIGH     if high is None else high

    num_t = SimpleImputer(strategy="median")
    cat_low = Pipeline([
        ("imput", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=0.01, sparse_output=False)),
    ])
    cat_high = TargetEncoder(target_type="binary", random_state=42)

    return ColumnTransformer([
        ("num", num_t, num),
        ("cat_low", cat_low, low),
        ("cat_high", cat_high, high),
    ])