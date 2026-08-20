from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.features import build_preprocessor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42


def build_logistic():
    """Baseline: preprocesamiento + escalado + regresión logística.
    class_weight='balanced' compensa el desbalance del 5%."""
    return Pipeline([
        ("pre", build_preprocessor()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])



def build_xgb(scale_pos_weight=None, params=None):
    """XGBoost: preprocesamiento + XGBoost. Sin escalado (los árboles son invariantes a la escala).
    scale_pos_weight compensa el desbalance.`params` permite pasar hiperparámetros a medida (p. ej. los que encuentra Optuna)"""
    base = dict(n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=1)
    if params:
        base.update(params)
    base["scale_pos_weight"] = scale_pos_weight
    return Pipeline([("pre", build_preprocessor()),
                     ("clf", XGBClassifier(**base))])


# Se guardan los parametros que se obtuvieron con Optuna para XGBoost, para poder reproducir el modelo final sin necesidad de reoptimizar.
BEST_XGB_PARAMS = {
    'n_estimators': 641, 'learning_rate': 0.03408884311323006, 'max_depth': 4, 
    'subsample': 0.9774081184289698, 'colsample_bytree': 0.8800318732815392, 
    'min_child_weight': 7, 'gamma': 4.370758204210594, 'reg_lambda': 8.848268281287996
}


def build_lgbm(scale_pos_weight=None, params=None):
    """LightGBM (otro gradient boosting) para comparación."""
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, bagging_freq=1,
                random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    if params:
        base.update(params)
    base["scale_pos_weight"] = scale_pos_weight
    return Pipeline([("pre", build_preprocessor()),
                     ("clf", LGBMClassifier(**base))])

# Se guardan los parametros que se obtuvieron con Optuna para LightGBM, para poder reproducir el modelo final sin necesidad de reoptimizar.
BEST_LGBM_PARAMS = {
    'n_estimators': 621, 'learning_rate': 0.029494148336943186, 'num_leaves': 16, 
    'max_depth': 12, 'min_child_samples': 23, 'feature_fraction': 0.7826173426603851, 
    'bagging_fraction': 0.8099926423362572, 'reg_lambda': 0.10243918246367029
}



def build_rf():
    """Random Forest (bagging) para comparación."""
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=1)),
    ])