from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.features import build_preprocessor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42

"""Baseline: preprocesamiento + escalado + regresión logística.
    class_weight='balanced' compensa el desbalance del 5%."""

def build_logistic():
    return Pipeline([
        ("pre", build_preprocessor()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])



"""XGBoost: preprocesamiento + XGBoost. Sin escalado (los árboles son invariantes a la escala).
    scale_pos_weight compensa el desbalance.`params` permite pasar hiperparámetros a medida (p. ej. los que encuentra Optuna)"""
def build_xgb(scale_pos_weight=None, params=None):

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
    'n_estimators': 664, 'learning_rate': 0.0286642261344634, 'max_depth': 4, 
    'subsample': 0.6756782218133349, 'colsample_bytree': 0.7922339615553947, 
    'min_child_weight': 5, 'gamma': 2.601477385704998, 'reg_lambda': 4.524540905246321
}



"""LightGBM (otro gradient boosting) para comparación."""
def build_lgbm(scale_pos_weight=None, params=None):
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, bagging_freq=1,
                random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    if params:
        base.update(params)
    base["scale_pos_weight"] = scale_pos_weight
    return Pipeline([("pre", build_preprocessor()),
                     ("clf", LGBMClassifier(**base))])



"""Random Forest (bagging) para comparación."""
def build_rf():
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=1)),
    ])