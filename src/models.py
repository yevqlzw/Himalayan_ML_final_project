"""Baselines, CatBoost / LightGBM (+ Optuna tuning), probability calibration (Student B).

All fitting uses rows from fit_frame(train) (anomalous seasons dropped). Early stopping NEVER looks at val:
it uses one expedition-grouped fold inside train (group_kfold(...)[0]).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import CATEGORICAL, FEATURES
from src.split import GROUP, RANDOM_STATE, TARGET, group_kfold

# YEAR is dropped from the model input: val/test years lie outside the training range (drift, not signal).
# To run the ablation, pass features=FEATURES to any function below.
MODEL_FEATURES = [f for f in FEATURES if f != "YEAR"]
HINGE_AGE = 40


# ----------------------------------------------------------------------------- data preparation
def _num(s: pd.Series) -> pd.Series:
    """bool / boolean / Int64 / float -> float64 with NaN for missing."""
    return pd.Series(s.astype("Float64").to_numpy(dtype=float, na_value=np.nan), index=s.index)


def _cat(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("NA").astype(str)


def to_frame(df: pd.DataFrame, features: list[str] = MODEL_FEATURES) -> pd.DataFrame:
    """Categoricals -> str ('NA' for missing), everything else -> float. Index is kept."""
    return pd.DataFrame({c: (_cat(df[c]) if c in CATEGORICAL else _num(df[c])) for c in features}, index=df.index)


def target(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET].astype(int).to_numpy()


def cat_names(features: list[str]) -> list[str]:
    return [c for c in features if c in CATEGORICAL]


def fit_categories(train: pd.DataFrame, features: list[str] = MODEL_FEATURES) -> dict:
    return {c: sorted(_cat(train[c]).unique()) for c in cat_names(features)}


def to_lgbm_frame(df: pd.DataFrame, features: list[str], categories: dict) -> pd.DataFrame:
    X = to_frame(df, features)
    for c, cats in categories.items():
        X[c] = pd.Categorical(X[c], categories=cats)  # level unseen in train -> NaN
    return X


class Fitted:
    """A fitted model that predicts P(summit) straight from a features_v1 frame."""

    def __init__(self, model, features, kind, categories=None, best_iteration=None):
        self.model, self.features, self.kind = model, list(features), kind
        self.categories, self.best_iteration = categories, best_iteration

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "lightgbm":
            X = to_lgbm_frame(df, self.features, self.categories)
        else:
            X = to_frame(df, self.features)
        return self.model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- baselines (a), (b)
def constant_baseline(train_fit: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), target(train_fit).mean())


class PeakSeasonRate:
    """Historical success rate of (peak, season), computed from the fitting rows only.

    For val/test every fitted row is from an earlier year, so only the past is used. Shrinkage:
    global rate -> peak rate -> (peak, season) rate, each with k pseudo-observations (k=20, as PEAK_SUCCESS_PRIOR).
    """

    def __init__(self, k: float = 20):
        self.k = k

    @staticmethod
    def _key(df):
        return df["PEAKID"].astype(str) + "|" + df["SEASON"].astype(str)

    def fit(self, train_fit: pd.DataFrame):
        y = pd.Series(target(train_fit), index=train_fit.index)
        self.g = float(y.mean())
        peak = y.groupby(train_fit["PEAKID"].astype(str)).agg(["sum", "count"])
        self.peak_rate = (peak["sum"] + self.k * self.g) / (peak["count"] + self.k)
        self.ps = y.groupby(self._key(train_fit)).agg(["sum", "count"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        peak = df["PEAKID"].astype(str).map(self.peak_rate).fillna(self.g).to_numpy(float)
        key = self._key(df)
        s = key.map(self.ps["sum"]).fillna(0).to_numpy(float)
        n = key.map(self.ps["count"]).fillna(0).to_numpy(float)
        return (s + self.k * peak) / (n + self.k)


def peak_success_prior_baseline(df: pd.DataFrame) -> np.ndarray:
    """The existing feature used directly as a probability (peak-level, not season-specific)."""
    return df["PEAK_SUCCESS_PRIOR"].to_numpy(float)


# ----------------------------------------------------------------------------- (c) logistic regression
def _add_age_hinge(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["AGE_OVER_40"] = (X["CALCAGE"] - HINGE_AGE).clip(lower=0)
    return X


def make_logreg(features: list[str] = MODEL_FEATURES, C: float = 1.0) -> Pipeline:
    """Linear age + hinge max(age-40, 0) (piecewise-linear age effect), standardised numerics, one-hot categoricals."""
    cats = cat_names(features)
    nums = [f for f in features if f not in cats] + (["AGE_OVER_40"] if "CALCAGE" in features else [])
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), nums),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cats),
    ])
    return Pipeline([("hinge", FunctionTransformer(_add_age_hinge if "CALCAGE" in features else None)),
                     ("pre", pre),
                     ("clf", LogisticRegression(C=C, max_iter=3000, random_state=RANDOM_STATE))])


def fit_logreg(train_fit: pd.DataFrame, features: list[str] = MODEL_FEATURES, C: float = 1.0) -> Fitted:
    model = make_logreg(features, C).fit(to_frame(train_fit, features), target(train_fit))
    return Fitted(model, features, "sklearn")


# ----------------------------------------------------------------------------- (d) CatBoost
def es_split(train_fit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Early-stopping split INSIDE train: fold 0 of the expedition-grouped 5-fold CV (positional indices)."""
    return group_kfold(train_fit, 5)[0]


def fit_catboost(train_fit: pd.DataFrame, params: dict | None = None,
                 features: list[str] = MODEL_FEATURES) -> Fitted:
    from catboost import CatBoostClassifier

    base = dict(iterations=1000, early_stopping_rounds=100, eval_metric="Logloss",
                random_seed=RANDOM_STATE, thread_count=-1, verbose=0)
    base.update(params or {})
    a, b = es_split(train_fit)
    X, y = to_frame(train_fit, features), target(train_fit)
    model = CatBoostClassifier(**base)
    model.fit(X.iloc[a], y[a], cat_features=cat_names(features), eval_set=(X.iloc[b], y[b]))
    return Fitted(model, features, "catboost", best_iteration=model.get_best_iteration())


def _score(y, p, metric: str) -> float:
    return log_loss(y, p, labels=[0, 1]) if metric == "logloss" else roc_auc_score(y, p)


def tune_catboost(train_fit: pd.DataFrame, n_trials: int = 30, metric: str = "logloss",
                  features: list[str] = MODEL_FEATURES, n_splits: int = 5, max_iter: int = 500):
    """Optuna TPE (seed 42). Objective = mean over expedition-grouped folds of val-fold log loss (or AUC).
    Speed cap: iterations <= max_iter (500), early stopping 50 on the fold itself, thread_count=-1."""
    import optuna
    from catboost import CatBoostClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y, cats = to_frame(train_fit, features), target(train_fit), cat_names(features)
    folds = [(X.iloc[a], y[a], X.iloc[b], y[b]) for a, b in group_kfold(train_fit, n_splits)]

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            random_strength=trial.suggest_float("random_strength", 0.5, 5.0, log=True),
        )
        scores, iters = [], []
        for Xa, ya, Xb, yb in folds:
            m = CatBoostClassifier(iterations=max_iter, early_stopping_rounds=50, eval_metric="Logloss",
                                   random_seed=RANDOM_STATE, thread_count=-1, verbose=0, **params)
            m.fit(Xa, ya, cat_features=cats, eval_set=(Xb, yb))
            scores.append(_score(yb, m.predict_proba(Xb)[:, 1], metric))
            iters.append(m.get_best_iteration())
        trial.set_user_attr("mean_best_iteration", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize" if metric == "logloss" else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study


def catboost_params_from(study, max_iter: int = 500) -> dict:
    return {**study.best_params, "iterations": max_iter, "early_stopping_rounds": 50}


# ----------------------------------------------------------------------------- LightGBM
def _lgbm(params: dict | None, max_iter: int):
    import lightgbm as lgb

    base = dict(n_estimators=max_iter, objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    base.update(params or {})
    return lgb, lgb.LGBMClassifier(**base)


def fit_lightgbm(train_fit: pd.DataFrame, params: dict | None = None,
                 features: list[str] = MODEL_FEATURES, max_iter: int = 500) -> Fitted:
    lgb, model = _lgbm(params, max_iter)
    cats = fit_categories(train_fit, features)
    a, b = es_split(train_fit)
    X, y = to_lgbm_frame(train_fit, features, cats), target(train_fit)
    model.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return Fitted(model, features, "lightgbm", categories=cats, best_iteration=model.best_iteration_)


def tune_lightgbm(train_fit: pd.DataFrame, n_trials: int = 12, metric: str = "logloss",
                  features: list[str] = MODEL_FEATURES, n_splits: int = 5, max_iter: int = 500):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cats = fit_categories(train_fit, features)
    X, y = to_lgbm_frame(train_fit, features, cats), target(train_fit)
    folds = [(X.iloc[a], y[a], X.iloc[b], y[b]) for a, b in group_kfold(train_fit, n_splits)]

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 96),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            cat_smooth=trial.suggest_float("cat_smooth", 5.0, 100.0, log=True),
            min_data_per_group=trial.suggest_int("min_data_per_group", 20, 200),
        )
        scores, iters = [], []
        for Xa, ya, Xb, yb in folds:
            lgb, m = _lgbm(params, max_iter)
            m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            scores.append(_score(yb, m.predict_proba(Xb)[:, 1], metric))
            iters.append(m.best_iteration_)
        trial.set_user_attr("mean_best_iteration", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize" if metric == "logloss" else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study


# ----------------------------------------------------------------------------- calibration
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """method='platt' (logistic regression on the logit of p) or 'isotonic'. Outputs are clipped to [0.001, 0.999]."""

    def __init__(self, method: str):
        assert method in ("platt", "isotonic")
        self.method = method

    def fit(self, p, y):
        if self.method == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        else:
            self.m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.asarray(p, float), y)
        return self

    def predict(self, p) -> np.ndarray:
        out = (self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1] if self.method == "platt"
               else self.m.predict(np.asarray(p, float)))
        return np.clip(out, 1e-3, 1 - 1e-3)


def crossfit_calibrate(val: pd.DataFrame, p, method: str, fit_rows=None, seed: int = RANDOM_STATE) -> np.ndarray:
    """Calibrated probabilities for ALL val rows without peeking at their own labels.

    Val expeditions are split into two random halves; the calibrator learned on one half is applied to the other.
    fit_rows (bool array) limits which rows may be used to learn a calibrator (e.g. exclude anomalous seasons).
    """
    p, y = np.asarray(p, float), target(val)
    ok = np.ones(len(val), bool) if fit_rows is None else np.asarray(fit_rows, bool)
    ids = val[GROUP].unique()
    half = set(np.random.default_rng(seed).permutation(ids)[: len(ids) // 2])
    in_a = val[GROUP].isin(half).to_numpy()
    out = np.empty(len(val))
    for learn in (in_a, ~in_a):
        cal = Calibrator(method).fit(p[learn & ok], y[learn & ok])
        out[~learn] = cal.predict(p[~learn])
    return out
