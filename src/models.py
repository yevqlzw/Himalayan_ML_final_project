import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import FINAL_MODEL_PATH, GROUP, RANDOM_STATE, TARGET
from src.features import CATEGORICAL, FEATURES, to_number
from src.leakage import check_no_leakage
from src.split import group_kfold, time_folds

MODEL_FEATURES = [f for f in FEATURES if f != "YEAR"]
HINGE_AGE = 40
SEEDS = (42, 1, 2, 3, 4)
DEFAULT_CATBOOST = {"learning_rate": 0.1}


def _cat(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("NA").astype(str)


def load_final_config(path: Path = FINAL_MODEL_PATH) -> dict:
    final = json.loads(Path(path).read_text())
    check_no_leakage(pd.DataFrame(columns=final["features"]))
    return final


def to_frame(df: pd.DataFrame, features: list[str] = MODEL_FEATURES) -> pd.DataFrame:
    return pd.DataFrame({c: (_cat(df[c]) if c in CATEGORICAL else to_number(df[c])) for c in features}, index=df.index)


def target(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET].astype(int).to_numpy()


def cat_names(features: list[str]) -> list[str]:
    return [c for c in features if c in CATEGORICAL]


def fit_categories(train: pd.DataFrame, features: list[str] = MODEL_FEATURES) -> dict:
    return {c: sorted(_cat(train[c]).unique()) for c in cat_names(features)}


def to_lgbm_frame(df: pd.DataFrame, features: list[str], categories: dict) -> pd.DataFrame:
    X = to_frame(df, features)
    for c, cats in categories.items():
        X[c] = pd.Categorical(X[c], categories=cats)
    return X


class Fitted:

    def __init__(self, model, features, kind, categories=None, best_iteration=None):
        self.model, self.features, self.kind = model, list(features), kind
        self.categories, self.best_iteration = categories, best_iteration

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "lightgbm":
            X = to_lgbm_frame(df, self.features, self.categories)
        else:
            X = to_frame(df, self.features)
        return self.model.predict_proba(X)[:, 1]


class SeedAverage:
    def __init__(self, fitted: list[Fitted]):
        self.fitted = list(fitted)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean([f.predict(df) for f in self.fitted], axis=0)


def constant_baseline(train_fit: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), target(train_fit).mean())


class PeakSeasonRate:

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
    return df["PEAK_SUCCESS_PRIOR"].to_numpy(float)


def _add_age_hinge(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["AGE_OVER_40"] = (X["CALCAGE"] - HINGE_AGE).clip(lower=0)
    return X


def make_logreg(features: list[str] = MODEL_FEATURES, C: float = 1.0) -> Pipeline:
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


def es_split(train_fit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return group_kfold(train_fit, 5)[0]


def fit_catboost(train_fit: pd.DataFrame, params: dict | None = None,
                 features: list[str] = MODEL_FEATURES) -> Fitted:
    base = dict(iterations=1000, early_stopping_rounds=100, eval_metric="Logloss",
                random_seed=RANDOM_STATE, thread_count=-1, verbose=0, allow_writing_files=False)
    base.update(params or {})
    a, b = es_split(train_fit)
    X, y = to_frame(train_fit, features), target(train_fit)
    model = CatBoostClassifier(**base)
    model.fit(X.iloc[a], y[a], cat_features=cat_names(features), eval_set=(X.iloc[b], y[b]))
    return Fitted(model, features, "catboost", best_iteration=model.get_best_iteration())


def _time_fold_frames(train_fit: pd.DataFrame, features: list[str]) -> list[tuple]:
    X, y = to_frame(train_fit, features), target(train_fit)
    return [(X.iloc[a], y[a], X.iloc[b], y[b]) for a, b in time_folds(train_fit)]


def _catboost_cv(folds: list[tuple], cats: list[str], params: dict, max_iter: int, seed: int) -> pd.DataFrame:
    rows = []
    for Xa, ya, Xb, yb in folds:
        m = CatBoostClassifier(iterations=max_iter, early_stopping_rounds=50, eval_metric="Logloss",
                               random_seed=seed, thread_count=-1, verbose=0, allow_writing_files=False, **params)
        m.fit(Xa, ya, cat_features=cats, eval_set=(Xb, yb))
        p = m.predict_proba(Xb)[:, 1]
        rows.append({"log_loss": log_loss(yb, p, labels=[0, 1]), "roc_auc": roc_auc_score(yb, p),
                     "iterations": m.get_best_iteration() + 1})
    return pd.DataFrame(rows)


def cv_catboost(train_fit: pd.DataFrame, params: dict | None = None, features: list[str] = MODEL_FEATURES,
                max_iter: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    return _catboost_cv(_time_fold_frames(train_fit, features), cat_names(features), params or {}, max_iter, seed)


def _run_study(objective, n_trials: int, metric: str, name: str, storage: str | None):
    study = optuna.create_study(study_name=name, storage=storage, load_if_exists=True,
                                direction="minimize" if metric == "logloss" else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    done = sum(t.state.is_finished() for t in study.trials)
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
    return study


def tune_catboost(train_fit: pd.DataFrame, n_trials: int = 30, metric: str = "logloss",
                  features: list[str] = MODEL_FEATURES, max_iter: int = 500, storage: str | None = None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cats = cat_names(features)
    folds = _time_fold_frames(train_fit, features)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            random_strength=trial.suggest_float("random_strength", 0.5, 5.0, log=True),
        )
        cv = _catboost_cv(folds, cats, params, max_iter, RANDOM_STATE)
        trial.set_user_attr("mean_best_iteration", int(cv["iterations"].mean()))
        return float(cv["log_loss" if metric == "logloss" else "roc_auc"].mean())

    return _run_study(objective, n_trials, metric, "catboost_time_folds", storage)


def refit_catboost(train_fit: pd.DataFrame, iterations: int, params: dict | None = None,
                   features: list[str] = MODEL_FEATURES, seeds: tuple[int, ...] = SEEDS) -> SeedAverage:
    X, y = to_frame(train_fit, features), target(train_fit)
    fitted = []
    for seed in seeds:
        model = CatBoostClassifier(iterations=iterations, eval_metric="Logloss", random_seed=seed, thread_count=-1,
                                   verbose=0, allow_writing_files=False, **(params or {}))
        model.fit(X, y, cat_features=cat_names(features))
        fitted.append(Fitted(model, features, "catboost", best_iteration=iterations))
    return SeedAverage(fitted)


def _lgbm(params: dict | None, max_iter: int):
    base = dict(n_estimators=max_iter, objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    base.update(params or {})
    return lgb.LGBMClassifier(**base)


def refit_lightgbm(train_fit: pd.DataFrame, iterations: int, params: dict | None = None,
                   features: list[str] = MODEL_FEATURES, seeds: tuple[int, ...] = SEEDS) -> SeedAverage:
    cats = fit_categories(train_fit, features)
    X, y = to_lgbm_frame(train_fit, features, cats), target(train_fit)
    fitted = []
    for seed in seeds:
        model = _lgbm({**(params or {}), "random_state": seed}, iterations)
        model.fit(X, y)
        fitted.append(Fitted(model, features, "lightgbm", categories=cats, best_iteration=iterations))
    return SeedAverage(fitted)


def _lightgbm_fold_frames(train_fit: pd.DataFrame, features: list[str]) -> list[tuple]:
    cats = fit_categories(train_fit, features)
    X, y = to_lgbm_frame(train_fit, features, cats), target(train_fit)
    return [(X.iloc[a], y[a], X.iloc[b], y[b]) for a, b in time_folds(train_fit)]


def _lightgbm_cv(folds: list[tuple], params: dict, max_iter: int, seed: int) -> pd.DataFrame:
    rows = []
    for Xa, ya, Xb, yb in folds:
        m = _lgbm({**params, "random_state": seed}, max_iter)
        m.fit(Xa, ya, eval_X=Xb, eval_y=yb, eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict_proba(Xb)[:, 1]
        rows.append({"log_loss": log_loss(yb, p, labels=[0, 1]), "roc_auc": roc_auc_score(yb, p),
                     "iterations": m.best_iteration_})
    return pd.DataFrame(rows)


def cv_lightgbm(train_fit: pd.DataFrame, params: dict | None = None, features: list[str] = MODEL_FEATURES,
                max_iter: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    return _lightgbm_cv(_lightgbm_fold_frames(train_fit, features), params or {}, max_iter, seed)


def tune_lightgbm(train_fit: pd.DataFrame, n_trials: int = 12, metric: str = "logloss",
                  features: list[str] = MODEL_FEATURES, max_iter: int = 500, storage: str | None = None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = _lightgbm_fold_frames(train_fit, features)

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
        cv = _lightgbm_cv(folds, params, max_iter, RANDOM_STATE)
        trial.set_user_attr("mean_best_iteration", int(cv["iterations"].mean()))
        return float(cv["log_loss" if metric == "logloss" else "roc_auc"].mean())

    return _run_study(objective, n_trials, metric, "lightgbm_time_folds", storage)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:

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
