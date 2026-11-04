import numpy as np
import pandas as pd

from src import models as M
from src.features import to_number

FEATURE_GROUPS = {
    "Age": ["CALCAGE"],
    "Experience": ["PRIOR_EXPEDITIONS", "PRIOR_SUMMITS", "PRIOR_EXPEDITIONS_PEAK", "PRIOR_SUMMITS_PEAK"],
    "Hired staff and team size": ["TOTMEMBERS", "TOTHIRED", "NOHIRED", "HIRED_PER_MEMBER"],
    "Crowding": ["CROWD_PEAK", "CROWD_PEAK_TOTAL", "CROWD_ROUTE", "CROWD_ROUTE_TOTAL", "OTHER_EXPEDITIONS_PEAK"],
    "Season": ["SEASON"],
    "Peak and route": ["PEAK_GROUP", "ROUTE1", "HEIGHTM", "HIMAL", "REGION", "HOST", "STDRTE", "COMRTE",
                       "PEAK_SUCCESS_PRIOR", "PEAK_ATTEMPTS_PRIOR", "PEAK_CLIMBED_BEFORE"],
    "Person": ["SEX", "CITIZEN", "STATUS", "LEADER", "DEPUTY", "SUPPORT", "DISABLED"],
    "Agency": ["AGENCY"],
}


def shap_values(model: M.SeedAverage, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    columns = model.fitted[0].features
    runs = [f.model.predict(M.to_lgbm_frame(df, f.features, f.categories), pred_contrib=True) for f in model.fitted]
    mean = np.mean(runs, axis=0)
    return pd.DataFrame(mean[:, :-1], columns=columns, index=df.index), mean[:, -1]


def raw_scores(model: M.SeedAverage, df: pd.DataFrame) -> np.ndarray:
    runs = [f.model.predict(M.to_lgbm_frame(df, f.features, f.categories), raw_score=True) for f in model.fitted]
    return np.mean(runs, axis=0)


def importance(shap: pd.DataFrame) -> pd.Series:
    return shap.abs().mean().sort_values(ascending=False)


def group_importance(shap: pd.DataFrame, groups: dict = FEATURE_GROUPS) -> pd.Series:
    return pd.Series({name: shap[columns].sum(axis=1).abs().mean() for name, columns in groups.items()}).sort_values(ascending=False)


def points_from_base(effect, base: float) -> np.ndarray:
    expit = lambda x: 1 / (1 + np.exp(-x))
    return 100 * (expit(base + np.asarray(effect, dtype=float)) - expit(base))


def binned_effect(x: pd.Series, effect: pd.Series, edges: list[float], base: float) -> pd.DataFrame:
    values = to_number(x)
    bins = pd.cut(values, edges, right=False)
    table = pd.DataFrame({"x": values, "effect": effect}).groupby(bins, observed=True).agg(
        n=("x", "size"), x_mean=("x", "mean"), shap_mean=("effect", "mean"))
    table["points"] = points_from_base(table["shap_mean"], base)
    return table


def category_effect(x: pd.Series, effect: pd.Series, base: float, names: dict | None = None) -> pd.DataFrame:
    labels = x.astype("string").fillna("NA").astype(str)
    if names:
        labels = labels.map(lambda v: names.get(v, v))
    table = pd.DataFrame({"category": labels, "effect": effect}).groupby("category").agg(
        n=("effect", "size"), shap_mean=("effect", "mean")).sort_values("shap_mean", ascending=False)
    table["points"] = points_from_base(table["shap_mean"], base)
    return table
