"""Metrics with 95% bootstrap CI resampled by expedition, calibration plot, metrics by group."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

RANDOM_STATE = 42
TARGET = "MSUCCESS"
EPS = 1e-15
METRICS = ("roc_auc", "brier", "log_loss")
AGE_BINS = [0, 25, 35, 45, 55, 120]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]


def _clean(y_true, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return y, p


def _point(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def evaluate(y_true, p, groups, n_boot: int = 1000, seed: int = RANDOM_STATE, alpha: float = 0.05) -> dict:
    """ROC AUC, Brier, log loss, each as {estimate, ci_low, ci_high}.

    The CI comes from resampling WHOLE expeditions (groups) with replacement n_boot times.
    Each bootstrap sample is expressed as integer row weights, so no data is copied.
    """
    y, p = _clean(y_true, p)
    codes, uniques = pd.factorize(np.asarray(groups))
    n_groups = len(uniques)
    estimate = _point(y, p)

    sq_err = (p - y) ** 2
    neg_ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    rng = np.random.default_rng(seed)
    boots = {m: np.full(n_boot, np.nan) for m in METRICS}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[codes]
        total = w.sum()
        boots["brier"][b] = (w * sq_err).sum() / total
        boots["log_loss"][b] = (w * neg_ll).sum() / total
        if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
            boots["roc_auc"][b] = roc_auc_score(y, p, sample_weight=w)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        m: {"estimate": float(estimate[m]),
            "ci_low": float(np.nanpercentile(boots[m], lo)),
            "ci_high": float(np.nanpercentile(boots[m], hi))}
        for m in METRICS
    }


def fmt(res: dict, digits: int = 3) -> dict:
    """{'roc_auc': '0.812 [0.801, 0.823]', ...}"""
    return {m: f"{r['estimate']:.{digits}f} [{r['ci_low']:.{digits}f}, {r['ci_high']:.{digits}f}]"
            for m, r in res.items()}


def summary_row(name: str, y_true, p, groups, n_boot: int = 1000) -> dict:
    res = evaluate(y_true, p, groups, n_boot=n_boot)
    text = fmt(res)
    return {"model": name, "ROC AUC": text["roc_auc"], "Brier": text["brier"], "Log loss": text["log_loss"],
            "_res": res, "n": len(np.asarray(y_true)), "mean_p": float(np.mean(p))}


def results_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).set_index("model")


def plot_calibration(y_true, probs, n_bins: int = 10, strategy: str = "quantile", ax=None,
                     title: str = "Calibration curve"):
    """Reliability diagram. probs: one array or a dict {name: array}. Returns the matplotlib axis."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    if not isinstance(probs, dict):
        probs = {"model": probs}
    y = np.asarray(y_true).astype(int)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in probs.items():
        frac, mean = calibration_curve(y, np.clip(np.asarray(p, float), EPS, 1 - EPS), n_bins=n_bins, strategy=strategy)
        ax.plot(mean, frac, marker="o", ms=4, lw=1.5, label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed share of summits")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return ax


def add_age_bins(df: pd.DataFrame) -> pd.Series:
    age = pd.to_numeric(df["CALCAGE"].astype("Float64"), errors="coerce").astype(float)
    return pd.cut(age, AGE_BINS, labels=AGE_LABELS, right=False).astype(object).fillna("unknown")


def metrics_by_group(df: pd.DataFrame, p, col: str, target: str = TARGET, min_rows: int = 30) -> pd.DataFrame:
    """n, mean outcome, mean prediction, AUC, Brier, log loss per group.

    col is e.g. 'PEAK_GROUP', 'SEASON' or 'AGE_BIN' (created from CALCAGE when missing).
    Groups with fewer than min_rows rows are skipped; a single-class group gets NaN AUC.
    """
    keys = add_age_bins(df) if col == "AGE_BIN" and "AGE_BIN" not in df else df[col]
    d = pd.DataFrame({"g": keys.astype(str).to_numpy(), "y": df[target].astype(int).to_numpy(),
                      "p": np.clip(np.asarray(p, float), EPS, 1 - EPS)})
    rows = []
    for g, s in d.groupby("g", sort=True):
        if len(s) < min_rows:
            continue
        both = s["y"].nunique() == 2
        rows.append({col: g, "n": len(s), "mean_y": s["y"].mean(), "mean_p": s["p"].mean(),
                     "roc_auc": roc_auc_score(s["y"], s["p"]) if both else np.nan,
                     "brier": brier_score_loss(s["y"], s["p"]),
                     "log_loss": log_loss(s["y"], s["p"], labels=[0, 1])})
    return pd.DataFrame(rows).set_index(col) if rows else pd.DataFrame()
