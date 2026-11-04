import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import GROUP, TARGET
from src.evaluate import evaluate, metrics_by_group


def group_report(df: pd.DataFrame, p, column: str, target: str = TARGET, min_rows: int = 30,
                 n_boot: int = 500) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    table = metrics_by_group(df, p, column, target=target, min_rows=min_rows)
    keys = df[column].astype(str).to_numpy()
    y = df[target].astype(int).to_numpy()
    expeditions = df[GROUP].to_numpy()
    rows = []
    for key in table.index:
        part = keys == key
        both = len(np.unique(y[part])) == 2
        res = evaluate(y[part], p[part], expeditions[part], n_boot=n_boot) if both else None
        rows.append({"expeditions": len(np.unique(expeditions[part])),
                     "auc_low": res["roc_auc"]["ci_low"] if both else np.nan,
                     "auc_high": res["roc_auc"]["ci_high"] if both else np.nan})
    extra = pd.DataFrame(rows, index=table.index)
    out = table.join(extra)
    out["gap"] = out["mean_p"] - out["mean_y"]
    return out[["n", "expeditions", "mean_y", "mean_p", "gap", "roc_auc", "auc_low", "auc_high", "brier", "log_loss"]]


def within_group_auc(y, p, keys) -> dict:
    y, p, keys = np.asarray(y, dtype=int), np.asarray(p, dtype=float), np.asarray(keys)
    total, weight, used_groups, used_rows = 0.0, 0, 0, 0
    for key in np.unique(keys):
        part = keys == key
        if len(np.unique(y[part])) == 2:
            n = int(part.sum())
            total += n * roc_auc_score(y[part], p[part])
            weight += n
            used_groups += 1
            used_rows += n
    return {"auc": total / weight if weight else np.nan, "groups": used_groups, "rows": used_rows}


def residual_icc(y, p, keys) -> float:
    residual = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    frame = pd.DataFrame({"key": np.asarray(keys), "r": residual})
    sizes = frame.groupby("key")["r"].size().to_numpy()
    means = frame.groupby("key")["r"].mean().to_numpy()
    n, k = len(frame), len(sizes)
    between = (sizes * (means - residual.mean()) ** 2).sum() / (k - 1)
    within = ((frame["r"] - frame.groupby("key")["r"].transform("mean")) ** 2).sum() / (n - k)
    typical = (n - (sizes ** 2).sum() / n) / (k - 1)
    return float((between - within) / (between + (typical - 1) * within))


def season_gaps(df: pd.DataFrame, p, target: str = TARGET, min_rows: int = 30) -> pd.DataFrame:
    frame = df[["PEAK_GROUP", "YEAR", "SEASON", GROUP]].copy()
    frame["y"] = df[target].astype(float).to_numpy()
    frame["p"] = np.asarray(p, dtype=float)
    table = frame.groupby(["PEAK_GROUP", "YEAR", "SEASON"], observed=True).agg(
        n=("y", "size"), expeditions=(GROUP, "nunique"), mean_y=("y", "mean"), mean_p=("p", "mean")).reset_index()
    table["gap"] = table["mean_p"] - table["mean_y"]
    return table[table["n"] >= min_rows].sort_values("gap", ascending=False).reset_index(drop=True)


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop") + 1
    return frame.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"))
