import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import TARGET
from src.features import to_number
from src.split import group_kfold

POST_CLIMB_EXPED = ["HIGHPOINT", "TERMREASON", "SMTMEMBERS"]
POST_CLIMB_MEMBER = ["MPERHIGHPT"]
POST_CLIMB = POST_CLIMB_EXPED + POST_CLIMB_MEMBER
OUTCOME_WORDS = r"summit|attempt|fail|abort|retreat|turn|death|died|accident|rescue|success|climbed|\bto \d"


def with_post_climb_fields(df: pd.DataFrame, exped: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    exped_fields = exped[["EXPID", "YEAR"] + POST_CLIMB_EXPED]
    member_fields = members[["EXPID", "MYEAR", "MEMBID"] + POST_CLIMB_MEMBER].rename(columns={"MYEAR": "YEAR"})
    out = df.merge(exped_fields, on=["EXPID", "YEAR"], how="left", validate="many_to_one")
    out = out.merge(member_fields, on=["EXPID", "YEAR", "MEMBID"], how="left", validate="one_to_one")
    assert len(out) == len(df)
    return out


def flip_labels_from(members: pd.DataFrame, year: int) -> pd.DataFrame:
    out = members.copy()
    later = (out["MYEAR"] >= year).fillna(False).astype(bool)
    out.loc[later, "MSUCCESS"] = ~out.loc[later, "MSUCCESS"]
    return out


def changed_rows(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> pd.Series:
    changed = pd.Series(False, index=before.index)
    for column in columns:
        equal = (before[column] == after[column]).fillna(False).astype(bool)
        both_missing = before[column].isna() & after[column].isna()
        changed |= ~(equal | both_missing)
    return changed


def single_feature_auc(df: pd.DataFrame, features: list[str], categorical: list[str], target: str = TARGET) -> pd.Series:
    y = df[target].astype(int).to_numpy()
    folds = group_kfold(df, 5)
    rows = {}
    for column in features:
        if column in categorical:
            key = df[column].astype("string").fillna("NA").astype(str).to_numpy()
            score = np.zeros(len(df))
            for fit_idx, held_idx in folds:
                means = pd.Series(y[fit_idx]).groupby(key[fit_idx]).mean()
                score[held_idx] = pd.Series(key[held_idx]).map(means).fillna(y[fit_idx].mean()).to_numpy()
        else:
            values = to_number(df[column])
            score = values.fillna(values.median()).to_numpy()
        auc = roc_auc_score(y, score)
        rows[column] = max(auc, 1 - auc)
    return pd.Series(rows).sort_values(ascending=False)


def missing_vs_success(df: pd.DataFrame, features: list[str], target: str = TARGET, min_rows: int = 30) -> pd.DataFrame:
    y = df[target].astype(float)
    rows = []
    for column in features:
        missing = df[column].isna()
        if missing.sum() >= min_rows:
            rows.append({"feature": column, "missing_share": missing.mean(), "success_if_missing": y[missing].mean(),
                         "success_if_known": y[~missing].mean()})
    return pd.DataFrame(rows).set_index("feature")


def _level_stats(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    y = df[target].astype(float)
    stats = y.groupby(df[column].astype("string")).agg(n="size", success="mean")
    return stats.rename_axis("level").reset_index().assign(column=column)


def extreme_levels(df: pd.DataFrame, columns: list[str], target: str = TARGET, min_rows: int = 30,
                   low: float = 0.02, high: float = 0.98) -> pd.DataFrame:
    stats = pd.concat([_level_stats(df, column, target) for column in columns], ignore_index=True)
    keep = (stats["n"] >= min_rows) & ((stats["success"] <= low) | (stats["success"] >= high))
    return stats[keep].sort_values("n", ascending=False)[["column", "level", "n", "success"]]


def keyword_levels(df: pd.DataFrame, columns: list[str], pattern: str = OUTCOME_WORDS, target: str = TARGET) -> pd.DataFrame:
    stats = pd.concat([_level_stats(df, column, target) for column in columns], ignore_index=True)
    keep = stats["level"].str.contains(pattern, case=False, regex=True, na=False)
    return stats[keep].sort_values("n", ascending=False)[["column", "level", "n", "success"]]
