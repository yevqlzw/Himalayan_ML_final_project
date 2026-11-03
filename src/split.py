"""Time split and expedition-grouped CV (Student B).

Rules: train 1990-2017, val 2018-2021, test 2022-2025; GroupKFold by EXPID inside train.
The test set is returned only by time_split(); during development use dev_split(), which never returns it.
"""
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.features import OUT_PATH

RANDOM_STATE = 42
TARGET = "MSUCCESS"
GROUP = "EXPID"
YEAR = "YEAR"

TRAIN_YEARS = (1990, 2017)
VAL_YEARS = (2018, 2021)
TEST_YEARS = (2022, 2025)


def load_features() -> pd.DataFrame:
    return pd.read_parquet(OUT_PATH)


def _years(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return df[df[YEAR].between(*years)].copy()


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) by year ranges. Call this for the test set ONCE, at the very end."""
    return _years(df, TRAIN_YEARS), _years(df, VAL_YEARS), _years(df, TEST_YEARS)


def dev_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, val) only: the test rows are never returned."""
    return _years(df, TRAIN_YEARS), _years(df, VAL_YEARS)


def fit_frame(train: pd.DataFrame, drop_anomalous: bool = True) -> pd.DataFrame:
    """Rows used for FITTING. Anomalous seasons (avalanche, earthquake, COVID) are dropped by default."""
    if drop_anomalous:
        return train[~train["ANOMALOUS_SEASON"].astype(bool)].copy()
    return train.copy()


def group_kfold(train: pd.DataFrame, n_splits: int = 5, groups: str = GROUP) -> list[tuple]:
    """List of (fit_idx, valid_idx) POSITIONAL indices; a whole expedition is always on one side.

    Shuffled with random_state=42 when the installed scikit-learn supports it (>= 1.6),
    otherwise the deterministic GroupKFold is used.
    """
    try:
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    except TypeError:
        cv = GroupKFold(n_splits=n_splits)
    return list(cv.split(train, groups=train[groups]))
