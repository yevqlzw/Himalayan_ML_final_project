import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import FEATURES_PATH, GROUP, RANDOM_STATE, TARGET, TEST_YEARS, TRAIN_YEARS, VAL_YEARS

YEAR = "YEAR"

TIME_FOLDS = (((1990, 2005), (2006, 2009)), ((1990, 2009), (2010, 2013)), ((1990, 2013), (2014, 2017)))


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def _years(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return df[df[YEAR].between(*years)].copy()


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _years(df, TRAIN_YEARS), _years(df, VAL_YEARS), _years(df, TEST_YEARS)


def dev_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _years(df, TRAIN_YEARS), _years(df, VAL_YEARS)


def fit_frame(train: pd.DataFrame, drop_anomalous: bool = True) -> pd.DataFrame:
    if drop_anomalous:
        return train[~train["ANOMALOUS_SEASON"].astype(bool)].copy()
    return train.copy()


def group_kfold(train: pd.DataFrame, n_splits: int = 5, groups: str = GROUP) -> list[tuple]:
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(train, groups=train[groups]))


def time_folds(train: pd.DataFrame) -> list[tuple]:
    years = train[YEAR]
    return [(np.flatnonzero(years.between(*past).to_numpy(dtype=bool)),
             np.flatnonzero(years.between(*future).to_numpy(dtype=bool)))
            for past, future in TIME_FOLDS]
