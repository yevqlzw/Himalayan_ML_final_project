import numpy as np
import pandas as pd
import pytest

from src import dataset
from src.config import FEATURES_PATH
from src.split import TEST_YEARS, TRAIN_YEARS, VAL_YEARS, dev_split, fit_frame, group_kfold, time_folds, time_split


def synthetic(n_exped_per_year: int = 6, members: int = 4) -> pd.DataFrame:
    rows = []
    for year in range(1988, 2027):
        for e in range(n_exped_per_year):
            for m in range(members):
                rows.append({"EXPID": f"X{year % 100:02d}{e}", "YEAR": year, "MEMBID": m,
                             "MSUCCESS": (e + m) % 2 == 0, "ANOMALOUS_SEASON": year == 2015})
    return pd.DataFrame(rows).assign(EXPID=lambda d: d["EXPID"] + d["YEAR"].astype(str))


def test_year_ranges_match_dataset_splits():
    assert [TRAIN_YEARS, VAL_YEARS, TEST_YEARS] == list(dataset.SPLITS.values())


def test_time_split_has_no_year_overlap():
    train, val, test = time_split(synthetic())
    assert train["YEAR"].between(1990, 2017).all() and train["YEAR"].nunique() == 28
    assert val["YEAR"].between(2018, 2021).all()
    assert test["YEAR"].between(2022, 2025).all()
    assert not set(train["YEAR"]) & set(val["YEAR"])
    assert not set(val["YEAR"]) & set(test["YEAR"])
    assert train["YEAR"].max() < val["YEAR"].min() <= val["YEAR"].max() < test["YEAR"].min()


def test_no_expedition_shared_across_splits():
    train, val, test = time_split(synthetic())
    assert not set(train["EXPID"]) & set(val["EXPID"])
    assert not set(train["EXPID"]) & set(test["EXPID"])
    assert not set(val["EXPID"]) & set(test["EXPID"])


def test_dev_split_never_returns_test_years():
    train, val = dev_split(synthetic())
    assert max(train["YEAR"].max(), val["YEAR"].max()) <= 2021


def test_fit_frame_drops_anomalous_rows():
    train, _ = dev_split(synthetic())
    assert train["ANOMALOUS_SEASON"].any()
    assert not fit_frame(train)["ANOMALOUS_SEASON"].any()
    assert len(fit_frame(train, drop_anomalous=False)) == len(train)


def test_group_kfold_never_splits_an_expedition():
    train, _ = dev_split(synthetic())
    folds = group_kfold(train, n_splits=5)
    assert len(folds) == 5
    seen = []
    for fit_idx, valid_idx in folds:
        assert not set(train["EXPID"].iloc[fit_idx]) & set(train["EXPID"].iloc[valid_idx])
        assert not set(fit_idx) & set(valid_idx)
        seen.extend(valid_idx)
    assert sorted(seen) == list(range(len(train)))


def test_group_kfold_is_reproducible():
    train, _ = dev_split(synthetic())
    a, b = group_kfold(train), group_kfold(train)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_time_folds_go_from_past_to_future():
    train, _ = dev_split(synthetic())
    folds = time_folds(train)
    assert len(folds) == 3
    for fit_idx, hold_idx in folds:
        assert train["YEAR"].iloc[fit_idx].max() < train["YEAR"].iloc[hold_idx].min()
        assert not set(train["EXPID"].iloc[fit_idx]) & set(train["EXPID"].iloc[hold_idx])
    assert [train["YEAR"].iloc[hold].max() for _, hold in folds] == [2009, 2013, 2017]


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run python -m src.features first")
def test_real_time_folds_stay_inside_train_and_share_no_expedition():
    train, _ = dev_split(pd.read_parquet(FEATURES_PATH))
    fit = fit_frame(train)
    for fit_idx, hold_idx in time_folds(fit):
        assert fit["YEAR"].iloc[hold_idx].max() <= 2017
        assert fit["YEAR"].iloc[fit_idx].max() < fit["YEAR"].iloc[hold_idx].min()
        assert not set(fit["EXPID"].iloc[fit_idx]) & set(fit["EXPID"].iloc[hold_idx])


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run python -m src.features first")
def test_real_features_split_has_no_overlap():
    df = pd.read_parquet(FEATURES_PATH)
    train, val, test = time_split(df)
    assert (len(train), len(val), len(test)) == (37_969, 5_042, 9_114)
    assert len(train) + len(val) + len(test) == len(df)
    assert not set(train["EXPID"]) & set(val["EXPID"])
    assert not set(train["EXPID"]) & set(test["EXPID"])
    assert not set(val["EXPID"]) & set(test["EXPID"])
    assert train["YEAR"].max() < val["YEAR"].min() and val["YEAR"].max() < test["YEAR"].min()
