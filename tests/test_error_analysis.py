import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.error_analysis import group_report, reliability_table, residual_icc, season_gaps, within_group_auc


def frame():
    rng = np.random.default_rng(0)
    rows = []
    for e in range(60):
        peak = "A" if e % 2 else "B"
        for _ in range(5):
            rows.append({"EXPID": f"E{e}", "PEAK_GROUP": peak, "YEAR": 2018, "SEASON": 1, "MSUCCESS": bool(rng.random() < (0.7 if peak == "A" else 0.3))})
    return pd.DataFrame(rows)


def test_group_report_has_gap_and_interval():
    df = frame()
    p = np.where(df["PEAK_GROUP"] == "A", 0.8, 0.2) + np.linspace(-0.05, 0.05, len(df))
    table = group_report(df, p, "PEAK_GROUP", n_boot=30)
    assert list(table.index) == ["A", "B"]
    assert table.loc["A", "expeditions"] == 30
    assert table.loc["A", "gap"] == pytest.approx(table.loc["A", "mean_p"] - table.loc["A", "mean_y"])
    assert table.loc["A", "auc_low"] <= table.loc["A", "roc_auc"] <= table.loc["A", "auc_high"]


def test_within_group_auc_ignores_differences_between_groups():
    y = [0, 1, 0, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    keys = ["a", "a", "b", "b"]
    assert roc_auc_score(y, p) == 0.75
    assert within_group_auc(y, p, keys) == {"auc": 1.0, "groups": 2, "rows": 4}


def test_within_group_auc_skips_groups_with_one_class():
    assert within_group_auc([0, 1, 1, 1], [0.1, 0.9, 0.5, 0.6], ["a", "a", "b", "b"])["groups"] == 1


def test_residual_icc_is_high_when_errors_are_shared():
    keys = np.repeat(np.arange(50), 5)
    shared = np.repeat(np.random.default_rng(1).normal(size=50), 5)
    noise = np.random.default_rng(2).normal(scale=0.01, size=250)
    assert residual_icc(shared + noise, np.zeros(250), keys) > 0.95


def test_residual_icc_is_near_zero_without_shared_errors():
    keys = np.repeat(np.arange(200), 5)
    noise = np.random.default_rng(3).normal(size=1000)
    assert abs(residual_icc(noise, np.zeros(1000), keys)) < 0.1


def test_season_gaps_sorts_by_gap_and_drops_small_seasons():
    df = frame()
    p = np.where(df["PEAK_GROUP"] == "A", 0.2, 0.9)
    table = season_gaps(df, p, min_rows=10)
    assert table.loc[0, "PEAK_GROUP"] == "B"
    assert table["gap"].is_monotonic_decreasing
    assert season_gaps(df, p, min_rows=1000).empty


def test_reliability_table_uses_equal_bins():
    table = reliability_table(np.arange(100) % 2, np.linspace(0, 1, 100), bins=4)
    assert table["n"].tolist() == [25, 25, 25, 25]
    assert table["mean_p"].is_monotonic_increasing
