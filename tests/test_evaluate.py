import numpy as np
import pandas as pd
import pytest

from src.evaluate import METRICS, evaluate, evaluate_views, metrics_by_group, paired_difference, plot_calibration


def toy(n_groups=200, per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    group_effect = rng.normal(size=n_groups)[groups]
    p_true = 1 / (1 + np.exp(-group_effect))
    y = (rng.random(len(groups)) < p_true).astype(int)
    return y, np.clip(p_true * 0.9 + 0.05, 0.01, 0.99), groups


def test_known_values():
    y = [0, 1, 0, 1]
    p = [0.1, 0.9, 0.2, 0.8]
    res = evaluate(y, p, groups=[1, 2, 3, 4], n_boot=50)
    assert res["roc_auc"]["estimate"] == 1.0
    assert res["brier"]["estimate"] == pytest.approx(0.025)
    assert res["log_loss"]["estimate"] == pytest.approx(-np.mean(np.log([0.9, 0.9, 0.8, 0.8])))


def test_ci_contains_estimate_and_is_ordered():
    y, p, g = toy()
    res = evaluate(y, p, g, n_boot=300)
    for r in res.values():
        assert r["ci_low"] <= r["estimate"] <= r["ci_high"]
        assert r["ci_low"] < r["ci_high"]


def test_bootstrap_is_reproducible():
    y, p, g = toy()
    assert evaluate(y, p, g, n_boot=100) == evaluate(y, p, g, n_boot=100)


def test_resampling_whole_expeditions_gives_wider_ci_than_rows():
    y, p, g = toy()
    by_exped = evaluate(y, p, g, n_boot=400)["brier"]
    by_row = evaluate(y, p, np.arange(len(y)), n_boot=400)["brier"]
    assert by_exped["ci_high"] - by_exped["ci_low"] > by_row["ci_high"] - by_row["ci_low"]


def test_metrics_by_group_handles_small_and_single_class_groups():
    df = pd.DataFrame({"MSUCCESS": [0, 1] * 40 + [1] * 40 + [0, 1] * 5,
                       "PEAK_GROUP": ["A"] * 80 + ["B"] * 40 + ["C"] * 10,
                       "CALCAGE": [20.0, 30.0, 40.0, 50.0, 60.0] * 26})
    p = np.full(len(df), 0.5)
    out = metrics_by_group(df, p, "PEAK_GROUP", min_rows=30)
    assert list(out.index) == ["A", "B"]
    assert np.isnan(out.loc["B", "roc_auc"])
    ages = metrics_by_group(df, p, "AGE_BIN", min_rows=1)
    assert ages["n"].sum() == len(df)


def test_plot_calibration_returns_axis():
    import matplotlib
    matplotlib.use("Agg")
    y, p, _ = toy()
    ax = plot_calibration(y, {"m": p})
    assert len(ax.lines) == 2


def test_paired_difference_of_identical_models_is_zero():
    y, p, g = toy()
    res = paired_difference(y, p, p, g, n_boot=50)
    for m in METRICS:
        assert res[m] == {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_paired_difference_finds_the_better_model():
    y, p, g = toy()
    res = paired_difference(y, p, np.full(len(y), 0.5), g, n_boot=200)
    assert res["roc_auc"]["ci_low"] > 0
    assert res["brier"]["ci_high"] < 0
    assert res["log_loss"]["ci_high"] < 0


def test_paired_difference_estimate_matches_evaluate():
    y, p, g = toy()
    other = np.clip(p + 0.05, 0.01, 0.99)
    res = paired_difference(y, p, other, g, n_boot=10)
    a, b = evaluate(y, p, g, n_boot=2), evaluate(y, other, g, n_boot=2)
    for m in METRICS:
        assert res[m]["estimate"] == pytest.approx(a[m]["estimate"] - b[m]["estimate"])


def test_evaluate_views_drops_anomalous_rows_from_the_second_view():
    y, p, g = toy()
    df = pd.DataFrame({"MSUCCESS": y, "EXPID": g, "ANOMALOUS_SEASON": g < 50})
    rows_all, rows_clean = evaluate_views({"model": p}, df, n_boot=10)
    assert rows_all[0]["n"] == len(df)
    assert rows_clean[0]["n"] == int((g >= 50).sum())
