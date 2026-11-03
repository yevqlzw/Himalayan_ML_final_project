import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate, metrics_by_group, plot_calibration


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
    assert list(out.index) == ["A", "B"]                 # C has < 30 rows
    assert np.isnan(out.loc["B", "roc_auc"])             # single class
    ages = metrics_by_group(df, p, "AGE_BIN", min_rows=1)
    assert ages["n"].sum() == len(df)


def test_plot_calibration_returns_axis():
    import matplotlib
    matplotlib.use("Agg")
    y, p, _ = toy()
    ax = plot_calibration(y, {"m": p})
    assert len(ax.lines) == 2
