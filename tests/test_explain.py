import numpy as np
import pandas as pd
import pytest

from src import models as M
from src.config import FEATURES_PATH
from src.explain import (FEATURE_GROUPS, binned_effect, category_effect, group_importance, importance,
                         points_from_base, raw_scores, shap_values)
from src.split import dev_split, fit_frame


def test_groups_cover_every_model_feature_once():
    grouped = [c for columns in FEATURE_GROUPS.values() for c in columns]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(M.MODEL_FEATURES)


def test_group_importance_lets_opposite_effects_cancel():
    shap = pd.DataFrame({"a": [1.0, -1.0], "b": [-1.0, 1.0], "c": [0.5, 0.5]})
    by_group = group_importance(shap, {"ab": ["a", "b"], "c": ["c"]})
    assert by_group["ab"] == 0
    assert by_group["c"] == 0.5
    assert list(importance(shap).index[:2]) == ["a", "b"]


def test_points_from_base_is_zero_without_effect():
    assert points_from_base([0.0], 0.3)[0] == 0
    assert points_from_base([1.0], 0.0)[0] == pytest.approx(100 * (1 / (1 + np.exp(-1.0)) - 0.5))


def test_binned_effect_uses_left_closed_bins():
    x = pd.Series([0, 1, 2, 3], dtype="Int64")
    table = binned_effect(x, pd.Series([1.0, 2.0, 3.0, 4.0]), [0, 2, 4], 0.0)
    assert table["n"].tolist() == [2, 2]
    assert table["shap_mean"].tolist() == [1.5, 3.5]


def test_category_effect_applies_names():
    x = pd.Series(["1", "3", "1"])
    table = category_effect(x, pd.Series([1.0, -1.0, 3.0]), 0.0, {"1": "Spring", "3": "Autumn"})
    assert table.loc["Spring", "n"] == 2
    assert table.loc["Spring", "shap_mean"] == 2.0


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run python -m src.features first")
def test_shap_adds_up_to_the_raw_score():
    train, val = dev_split(pd.read_parquet(FEATURES_PATH))
    fit = fit_frame(train).sample(3000, random_state=0)
    model = M.refit_lightgbm(fit, 5, seeds=(42, 1))
    rows = val.head(200)
    shap, base = shap_values(model, rows)
    assert list(shap.columns) == list(M.MODEL_FEATURES)
    assert (shap.sum(axis=1).to_numpy() + base) == pytest.approx(raw_scores(model, rows))
