import json

import pandas as pd
import pytest

from src import models as M
from src.config import DATASET_PATH, FINAL_MODEL_PATH, PROCESSED_DIR
from src.features import CATEGORICAL, DERIVED, build_features
from src.leakage_audit import (changed_rows, extreme_levels, flip_labels_from, keyword_levels, missing_vs_success,
                               single_feature_auc)
from src.load import TABLES


def test_flip_labels_changes_only_later_years():
    members = pd.DataFrame({"MYEAR": pd.Series([2000, 2001, None], dtype="Int64"), "MSUCCESS": [True, True, True]})
    assert flip_labels_from(members, 2001)["MSUCCESS"].tolist() == [True, False, True]


def test_changed_rows_treats_two_missing_values_as_equal():
    before = pd.DataFrame({"a": pd.Series([1, None, 3], dtype="Int64")})
    after = pd.DataFrame({"a": pd.Series([1, None, 4], dtype="Int64")})
    assert changed_rows(before, after, ["a"]).tolist() == [False, False, True]


def test_single_feature_auc_finds_a_perfect_numeric_feature():
    df = pd.DataFrame({"EXPID": [f"E{i}" for i in range(40)], "MSUCCESS": [i % 2 == 0 for i in range(40)],
                       "perfect": [float(i % 2) for i in range(40)], "city": ["a" if i % 2 else "b" for i in range(40)]})
    auc = single_feature_auc(df, ["perfect", "city"], ["city"])
    assert auc["perfect"] == 1.0
    assert auc["city"] == 1.0


def test_missing_vs_success_reports_rates():
    df = pd.DataFrame({"MSUCCESS": [True, True, False, False], "x": [None, None, 1.0, 2.0]})
    table = missing_vs_success(df, ["x"], min_rows=2)
    assert table.loc["x", "missing_share"] == 0.5
    assert table.loc["x", "success_if_missing"] == 1.0
    assert table.loc["x", "success_if_known"] == 0.0


def test_extreme_levels_and_keywords():
    df = pd.DataFrame({"MSUCCESS": [True] * 30 + [False] * 30 + [True, False],
                       "ROUTE1": ["always"] * 30 + ["never"] * 30 + ["SE Ridge attempt"] * 2})
    extreme = extreme_levels(df, ["ROUTE1"])
    assert set(extreme["level"]) == {"always", "never"}
    assert keyword_levels(df, ["ROUTE1"])["level"].tolist() == ["SE Ridge attempt"]


def test_final_config_rejects_a_forbidden_field(tmp_path):
    path = tmp_path / "final.json"
    path.write_text(json.dumps({"features": ["CALCAGE", "HIGHPOINT"]}))
    with pytest.raises(ValueError):
        M.load_final_config(path)


@pytest.mark.skipif(not FINAL_MODEL_PATH.exists(), reason="run notebook 06 first")
def test_final_config_passes_the_guard():
    final = M.load_final_config()
    assert "YEAR" not in final["features"]
    assert final["features"] == list(M.MODEL_FEATURES)


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="run python -m src.dataset first")
def test_future_labels_do_not_change_past_features():
    tables = {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in TABLES}
    dataset = pd.read_parquet(DATASET_PATH)
    base = build_features(dataset, tables)
    flipped = build_features(dataset, {**tables, "members": flip_labels_from(tables["members"], 2018)})
    changed = changed_rows(base, flipped, DERIVED)
    past = (base["YEAR"] < 2018) | ((base["YEAR"] == 2018) & (base["SEASON"] == 1))
    later = (base["YEAR"] > 2018) & (base["YEAR"] <= 2021)
    assert not changed[past].any()
    assert changed[later].any()
    assert set(CATEGORICAL).isdisjoint(DERIVED)
