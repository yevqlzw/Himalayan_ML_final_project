import pandas as pd
import pytest

from src.features import (CATEGORICAL, FEATURES, OUT_PATH, crowd, experience, hired_per_member,
                          peak_climbed_before, peak_success_prior, person_id, time_key)
from src.leakage import check_no_leakage


def ints(*values) -> pd.Series:
    return pd.Series(values, dtype="Int64")


def test_time_key_orders_seasons_and_puts_unknown_last():
    keys = time_key(ints(2000, 2000, 2000, 2000, 2001), ints(1, 3, 4, 0, 1)).tolist()
    assert keys[0] < keys[1] < keys[2] < keys[3] < keys[4]


def test_person_id_normalizes_names_and_needs_yob():
    df = pd.DataFrame({
        "FNAME": pd.Series(["José  María", "jose maria", "José María"], dtype="str"),
        "LNAME": pd.Series(["García", "GARCIA", "García"], dtype="str"),
        "YOB": ints(1970, 1970, None),
    })
    pid = person_id(df)
    assert pid[0] == pid[1] == "jose maria|garcia|1970"
    assert pd.isna(pid[2])


def test_experience_counts_only_earlier_seasons():
    df = pd.DataFrame({
        "PERSON_ID": pd.Series(["a", "a", "a", "a", None], dtype="str"),
        "PEAKID": ["EVER", "LHOT", "EVER", "EVER", "EVER"],
        "TIME": time_key(ints(2000, 2000, 2000, 2001, 2001), ints(1, 1, 3, 1, 1)),
        "MSUCCESS": [True, False, False, False, True],
    })
    out = experience(df, df)
    assert out["PRIOR_EXPEDITIONS"].tolist()[:4] == [0, 0, 2, 3]
    assert out["PRIOR_SUMMITS"].tolist()[:4] == [0, 0, 1, 1]
    assert out["PRIOR_EXPEDITIONS_PEAK"].tolist()[:4] == [0, 0, 1, 2]
    assert out["PRIOR_SUMMITS_PEAK"].tolist()[:4] == [0, 0, 1, 1]
    assert out.iloc[4].isna().all()


def test_crowd_excludes_own_expedition_and_other_seasons():
    exped = pd.DataFrame({
        "EXPID": ["A", "B", "C", "D", "E"],
        "YEAR": ints(2019, 2019, 2019, 2019, 2019),
        "SEASON": ints(1, 1, 1, 3, 1),
        "PEAKID": ["EVER"] * 5,
        "ROUTE1": pd.Series(["S Col-SE Ridge", "S Col-SE Ridge (to 8000m)", "N Col-NE Ridge", "S Col-SE Ridge", None],
                            dtype="str"),
        "TOTMEMBERS": ints(5, 10, 20, 7, 1),
        "TOTHIRED": ints(5, 0, 10, 7, 1),
    })
    out = crowd(exped).set_index("EXPID")
    assert out["CROWD_PEAK"].tolist() == [31, 26, 16, 0, 35]
    assert out["CROWD_PEAK_TOTAL"].tolist() == [42, 42, 22, 0, 50]
    assert out["CROWD_ROUTE"].tolist()[:4] == [10, 5, 0, 0]
    assert pd.isna(out.loc["E", "CROWD_ROUTE"])
    assert out["OTHER_EXPEDITIONS_PEAK"].tolist() == [3, 3, 3, 0, 3]


def test_hired_per_member_is_nan_without_members():
    out = hired_per_member(pd.DataFrame({"TOTHIRED": ints(4, 3), "TOTMEMBERS": ints(8, 0)}))
    assert out[0] == 0.5
    assert pd.isna(out[1])


def test_peak_success_prior_uses_only_earlier_years_with_smoothing():
    members = pd.DataFrame({
        "PEAKID": ["X", "X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y"],
        "MYEAR": ints(2000, 2000, 2000, 2000, 2001, 2001, 2000, 2000, 2000, 2000, 2001),
        "MSUCCESS": [True, True, True, True, False, False, False, False, False, False, True],
        "HIRED": [False, False, True, False, False, False, False, False, False, False, False],
        "BCONLY": [False] * 11,
        "NOTTOBC": [False] * 11,
        "MSMTTERM": ints(1, 1, 1, 17, 3, 3, 3, 3, 3, 3, 1),
    })
    rows = pd.DataFrame({"PEAKID": ["X", "Y"], "YEAR": ints(2001, 2001)})
    out = peak_success_prior(rows, members, smoothing=2)
    assert out["PEAK_ATTEMPTS_PRIOR"].tolist() == [2, 4]
    assert out["PEAK_SUCCESS_PRIOR"].tolist() == pytest.approx([2 / 3, 1 / 9])


def test_peak_climbed_before_is_strict_and_conservative():
    peaks = pd.DataFrame({"PEAKID": ["A", "B", "C"], "PYEAR": ints(2000, None, 2000), "PSEASON": ints(3, 0, 0)})
    rows = pd.DataFrame({
        "PEAKID": ["A", "A", "B", "C", "C"],
        "TIME": time_key(ints(2000, 2001, 2001, 2000, 2001), ints(3, 1, 1, 4, 1)),
    })
    assert peak_climbed_before(rows, peaks).tolist() == [False, True, False, False, True]


def test_feature_lists_are_consistent():
    check_no_leakage(pd.DataFrame(columns=FEATURES))
    assert set(CATEGORICAL) <= set(FEATURES)
    assert len(FEATURES) == len(set(FEATURES))


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    if not OUT_PATH.exists():
        pytest.skip("run python -m src.features first")
    return pd.read_parquet(OUT_PATH).set_index(["EXPID", "YEAR", "MEMBID"]).sort_index()


def test_features_file_has_all_features(features):
    assert set(FEATURES) <= set(features.columns) | {"YEAR"}


@pytest.mark.parametrize("key, expected", [
    (("AMAD91105", 1991, "03"), (4, 2, 1, 1)),
    (("TAWA10301", 2010, "06"), (4, 1, 1, 0)),
    (("AMAD92309", 1992, "09"), (4, 0, 0, 0)),
    (("AMAD94103", 1994, "04"), (5, 2, 1, 0)),
])
def test_experience_real_records(features, key, expected):
    row = features.loc[key]
    assert tuple(row[["PRIOR_EXPEDITIONS", "PRIOR_SUMMITS", "PRIOR_EXPEDITIONS_PEAK", "PRIOR_SUMMITS_PEAK"]]) == expected


def test_experience_real_record_without_yob(features):
    assert features.loc[("AMAD90307", 1990, "05"), ["PRIOR_EXPEDITIONS", "PRIOR_SUMMITS"]].isna().all()


@pytest.mark.parametrize("expedition, expected", [
    (("EVER19102", 2019), (547, 1202, 388, 861, 91, 1.0)),
    (("AMAD17301", 2017), (246, 349, 246, 349, 58, 1 / 3)),
    (("ANN113302", 2013), (2, 2, 2, 2, 1, 0.0)),
])
def test_crowd_and_team_real_records(features, expedition, expected):
    row = features.loc[expedition].iloc[0]
    columns = ["CROWD_PEAK", "CROWD_PEAK_TOTAL", "CROWD_ROUTE", "CROWD_ROUTE_TOTAL", "OTHER_EXPEDITIONS_PEAK",
               "HIRED_PER_MEMBER"]
    assert tuple(row[columns]) == pytest.approx(expected)


@pytest.mark.parametrize("expedition, rate, attempts", [
    (("EVER19102", 2019), 0.387476, 12_208),
    (("AMAD17301", 2017), 0.507478, 5_832),
    (("ANN113302", 2013), 0.140837, 1_170),
])
def test_peak_success_prior_real_records(features, expedition, rate, attempts):
    row = features.loc[expedition].iloc[0]
    assert row["PEAK_SUCCESS_PRIOR"] == pytest.approx(rate, abs=1e-6)
    assert row["PEAK_ATTEMPTS_PRIOR"] == attempts


@pytest.mark.parametrize("expedition, expected", [
    (("BOBA96301", 1996), False),
    (("CHAN10301", 2010), False),
    (("CHAN25201", 2025), True),
])
def test_peak_climbed_before_real_records(features, expedition, expected):
    assert features.loc[expedition].iloc[0]["PEAK_CLIMBED_BEFORE"] == expected
