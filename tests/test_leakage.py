import pandas as pd
import pytest

from src.config import DATASET_PATH
from src.data_dictionary import OUT_CSV, parse_guide
from src.dataset import ROUTE_POSTFACTUM_RE, TARGETS, clean_values
from src.leakage import FORBIDDEN_RAW, TECHNICAL, check_no_leakage

PLAN_FEATURES = [
    "CALCAGE", "SEX", "CITIZEN", "LEADER", "DEPUTY", "STATUS", "SUPPORT", "DISABLED",
    "PEAKID", "ROUTE1", "SEASON", "YEAR", "HOST", "TOTMEMBERS", "TOTHIRED", "NOHIRED", "STDRTE", "COMRTE", "AGENCY",
    "HEIGHTM", "HIMAL", "REGION", "PEAK_GROUP",
]


def frame(*columns: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def test_plan_features_pass():
    check_no_leakage(frame(*PLAN_FEATURES))


@pytest.mark.parametrize("column", [
    "MSUCCESS", "DEATH", "O2CLIMB", "MO2USED", "HIGHPOINT", "highpoint_bin", "PYEAR_lag",
    "BCDATE", "PSTATUS", "ANOMALOUS_SEASON", "EXPID", "MEMBID",
])


def test_forbidden_or_technical_fail(column):
    with pytest.raises(ValueError):
        check_no_leakage(frame("CALCAGE", column))


def test_similar_names_pass():
    check_no_leakage(frame("SKILL", "OPENNESS", "DEATHLESS_RATE"))


def test_dictionary_matches_leakage_lists():
    dictionary = pd.read_csv(OUT_CSV, encoding="utf-8-sig", keep_default_na=False)
    assert set(dictionary.loc[dictionary["timing"] == "after", "field"]) == FORBIDDEN_RAW
    assert set(dictionary.loc[dictionary["timing"] == "technical", "field"]) == TECHNICAL


def test_guide_codes_are_parsed():
    guide = parse_guide()
    assert guide[("exped", "SEASON")]["codes"] == {0: "Unknown", 1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
    assert sum(bool(info["codes"]) for info in guide.values()) == 13


def test_clean_values_turns_unknown_into_nan():
    df = pd.DataFrame({
        "SEX": pd.Series(["M", "F", "X", None], dtype="str"),
        "HOST": pd.Series([1, 0, 2, None], dtype="Int64"),
    })
    out = clean_values(df)
    assert out["SEX"].isna().tolist() == [False, False, True, True]
    assert out["HOST"].isna().tolist() == [False, True, False, True]


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="run python -m src.dataset first")
def test_dataset_v1_has_no_unknown_codes():
    df = pd.read_parquet(DATASET_PATH)
    assert set(df["SEX"].dropna()) <= {"M", "F"}
    assert not df["HOST"].eq(0).any()


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="run python -m src.dataset first")
def test_dataset_v1_features_have_no_leakage():
    df = pd.read_parquet(DATASET_PATH)
    check_no_leakage(df.drop(columns=["EXPID", "MEMBID", "ANOMALOUS_SEASON"] + TARGETS))
    assert not df["ROUTE1"].str.contains(ROUTE_POSTFACTUM_RE, regex=True, na=False).any()
