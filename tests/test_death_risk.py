import numpy as np
import pandas as pd
import pytest

from src.death_risk import (age_group, compare_clients_hired, counter_rates, death_class_shares, expedition_counts,
                            experience_group, member_rates, peak_type, risk_ratio_interval, shared_expeditions,
                            wilson_interval)


def ints(*values) -> pd.Series:
    return pd.Series(values, dtype="Int64")


def test_wilson_interval_known_values():
    low, high = wilson_interval([5, 0], [100, 10])
    assert low == pytest.approx([0.02154, 0.0], abs=1e-4)
    assert high == pytest.approx([0.11175, 0.27753], abs=1e-4)


def test_member_rates_per_1000():
    df = pd.DataFrame({"GROUP": ["a"] * 1000 + ["b"] * 500, "DEATH": [True] * 5 + [False] * 995 + [True] * 10 + [False] * 490})
    out = member_rates(df, "GROUP")
    assert out["per_1000"].tolist() == pytest.approx([5.0, 20.0])
    assert (out["low"] < out["per_1000"]).all() and (out["high"] > out["per_1000"]).all()


def test_groupings():
    groups = age_group(ints(25, 30, 49, 60, None))
    assert groups.astype(str).tolist()[:4] == ["<30", "30-39", "40-49", "60+"]
    assert pd.isna(groups.iloc[4])
    assert experience_group(ints(0, 3, None)).tolist() == ["first expedition", "experienced", "unknown"]
    assert peak_type(pd.Series(["EVER", "CHOY", "AMAD"]), ints(8849, 8188, 6814)).tolist() == [
        "Everest", "Other 8000 m", "Below 8000 m"]


def synthetic_tables():
    exped = pd.DataFrame({
        "EXPID": ["A", "B", "C"], "YEAR": ints(2000, 2001, 1940), "SEASON": ints(1, 3, 1),
        "PEAKID": ["EVER", "AMAD", "EVER"], "TOTHIRED": ints(4, 0, 2), "HDEATHS": ints(1, 0, 0), "MDEATHS": ints(1, 0, 0),
    })
    members = pd.DataFrame({
        "EXPID": ["A", "A", "A", "A", "B", "B", "C"],
        "MYEAR": ints(2000, 2000, 2000, 2000, 2001, 2001, 1940),
        "HIRED": [False, False, False, True, False, False, False],
        "BCONLY": [False, False, True, False, False, False, False],
        "NOTTOBC": [False] * 7,
        "DEATH": [True, False, False, True, False, False, False],
        "DEATHCLASS": ints(5, 0, 0, 3, 0, 0, 0),
    })
    peaks = pd.DataFrame({"PEAKID": ["EVER", "AMAD"], "HEIGHTM": ints(8849, 6814)})
    return exped, members, peaks


def test_expedition_counts_use_clients_above_base_camp():
    counts = expedition_counts(*synthetic_tables()).set_index("EXPID")
    assert list(counts.index) == ["A", "B"]
    assert counts.loc["A", ["CLIENTS", "CLIENT_DEATHS", "HIRED", "HIRED_DEATHS"]].tolist() == [2, 1, 4, 1]
    assert counts.loc["B", "CLIENTS"] == 2 and counts.loc["B", "HIRED"] == 0
    assert shared_expeditions(counts.reset_index())["EXPID"].tolist() == ["A"]


def test_counter_rates_and_risk_ratio():
    counts = pd.DataFrame({"G": ["x"] * 100, "CLIENTS": 10, "CLIENT_DEATHS": [1] * 10 + [0] * 90,
                           "HIRED": 5, "HIRED_DEATHS": [0] * 50 + [1] * 10 + [0] * 40})
    rates = counter_rates(counts, "G")
    assert rates.loc["x", "client_per_1000"] == pytest.approx(10.0)
    assert rates.loc["x", "hired_per_1000"] == pytest.approx(20.0)
    ratio, low, high = risk_ratio_interval(counts, n_boot=200)
    assert ratio == pytest.approx(2.0) and low < 2.0 < high
    table = compare_clients_hired(counts, "G")
    assert table.loc["x", "ratio"] == pytest.approx(2.0)


def test_death_class_shares_by_group():
    _, members, _ = synthetic_tables()
    shares = death_class_shares(members)
    assert shares.loc["Descending from summit bid", "Clients"] == pytest.approx(100.0)
    assert shares.loc["Route preparation", "Hired"] == pytest.approx(100.0)
    assert np.allclose(shares.sum(), 100.0)
