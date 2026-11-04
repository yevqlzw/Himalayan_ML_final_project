import unicodedata

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, FEATURES_PATH, PROCESSED_DIR, ROOT
from src.dataset import KEYS, TARGETS, clean_route
from src.leakage import check_no_leakage
from src.load import TABLES, fmt

EXPECTED_ROWS = 52_125
SMOOTHING = 20

PARTICIPANT = ["CALCAGE", "SEX", "CITIZEN", "LEADER", "DEPUTY", "STATUS", "SUPPORT", "DISABLED"]
EXPEDITION = ["PEAK_GROUP", "ROUTE1", "SEASON", "YEAR", "HOST", "TOTMEMBERS", "TOTHIRED", "NOHIRED",
              "STDRTE", "COMRTE", "AGENCY"]
PEAK = ["HEIGHTM", "HIMAL", "REGION"]
EXPERIENCE = ["PRIOR_EXPEDITIONS", "PRIOR_SUMMITS", "PRIOR_EXPEDITIONS_PEAK", "PRIOR_SUMMITS_PEAK"]
CROWD = ["CROWD_PEAK", "CROWD_PEAK_TOTAL", "CROWD_ROUTE", "CROWD_ROUTE_TOTAL", "OTHER_EXPEDITIONS_PEAK"]
TEAM = ["HIRED_PER_MEMBER"]
PEAK_HISTORY = ["PEAK_SUCCESS_PRIOR", "PEAK_ATTEMPTS_PRIOR", "PEAK_CLIMBED_BEFORE"]
DERIVED = EXPERIENCE + CROWD + TEAM + PEAK_HISTORY
FEATURES = PARTICIPANT + EXPEDITION + PEAK + DERIVED
CATEGORICAL = ["SEX", "CITIZEN", "STATUS", "PEAK_GROUP", "ROUTE1", "SEASON", "HOST", "AGENCY", "HIMAL", "REGION"]


def to_number(s: pd.Series) -> pd.Series:
    return pd.Series(s.astype("Float64").to_numpy(dtype=float, na_value=np.nan), index=s.index)


def time_key(year: pd.Series, season: pd.Series) -> pd.Series:
    return year * 10 + season.where(season != 0, 9)


def normalize_name(s: pd.Series) -> pd.Series:
    ascii_text = s.fillna("").map(
        lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower())
    return ascii_text.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


def person_id(df: pd.DataFrame) -> pd.Series:
    pid = normalize_name(df["FNAME"]) + "|" + normalize_name(df["LNAME"]) + "|" + df["YOB"].astype(str)
    return pid.where(df["YOB"].notna())


def prior_counts(history: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    per_time = (history.dropna(subset=keys).groupby(keys + ["TIME"])["MSUCCESS"]
                .agg(n="size", s="sum").reset_index())
    cumulative = per_time.groupby(keys)[["n", "s"]].cumsum()
    per_time["PRIOR_N"] = cumulative["n"] - per_time["n"]
    per_time["PRIOR_S"] = cumulative["s"] - per_time["s"]
    return per_time[keys + ["TIME", "PRIOR_N", "PRIOR_S"]]


def experience(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for suffix, keys in (("", ["PERSON_ID"]), ("_PEAK", ["PERSON_ID", "PEAKID"])):
        prior = rows[keys + ["TIME"]].merge(prior_counts(history, keys), on=keys + ["TIME"], how="left")
        assert prior.loc[rows["PERSON_ID"].notna().to_numpy(), "PRIOR_N"].notna().all()
        out[f"PRIOR_EXPEDITIONS{suffix}"] = prior["PRIOR_N"].to_numpy()
        out[f"PRIOR_SUMMITS{suffix}"] = prior["PRIOR_S"].to_numpy()
    return out.astype("Int64")


def crowd(exped: pd.DataFrame) -> pd.DataFrame:
    e = exped.assign(ROUTE_KEY=clean_route(exped["ROUTE1"]).str.lower(),
                     TOTAL=exped["TOTMEMBERS"] + exped["TOTHIRED"])
    out = e[["EXPID", "YEAR"]].copy()
    for name, keys in (("PEAK", ["PEAKID", "YEAR", "SEASON"]), ("ROUTE", ["PEAKID", "YEAR", "SEASON", "ROUTE_KEY"])):
        sums = e.groupby(keys)[["TOTMEMBERS", "TOTAL"]].transform("sum")
        out[f"CROWD_{name}"] = sums["TOTMEMBERS"] - e["TOTMEMBERS"]
        out[f"CROWD_{name}_TOTAL"] = sums["TOTAL"] - e["TOTAL"]
    out["OTHER_EXPEDITIONS_PEAK"] = e.groupby(["PEAKID", "YEAR", "SEASON"])["EXPID"].transform("size") - 1
    return out


def hired_per_member(df: pd.DataFrame) -> pd.Series:
    return df["TOTHIRED"] / df["TOTMEMBERS"].where(df["TOTMEMBERS"] > 0)


def climbing_population(members: pd.DataFrame) -> pd.DataFrame:
    return members[~members["HIRED"] & ~members["BCONLY"] & ~members["NOTTOBC"] & (members["MSMTTERM"] != 17)]


def prior_by_year(population: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    per_year = population.groupby(keys + ["MYEAR"])["MSUCCESS"].agg(n="size", s="sum").reset_index()
    cumulative = per_year.groupby(keys)[["n", "s"]].cumsum() if keys else per_year[["n", "s"]].cumsum()
    per_year["PRIOR_N"] = cumulative["n"] - per_year["n"]
    per_year["PRIOR_S"] = cumulative["s"] - per_year["s"]
    return per_year.rename(columns={"MYEAR": "YEAR"})[keys + ["YEAR", "PRIOR_N", "PRIOR_S"]]


def peak_success_prior(rows: pd.DataFrame, members: pd.DataFrame, smoothing: float = SMOOTHING) -> pd.DataFrame:
    population = climbing_population(members)
    peak = rows[["PEAKID", "YEAR"]].merge(prior_by_year(population, ["PEAKID"]), on=["PEAKID", "YEAR"], how="left")
    overall = rows[["YEAR"]].merge(prior_by_year(population, []), on="YEAR", how="left")
    assert peak["PRIOR_N"].notna().all() and overall["PRIOR_N"].notna().all()
    base_rate = overall["PRIOR_S"] / overall["PRIOR_N"]
    rate = (peak["PRIOR_S"] + smoothing * base_rate) / (peak["PRIOR_N"] + smoothing)
    return pd.DataFrame({"PEAK_SUCCESS_PRIOR": rate.to_numpy(), "PEAK_ATTEMPTS_PRIOR": peak["PRIOR_N"].to_numpy()},
                        index=rows.index).astype({"PEAK_ATTEMPTS_PRIOR": "Int64"})


def peak_climbed_before(rows: pd.DataFrame, peaks: pd.DataFrame) -> pd.Series:
    first = time_key(peaks["PYEAR"], peaks["PSEASON"]).set_axis(peaks["PEAKID"])
    return (rows["PEAKID"].map(first) < rows["TIME"]).fillna(False).astype(bool)


def build_features(dataset: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    members, exped, peaks = tables["members"], tables["exped"], tables["peaks"]
    history = members.assign(PERSON_ID=person_id(members), TIME=time_key(members["MYEAR"], members["MSEASON"]))
    rows = dataset.assign(PERSON_ID=person_id(dataset), TIME=time_key(dataset["YEAR"], dataset["SEASON"]))

    crowd_rows = rows[["EXPID", "YEAR"]].merge(crowd(exped), on=["EXPID", "YEAR"], how="left", validate="many_to_one")
    derived = pd.concat([
        experience(rows, history[["PERSON_ID", "PEAKID", "TIME", "MSUCCESS"]]),
        crowd_rows[CROWD].set_axis(rows.index),
        hired_per_member(rows).rename("HIRED_PER_MEMBER"),
        peak_success_prior(rows, members),
        peak_climbed_before(rows, peaks).rename("PEAK_CLIMBED_BEFORE"),
    ], axis=1)

    base = [c for c in PARTICIPANT + EXPEDITION + PEAK if c not in KEYS]
    df = pd.concat([rows[KEYS + TARGETS + ["ANOMALOUS_SEASON"] + base], derived[DERIVED]], axis=1)
    df = df.reset_index(drop=True)
    check_no_leakage(df[FEATURES])
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{fmt(len(df))} rows x {df.shape[1]} columns; {len(FEATURES)} features "
          f"({len(CATEGORICAL)} categorical)")
    stats = pd.DataFrame({
        "NaN %": df[DERIVED].isna().mean().mul(100).round(1),
        "median": df[DERIVED].median(),
        "max": df[DERIVED].max(),
    })
    print("\nNew features:")
    print(stats.to_string())
    first_timers = (df["PRIOR_EXPEDITIONS"] == 0).sum()
    print(f"\nFirst expedition in the database: {fmt(first_timers)} "
          f"({first_timers / df['PRIOR_EXPEDITIONS'].notna().sum():.1%} of members with known YOB)")
    print(f"Peak not yet climbed at expedition time: {fmt((~df['PEAK_CLIMBED_BEFORE']).sum())} rows")


def main() -> None:
    tables = {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in TABLES}
    dataset = pd.read_parquet(DATASET_PATH)
    df = build_features(dataset, tables)
    assert len(df) == EXPECTED_ROWS, f"features_v1: {len(df)} rows, expected {EXPECTED_ROWS}"
    assert not df.duplicated(["EXPID", "YEAR", "MEMBID"]).any()

    df.to_parquet(FEATURES_PATH, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(FEATURES_PATH), df)
    print(f"{FEATURES_PATH.relative_to(ROOT)}  {FEATURES_PATH.stat().st_size / 2**20:.1f} MB")

    print_summary(df)


if __name__ == "__main__":
    main()
