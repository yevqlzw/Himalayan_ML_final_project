import pandas as pd

from src.config import DATASET_PATH, PROCESSED_DIR, ROOT, TEST_YEARS, TRAIN_YEARS, VAL_YEARS
from src.leakage import FORBIDDEN_RAW, TECHNICAL, check_no_leakage
from src.load import EXPECTED_NOT_HIRED_1990_2025, TABLES, fmt

EXPECTED_ROWS = 52_125

YEARS = (1990, 2025)
SPLITS = {"train 1990-2017": TRAIN_YEARS, "val 2018-2021": VAL_YEARS, "test 2022-2025": TEST_YEARS}
MIN_PEAK_ROWS = 100

KEYS = ["EXPID", "MEMBID", "YEAR", "SEASON", "PEAKID"]
TARGETS = ["MSUCCESS", "DEATH"]
DERIVED = ["PEAK_GROUP", "ANOMALOUS_SEASON"]
DROP = {"MYEAR", "MSEASON", "HIRED", "AGE", "BIRTHDATE", "ROUTE2", "ROUTE3", "ROUTE4"}

ANOMALOUS_SEASONS = (
    {"YEAR": 2014, "SEASON": 1, "PEAKID": ("EVER", "LHOT")},
    {"YEAR": 2015, "SEASON": 1},
    {"YEAR": 2020},
    {"YEAR": 2021, "SEASON": 1},
)

VALID_SEX = ("M", "F")
UNKNOWN_HOST = 0

ROUTE_POSTFACTUM_RE = r"\s*\((?:to [^)]*|intended)\)"
ROUTE_UP_DOWN_RE = r"^(.*?)\s*\(?\bup\b\)?\s*[;,].*$"

BASE_FILTERS = (
    ("years 1990-2025", lambda d: d["YEAR"].between(*YEARS)),
    ("HIRED == False", lambda d: ~d["HIRED"]),
)
CLIMB_FILTERS = (
    ("MSMTTERM != 17", lambda d: d["MSMTTERM"] != 17),
    ("NOTTOBC == False", lambda d: ~d["NOTTOBC"]),
    ("BCONLY == False", lambda d: ~d["BCONLY"]),
)
DISPUTED_FILTER = ("no MDISPUTED or MCLAIMED", lambda d: ~(d["MDISPUTED"] | d["MCLAIMED"]))


def join(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    members, exped, peaks = tables["members"], tables["exped"], tables["peaks"]
    df = members.merge(exped.drop(columns="PEAKID"), left_on=["EXPID", "MYEAR"], right_on=["EXPID", "YEAR"],
                       how="inner", validate="many_to_one")
    df = df.merge(peaks, on="PEAKID", how="left", validate="many_to_one")
    assert len(df) == len(members), f"join gave {len(df)} rows instead of {len(members)}"
    return df


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    for name, condition in filters:
        before = len(df)
        df = df[condition(df)]
        print(f"  {name:26s} {fmt(before):>7} -> {fmt(len(df)):>7}  (-{fmt(before - len(df))})")
    return df


def clean_route(s: pd.Series) -> pd.Series:
    s = s.str.replace(ROUTE_POSTFACTUM_RE, "", regex=True).str.replace(ROUTE_UP_DOWN_RE, r"\1", regex=True).str.strip()
    return s.mask(s == "")


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    other_sex = df["SEX"].notna() & ~df["SEX"].isin(VALID_SEX)
    unknown_host = df["HOST"].eq(UNKNOWN_HOST).fillna(False).astype(bool)
    print(f"  {'SEX not M/F -> NaN':26s} {fmt(other_sex.sum()):>7}")
    print(f"  {'HOST = 0 -> NaN':26s} {fmt(unknown_host.sum()):>7}")
    return df.assign(SEX=df["SEX"].mask(other_sex), HOST=df["HOST"].mask(unknown_host))


def peak_group(df: pd.DataFrame) -> pd.Series:
    counts = df.loc[df["YEAR"].between(*TRAIN_YEARS), "PEAKID"].value_counts()
    keep = counts.index[counts >= MIN_PEAK_ROWS]
    return df["PEAKID"].where(df["PEAKID"].isin(keep), "OTHER")


def anomalous_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in rule.items():
        mask &= df[column].isin(value if isinstance(value, tuple) else [value])
    return mask


def anomalous_season(df: pd.DataFrame) -> pd.Series:
    flag = pd.Series(False, index=df.index)
    for rule in ANOMALOUS_SEASONS:
        flag |= anomalous_mask(df, rule)
    return flag


def build_dataset(tables: dict[str, pd.DataFrame], exclude_disputed: bool = True) -> pd.DataFrame:
    df = join(tables)
    print(f"  {'join members+exped+peaks':26s} {fmt(len(df)):>7}")
    df = apply_filters(df, BASE_FILTERS)
    assert len(df) == EXPECTED_NOT_HIRED_1990_2025, f"non-hired members 1990-2025: {len(df)}"
    df = apply_filters(df, CLIMB_FILTERS + ((DISPUTED_FILTER,) if exclude_disputed else ()))
    df = clean_values(df)

    df = df.assign(ROUTE1=clean_route(df["ROUTE1"]), PEAK_GROUP=peak_group(df), ANOMALOUS_SEASON=anomalous_season(df))
    assert not df["ROUTE1"].str.contains(ROUTE_POSTFACTUM_RE, regex=True, na=False).any()

    excluded = FORBIDDEN_RAW | TECHNICAL | DROP | set(KEYS) | set(DERIVED)
    features = [c for c in df.columns if c not in excluded]
    df = df[KEYS + TARGETS + DERIVED + features].reset_index(drop=True)

    assert not df.duplicated(["EXPID", "YEAR", "MEMBID"]).any()
    check_no_leakage(df.drop(columns=["EXPID", "MEMBID", "ANOMALOUS_SEASON"] + TARGETS))
    return df


def print_summary(df: pd.DataFrame, raw_route1: pd.Series) -> None:
    known = df["YEAR"] <= SPLITS["val 2018-2021"][1]
    print("\ndataset_v1 summary")
    print(f"  {fmt(len(df))} rows x {df.shape[1]} columns, {fmt(df['EXPID'].nunique())} expeditions")
    print(f"  Success / death rate (1990-2021 only, the test stays closed): "
          f"{df.loc[known, 'MSUCCESS'].mean():.1%} / {df.loc[known, 'DEATH'].mean():.2%}")
    print("  Rows per period: " + ", ".join(
        f"{name} {fmt(df['YEAR'].between(lo, hi).sum())}" for name, (lo, hi) in SPLITS.items()))

    groups = df["PEAK_GROUP"].value_counts()
    print(f"\n  PEAK_GROUP: {len(groups) - 1} peaks with >= {MIN_PEAK_ROWS} rows in 1990-2017 + OTHER "
          f"({fmt(groups['OTHER'])} rows)")
    print(f"  ROUTE1 cleaned of after-the-fact notes in {fmt((raw_route1 != df['ROUTE1']).sum())} rows")

    print("\n  ANOMALOUS_SEASON:")
    for rule in ANOMALOUS_SEASONS:
        print(f"    {rule}: {fmt(anomalous_mask(df, rule).sum())} rows")
    print(f"    total: {fmt(df['ANOMALOUS_SEASON'].sum())} ({df['ANOMALOUS_SEASON'].mean():.1%})")

    print(f"\n  Columns: {', '.join(df.columns)}")


def main() -> None:
    tables = {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in TABLES}
    print("Filters (rows before -> after):")
    df = build_dataset(tables)
    assert len(df) == EXPECTED_ROWS, f"dataset_v1: {len(df)} rows, expected {EXPECTED_ROWS}"

    df.to_parquet(DATASET_PATH, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(DATASET_PATH), df)
    print(f"\n{DATASET_PATH.relative_to(ROOT)}  {DATASET_PATH.stat().st_size / 2**20:.1f} MB")

    raw = join(tables).set_index(["EXPID", "YEAR", "MEMBID"])["ROUTE1"]
    print_summary(df, raw.reindex(pd.MultiIndex.from_frame(df[["EXPID", "YEAR", "MEMBID"]])).reset_index(drop=True))


if __name__ == "__main__":
    main()
