from pathlib import Path

import pandas as pd
from dbfread import DBF

from src.config import PROCESSED_DIR, RAW_DIR, ROOT

TABLES = ("exped", "members", "peaks")

EXPECTED_SHAPE = {"exped": (11_854, 66), "members": (94_999, 78), "peaks": (503, 25)}
EXPECTED_YEARS = (1905, 2025)
EXPECTED_NOT_HIRED_1990_2025 = 58_100

YEAR_COLUMNS = {"YEAR", "MYEAR", "YOB", "TREKYEAR", "PYEAR"}
ZERO_AS_MISSING = {"AGE", "CALCAGE"}

KEYS = {"exped": ["EXPID", "YEAR"], "members": ["EXPID", "MYEAR", "MEMBID"], "peaks": ["PEAKID"]}


def find_table(raw_dir: Path, name: str) -> Path:
    files = {p.name.lower(): p for p in raw_dir.iterdir()}
    dbf = files.get(f"{name}.dbf")
    if dbf is None:
        raise FileNotFoundError(f"{name}.DBF not found in {raw_dir}")
    if f"{name}.fpt" not in files:
        raise FileNotFoundError(f"Memo file {name}.FPT not found next to {dbf.name}")
    return dbf


def _to_text(s: pd.Series) -> pd.Series:
    s = s.astype("str").str.strip()
    return s.mask(s == "")


def convert_column(s: pd.Series, field) -> pd.Series:
    if field.name in YEAR_COLUMNS:
        x = pd.to_numeric(_to_text(s))
        return x.where(x != 0).astype("Int64")
    if field.type in ("C", "M"):
        return _to_text(s)
    if field.type == "N":
        x = pd.to_numeric(s)
        if field.name in ZERO_AS_MISSING:
            x = x.where(x != 0)
        return x.astype("Int64") if field.decimal_count == 0 else x.astype("float64")
    if field.type == "L":
        return s.astype(bool) if s.notna().all() else s.astype("boolean")
    if field.type == "D":
        return pd.to_datetime(s).astype("datetime64[ms]")
    raise ValueError(f"Unknown field type {field.name}: {field.type}")


def load_table(path: Path) -> pd.DataFrame:
    table = DBF(path, load=False, encoding="cp1252", char_decode_errors="replace")
    raw = pd.DataFrame(list(table), columns=table.field_names, dtype=object)
    return pd.DataFrame({f.name: convert_column(raw[f.name], f) for f in table.fields})


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {name: load_table(find_table(raw_dir, name)) for name in TABLES}


def not_hired_1990_2025(members: pd.DataFrame) -> int:
    return int((~members["HIRED"] & members["MYEAR"].between(1990, 2025)).sum())


def check(tables: dict[str, pd.DataFrame]) -> None:
    for name, expected in EXPECTED_SHAPE.items():
        assert tables[name].shape == expected, f"{name}: {tables[name].shape}, expected {expected}"
    for name, key in KEYS.items():
        n_dup = tables[name].duplicated(key).sum()
        assert n_dup == 0, f"{name}: {n_dup} duplicates of key {key}"

    exped, members, peaks = tables["exped"], tables["members"], tables["peaks"]
    years = exped["YEAR"]
    assert years.notna().all(), "exped: empty YEAR values"
    assert (years.min(), years.max()) == EXPECTED_YEARS, f"years {years.min()}-{years.max()}"
    n_not_hired = not_hired_1990_2025(members)
    assert n_not_hired == EXPECTED_NOT_HIRED_1990_2025, f"non-hired members 1990-2025: {n_not_hired}"

    joined = members.merge(exped[["EXPID", "YEAR", "PEAKID", "SEASON"]], how="left",
                           left_on=["EXPID", "MYEAR"], right_on=["EXPID", "YEAR"],
                           suffixes=("", "_EXPED"), indicator=True)
    assert len(joined) == len(members)
    assert (joined["_merge"] == "both").all(), "members without an expedition in exped"
    assert (joined["PEAKID"] == joined["PEAKID_EXPED"]).all(), "members.PEAKID != exped.PEAKID"
    assert (joined["MSEASON"] == joined["SEASON"]).all(), "members.MSEASON != exped.SEASON"
    assert exped["PEAKID"].isin(peaks["PEAKID"]).all(), "expeditions on peaks missing from peaks"


def fmt(n: int) -> str:
    return f"{n:,}"


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    exped, members = tables["exped"], tables["members"]
    print("\nSummary")
    for name, df in tables.items():
        dtypes = ", ".join(f"{t} {n}" for t, n in df.dtypes.astype(str).value_counts().items())
        print(f"  {name:8s} {fmt(len(df)):>7} rows x {df.shape[1]} fields  ({dtypes})")

    print(f"\n  Expedition years: {exped['YEAR'].min()}-{exped['YEAR'].max()}")
    print(f"  Non-hired members 1990-2025: {fmt(not_hired_1990_2025(members))}")
    print(f"  Expeditions without any members record: {(~exped['EXPID'].isin(members['EXPID'])).sum()}")

    dup = exped[exped["EXPID"].duplicated(keep=False)]
    pairs = dup.groupby("EXPID")["YEAR"].agg(lambda y: "/".join(map(str, sorted(y))))
    print(f"  EXPIDs repeated 100 years apart ({len(pairs)}): "
          + ", ".join(f"{i} ({y})" for i, y in pairs.items()))
    print(f"  Joining members and exped on EXPID alone would give "
          f"{fmt(len(members.merge(exped[['EXPID']], on='EXPID')))} rows instead of {fmt(len(members))}")

    print("\n  Logical fields with unknown values (<NA>):")
    for name, df in tables.items():
        for col in df.columns[df.dtypes.astype(str) == "boolean"]:
            print(f"    {name}.{col}: {fmt(df[col].isna().sum())} ({df[col].isna().mean():.1%})")

    empty = [f"{name}.{c}" for name, df in tables.items() for c in df.columns if df[c].isna().all()]
    print(f"\n  Completely empty fields: {', '.join(empty) or 'none'}")

    base = members[~members["HIRED"] & members["MYEAR"].between(1990, 2025)]
    print(f"  Among non-hired 1990-2025: CALCAGE = NaN {fmt(base['CALCAGE'].isna().sum())}, "
          f"YOB = NaN {fmt(base['YOB'].isna().sum())}")


def main() -> None:
    tables = load_all()
    check(tables)
    print("All control checks passed.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(path), df)
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 2**20:.1f} MB")

    print_summary(tables)


if __name__ == "__main__":
    main()
