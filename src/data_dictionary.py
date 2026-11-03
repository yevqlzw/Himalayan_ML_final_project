import re
from pathlib import Path

import pandas as pd

from src.leakage import FORBIDDEN_RAW, TECHNICAL
from src.load import EXPECTED_SHAPE, OUT_DIR, ROOT, TABLES





GUIDE_PATH = ROOT / "docs" / "himalayan_fields_guide.txt"
OUT_CSV = ROOT / "docs" / "data_dictionary.csv"

SECTIONS = {
    "Peaks Table (PEAKS)": "peaks",
    "Expeditions Table (EXPED)": "exped",
    "Members Table (MEMBERS)": "members",
    "Literature References Table (REFER)": None,
}
FIELD_RE = re.compile(
    r"^\s*(?P<desc>\S.*?)\s{2,}(?P<name>[A-Z][A-Z0-9]+)\s+[CNDLM]\s+\d+(?:\s+\((?P<extra>[^)]*)\))?\s*$"
)
DASH = "\N{EN DASH}"
CODE_LINE_RE = re.compile(rf"^\s*\d+\s+{DASH}\s")
CODE_RE = re.compile(rf"(\d+)\s+{DASH}\s+(.+?)(?=\s+\d+\s+{DASH}\s|\s*$)")

FIRST_ASCENT = ("First ascent data: for 378 expeditions this is the expedition itself (PEXPID). Safe derived "
                "feature: 'peak already climbed' = (PYEAR, PSEASON) earlier than (YEAR, SEASON).")
CURRENT_PEAK_STATE = "Current state of the peak, not as of the expedition year."
CURRENT_DB_STATE = "Current state of the database, contains later information."
FILTER_ONLY = "Recorded after the fact. Filter only, not a feature."
DISPUTED = "Filter: exclude disputed ascents and check their impact separately."
UNKNOWN_IS_NA = "Feature. NA means 'unknown', not False."
SHERPA_RISK = "Used for Sherpa risk (not a feature)."

NOTES = {
    "EXPID": "Identifier, not a feature. Not unique on its own: the key is (EXPID, YEAR). Groups for GroupKFold.",
    "MEMBID": "Identifier, not a feature. Members key: (EXPID, MYEAR, MEMBID).",
    "YEAR": "Key together with EXPID. Text in DBF; 0 or empty -> NaN.",
    "MYEAR": "Matches exped.YEAR when joined on (EXPID, MYEAR).",
    "YOB": "Used for the person ID (FNAME + LNAME + YOB). Text in DBF; empty -> NaN.",
    "AGE": "Empty in the public version (all 0 -> NaN). Age comes from CALCAGE.",
    "BIRTHDATE": "Empty in the public version.",
    "CALCAGE": "Age; 0 -> NaN.",
    "HIRED": "Filter: the main model uses HIRED == False only.",
    "BCONLY": FILTER_ONLY,
    "NOTTOBC": FILTER_ONLY,
    "BCDATE": "Known only after reaching base camp; empty if base camp was not reached.",
    "MSUCCESS": "Target of the main task.",
    "DEATH": "Target of the death risk analysis.",
    "MSMTTERM": "Filter: drop 17 (Did not climb or intent to summit).",
    "MDISPUTED": DISPUTED,
    "MCLAIMED": DISPUTED,
    "O2CLIMB": "Actual use, not a plan: excluded from prediction. Treatment in the causal analysis.",
    "ROUTE1": ("Feature. Contains after-the-fact notes '(to 6000m)' (in 1990-2025 only in unsuccessful "
               "expeditions), '(intended)', '(up)/(down)'; removed in dataset_v1."),
    **{f"ROUTE{i}": "Not a feature. Contains the same after-the-fact notes as ROUTE1." for i in range(2, 5)},
    "SEX": "Feature. Values other than M/F (one 'X') -> NaN in dataset_v1.",
    "HOST": "Feature. 0 = Unknown -> NaN in dataset_v1.",
    "TOTMEMBERS": "Feature. 0 in one 2025 expedition that has member records; kept, HIRED_PER_MEMBER is NaN there.",
    "COMRTE": UNKNOWN_IS_NA,
    "STDRTE": UNKNOWN_IS_NA,
    "PEAKID": "Feature; peaks with < 100 records -> OTHER.",
    "TOTHIRED": "Feature; also used for Sherpa risk.",
    "SMTHIRED": SHERPA_RISK,
    "HDEATHS": SHERPA_RISK,
    "OPEN": CURRENT_PEAK_STATE,
    "RESTRICT": CURRENT_PEAK_STATE,
    "UNLISTED": CURRENT_PEAK_STATE,
    "PSTATUS": f"{CURRENT_PEAK_STATE} Unclimbed means there was no success.",
    "TREKKING": f"{CURRENT_PEAK_STATE} The year it became a trekking peak is in TREKYEAR.",
    "TREKYEAR": CURRENT_PEAK_STATE,
    **dict.fromkeys(["PYEAR", "PSEASON", "PEXPID", "PSMTDATE", "PCOUNTRY", "PSUMMITERS", "PSMTNOTE"], FIRST_ASCENT),
    **dict.fromkeys(["PEAKMEMO", "REFERMEMO", "PHOTOMEMO"], CURRENT_DB_STATE),
}








def parse_guide(path: Path = GUIDE_PATH) -> dict[tuple[str, str], dict]:
    fields, table, current = {}, None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() in SECTIONS:
            table, current = SECTIONS[line.strip()], None
            continue
        if table is None:
            continue
        match = FIELD_RE.match(line)
        if match:
            desc = match["desc"] + (f" ({match['extra']})" if match["extra"] else "")
            current = fields[(table, match["name"])] = {"description": desc, "codes": {}}
        elif current is not None and CODE_LINE_RE.match(line):
            current["codes"].update({int(k): v.strip() for k, v in CODE_RE.findall(line)})
    return fields







def timing(field: str) -> str:
    if field in FORBIDDEN_RAW:
        return "after"
    if field in TECHNICAL:
        return "technical"
    return "before"







def examples(s: pd.Series, has_codes: bool) -> str:
    v = s.dropna()
    if v.empty:
        return ""
    if pd.api.types.is_datetime64_any_dtype(v):
        return f"{v.min().date()} ... {v.max().date()}"
    if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v) and not has_codes:
        return f"{v.min()} ... {v.max()}"
    text = v.astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 40)
    return " | ".join(text.value_counts().head(3).index)







def model_rows(tables: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    exped, members, peaks = tables["exped"], tables["members"], tables["peaks"]
    exped_rows = exped["YEAR"].between(1990, 2025)
    return {
        "exped": exped_rows,
        "members": members["MYEAR"].between(1990, 2025) & ~members["HIRED"],
        "peaks": peaks["PEAKID"].isin(exped.loc[exped_rows, "PEAKID"]),
    }







def build(tables: dict[str, pd.DataFrame], guide: dict[tuple[str, str], dict]) -> pd.DataFrame:
    rows_model = model_rows(tables)
    records = []
    for table in TABLES:
        df = tables[table]
        for field in df.columns:
            info = guide[(table, field)]
            records.append({
                "table": table,
                "field": field,
                "dtype": str(df[field].dtype),
                "missing_pct": round(100 * df[field].isna().mean(), 1),
                "missing_pct_model": round(100 * df.loc[rows_model[table], field].isna().mean(), 1),
                "examples": examples(df[field], bool(info["codes"])),
                "description": info["description"],
                "codes": "; ".join(f"{k}={v}" for k, v in sorted(info["codes"].items())),
                "timing": timing(field),
                "note": NOTES.get(field, ""),
            })
    return pd.DataFrame(records)








def check(tables: dict[str, pd.DataFrame], guide: dict[tuple[str, str], dict]) -> None:
    data_fields = {(t, f) for t in TABLES for f in tables[t].columns}
    assert len(data_fields) == sum(cols for _, cols in EXPECTED_SHAPE.values())
    assert not data_fields - guide.keys(), f"no description in the Guide: {sorted(data_fields - guide.keys())}"
    assert not guide.keys() - data_fields, f"in the Guide but not in the data: {sorted(guide.keys() - data_fields)}"
    names = {f for _, f in data_fields}
    listed = FORBIDDEN_RAW | TECHNICAL | NOTES.keys()
    assert listed <= names, f"unknown fields in the lists: {sorted(listed - names)}"







def unknown_codes(tables: dict[str, pd.DataFrame], guide: dict[tuple[str, str], dict]) -> dict[str, list]:
    found = {}
    for (table, field), info in guide.items():
        if info["codes"]:
            extra = sorted(set(tables[table][field].dropna().astype(int)) - info["codes"].keys())
            if extra:
                found[f"{table}.{field}"] = extra
    return found








def main() -> None:
    tables = {name: pd.read_parquet(OUT_DIR / f"{name}.parquet") for name in TABLES}
    guide = parse_guide()
    check(tables, guide)
    dictionary = build(tables, guide)
    dictionary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"{OUT_CSV.relative_to(ROOT)}: {len(dictionary)} fields, all described in the Guide")
    print(f"Fields with codes: {(dictionary['codes'] != '').sum()}\n")
    print(pd.crosstab(dictionary["table"], dictionary["timing"], margins=True, margins_name="total")
          .reindex(list(TABLES) + ["total"]).to_string())

    for when in ("before", "technical"):
        print(f"\n{when}:")
        for table, group in dictionary[dictionary["timing"] == when].groupby("table", sort=False):
            print(f"  {table}: {', '.join(group['field'])}")

    codes = unknown_codes(tables, guide)
    print("\nCodes in the data missing from the Guide:", codes or "none")


if __name__ == "__main__":
    main()
