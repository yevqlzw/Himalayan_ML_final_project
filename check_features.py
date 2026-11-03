import pandas as pd

from src.dataset import SPLITS
from src.features import CATEGORICAL, FEATURES, OUT_PATH
from src.leakage import check_no_leakage

TARGET = "MSUCCESS"
df = pd.read_parquet(OUT_PATH)
print("shape:", df.shape)

# assign split from dataset.SPLITS (inclusive year ranges); every row must land in exactly one
part = pd.Series(pd.NA, index=df.index, dtype="object")
for name, (lo, hi) in SPLITS.items():
    part[df["YEAR"].between(lo, hi)] = name.split()[0]
assert part.notna().all(), "rows outside the split ranges"

# test: row counts only, labels stay closed
for s in ("train", "val", "test"):
    m = part == s
    line = f"{s:5s} rows={m.sum():>6,} expeditions={df.loc[m, 'EXPID'].nunique():>5,}"
    if s != "test":
        line += (f"  {TARGET} mean={df.loc[m, TARGET].mean():.3f}"
                 f"  anomalous share={df.loc[m, 'ANOMALOUS_SEASON'].mean():.1%}")
    print(line)

# expeditions must not be shared across splits
sets = {s: set(df.loc[part == s, "EXPID"]) for s in ("train", "val", "test")}
assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])

# missing values on train+val only
dev = df[part.isin(["train", "val"])]
miss = dev[FEATURES].isna().mean().sort_values(ascending=False)
print("\nMissing share in train+val (features with >0):")
print(miss[miss > 0].round(4).to_string())

# leakage: raises ValueError if any banned/technical column is among FEATURES
check_no_leakage(df[FEATURES])
assert set(CATEGORICAL) <= set(FEATURES) and len(FEATURES) == len(set(FEATURES))
print("\nOK: no forbidden or technical column in FEATURES.")
print("In the parquet but NOT features:", [c for c in df.columns if c not in FEATURES])

print("\nDtypes of columns needing care:")
print(df[FEATURES].dtypes.astype(str).value_counts().to_string())
print(df[CATEGORICAL].dtypes.astype(str).to_string())