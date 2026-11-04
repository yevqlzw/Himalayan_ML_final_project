import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.data_dictionary import parse_guide

TREND_YEARS = (1950, 2025)
PER = 1000
Z = 1.96
N_BOOTSTRAP = 500
AGE_BINS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]
COUNT_COLUMNS = ["CLIENTS", "CLIENT_DEATHS", "HIRED", "HIRED_DEATHS"]


def wilson_interval(deaths, n, z: float = Z) -> tuple[np.ndarray, np.ndarray]:
    deaths, n = np.asarray(deaths, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = deaths / n
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def rate_table(deaths: pd.Series, n: pd.Series) -> pd.DataFrame:
    low, high = wilson_interval(deaths, n)
    return pd.DataFrame({
        "deaths": deaths.astype(int),
        "n": n.astype(int),
        "per_1000": PER * deaths.astype(float) / n.astype(float),
        "low": PER * low,
        "high": PER * high,
    }, index=deaths.index)


def member_rates(df: pd.DataFrame, by) -> pd.DataFrame:
    grouped = df.groupby(by, observed=True)["DEATH"]
    return rate_table(grouped.sum(), grouped.size())


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age.astype(float), AGE_BINS, right=False, labels=AGE_LABELS)


def experience_group(prior: pd.Series) -> pd.Series:
    labels = np.where(prior.isna(), "unknown", np.where(prior.fillna(0) == 0, "first expedition", "experienced"))
    return pd.Series(labels, index=prior.index, name="EXPERIENCE")


def peak_type(peak_id: pd.Series, height: pd.Series) -> pd.Series:
    labels = np.select([peak_id.eq("EVER").to_numpy(), height.ge(8000).fillna(False).to_numpy(bool)],
                       ["Everest", "Other 8000 m"], "Below 8000 m")
    return pd.Series(labels, index=peak_id.index, name="PEAK_TYPE")


def clients_above_bc(members: pd.DataFrame) -> pd.DataFrame:
    return members[~members["HIRED"] & ~members["BCONLY"] & ~members["NOTTOBC"]]


def expedition_counts(exped: pd.DataFrame, members: pd.DataFrame, peaks: pd.DataFrame,
                      years: tuple[int, int] = TREND_YEARS) -> pd.DataFrame:
    clients = (clients_above_bc(members).groupby(["EXPID", "MYEAR"])["DEATH"]
               .agg(CLIENTS="size", CLIENT_DEATHS="sum").reset_index().rename(columns={"MYEAR": "YEAR"}))
    df = exped.loc[exped["YEAR"].between(*years), ["EXPID", "YEAR", "SEASON", "PEAKID", "TOTHIRED", "HDEATHS", "MDEATHS"]]
    df = df.merge(clients, on=["EXPID", "YEAR"], how="left", validate="one_to_one")
    df = df.merge(peaks[["PEAKID", "HEIGHTM"]], on="PEAKID", how="left", validate="many_to_one")
    return df.assign(
        CLIENTS=df["CLIENTS"].fillna(0).astype(int),
        CLIENT_DEATHS=df["CLIENT_DEATHS"].fillna(0).astype(int),
        HIRED=df["TOTHIRED"].astype(int),
        HIRED_DEATHS=df["HDEATHS"].astype(int),
        DECADE=(df["YEAR"] // 10 * 10).astype(int),
        PEAK_TYPE=peak_type(df["PEAKID"], df["HEIGHTM"]),
    ).reset_index(drop=True)


def shared_expeditions(counts: pd.DataFrame) -> pd.DataFrame:
    return counts[(counts["CLIENTS"] > 0) & (counts["HIRED"] > 0)].reset_index(drop=True)


def counter_rates(counts: pd.DataFrame, by) -> pd.DataFrame:
    sums = counts.groupby(by, observed=True)[COUNT_COLUMNS].sum()
    clients = rate_table(sums["CLIENT_DEATHS"], sums["CLIENTS"]).add_prefix("client_")
    hired = rate_table(sums["HIRED_DEATHS"], sums["HIRED"]).add_prefix("hired_")
    return pd.concat([clients, hired], axis=1)


def risk_ratio(sums: np.ndarray) -> float:
    clients, client_deaths, hired, hired_deaths = sums
    if client_deaths == 0 or hired == 0 or clients == 0:
        return np.nan
    return (hired_deaths / hired) / (client_deaths / clients)


def risk_ratio_interval(counts: pd.DataFrame, n_boot: int = N_BOOTSTRAP,
                        seed: int = RANDOM_STATE) -> tuple[float, float, float]:
    values = counts[COUNT_COLUMNS].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    ratios = [risk_ratio(values[rng.integers(0, len(values), len(values))].sum(axis=0)) for _ in range(n_boot)]
    low, high = np.nanpercentile(ratios, [2.5, 97.5])
    return risk_ratio(values.sum(axis=0)), low, high


def compare_clients_hired(counts: pd.DataFrame, by) -> pd.DataFrame:
    table = counter_rates(counts, by)
    intervals = {key: risk_ratio_interval(group) for key, group in counts.groupby(by, observed=True)}
    ratio = pd.DataFrame(intervals, index=["ratio", "ratio_low", "ratio_high"]).T
    return table.join(ratio)


def death_class_shares(members: pd.DataFrame, years: tuple[int, int] = TREND_YEARS) -> pd.DataFrame:
    labels = parse_guide()[("members", "DEATHCLASS")]["codes"]
    deaths = members[members["DEATH"] & members["MYEAR"].between(*years)]
    who = pd.Series(np.where(deaths["HIRED"], "Hired", "Clients"), index=deaths.index, name="WHO")
    phase = deaths["DEATHCLASS"].map(labels).rename("PHASE")
    return pd.crosstab(phase, who, normalize="columns").mul(100)
