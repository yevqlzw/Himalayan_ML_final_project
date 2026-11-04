import pandas as pd


def _numbered(prefix: str, n: int) -> set[str]:
    return {f"{prefix}{i}" for i in range(1, n + 1)}


_OUTCOME_FIELDS = (
    _numbered("SUCCESS", 4)
    | {"SMTMEMBERS", "SMTHIRED", "SMTDATE", "SMTDAYS", "HIGHPOINT", "TERMREASON", "TERMDATE",
       "TERMNOTE", "TOTDAYS", "MDEATHS", "HDEATHS", "ACCIDENTS"}
    | _numbered("MSMTDATE", 3)
    | _numbered("MSMTTIME", 3)
    | {"MSMTBID", "MSMTTERM", "MHIGHPT", "MPERHIGHPT", "MO2DESCENT", "MO2MEDICAL"}
    | {"DEATH", "DEATHDATE", "DEATHTIME", "DEATHTYPE", "DEATHHGTM", "DEATHCLASS", "DEATHNOTE", "DEATHRTE"}
    | {"INJURY", "INJURYDATE", "INJURYTIME", "INJURYTYPE", "INJURYHGTM"}
)


_AFTER_CLIMB_FIELDS = (
    _numbered("ASCENT", 4)
    | {"CLAIMED", "DISPUTED", "SMTTIME", "OTHERSMTS", "CAMPSITES", "ROUTEMEMO", "ACHIEVMENT",
       "TRAVERSE", "SKI", "PARAPENTE", "CAMPS", "ROPE", "O2USED", "O2NONE", "O2DESCENT", "O2SLEEP",
       "O2MEDICAL", "O2TAKEN", "O2UNKWN"}
    | {"MSUCCESS", "MCLAIMED", "MDISPUTED", "MSOLO", "MTRAVERSE", "MSKI", "MPARAPENTE", "MSPEED",
       "MO2USED", "MO2NONE", "MO2CLIMB", "MO2SLEEP", "MO2NOTE", "AMS", "WEATHER", "MEMBERMEMO", "NECROLOGY"}
    | _numbered("MROUTE", 3)
    | _numbered("MASCENT", 3)
    | _numbered("MSMTNOTE", 3)
)


_NOT_KNOWN_AT_START = {
    "O2CLIMB", "BCDATE", "BCONLY", "NOTTOBC",
    "OPEN", "RESTRICT", "PSTATUS", "UNLISTED", "TREKKING", "TREKYEAR",
    "PYEAR", "PSEASON", "PEXPID", "PSMTDATE", "PCOUNTRY", "PSUMMITERS", "PSMTNOTE",
    "PEAKMEMO", "REFERMEMO", "PHOTOMEMO",
}


FORBIDDEN_RAW = frozenset(_OUTCOME_FIELDS | _AFTER_CLIMB_FIELDS | _NOT_KNOWN_AT_START)
FORBIDDEN = FORBIDDEN_RAW | {"ANOMALOUS_SEASON"}

TECHNICAL = frozenset({"CHKSUM", "MCHKSUM", "PRIMRTE", "PRIMMEM", "PRIMREF", "PRIMID", "HCN", "EXPID", "MEMBID"})


def _blocked(column, names: frozenset[str]) -> bool:
    name = str(column).upper()
    return any(name == field or name.startswith(f"{field}_") for field in names)


def check_no_leakage(df: pd.DataFrame) -> None:
    forbidden = [c for c in df.columns if _blocked(c, FORBIDDEN)]
    technical = [c for c in df.columns if _blocked(c, TECHNICAL)]
    if forbidden or technical:
        raise ValueError(f"Features include forbidden fields {forbidden} or technical fields {technical}")
