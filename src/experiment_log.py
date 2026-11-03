"""Append results to the Experiment log table in README.md and to reports/experiment_log.csv (idempotent)."""
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CSV = ROOT / "reports" / "experiment_log.csv"
HEADER = "## Experiment log"
COLUMNS = ["date", "who", "model", "view", "roc_auc", "roc_auc_lo", "roc_auc_hi", "brier", "brier_lo", "brier_hi",
           "log_loss", "log_loss_lo", "log_loss_hi", "note"]


def _cell(r: dict, m: str) -> str:
    return f"{r[m]['estimate']:.3f} [{r[m]['ci_low']:.3f}, {r[m]['ci_high']:.3f}]"


def _csv_rows(model, view, res, note, who, day):
    row = {"date": day, "who": who, "model": model, "view": view, "note": note}
    for m in ("roc_auc", "brier", "log_loss"):
        row.update({m: res[m]["estimate"], f"{m}_lo": res[m]["ci_low"], f"{m}_hi": res[m]["ci_high"]})
    return row


def log_experiment(model: str, res_all: dict, res_clean: dict, conclusion: str, who: str = "B") -> None:
    """res_all: evaluate() on all val rows (goes into README); res_clean: val without anomalous seasons."""
    day = date.today().isoformat()
    model, conclusion = model.replace("|", "/"), conclusion.replace("|", "/")

    # --- CSV (replace rows of the same model)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=COLUMNS)
    old = old[old["model"] != model]
    new = pd.DataFrame([_csv_rows(model, "val_all", res_all, conclusion, who, day),
                        _csv_rows(model, "val_no_anomalous", res_clean, conclusion, who, day)])
    pd.concat([old, new], ignore_index=True)[COLUMNS].to_csv(CSV, index=False)

    # --- README row
    clean = (f"no-anomalous val: AUC {res_clean['roc_auc']['estimate']:.3f}, "
             f"Brier {res_clean['brier']['estimate']:.3f}, LL {res_clean['log_loss']['estimate']:.3f}")
    row = (f"| {day} | {who} | {model} | {_cell(res_all, 'roc_auc')} | {_cell(res_all, 'brier')} | "
           f"LL {res_all['log_loss']['estimate']:.3f}; {clean}. {conclusion} |")
    lines = README.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == HEADER)
    end = start + 1
    while end < len(lines) and not lines[end].lstrip().startswith("|"):
        end += 1
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    block = [l for l in lines[start + 1:end]]
    block = [l for l in block if not (f"| {who} | {model} |" in l)]
    lines[start + 1:end] = block + [row]
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")
