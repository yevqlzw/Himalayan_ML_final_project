from datetime import date

import pandas as pd

from src.config import EXPERIMENT_LOG_PATH, README_PATH

HEADER = "## Experiment log"
COLUMNS = ["date", "model", "view", "roc_auc", "roc_auc_lo", "roc_auc_hi", "brier", "brier_lo", "brier_hi",
           "log_loss", "log_loss_lo", "log_loss_hi", "note"]


def _cell(result: dict, metric: str) -> str:
    return f"{result[metric]['estimate']:.3f} [{result[metric]['ci_low']:.3f}, {result[metric]['ci_high']:.3f}]"


def _csv_row(model: str, view: str, result: dict, note: str, day: str) -> dict:
    row = {"date": day, "model": model, "view": view, "note": note}
    for metric in ("roc_auc", "brier", "log_loss"):
        row.update({metric: result[metric]["estimate"], f"{metric}_lo": result[metric]["ci_low"],
                    f"{metric}_hi": result[metric]["ci_high"]})
    return row


def log_experiment(model: str, res_all: dict, res_clean: dict, conclusion: str) -> None:
    day = date.today().isoformat()
    model, conclusion = model.replace("|", "/"), conclusion.replace("|", "/")

    old = pd.read_csv(EXPERIMENT_LOG_PATH) if EXPERIMENT_LOG_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    old = old[old["model"] != model]
    new = pd.DataFrame([_csv_row(model, "val_all", res_all, conclusion, day),
                        _csv_row(model, "val_no_anomalous", res_clean, conclusion, day)])
    table = new if old.empty else pd.concat([old, new], ignore_index=True)
    table[COLUMNS].to_csv(EXPERIMENT_LOG_PATH, index=False)

    clean = (f"no-anomalous val: AUC {res_clean['roc_auc']['estimate']:.3f}, "
             f"Brier {res_clean['brier']['estimate']:.3f}, LL {res_clean['log_loss']['estimate']:.3f}")
    row = (f"| {day} | {model} | {_cell(res_all, 'roc_auc')} | {_cell(res_all, 'brier')} | "
           f"LL {res_all['log_loss']['estimate']:.3f}; {clean}. {conclusion} |")
    lines = README_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == HEADER)
    end = start + 1
    while end < len(lines) and not lines[end].lstrip().startswith("|"):
        end += 1
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    block = [line for line in lines[start + 1:end] if f"| {model} |" not in line]
    lines[start + 1:end] = block + [row]
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
