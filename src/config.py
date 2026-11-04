from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "HIMDATA"
PROCESSED_DIR = ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset_v1.parquet"
FEATURES_PATH = PROCESSED_DIR / "features_v1.parquet"
DOCS_DIR = ROOT / "docs"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
FINAL_MODEL_PATH = REPORTS_DIR / "final_model.json"
EXPERIMENT_LOG_PATH = REPORTS_DIR / "experiment_log.csv"
README_PATH = ROOT / "README.md"

RANDOM_STATE = 42
TARGET = "MSUCCESS"
GROUP = "EXPID"

TRAIN_YEARS = (1990, 2017)
VAL_YEARS = (2018, 2021)
TEST_YEARS = (2022, 2025)
