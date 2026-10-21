"""Shared constants for the loan default project.

Why this file exists:
- One place to change paths / random_state / thresholds.
- Both 01 and 02 scripts import from here, so train/test stay consistent.
"""
from pathlib import Path

# loan/ is the project root (this file lives in loan/src/)
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_CSV = BASE_DIR / "accepted_2007_to_2018Q4.csv"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
NROWS = 10_000  # same 10k sample you used in eda.ipynb; raise later for full run
MISSING_THRESHOLD = 80.0  # drop cols with >80% missing (your step 3)
TARGET_COL = "target"

# 10 cols you already picked for deep univariate look
IMPORTANT_NUM_COLS = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "tot_cur_bal",
]
