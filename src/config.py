"""Central configuration for Group 12 Credit Card Fraud Detection project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
REPORT_DIR = PROJECT_ROOT / "report"

# Default dataset locations checked in order
CANDIDATE_DATA_PATHS = [
    PROJECT_ROOT / "data" / "creditcard.csv",
    Path(r"D:\creditcard.csv\creditcard.csv"),
    Path(r"D:\creditcard.csv"),
]

# Reproducibility and data split settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # 20% of full dataset (0.25 of remaining 80% -> 60/20/20 split)

TARGET_COL = "Class"
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
KEY_CORRELATED_FEATURES = ["V17", "V14", "V12", "V10", "V16", "V3", "V7", "V11", "V4"]


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    for directory in (OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
