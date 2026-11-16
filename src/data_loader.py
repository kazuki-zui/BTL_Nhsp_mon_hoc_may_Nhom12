"""Dataset loading and schema validation utilities."""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CANDIDATE_DATA_PATHS, EXPECTED_COLUMNS, TARGET_COL


def resolve_data_path(custom_path: Optional[str] = None) -> Path:
    """Resolve the path to creditcard.csv from CLI arg, env var, or default locations."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    env_path = os.environ.get("CREDITCARD_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(CANDIDATE_DATA_PATHS)

    for path in candidates:
        if path.is_file():
            return path
        if path.is_dir() and (path / "creditcard.csv").is_file():
            return path / "creditcard.csv"

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find creditcard.csv. Checked: {searched}. "
        "Pass --data-path or place creditcard.csv inside the data/ directory."
    )


def load_creditcard_data(custom_path: Optional[str] = None) -> pd.DataFrame:
    """Load creditcard.csv and validate expected columns and target labels."""
    data_path = resolve_data_path(custom_path)
    df = pd.read_csv(data_path)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    unique_labels = set(df[TARGET_COL].dropna().unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Unexpected target labels in '{TARGET_COL}': {unique_labels}")

    return df
