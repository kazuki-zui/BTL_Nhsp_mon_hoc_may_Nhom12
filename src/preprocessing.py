"""Preprocessing pipelines (Flow A vs Flow B) with strict train-first splitting to prevent data leakage."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.utils import resample

from .config import PCA_COLUMNS, RANDOM_STATE, TARGET_COL, TEST_SIZE, VAL_SIZE

FLOW_A_FEATURES = ["Time"] + PCA_COLUMNS + ["Amount"]
FLOW_B_CONT_FEATURES = PCA_COLUMNS + ["Log_Amount"]
FLOW_B_PASSTHROUGH_FEATURES = ["Hour_sin", "Hour_cos", "Is_Zero_Amount"]
FLOW_B_FEATURES = FLOW_B_CONT_FEATURES + FLOW_B_PASSTHROUGH_FEATURES


@dataclass
class PreparedSplit:
    """Container for prepared train/validation/test arrays and metadata."""

    name: str
    feature_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_raw_df: pd.DataFrame
    transformer: object
    metadata: Dict[str, object]


def split_train_val_test(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    strategy: str = "stratified",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train, validation, and test partitions before any feature fitting."""
    if strategy == "stratified":
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col],
        )
        val_relative = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative,
            random_state=random_state,
            stratify=train_val_df[target_col],
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    if strategy == "time":
        ordered_df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
        n = len(ordered_df)
        train_end = int(n * (1.0 - test_size - val_size))
        val_end = int(n * (1.0 - test_size))
        return (
            ordered_df.iloc[:train_end].reset_index(drop=True),
            ordered_df.iloc[train_end:val_end].reset_index(drop=True),
            ordered_df.iloc[val_end:].reset_index(drop=True),
        )

    raise ValueError(f"Unsupported split strategy: {strategy}")


def engineer_flow_b_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-motivated cyclical time and log-amount features for Flow B."""
    engineered = df.copy()
    hours = (engineered["Time"] // 3600.0) % 24.0
    engineered["Hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    engineered["Hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    engineered["Log_Amount"] = np.log1p(engineered["Amount"].clip(lower=0.0))
    engineered["Is_Zero_Amount"] = (engineered["Amount"] == 0.0).astype(float)
    return engineered


def _build_flow_b_transformer(scaler_type: str = "robust") -> ColumnTransformer:
    """Build ColumnTransformer scaling continuous columns while preserving engineered indicators."""
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "power":
        scaler = PowerTransformer(method="yeo-johnson")
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    return ColumnTransformer(
        transformers=[
            ("cont", scaler, FLOW_B_CONT_FEATURES),
            ("pass", "passthrough", FLOW_B_PASSTHROUGH_FEATURES),
        ],
        remainder="drop",
    )


def prepare_flow_a(
    df: pd.DataFrame,
    split_strategy: str = "stratified",
    scale: bool = True,
    random_state: int = RANDOM_STATE,
) -> PreparedSplit:
    """Flow A (Baseline pipeline): raw 30 features, train/val/test split, optional StandardScaler."""
    train_df, val_df, test_df = split_train_val_test(
        df, random_state=random_state, strategy=split_strategy
    )

    X_train_raw = train_df[FLOW_A_FEATURES].to_numpy(dtype=np.float64)
    X_val_raw = val_df[FLOW_A_FEATURES].to_numpy(dtype=np.float64)
    X_test_raw = test_df[FLOW_A_FEATURES].to_numpy(dtype=np.float64)

    scaler: Optional[StandardScaler] = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

    return PreparedSplit(
        name=f"FlowA_{split_strategy}_{'scaled' if scale else 'unscaled'}",
        feature_names=list(FLOW_A_FEATURES),
        X_train=X_train,
        y_train=train_df[TARGET_COL].to_numpy(dtype=np.int64),
        X_val=X_val,
        y_val=val_df[TARGET_COL].to_numpy(dtype=np.int64),
        X_test=X_test,
        y_test=test_df[TARGET_COL].to_numpy(dtype=np.int64),
        test_raw_df=test_df.copy(),
        transformer=scaler,
        metadata={
            "deduplicated": False,
            "n_rows_used": len(df),
            "split_strategy": split_strategy,
            "scaler": "StandardScaler" if scale else "None",
        },
    )


def prepare_flow_b(
    df: pd.DataFrame,
    split_strategy: str = "stratified",
    scaler_type: str = "robust",
    random_state: int = RANDOM_STATE,
) -> PreparedSplit:
    """Flow B (Improved pipeline): deduplicate -> split -> cyclical time & log-amount -> Robust/Power scaling."""
    n_before = len(df)
    dedup_df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_after = len(dedup_df)

    engineered_df = engineer_flow_b_features(dedup_df)
    train_df, val_df, test_df = split_train_val_test(
        engineered_df, random_state=random_state, strategy=split_strategy
    )

    transformer = _build_flow_b_transformer(scaler_type=scaler_type)
    X_train = transformer.fit_transform(train_df[FLOW_B_FEATURES])
    X_val = transformer.transform(val_df[FLOW_B_FEATURES])
    X_test = transformer.transform(test_df[FLOW_B_FEATURES])

    return PreparedSplit(
        name=f"FlowB_{split_strategy}_{scaler_type}",
        feature_names=list(FLOW_B_FEATURES),
        X_train=X_train,
        y_train=train_df[TARGET_COL].to_numpy(dtype=np.int64),
        X_val=X_val,
        y_val=val_df[TARGET_COL].to_numpy(dtype=np.int64),
        X_test=X_test,
        y_test=test_df[TARGET_COL].to_numpy(dtype=np.int64),
        test_raw_df=test_df.copy(),
        transformer=transformer,
        metadata={
            "deduplicated": True,
            "duplicates_removed": n_before - n_after,
            "n_rows_used": n_after,
            "split_strategy": split_strategy,
            "scaler": scaler_type,
        },
    )


def balance_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "undersample",
    neg_to_pos_ratio: int = 10,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample ONLY training data to mitigate extreme class imbalance without touching val/test."""
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    if method == "undersample":
        target_neg = min(len(neg_idx), len(pos_idx) * neg_to_pos_ratio)
        sampled_neg_idx = resample(
            neg_idx, replace=False, n_samples=target_neg, random_state=random_state
        )
        combined_idx = np.concatenate([pos_idx, sampled_neg_idx])
    elif method == "oversample":
        target_pos = max(len(pos_idx), len(neg_idx) // neg_to_pos_ratio)
        sampled_pos_idx = resample(
            pos_idx, replace=True, n_samples=target_pos, random_state=random_state
        )
        combined_idx = np.concatenate([neg_idx, sampled_pos_idx])
    else:
        raise ValueError(f"Unsupported balancing method: {method}")

    rng = np.random.default_rng(random_state)
    rng.shuffle(combined_idx)
    return X_train[combined_idx], y_train[combined_idx]
