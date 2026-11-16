"""Error analysis utilities for False Negative and False Positive inspection."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_config_nhom12"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, METRICS_DIR, TARGET_COL

sns.set_theme(style="whitegrid", context="notebook")


def analyze_errors(
    test_raw_df: pd.DataFrame,
    y_true: np.ndarray,
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    primary_model_key: str,
) -> dict:
    """Inspect False Negatives and False Positives and save structured examples & plots."""
    primary = model_predictions[primary_model_key]
    y_pred = primary["y_pred"]
    y_prob = primary["y_prob"]

    analysis_df = test_raw_df.copy()
    analysis_df["y_true"] = y_true
    analysis_df["y_pred"] = y_pred
    analysis_df["y_prob"] = y_prob
    analysis_df["Hour"] = (analysis_df["Time"] // 3600.0) % 24.0

    conditions = [
        (analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 1),
        (analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 0),
        (analysis_df["y_true"] == 0) & (analysis_df["y_pred"] == 1),
        (analysis_df["y_true"] == 0) & (analysis_df["y_pred"] == 0),
    ]
    labels = ["TP", "FN", "FP", "TN"]
    analysis_df["Outcome"] = np.select(conditions, labels, default="TN")

    key_cols = ["Amount", "Hour", "V17", "V14", "V12", "V10", "V4", "V11"]
    group_stats = {}
    for outcome in labels:
        sub = analysis_df[analysis_df["Outcome"] == outcome]
        group_stats[outcome] = {
            "count": int(len(sub)),
            "amount_sum": round(float(sub["Amount"].sum()), 2) if len(sub) else 0.0,
            "amount_mean": round(float(sub["Amount"].mean()), 2) if len(sub) else 0.0,
            "amount_median": round(float(sub["Amount"].median()), 2) if len(sub) else 0.0,
            "feature_medians": {col: round(float(sub[col].median()), 4) for col in key_cols} if len(sub) else {},
        }

    fn_cases = (
        analysis_df[analysis_df["Outcome"] == "FN"]
        .sort_values("Amount", ascending=False)
        .head(5)[["Time", "Hour", "Amount", "V17", "V14", "V12", "V10", "y_prob"]]
        .round(4)
        .to_dict(orient="records")
    )
    fp_cases = (
        analysis_df[analysis_df["Outcome"] == "FP"]
        .sort_values("y_prob", ascending=False)
        .head(5)[["Time", "Hour", "Amount", "V17", "V14", "V12", "V10", "y_prob"]]
        .round(4)
        .to_dict(orient="records")
    )

    # Cross-model overlap on actual fraud test samples
    fraud_idx = np.where(y_true == 1)[0]
    fraud_overlap = {}
    for model_key, preds in model_predictions.items():
        missed = int(np.sum(preds["y_pred"][fraud_idx] == 0))
        caught = int(np.sum(preds["y_pred"][fraud_idx] == 1))
        fraud_overlap[model_key] = {"caught_tp": caught, "missed_fn": missed}

    hard_for_all = []
    for idx in fraud_idx:
        missed_by = [k for k, preds in model_predictions.items() if preds["y_pred"][idx] == 0]
        if len(missed_by) >= max(2, len(model_predictions) - 1):
            row = analysis_df.iloc[idx]
            hard_for_all.append(
                {
                    "test_row_pos": int(idx),
                    "Amount": round(float(row["Amount"]), 2),
                    "Hour": round(float(row["Hour"]), 2),
                    "V17": round(float(row["V17"]), 4),
                    "V14": round(float(row["V14"]), 4),
                    "V12": round(float(row["V12"]), 4),
                    "missed_by": missed_by,
                }
            )

    summary = {
        "primary_model": primary_model_key,
        "outcome_stats": group_stats,
        "top_fn_high_amount": fn_cases,
        "top_fp_high_confidence": fp_cases,
        "fraud_detection_by_model": fraud_overlap,
        "hard_fraud_cases_count": len(hard_for_all),
        "sample_hard_fraud_cases": sorted(hard_for_all, key=lambda x: x["Amount"], reverse=True)[:5],
    }

    with (METRICS_DIR / "error_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _plot_error_analysis(analysis_df, primary_model_key)
    return summary


def _plot_error_analysis(analysis_df: pd.DataFrame, primary_model_key: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # 1. V14 vs V17 scatter for TN sample + all TP, FN, FP
    tn_sample = analysis_df[analysis_df["Outcome"] == "TN"].sample(
        n=min(3000, int((analysis_df["Outcome"] == "TN").sum())), random_state=42
    )
    axes[0].scatter(tn_sample["V14"], tn_sample["V17"], c="#cbd5e0", s=10, alpha=0.5, label="TN (mẫu)")
    for outcome, color, marker, size in [
        ("TP", "#2f855a", "o", 28),
        ("FN", "#e53e3e", "X", 65),
        ("FP", "#dd6b20", "^", 55),
    ]:
        sub = analysis_df[analysis_df["Outcome"] == outcome]
        if not sub.empty:
            axes[0].scatter(sub["V14"], sub["V17"], c=color, marker=marker, s=size, alpha=0.85, label=f"{outcome} ({len(sub)})")
    axes[0].set_title(f"Phân bố lỗi trên không gian (V14, V17)\n[{primary_model_key}]")
    axes[0].set_xlabel("V14")
    axes[0].set_ylabel("V17")
    axes[0].legend(fontsize=8)

    # 2. Median feature profile across TP, FN, FP, TN
    profile_cols = ["V17", "V14", "V12", "V10", "V4", "V11"]
    medians = analysis_df.groupby("Outcome")[profile_cols].median().reindex(["TP", "FN", "FP", "TN"])
    sns.heatmap(medians, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1])
    axes[1].set_title("Giá trị trung vị các đặc trưng theo nhóm dự đoán")
    axes[1].set_xlabel("Đặc trưng")
    axes[1].set_ylabel("Nhóm kết quả")

    # 3. Amount distribution for TP vs FN vs FP
    err_sub = analysis_df[analysis_df["Outcome"].isin(["TP", "FN", "FP"])].copy()
    if not err_sub.empty:
        err_sub["Log_Amount"] = np.log1p(err_sub["Amount"])
        sns.boxplot(
            data=err_sub,
            x="Outcome",
            y="Log_Amount",
            order=["TP", "FN", "FP"],
            hue="Outcome",
            palette={"TP": "#2f855a", "FN": "#e53e3e", "FP": "#dd6b20"},
            legend=False,
            ax=axes[2],
        )
    axes[2].set_title("Phân phối log(1+Amount) của TP, FN và FP")
    axes[2].set_xlabel("Nhóm kết quả")
    axes[2].set_ylabel("log(1 + Amount)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_analysis_fn_fp.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
