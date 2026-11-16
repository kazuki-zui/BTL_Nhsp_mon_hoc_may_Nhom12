"""Exploratory Data Analysis (EDA) script generating summary JSON and report figures."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_config_nhom12"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, KEY_CORRELATED_FEATURES, METRICS_DIR, TARGET_COL, ensure_output_dirs
from .data_loader import load_creditcard_data

sns.set_theme(style="whitegrid", context="notebook")


def run_eda(custom_data_path: Optional[str] = None) -> dict:
    """Compute EDA statistics and save figures for the report and notebook."""
    ensure_output_dirs()
    df = load_creditcard_data(custom_data_path)

    n_rows, n_cols = df.shape
    missing_total = int(df.isna().sum().sum())
    dup_mask_drop = df.duplicated(keep="first")
    n_duplicates = int(dup_mask_drop.sum())
    dup_by_class_removed = df.loc[dup_mask_drop, TARGET_COL].value_counts().to_dict()
    class_counts = df[TARGET_COL].value_counts().to_dict()
    fraud_rate = float(df[TARGET_COL].mean() * 100.0)

    corr_series = df.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)
    top_corr = (
        corr_series.reindex(corr_series.abs().sort_values(ascending=False).index[:12])
        .round(4)
        .to_dict()
    )

    amount_stats = {
        "overall": df["Amount"].describe().round(4).to_dict(),
        "class_0": df.loc[df[TARGET_COL] == 0, "Amount"].describe().round(4).to_dict(),
        "class_1": df.loc[df[TARGET_COL] == 1, "Amount"].describe().round(4).to_dict(),
        "zero_amount_class_0": int(((df["Amount"] == 0) & (df[TARGET_COL] == 0)).sum()),
        "zero_amount_class_1": int(((df["Amount"] == 0) & (df[TARGET_COL] == 1)).sum()),
    }

    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_total": missing_total,
        "duplicates_to_remove": n_duplicates,
        "duplicates_removed_by_class": {str(k): int(v) for k, v in dup_by_class_removed.items()},
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "fraud_rate_percent": round(fraud_rate, 4),
        "time_min_sec": float(df["Time"].min()),
        "time_max_sec": float(df["Time"].max()),
        "time_monotonic": bool(df["Time"].is_monotonic_increasing),
        "amount_stats": amount_stats,
        "top_correlations_with_class": top_corr,
    }

    with (METRICS_DIR / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _plot_class_distribution(df)
    _plot_amount_and_time(df)
    _plot_top_correlations(corr_series, df)
    _plot_key_pca_distributions(df)

    return summary


def _plot_class_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["0: Hợp lệ", "1: Gian lận"]
    colors = ["#2b6cb0", "#e53e3e"]

    bars0 = axes[0].bar(labels, counts.values, color=colors, width=0.55)
    axes[0].set_title("Phân phối nhãn (Tỉ lệ tuyến tính)")
    axes[0].set_ylabel("Số lượng giao dịch")
    for bar, val in zip(bars0, counts.values):
        pct = val / len(df) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 4000,
            f"{val:,}\n({pct:.4f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_ylim(0, counts.max() * 1.18)

    bars1 = axes[1].bar(labels, counts.values, color=colors, width=0.55)
    axes[1].set_yscale("log")
    axes[1].set_title("Phân phối nhãn (Thang đo Log)")
    axes[1].set_ylabel("Số lượng giao dịch (Log)")
    for bar, val in zip(bars1, counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.25,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_ylim(1, counts.max() * 4)

    fig.suptitle("Mất cân bằng lớp nghiêm trọng trong creditcard.csv (578 : 1)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_amount_and_time(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # 1. Log1p Amount KDE
    for cls, label, color in [(0, "Hợp lệ (0)", "#2b6cb0"), (1, "Gian lận (1)", "#e53e3e")]:
        vals = np.log1p(df.loc[df[TARGET_COL] == cls, "Amount"])
        sns.kdeplot(vals, ax=axes[0], label=label, color=color, fill=True, alpha=0.25)
    axes[0].set_title("Phân phối log(1 + Amount)")
    axes[0].set_xlabel("log(1 + Amount)")
    axes[0].legend()

    # 2. Boxplot Amount (log scale) for outlier awareness
    plot_df = df[[TARGET_COL, "Amount"]].copy()
    plot_df["Nhãn"] = plot_df[TARGET_COL].map({0: "Hợp lệ", 1: "Gian lận"})
    plot_df["Amount_plus_1"] = plot_df["Amount"] + 1.0
    sns.boxplot(
        data=plot_df,
        x="Nhãn",
        y="Amount_plus_1",
        hue="Nhãn",
        palette={"Hợp lệ": "#2b6cb0", "Gian lận": "#e53e3e"},
        legend=False,
        ax=axes[1],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Boxplot Amount (+1, thang Log)")
    axes[1].set_ylabel("Amount + 1 (Log)")

    # 3. Hourly transaction rate (normalized within class)
    hours = (df["Time"] // 3600) % 24
    hourly = pd.DataFrame({"Hour": hours, TARGET_COL: df[TARGET_COL]})
    for cls, label, color in [(0, "Hợp lệ (0)", "#2b6cb0"), (1, "Gian lận (1)", "#e53e3e")]:
        dist = hourly.loc[hourly[TARGET_COL] == cls, "Hour"].value_counts(normalize=True).sort_index() * 100
        axes[2].plot(dist.index, dist.values, marker="o", markersize=4, label=label, color=color)
    axes[2].set_title("Tỉ lệ giao dịch theo giờ trong ngày")
    axes[2].set_xlabel("Giờ trong ngày (0 - 23h)")
    axes[2].set_ylabel("Tỉ lệ trong từng lớp (%)")
    axes[2].set_xticks(range(0, 24, 3))
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_amount_time_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_top_correlations(corr_series: pd.Series, df: pd.DataFrame) -> None:
    top_idx = corr_series.abs().sort_values(ascending=False).index[:12]
    top_corr = corr_series.loc[top_idx].sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})

    colors = ["#e53e3e" if v < 0 else "#2b6cb0" for v in top_corr.values]
    axes[0].barh(top_corr.index, top_corr.values, color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Top 12 đặc trưng tương quan mạnh nhất với Class")
    axes[0].set_xlabel("Hệ số tương quan Pearson")

    heat_cols = list(top_idx[:8]) + [TARGET_COL]
    corr_mat = df[heat_cols].corr()
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Ma trận tương quan Top 8 đặc trưng & Class")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_top_correlations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_key_pca_distributions(df: pd.DataFrame) -> None:
    features = KEY_CORRELATED_FEATURES[:4]  # V17, V14, V12, V10
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    axes = axes.ravel()

    for ax, feat in zip(axes, features):
        for cls, label, color in [(0, "Hợp lệ (0)", "#2b6cb0"), (1, "Gian lận (1)", "#e53e3e")]:
            vals = df.loc[df[TARGET_COL] == cls, feat]
            sns.kdeplot(vals, ax=ax, label=label, color=color, fill=True, alpha=0.25)
        ax.set_title(f"Phân phối đặc trưng {feat}")
        ax.set_xlim(-20, 10)
        ax.legend()

    fig.suptitle("So sánh phân phối 4 biến PCA phân tách mạnh nhất giữa Hợp lệ và Gian lận", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_key_features_dist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    result = run_eda()
    print(json.dumps(result, indent=2, ensure_ascii=False))
