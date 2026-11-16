"""Evaluation metrics, threshold optimization, and visualization utilities."""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_config_nhom12"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(style="whitegrid", context="notebook")


def get_positive_probabilities(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """Return predicted probability or normalized score for positive class (Class 1)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
    return None


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute imbalanced-aware classification metrics and confusion matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if y_prob is not None else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def find_best_threshold(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Find probability threshold maximizing F1-score strictly on the validation set."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        preds = (y_prob_val >= thr).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, float(best_f1)


def fit_and_evaluate_model(
    model_name: str,
    flow_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    notes: str = "",
    tune_threshold: bool = False,
    use_float32: bool = False,
) -> Tuple[List[Dict[str, Any]], Any, Optional[np.ndarray], np.ndarray]:
    """Fit model on Train, assess on Val/Test, and optionally tune decision threshold on Val."""
    if use_float32:
        X_tr = X_train.astype(np.float32, copy=False)
        X_va = X_val.astype(np.float32, copy=False)
        X_te = X_test.astype(np.float32, copy=False)
    else:
        X_tr, X_va, X_te = X_train, X_val, X_test

    t0 = time.perf_counter()
    model.fit(X_tr, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_test_prob = get_positive_probabilities(model, X_te)
    if y_test_prob is not None:
        y_test_pred = (y_test_prob >= 0.5).astype(int)
    else:
        y_test_pred = model.predict(X_te)
    predict_time = time.perf_counter() - t1

    y_val_prob = get_positive_probabilities(model, X_va)
    y_val_pred = (y_val_prob >= 0.5).astype(int) if y_val_prob is not None else model.predict(X_va)

    val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_prob)
    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_prob)

    records: List[Dict[str, Any]] = [
        {
            "flow": flow_name,
            "model": model_name,
            "threshold": 0.50,
            "val_f1": round(val_metrics["f1"], 4),
            "val_pr_auc": round(val_metrics["pr_auc"], 4) if not np.isnan(val_metrics["pr_auc"]) else None,
            "accuracy": round(test_metrics["accuracy"], 4),
            "precision": round(test_metrics["precision"], 4),
            "recall": round(test_metrics["recall"], 4),
            "f1": round(test_metrics["f1"], 4),
            "roc_auc": round(test_metrics["roc_auc"], 4) if not np.isnan(test_metrics["roc_auc"]) else None,
            "pr_auc": round(test_metrics["pr_auc"], 4) if not np.isnan(test_metrics["pr_auc"]) else None,
            "tn": test_metrics["tn"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
            "tp": test_metrics["tp"],
            "fit_time_sec": round(fit_time, 3),
            "predict_time_sec": round(predict_time, 3),
            "notes": notes,
        }
    ]

    final_pred = y_test_pred
    if tune_threshold and y_val_prob is not None and y_test_prob is not None:
        best_thr, best_val_f1 = find_best_threshold(y_val, y_val_prob)
        if abs(best_thr - 0.50) > 1e-6:
            tuned_test_pred = (y_test_prob >= best_thr).astype(int)
            tuned_metrics = compute_classification_metrics(y_test, tuned_test_pred, y_test_prob)
            records.append(
                {
                    "flow": flow_name,
                    "model": f"{model_name} (Threshold={best_thr:.2f})",
                    "threshold": best_thr,
                    "val_f1": round(best_val_f1, 4),
                    "val_pr_auc": round(val_metrics["pr_auc"], 4),
                    "accuracy": round(tuned_metrics["accuracy"], 4),
                    "precision": round(tuned_metrics["precision"], 4),
                    "recall": round(tuned_metrics["recall"], 4),
                    "f1": round(tuned_metrics["f1"], 4),
                    "roc_auc": round(tuned_metrics["roc_auc"], 4),
                    "pr_auc": round(tuned_metrics["pr_auc"], 4),
                    "tn": tuned_metrics["tn"],
                    "fp": tuned_metrics["fp"],
                    "fn": tuned_metrics["fn"],
                    "tp": tuned_metrics["tp"],
                    "fit_time_sec": round(fit_time, 3),
                    "predict_time_sec": round(predict_time, 3),
                    "notes": f"{notes}; threshold tuned on validation".strip("; "),
                }
            )
            final_pred = tuned_test_pred

    return records, model, y_test_prob, final_pred


def plot_confusion_matrices_grid(
    entries: List[Tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot a grid of confusion matrices with counts and row percentages."""
    n = len(entries)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.3 * rows))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, (title, y_true, y_pred) in zip(axes_arr, entries):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["Dự đoán 0", "Dự đoán 1"],
            yticklabels=["Thực tế 0", "Thực tế 1"],
            ax=ax,
        )
        ax.set_title(title, fontsize=10, fontweight="bold")

    for ax in axes_arr[n:]:
        ax.axis("off")

    fig.suptitle("Ma trận nhầm lẫn (Confusion Matrix) trên tập Test", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_and_roc_curves(
    curve_entries: List[Tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot Precision-Recall and ROC curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for label, y_true, y_prob in curve_entries:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[0].plot(recall, precision, linewidth=2, label=f"{label} (PR-AUC={ap:.4f})")

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        axes[1].plot(fpr, tpr, linewidth=2, label=f"{label} (ROC-AUC={roc_auc:.4f})")

    baseline_rate = float(np.mean(curve_entries[0][1])) if curve_entries else 0.0017
    axes[0].axhline(baseline_rate, linestyle="--", color="gray", label=f"Ngẫu nhiên ({baseline_rate:.4f})")
    axes[0].set_title("Đường cong Precision - Recall (Ưu tiên cho dữ liệu lệch)")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left", fontsize=8.5)

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ngẫu nhiên (0.5000)")
    axes[1].set_title("Đường cong ROC")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_flow_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot bar comparison between Flow A and Flow B across primary models."""
    subset = results_df[results_df["flow"].isin(["Flow A (Cơ bản)", "Flow B (Cải tiến)"])].copy()
    subset = subset[~subset["model"].str.contains("Threshold=|Undersample|Oversample|Power|Baseline", regex=True)]
    if subset.empty:
        return

    name_map = {
        "K-NN (k=5)": "K-NN",
        "K-NN (Tuned)": "K-NN",
        "Gaussian Naive Bayes": "Naive Bayes",
        "Decision Tree (Tuned)": "Decision Tree",
        "Random Forest (Mở rộng)": "Random Forest",
        "HistGradientBoosting (Mở rộng)": "HistGradientBoosting",
    }
    subset["model_group"] = subset["model"].replace(name_map)
    model_order = ["K-NN", "Naive Bayes", "Decision Tree", "Random Forest", "HistGradientBoosting"]
    subset = subset[subset["model_group"].isin(model_order)]

    melted = subset.melt(
        id_vars=["flow", "model_group"],
        value_vars=["precision", "recall", "f1", "pr_auc"],
        var_name="Metric",
        value_name="Score",
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), sharey=True)
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("pr_auc", "PR-AUC")]
    handles, labels = None, None
    for ax, (metric_key, metric_title) in zip(axes.ravel(), metrics):
        sub_m = melted[melted["Metric"] == metric_key]
        sns.barplot(data=sub_m, x="model_group", y="Score", hue="flow", order=model_order, ax=ax, palette="Set2")
        ax.set_title(metric_title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="x", rotation=15)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=2)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if handles and labels:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=True)
    fig.suptitle("So sánh hiệu quả giữa Luồng A (Cơ bản) và Luồng B (Cải tiến) trên tập Test", fontsize=12.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hyperparameter_sensitivity(tuning_df: pd.DataFrame, output_path: Path) -> None:
    """Plot hyperparameter sensitivity curves for K-NN, Decision Tree, and Naive Bayes."""
    if tuning_df.empty:
        return

    def _format_depth_label(val: Any) -> str:
        if pd.isna(val) or str(val) == "None":
            return "None"
        try:
            num = float(val)
            return str(int(num)) if num.is_integer() else str(val)
        except (TypeError, ValueError):
            return str(val)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    knn_sub = tuning_df[tuning_df["family"] == "K-NN"]
    if not knn_sub.empty:
        for weight_type, grp in knn_sub.groupby("variant"):
            x_knn = pd.to_numeric(grp["param_val"], errors="coerce")
            axes[0].plot(x_knn, grp["val_f1"], marker="o", label=weight_type)
        axes[0].set_title("K-NN: F1 (Validation) theo k")
        axes[0].set_xlabel("Số láng giềng k")
        axes[0].set_ylabel("F1-score")
        axes[0].legend()

    dt_sub = tuning_df[tuning_df["family"] == "Decision Tree"]
    if not dt_sub.empty:
        dt_groups = sorted(dt_sub.groupby("variant"), key=lambda item: (len(item[1]) == 1, item[0]))
        for cw_label, grp in dt_groups:
            x_dt = [_format_depth_label(v) for v in grp["param_val"]]
            if len(grp) == 1:
                axes[1].plot(
                    x_dt,
                    grp["val_f1"],
                    marker="*",
                    markersize=12,
                    linestyle="None",
                    color="#d97706",
                    label=cw_label,
                )
            else:
                axes[1].plot(x_dt, grp["val_f1"], marker="s", label=cw_label)
        axes[1].set_title("Decision Tree: F1 (Validation) theo max_depth")
        axes[1].set_xlabel("max_depth")
        axes[1].set_ylabel("F1-score")
        axes[1].legend(fontsize=8.5)

    nb_sub = tuning_df[tuning_df["family"] == "Naive Bayes"]
    if not nb_sub.empty:
        for variant_label, grp in nb_sub.groupby("variant"):
            x_nb = [str(v) for v in grp["param_val"]]
            axes[2].plot(x_nb, grp["val_f1"], marker="^", label=variant_label)
        axes[2].set_title("GaussianNB: F1 (Validation) theo var_smoothing")
        axes[2].set_xlabel("var_smoothing")
        axes[2].set_ylabel("F1-score")
        axes[2].tick_params(axis="x", rotation=15)
        axes[2].legend()

    fig.suptitle("Khảo sát độ nhạy siêu tham số trên tập Validation (Luồng B)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_overview(output_path: Path) -> None:
    """Generate professional pipeline architecture flowchart matching report Figure 1."""
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=200)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    c_blue, c_blue_light = "#1e40af", "#dbeafe"
    c_green, c_green_light = "#065f46", "#d1fae5"
    c_purple, c_purple_light = "#5b21b6", "#ede9fe"
    c_orange, c_orange_light = "#9a3412", "#ffedd5"
    c_red, c_red_light = "#991b1b", "#fee2e2"
    c_gray, c_gray_light = "#374151", "#f3f4f6"

    def draw_box(x, y, w, h, title, subtitle="", color=c_blue, bg=c_blue_light):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", facecolor=bg, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.62 if subtitle else y + h / 2, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=8.5, color="#374151")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#4b5563", lw=1.8))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1f2937")

    draw_box(0.5, 4.8, 2.2, 1.4, "Dữ liệu thô", "creditcard.csv\n(284.807 dòng, 31 cột)", c_gray, c_gray_light)
    draw_box(3.3, 4.8, 2.4, 1.4, "Khám phá dữ liệu (EDA)", "0 NaN, 1.081 trùng lặp\nMất cân bằng 578 : 1", c_orange, c_orange_light)
    draw_arrow(2.7, 5.5, 3.3, 5.5)

    draw_box(6.3, 4.8, 2.6, 1.4, "Tách tập chống Leakage", "Train 60% | Val 20% | Test 20%\n(random_state = 42)", c_purple, c_purple_light)
    draw_arrow(5.7, 5.5, 6.3, 5.5)

    draw_box(9.5, 5.4, 3.0, 1.2, "Luồng A: Pipeline Cơ bản", "30 đặc trưng thô\nStandardScaler (chỉ fit Train)", c_blue, c_blue_light)
    draw_box(9.5, 3.7, 3.0, 1.4, "Luồng B: Pipeline Cải tiến", "Khử 1.081 trùng + Hour_sin/cos\n+ log1p(Amount) + Is_Zero\nRobust/PowerTransformer (fit Train)", c_green, c_green_light)

    draw_arrow(8.9, 5.8, 9.5, 6.0, "Luồng A")
    draw_arrow(8.9, 5.2, 9.5, 4.4, "Luồng B")

    draw_box(9.5, 1.6, 3.0, 1.6, "Huấn luyện & Tuning", "Tuning Validation: K-NN, DT, NB\nEnsemble: RF, HistGB\n+ Threshold Tuning", c_purple, c_purple_light)
    draw_arrow(11.0, 3.7, 11.0, 3.2)

    draw_box(5.8, 1.6, 3.0, 1.6, "Đánh giá trên Test Độc lập", "PR-AUC, F1, Recall, Precision\nMa trận nhầm lẫn\nKiểm chứng Time Split 60/20/20", c_blue, c_blue_light)
    draw_arrow(9.5, 2.4, 8.8, 2.4)

    draw_box(2.8, 1.6, 2.4, 1.6, "Phân tích lỗi (Error Analysis)", "Phân tích 27 ca FN ($5,874.20)\n3 ca FP & 26 ca khó bỏ sót\nĐộ chồng lấn đa mô hình", c_red, c_red_light)
    draw_arrow(5.8, 2.4, 5.2, 2.4)

    draw_box(0.5, 1.6, 1.8, 1.6, "Demo & Báo cáo", "Web GUI Demo (app.py)\nBáo cáo LaTeX 6 phần\nReproducible Pipeline", c_green, c_green_light)
    draw_arrow(2.8, 2.4, 2.3, 2.4)

    plt.title("SƠ ĐỒ QUY TRÌNH HỌC MÁY TOÀN DIỆN CHỐNG RÒ RỈ DỮ LIỆU — NHÓM 12 (ĐỀ TÀI 12)", fontsize=12, fontweight="bold", pad=15, color="#0f172a")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
