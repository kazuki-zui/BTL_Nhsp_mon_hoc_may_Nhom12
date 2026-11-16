"""End-to-end reproducible experiment runner for Group 12 Credit Card Fraud Detection."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_STATE, ensure_output_dirs
from .data_loader import load_creditcard_data
from .eda import run_eda
from .error_analysis import analyze_errors
from .evaluate import (
    fit_and_evaluate_model,
    plot_confusion_matrices_grid,
    plot_flow_comparison,
    plot_hyperparameter_sensitivity,
    plot_pipeline_overview,
    plot_pr_and_roc_curves,
)
from .models import (
    build_baseline_models,
    build_extension_models,
    tune_decision_tree_on_validation,
    tune_knn_on_validation,
    tune_naive_bayes_on_validation,
)
from .preprocessing import (
    balance_training_data,
    prepare_flow_a,
    prepare_flow_b,
)


def _save_demo_artifacts(
    flow_b,
    trained_models: Dict[str, Any],
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    primary_key: str,
) -> None:
    """Persist fitted Flow B transformer, trained models, and sample test cases for the GUI demo."""
    bundle = {
        "feature_names": flow_b.feature_names,
        "transformer": flow_b.transformer,
        "models": trained_models,
        "normal_medians": flow_b.test_raw_df.loc[flow_b.y_test == 0, ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]].median().to_dict(),
        "fraud_medians": flow_b.test_raw_df.loc[flow_b.y_test == 1, ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]].median().to_dict(),
    }
    joblib.dump(bundle, MODELS_DIR / "demo_bundle.joblib", compress=3)

    raw_test = flow_b.test_raw_df.copy()
    raw_test["y_true"] = flow_b.y_test
    raw_test["y_pred"] = model_predictions[primary_key]["y_pred"]
    raw_test["y_prob"] = model_predictions[primary_key]["y_prob"]

    sample_groups = [
        ("Giao dịch hợp lệ điển hình", raw_test[(raw_test["y_true"] == 0) & (raw_test["y_prob"] < 0.05)].head(4)),
        ("Gian lận phát hiện đúng (TP)", raw_test[(raw_test["y_true"] == 1) & (raw_test["y_pred"] == 1)].sort_values("Amount", ascending=False).head(4)),
        ("Gian lận khó / bỏ sót (FN hoặc biên)", raw_test[(raw_test["y_true"] == 1)].sort_values("y_prob", ascending=True).head(4)),
    ]

    samples = []
    cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)] + ["y_true", "y_prob"]
    for category, df_sub in sample_groups:
        for idx, row in df_sub.iterrows():
            item = {"category": category, "test_index": int(idx)}
            for col in cols:
                item[col] = round(float(row[col]), 6)
            samples.append(item)

    with (MODELS_DIR / "sample_transactions.json").open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def run_all_experiments(custom_data_path: Optional[str] = None, skip_eda: bool = False) -> None:
    """Execute full reproducible pipeline from raw data to metrics, plots, error analysis, and saved models."""
    ensure_output_dirs()
    print("[1/6] Loading dataset...")
    df = load_creditcard_data(custom_data_path)

    if not skip_eda:
        print("[2/6] Running EDA and generating EDA figures...")
        run_eda(custom_data_path)
    else:
        print("[2/6] Skipping EDA as requested.")

    print("[3/6] Preparing Flow A (Baseline) and Flow B (Improved) splits without data leakage...")
    flow_a_unscaled = prepare_flow_a(df, split_strategy="stratified", scale=False, random_state=RANDOM_STATE)
    flow_a = prepare_flow_a(df, split_strategy="stratified", scale=True, random_state=RANDOM_STATE)
    flow_b = prepare_flow_b(df, split_strategy="stratified", scaler_type="robust", random_state=RANDOM_STATE)
    flow_b_power = prepare_flow_b(df, split_strategy="stratified", scaler_type="power", random_state=RANDOM_STATE)
    flow_b_time = prepare_flow_b(df, split_strategy="time", scaler_type="robust", random_state=RANDOM_STATE)

    all_records: List[Dict[str, Any]] = []

    # --- FLOW A EXPERIMENTS ---
    print("[4/6] Training & evaluating Flow A models...")
    for name, (model, cfg) in build_baseline_models(RANDOM_STATE).items():
        recs, _, _, _ = fit_and_evaluate_model(
            name, "Flow A (Cơ bản)", model,
            flow_a.X_train, flow_a.y_train, flow_a.X_val, flow_a.y_val, flow_a.X_test, flow_a.y_test,
            **cfg,
        )
        all_records.extend(recs)

    # Show impact of unscaled Time & Amount on distance-based K-NN
    recs, _, _, _ = fit_and_evaluate_model(
        "K-NN (k=5, Unscaled)",
        "Flow A (Chưa chuẩn hóa)",
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        flow_a_unscaled.X_train, flow_a_unscaled.y_train,
        flow_a_unscaled.X_val, flow_a_unscaled.y_val,
        flow_a_unscaled.X_test, flow_a_unscaled.y_test,
        notes="Không scale: Time/Amount chi phối khoảng cách",
        use_float32=True,
    )
    all_records.extend(recs)

    flow_a_models = [
        ("K-NN (k=5)", KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True, "StandardScaler trên tập Train"),
        ("Gaussian Naive Bayes", GaussianNB(), False, "Tham số mặc định"),
        ("Decision Tree (Tuned)", DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE), False, "depth=6, balanced"),
    ]
    for name, model, use_f32, notes in flow_a_models:
        recs, _, _, _ = fit_and_evaluate_model(
            name, "Flow A (Cơ bản)", model,
            flow_a.X_train, flow_a.y_train, flow_a.X_val, flow_a.y_val, flow_a.X_test, flow_a.y_test,
            notes=notes, use_float32=use_f32,
        )
        all_records.extend(recs)

    for name, model in build_extension_models(RANDOM_STATE).items():
        recs, _, _, _ = fit_and_evaluate_model(
            name, "Flow A (Cơ bản)", model,
            flow_a.X_train, flow_a.y_train, flow_a.X_val, flow_a.y_val, flow_a.X_test, flow_a.y_test,
            notes="Ensemble mở rộng trên Luồng A",
        )
        all_records.extend(recs)

    # --- FLOW B HYPERPARAMETER TUNING ON VALIDATION ---
    print("[5/6] Tuning K-NN, GaussianNB, Decision Tree on Flow B Validation & evaluating Test...")
    best_knn_cfg, knn_tune_df = tune_knn_on_validation(flow_b.X_train, flow_b.y_train, flow_b.X_val, flow_b.y_val)
    best_dt_cfg, dt_tune_df = tune_decision_tree_on_validation(flow_b.X_train, flow_b.y_train, flow_b.X_val, flow_b.y_val, RANDOM_STATE)
    best_nb_cfg, nb_tune_df = tune_naive_bayes_on_validation(
        flow_b.X_train, flow_b.y_train, flow_b.X_val, flow_b.y_val,
        flow_b_power.X_train, flow_b_power.X_val,
    )
    tuning_df = pd.concat([knn_tune_df, dt_tune_df, nb_tune_df], ignore_index=True)
    tuning_df.to_csv(METRICS_DIR / "tuning_results.csv", index=False)

    # Evaluate Flow B primary & tuned models
    flow_b_trained_models: Dict[str, Any] = {}
    flow_b_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    cm_entries: List[Tuple[str, np.ndarray, np.ndarray]] = []
    curve_entries: List[Tuple[str, np.ndarray, np.ndarray]] = []

    knn_best = KNeighborsClassifier(
        n_neighbors=best_knn_cfg["n_neighbors"],
        weights=best_knn_cfg["weights"],
        metric="euclidean",
        n_jobs=-1,
    )
    recs, fitted_knn, prob_knn, pred_knn = fit_and_evaluate_model(
        "K-NN (Tuned)",
        "Flow B (Cải tiến)",
        knn_best,
        flow_b.X_train, flow_b.y_train, flow_b.X_val, flow_b.y_val, flow_b.X_test, flow_b.y_test,
        notes=f"k={best_knn_cfg['n_neighbors']}, weights={best_knn_cfg['weights']}",
        tune_threshold=True,
        use_float32=True,
    )
    all_records.extend(recs)
    flow_b_trained_models["K-NN (Tuned)"] = fitted_knn
    flow_b_predictions["K-NN (Tuned)"] = {"y_pred": pred_knn, "y_prob": prob_knn}
    cm_entries.append(("Flow B: K-NN (Tuned)", flow_b.y_test, pred_knn))
    curve_entries.append(("K-NN (Tuned)", flow_b.y_test, prob_knn))

    # K-NN with train-only undersampling comparison (fast & shows recall/precision tradeoff)
    X_tr_under, y_tr_under = balance_training_data(
        flow_b.X_train, flow_b.y_train, method="undersample", neg_to_pos_ratio=10, random_state=RANDOM_STATE
    )
    recs, _, _, _ = fit_and_evaluate_model(
        "K-NN (Undersample 10:1 Train)",
        "Flow B (Cân bằng Train)",
        KNeighborsClassifier(n_neighbors=best_knn_cfg["n_neighbors"], weights=best_knn_cfg["weights"], n_jobs=-1),
        X_tr_under, y_tr_under, flow_b.X_val, flow_b.y_val, flow_b.X_test, flow_b.y_test,
        notes="Chỉ undersample tập Train (tỉ lệ 10:1); giữ nguyên Val/Test",
        use_float32=True,
    )
    all_records.extend(recs)

    # GaussianNB on RobustScaler vs PowerTransformer
    nb_robust = GaussianNB(var_smoothing=1e-9)
    recs, fitted_nb_robust, prob_nb_robust, pred_nb_robust = fit_and_evaluate_model(
        "Gaussian Naive Bayes",
        "Flow B (Cải tiến)",
        nb_robust,
        flow_b.X_train, flow_b.y_train, flow_b.X_val, flow_b.y_val, flow_b.X_test, flow_b.y_test,
        notes="RobustScaler, var_smoothing=1e-9",
        tune_threshold=True,
    )
    all_records.extend(recs)

    nb_power = GaussianNB(var_smoothing=best_nb_cfg["var_smoothing"])
    recs, fitted_nb_power, prob_nb_power, pred_nb_power = fit_and_evaluate_model(
        "GaussianNB (PowerTransformer)",
        "Flow B (PowerTransformer)",
        nb_power,
        flow_b_power.X_train, flow_b_power.y_train,
        flow_b_power.X_val, flow_b_power.y_val,
        flow_b_power.X_test, flow_b_power.y_test,
        notes=f"Yeo-Johnson, var_smoothing={best_nb_cfg['var_smoothing']}",
        tune_threshold=True,
    )
    all_records.extend(recs)
    # Keep RobustScaler GaussianNB in demo bundle so all models share the same Flow B transformer
    flow_b_trained_models["Gaussian Naive Bayes"] = fitted_nb_robust
    flow_b_predictions["Gaussian Naive Bayes"] = {"y_pred": pred_nb_robust, "y_prob": prob_nb_robust}
    cm_entries.append(("Flow B: GaussianNB", flow_b.y_test, pred_nb_robust))
    curve_entries.append(("GaussianNB", flow_b.y_test, prob_nb_robust))

    # Decision Tree Tuned
    dt_best = DecisionTreeClassifier(
        criterion=best_dt_cfg["criterion"],
        max_depth=best_dt_cfg["max_depth"],
        min_samples_leaf=best_dt_cfg["min_samples_leaf"],
        class_weight=best_dt_cfg["class_weight"],
        random_state=RANDOM_STATE,
    )
    recs, fitted_dt, prob_dt, pred_dt = fit_and_evaluate_model(
        "Decision Tree (Tuned)",
        "Flow B (Cải tiến)",
        dt_best,
        flow_b.X_train, flow_b.y_train, flow_b.X_val, flow_b.y_val, flow_b.X_test, flow_b.y_test,
        notes=f"{best_dt_cfg['criterion']}, depth={best_dt_cfg['max_depth']}, cw={best_dt_cfg['class_weight']}",
        tune_threshold=True,
    )
    all_records.extend(recs)
    flow_b_trained_models["Decision Tree (Tuned)"] = fitted_dt
    flow_b_predictions["Decision Tree (Tuned)"] = {"y_pred": pred_dt, "y_prob": prob_dt}
    cm_entries.append(("Flow B: Decision Tree", flow_b.y_test, pred_dt))
    curve_entries.append(("Decision Tree (Tuned)", flow_b.y_test, prob_dt))

    # Bonus Ensemble models on Flow B
    for name, model in build_extension_models(RANDOM_STATE).items():
        recs, fitted_ens, prob_ens, pred_ens = fit_and_evaluate_model(
            name, "Flow B (Cải tiến)", model,
            flow_b.X_train, flow_b.y_train, flow_b.X_val, flow_b.y_val, flow_b.X_test, flow_b.y_test,
            notes="Ensemble mở rộng trên Luồng B",
            tune_threshold=True,
        )
        all_records.extend(recs)
        flow_b_trained_models[name] = fitted_ens
        flow_b_predictions[name] = {"y_pred": pred_ens, "y_prob": prob_ens}
        cm_entries.append((f"Flow B: {name}", flow_b.y_test, pred_ens))
        curve_entries.append((name, flow_b.y_test, prob_ens))

    # Time-based split robustness check on top models
    time_models = [
        ("K-NN (Tuned)", KNeighborsClassifier(n_neighbors=best_knn_cfg["n_neighbors"], weights=best_knn_cfg["weights"], n_jobs=-1), True),
        ("Decision Tree (Tuned)", DecisionTreeClassifier(**best_dt_cfg, random_state=RANDOM_STATE), False),
        ("Random Forest (Mở rộng)", build_extension_models(RANDOM_STATE)["Random Forest (Mở rộng)"], False),
        ("HistGradientBoosting (Mở rộng)", build_extension_models(RANDOM_STATE)["HistGradientBoosting (Mở rộng)"], False),
    ]
    for name, model, use_f32 in time_models:
        recs, _, _, _ = fit_and_evaluate_model(
            name, "Flow B (Chia theo Time)", model,
            flow_b_time.X_train, flow_b_time.y_train,
            flow_b_time.X_val, flow_b_time.y_val,
            flow_b_time.X_test, flow_b_time.y_test,
            notes="Kiểm chứng phân tách thời gian 60/20/20",
            use_float32=use_f32,
        )
        all_records.extend(recs)

    # --- SAVE METRICS, FIGURES, ERROR ANALYSIS, AND DEMO BUNDLE ---
    print("[6/6] Saving comparison tables, figures, error analysis, and demo models...")
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    with (METRICS_DIR / "best_configs.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "random_state": RANDOM_STATE,
                "flow_a_meta": flow_a.metadata,
                "flow_b_meta": flow_b.metadata,
                "best_knn": best_knn_cfg,
                "best_decision_tree": best_dt_cfg,
                "best_naive_bayes": best_nb_cfg,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    plot_pipeline_overview(FIGURES_DIR / "pipeline_overview.png")
    plot_confusion_matrices_grid(cm_entries, FIGURES_DIR / "confusion_matrices_flow_b.png")
    plot_pr_and_roc_curves(curve_entries, FIGURES_DIR / "pr_roc_curves_flow_b.png")
    plot_flow_comparison(results_df, FIGURES_DIR / "flow_a_vs_flow_b.png")
    plot_hyperparameter_sensitivity(tuning_df, FIGURES_DIR / "hyperparameter_sensitivity.png")

    # Determine highest F1 model on Flow B for primary error analysis
    fb_rows = results_df[results_df["flow"] == "Flow B (Cải tiến)"]
    primary_model_key = max(
        flow_b_predictions.keys(),
        key=lambda k: float(fb_rows.loc[fb_rows["model"].str.startswith(k), "f1"].max()),
    )
    error_summary = analyze_errors(flow_b.test_raw_df, flow_b.y_test, flow_b_predictions, primary_model_key)
    _save_demo_artifacts(flow_b, flow_b_trained_models, flow_b_predictions, primary_model_key)

    print("\n=== DONE! TOP FLOW B RESULTS (TEST SET) ===")
    cols_show = ["flow", "model", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "fp", "fn", "tp"]
    print(results_df[cols_show].to_string(index=False))
    print("\nError Analysis Primary Model:", primary_model_key)
    print("Fraud detection overlap:", json.dumps(error_summary["fraud_detection_by_model"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Group 12 Credit Card Fraud ML Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Optional path to creditcard.csv")
    parser.add_argument("--skip-eda", action="store_true", help="Skip regenerating EDA figures")
    args = parser.parse_args()
    run_all_experiments(custom_data_path=args.data_path, skip_eda=args.skip_eda)


if __name__ == "__main__":
    main()
