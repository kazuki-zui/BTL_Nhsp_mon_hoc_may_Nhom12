"""Model factories and hyperparameter tuning helpers."""

import os
from typing import Any, Dict, List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .evaluate import compute_classification_metrics, get_positive_probabilities


def build_baseline_models(random_state: int = RANDOM_STATE) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Return baseline models for benchmarking."""
    return {
        "Baseline: Dummy (Most Frequent)": (
            DummyClassifier(strategy="most_frequent"),
            {"tune_threshold": False, "use_float32": False, "notes": "Dự đoán toàn bộ là lớp đa số (0)"},
        ),
        "Baseline: Decision Tree (Mặc định)": (
            DecisionTreeClassifier(random_state=random_state),
            {"tune_threshold": False, "use_float32": False, "notes": "Cây quyết định tham số mặc định"},
        ),
    }


def tune_knn_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Tune K-NN (k and weights) on Validation using float32 distances."""
    X_tr = X_train.astype(np.float32, copy=False)
    X_va = X_val.astype(np.float32, copy=False)

    records: List[Dict[str, Any]] = []
    best_cfg: Dict[str, Any] = {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"}
    best_val_f1 = -1.0

    for weights in ("uniform", "distance"):
        for k in (3, 5, 7, 11):
            model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric="euclidean", n_jobs=-1)
            model.fit(X_tr, y_train)
            probs = get_positive_probabilities(model, X_va)
            preds = (probs >= 0.5).astype(int)
            m = compute_classification_metrics(y_val, preds, probs)
            records.append(
                {
                    "family": "K-NN",
                    "variant": f"weights={weights}",
                    "param_name": "n_neighbors",
                    "param_val": k,
                    "val_precision": round(m["precision"], 4),
                    "val_recall": round(m["recall"], 4),
                    "val_f1": round(m["f1"], 4),
                    "val_pr_auc": round(m["pr_auc"], 4),
                }
            )
            if m["f1"] > best_val_f1:
                best_val_f1 = m["f1"]
                best_cfg = {"n_neighbors": k, "weights": weights, "metric": "euclidean"}

    return best_cfg, pd.DataFrame(records)


def tune_decision_tree_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Tune Decision Tree (max_depth, criterion, class_weight) strictly on Validation."""
    records: List[Dict[str, Any]] = []
    best_cfg: Dict[str, Any] = {
        "criterion": "gini",
        "max_depth": 6,
        "min_samples_leaf": 5,
        "class_weight": None,
    }
    best_val_f1 = -1.0

    for cw in (None, "balanced"):
        cw_label = "class_weight=balanced" if cw else "class_weight=None"
        for depth in (4, 6, 10, 15, None):
            model = DecisionTreeClassifier(
                criterion="gini",
                max_depth=depth,
                min_samples_leaf=5,
                class_weight=cw,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            probs = get_positive_probabilities(model, X_val)
            preds = (probs >= 0.5).astype(int)
            m = compute_classification_metrics(y_val, preds, probs)
            records.append(
                {
                    "family": "Decision Tree",
                    "variant": cw_label,
                    "param_name": "max_depth",
                    "param_val": "None" if depth is None else depth,
                    "val_precision": round(m["precision"], 4),
                    "val_recall": round(m["recall"], 4),
                    "val_f1": round(m["f1"], 4),
                    "val_pr_auc": round(m["pr_auc"], 4),
                }
            )
            if m["f1"] > best_val_f1:
                best_val_f1 = m["f1"]
                best_cfg = {
                    "criterion": "gini",
                    "max_depth": depth,
                    "min_samples_leaf": 5,
                    "class_weight": cw,
                }

    # Check entropy criterion at the best depth & class_weight
    ent_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=best_cfg["max_depth"],
        min_samples_leaf=5,
        class_weight=best_cfg["class_weight"],
        random_state=random_state,
    )
    ent_model.fit(X_train, y_train)
    ent_probs = get_positive_probabilities(ent_model, X_val)
    ent_preds = (ent_probs >= 0.5).astype(int)
    ent_m = compute_classification_metrics(y_val, ent_preds, ent_probs)
    records.append(
        {
            "family": "Decision Tree",
            "variant": f"criterion=entropy ({'cw=balanced' if best_cfg['class_weight'] else 'cw=None'})",
            "param_name": "max_depth",
            "param_val": "None" if best_cfg["max_depth"] is None else best_cfg["max_depth"],
            "val_precision": round(ent_m["precision"], 4),
            "val_recall": round(ent_m["recall"], 4),
            "val_f1": round(ent_m["f1"], 4),
            "val_pr_auc": round(ent_m["pr_auc"], 4),
        }
    )
    if ent_m["f1"] > best_val_f1:
        best_cfg["criterion"] = "entropy"

    return best_cfg, pd.DataFrame(records)


def tune_naive_bayes_on_validation(
    X_train_robust: np.ndarray,
    y_train: np.ndarray,
    X_val_robust: np.ndarray,
    y_val: np.ndarray,
    X_train_power: np.ndarray,
    X_val_power: np.ndarray,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Tune GaussianNB var_smoothing on RobustScaler vs PowerTransformer."""
    records: List[Dict[str, Any]] = []
    best_cfg: Dict[str, Any] = {"var_smoothing": 1e-9, "scaler": "robust"}
    best_val_f1 = -1.0

    variants = [
        ("RobustScaler", "robust", X_train_robust, X_val_robust),
        ("PowerTransformer", "power", X_train_power, X_val_power),
    ]
    for label, scaler_key, X_tr, X_va in variants:
        for vs in (1e-9, 1e-6, 1e-3, 1e-1):
            model = GaussianNB(var_smoothing=vs)
            model.fit(X_tr, y_train)
            probs = get_positive_probabilities(model, X_va)
            preds = (probs >= 0.5).astype(int)
            m = compute_classification_metrics(y_val, preds, probs)
            records.append(
                {
                    "family": "Naive Bayes",
                    "variant": label,
                    "param_name": "var_smoothing",
                    "param_val": vs,
                    "val_precision": round(m["precision"], 4),
                    "val_recall": round(m["recall"], 4),
                    "val_f1": round(m["f1"], 4),
                    "val_pr_auc": round(m["pr_auc"], 4),
                }
            )
            if m["f1"] > best_val_f1:
                best_val_f1 = m["f1"]
                best_cfg = {"var_smoothing": vs, "scaler": scaler_key}

    return best_cfg, pd.DataFrame(records)


def build_extension_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Return bonus ensemble models built strictly with scikit-learn."""
    return {
        "Random Forest (Mở rộng)": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "HistGradientBoosting (Mở rộng)": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
