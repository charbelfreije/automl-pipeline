# reporting.py
"""
Visualisation and run-summary generation for the completed AutoML run.

Produces and saves PNG plots: confusion matrix, ROC curve, and
precision-recall curve for classification; predicted-vs-actual, residuals-vs-
predicted, and residuals histogram for regression; plus a horizontal bar chart
of feature importances. Also generates a structured plain-text run summary
suitable for inclusion in a course report.
"""

from typing import Dict, Any, List, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from config import Config
from data_io import DatasetBundle
from analysis import ColumnInfo


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION PLOTS
# ---------------------------------------------------------------------------

def plot_classification_results(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    output_dir: str = ".",
    prefix: str = "classification",
) -> Dict[str, str]:
    """
    Generate standard classification plots:
        - Confusion matrix
        - ROC curve (if y_proba is provided)
        - Precision-Recall curve (if y_proba is provided)

    Saves plots to PNG files in output_dir and returns a dict of
    plot_name -> file_path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)
    paths["confusion_matrix"] = cm_path

    # --- ROC curve ---
    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, label="ROC curve")
            ax.plot([0, 1], [0, 1], "k--", label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")

            roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
            fig.tight_layout()
            fig.savefig(roc_path)
            plt.close(fig)
            paths["roc_curve"] = roc_path
        except Exception:
            # If ROC fails (e.g. labels not binary), just skip
            pass

        # --- Precision-Recall curve ---
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(recall, precision, label="Precision-Recall")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc="lower left")

            pr_path = os.path.join(output_dir, f"{prefix}_precision_recall.png")
            fig.tight_layout()
            fig.savefig(pr_path)
            plt.close(fig)
            paths["precision_recall_curve"] = pr_path
        except Exception:
            pass

    return paths


# ---------------------------------------------------------------------------
# 2. REGRESSION PLOTS
# ---------------------------------------------------------------------------

def plot_regression_results(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: str = ".",
    prefix: str = "regression",
) -> Dict[str, str]:
    """
    Generate standard regression plots:
        - Predicted vs Actual
        - Residuals vs Predicted
        - Histogram of residuals

    Saves plots and returns a dict of plot_name -> file_path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    residuals = y_true_arr - y_pred_arr

    # --- Predicted vs Actual ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true_arr, y_pred_arr, alpha=0.7)
    min_val = min(y_true_arr.min(), y_pred_arr.min())
    max_val = max(y_true_arr.max(), y_pred_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.legend()

    pva_path = os.path.join(output_dir, f"{prefix}_pred_vs_actual.png")
    fig.tight_layout()
    fig.savefig(pva_path)
    plt.close(fig)
    paths["pred_vs_actual"] = pva_path

    # --- Residuals vs Predicted ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred_arr, residuals, alpha=0.7)
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs Predicted")

    rvsp_path = os.path.join(output_dir, f"{prefix}_residuals_vs_pred.png")
    fig.tight_layout()
    fig.savefig(rvsp_path)
    plt.close(fig)
    paths["residuals_vs_predicted"] = rvsp_path

    # --- Histogram of residuals ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Distribution")

    hist_path = os.path.join(output_dir, f"{prefix}_residuals_hist.png")
    fig.tight_layout()
    fig.savefig(hist_path)
    plt.close(fig)
    paths["residuals_hist"] = hist_path

    return paths


# ---------------------------------------------------------------------------
# 3. FEATURE IMPORTANCE PLOT
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_importances: List[Dict[str, Any]],
    output_dir: str = ".",
    filename: str = "feature_importance.png",
    top_k: int = 20,
) -> Optional[str]:
    """
    Plot feature importances as a horizontal bar chart.

    feature_importances is a list of dicts with keys:
        - "feature"
        - "importance"

    Returns the path to the saved figure, or None if list is empty.
    """
    if not feature_importances:
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Take top_k
    sorted_feats = sorted(
        feature_importances,
        key=lambda x: x["importance"],
        reverse=True,
    )[:top_k]

    feat_names = [d["feature"] for d in sorted_feats]
    feat_vals = [d["importance"] for d in sorted_feats]

    fig, ax = plt.subplots(figsize=(6, max(4, len(sorted_feats) * 0.3)))
    y_positions = np.arange(len(sorted_feats))

    ax.barh(y_positions, feat_vals)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feat_names)
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")

    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# 4. RUN SUMMARY TEXT (FOR REPORT)
# ---------------------------------------------------------------------------

def generate_run_summary(
    config: Config,
    bundle: DatasetBundle,
    schema: Dict[str, ColumnInfo],
    all_results: List[Dict[str, Any]],
    best_per_model: List[Dict[str, Any]],
    best_overall: Dict[str, Any],
    test_result: Dict[str, Any],
) -> str:
    """
    Create a human-readable text summary for the run.

    This can be pasted into the course report under sections like:
    - Data Description
    - Methods
    - Results
    - Discussion
    """
    lines: List[str] = []

    # --- High-level setup ---
    lines.append("=== Goal-Aware AutoML Run Summary ===")
    lines.append("")
    lines.append(f"Task type      : {config.task_type}")
    lines.append(f"Goal           : {config.goal_text!r}")
    lines.append(f"Metric (CV)    : {config.metric_name}")
    lines.append(f"CV folds       : {config.cv_folds}")
    lines.append("")

    # --- Dataset ---
    n_train = len(bundle.X_train) if bundle.X_train is not None else 0
    n_test = len(bundle.X_test) if bundle.X_test is not None else 0
    n_features = len(schema)

    lines.append("Dataset:")
    lines.append(f"  Train size   : {n_train} rows")
    lines.append(f"  Test size    : {n_test} rows")
    lines.append(f"  Features used: {n_features}")
    lines.append(f"  Target column: {bundle.target_name}")
    lines.append("")

    # --- Models & strategies tried ---
    lines.append("Pipelines evaluated (model × preprocessing × hyperparameters):")
    lines.append(f"  Total pipelines evaluated: {len(all_results)}")
    lines.append("")

    # Show a few examples
    max_show = min(5, len(all_results))
    if max_show > 0:
        lines.append("  Example pipelines:")
        for row in all_results[:max_show]:
            lines.append(
                f"    - Model={row['model_name']}, "
                f"Strategy={row['strategy_name']}, "
                f"Hyperparams={row['hyperparams']}, "
                f"CV score={row['mean_score']:.4f} (std={row['std_score']:.4f})"
            )
        lines.append("")

    # --- Best per model ---
    lines.append("Best pipeline per model:")
    for row in best_per_model:
        lines.append(
            f"  - {row['model_name']}: "
            f"Strategy={row['strategy_name']}, "
            f"Hyperparams={row['hyperparams']}, "
            f"CV score={row['mean_score']:.4f}"
        )
    lines.append("")

    # --- Best overall ---
    lines.append("Best overall pipeline:")
    lines.append(
        f"  Model        : {best_overall['model_name']}\n"
        f"  Strategy     : {best_overall['strategy_name']}\n"
        f"  Hyperparams  : {best_overall['hyperparams']}\n"
        f"  CV mean score: {best_overall['mean_score']:.4f}\n"
        f"  CV std score : {best_overall['std_score']:.4f}"
    )
    lines.append("")

    # --- Test metrics ---
    lines.append("Final test set performance:")
    metrics = test_result.get("metrics", {})
    if metrics:
        for name, value in metrics.items():
            if value is None:
                continue
            lines.append(f"  {name:12s}: {value:.4f}")
    else:
        lines.append("  (No test metrics available; y_test was None.)")
    lines.append("")

    lines.append("Notes:")
    lines.append("  - Cross-validation was used to compare all pipelines and choose the best one.")
    lines.append("  - Multiple preprocessing strategies were evaluated, including different")
    lines.append("    imputation and scaling/encoding schemes.")
    lines.append("  - The selected model and preprocessing strategy achieved the best")
    lines.append("    cross-validation score according to the chosen metric.")
    lines.append("")
    lines.append("This summary can be adapted into the report's Methodology and Results sections.")

    return "\n".join(lines)
