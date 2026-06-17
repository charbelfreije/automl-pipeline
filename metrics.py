# metrics.py
"""
Metric function registry and evaluation utilities.

Maps metric name strings to sklearn scoring callables for use during CV search
(get_metric_function) and final reporting (compute_additional_metrics). Also
exposes is_higher_better() so the search and selection logic knows the
optimisation direction for each metric without hard-coding it elsewhere.

Note: RMSE is computed manually as sqrt(MSE) to remain compatible with all
supported sklearn versions.
"""

from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# 1. CLASSIFICATION METRIC MAP
# ---------------------------------------------------------------------------

CLASSIFICATION_METRIC_FUNCTIONS = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "recall":    lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
    "f1":        lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
    "f1_score":  lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
    "roc_auc":   roc_auc_score,  # requires y_proba
}

# ---------------------------------------------------------------------------
# 2. REGRESSION METRIC MAP (FIXED RMSE — NO squared=False)
# ---------------------------------------------------------------------------

REGRESSION_METRIC_FUNCTIONS = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,

    # FIXED: manual RMSE computation (works on all sklearn versions)
    "rmse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred) ** 0.5,

    "r2": r2_score,
    "r2_score": r2_score,
}

# ---------------------------------------------------------------------------
# 3. DETERMINE OPTIMIZATION DIRECTION
# ---------------------------------------------------------------------------

def is_higher_better(metric_name: str) -> bool:
    """
    Returns True if the metric should be maximized, False if minimized.
    """
    metric_name = metric_name.lower().strip()

    # Higher is better
    if metric_name in {"accuracy", "precision", "recall", "f1", "f1_score", "roc_auc"}:
        return True
    if metric_name in {"r2", "r2_score"}:
        return True

    # Lower is better
    if metric_name in {"mae", "mse", "rmse"}:
        return False

    raise ValueError(f"Unknown metric name {metric_name!r} in is_higher_better().")

# ---------------------------------------------------------------------------
# 4. GET METRIC FUNCTION
# ---------------------------------------------------------------------------

def get_metric_function(metric_name: str, task_type: str):
    """
    Returns the metric function for CV scoring.
    """
    metric_name = metric_name.lower().strip()
    task_type = task_type.lower().strip()

    if task_type == "classification":
        if metric_name not in CLASSIFICATION_METRIC_FUNCTIONS:
            raise ValueError(f"Metric '{metric_name}' is not valid for classification.")
        return CLASSIFICATION_METRIC_FUNCTIONS[metric_name]

    elif task_type == "regression":
        if metric_name not in REGRESSION_METRIC_FUNCTIONS:
            raise ValueError(f"Metric '{metric_name}' is not valid for regression.")
        return REGRESSION_METRIC_FUNCTIONS[metric_name]

    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

# ---------------------------------------------------------------------------
# 5. COMPUTE ADDITIONAL METRICS FOR FINAL REPORT (ALSO FIXED RMSE)
# ---------------------------------------------------------------------------

def compute_additional_metrics(y_true, y_pred, task_type: str, y_proba=None) -> Dict[str, Any]:
    """
    Compute metrics for the final evaluation phase.
    """
    task_type = task_type.lower().strip()
    results = {}

    if task_type == "classification":
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["precision"] = precision_score(y_true, y_pred, zero_division=0)
        results["recall"] = recall_score(y_true, y_pred, zero_division=0)
        results["f1"] = f1_score(y_true, y_pred, zero_division=0)

        if y_proba is not None:
            try:
                results["roc_auc"] = roc_auc_score(y_true, y_proba)
            except Exception:
                results["roc_auc"] = None

    elif task_type == "regression":
        results["mae"]  = mean_absolute_error(y_true, y_pred)

        # FIXED RMSE
        results["rmse"] = mean_squared_error(y_true, y_pred) ** 0.5

        results["r2"]   = r2_score(y_true, y_pred)

    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    return results
