# evaluation.py
"""
Final pipeline training and held-out test-set evaluation.

Rebuilds and fits the winning pipeline (selected by the search module) on the
full training set, generates predictions on the test set, computes the full
suite of evaluation metrics, and extracts feature importances or coefficients
from the fitted estimator with human-readable feature names.
"""

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from data_io import DatasetBundle
from analysis import ColumnInfo
from preprocessing import (
    get_feature_groups,
    apply_feature_filtering,
    build_preprocessing_transformer,
)
from models import ModelConfig, instantiate_model
from metrics import compute_additional_metrics
from search import build_pipeline


# ---------------------------------------------------------------------------
# 1. TRAIN THE BEST PIPELINE ON FULL TRAINING DATA
# ---------------------------------------------------------------------------

def train_best_pipeline(
    best_overall: Dict[str, Any],
    bundle: DatasetBundle,
    schema: Dict[str, ColumnInfo],
    strategy_configs: List[Dict[str, Any]],
    model_configs: List[ModelConfig],
    task_type: str,
) -> Pipeline:
    """
    Rebuild and train the best pipeline (preprocessing + model) on all training data.
    """
    X_train = bundle.X_train
    y_train = bundle.y_train

    if X_train is None or y_train is None:
        raise ValueError(
            "X_train and y_train must not be None in train_best_pipeline. "
            "Make sure target has been applied and search has been run."
        )

    model_name = best_overall["model_name"]
    strategy_name = best_overall["strategy_name"]
    hyperparams = best_overall.get("hyperparams", {})

    # --- Find the matching strategy config ---
    strategy_config = None
    for cfg in strategy_configs:
        if cfg["name"] == strategy_name:
            strategy_config = cfg
            break
    if strategy_config is None:
        raise ValueError(f"Strategy config {strategy_name!r} not found.")

    # --- Find the matching model config ---
    model_config = None
    for mcfg in model_configs:
        if mcfg.name == model_name:
            model_config = mcfg
            break
    if model_config is None:
        raise ValueError(f"Model config {model_name!r} not found.")

    # --- Apply feature filtering consistent with search ---
    X_tr_filtered, X_te_filtered, schema_filtered = apply_feature_filtering(
        X_train,
        bundle.X_test,
        schema,
        strategy_config,
    )

    # --- Build feature groups & preprocessor ---
    feature_groups = get_feature_groups(schema_filtered)
    preprocessor = build_preprocessing_transformer(strategy_config, feature_groups)

    # --- Instantiate model and build pipeline ---
    model = instantiate_model(model_config, hyperparams)
    pipeline = build_pipeline(preprocessor, model)

    # --- Fit pipeline ---
    pipeline.fit(X_tr_filtered, y_train)

    # Optionally update bundle to hold filtered X for consistency
    bundle.X_train = X_tr_filtered
    bundle.X_test = X_te_filtered

    return pipeline


# ---------------------------------------------------------------------------
# 2. EVALUATE TRAINED PIPELINE ON TEST SET
# ---------------------------------------------------------------------------

def evaluate_on_test(
    trained_pipeline: Pipeline,
    bundle: DatasetBundle,
    task_type: str,
) -> Dict[str, Any]:
    """
    Use the trained pipeline to predict on the test set and compute metrics.
    """
    X_test = bundle.X_test
    y_test = bundle.y_test

    if X_test is None:
        raise ValueError(
            "X_test is None in evaluate_on_test. "
            "You need a test set to perform final evaluation."
        )

    # --- Predictions ---
    y_pred = trained_pipeline.predict(X_test)

    # For classification, try to get probabilities if possible
    y_proba = None
    if task_type.lower() == "classification":
        if hasattr(trained_pipeline, "predict_proba"):
            try:
                proba = trained_pipeline.predict_proba(X_test)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    # probability of positive class
                    y_proba = proba[:, 1]
            except Exception:
                y_proba = None

    # --- Metrics ---
    if y_test is not None:
        metrics_dict = compute_additional_metrics(
            y_true=y_test,
            y_pred=y_pred,
            task_type=task_type,
            y_proba=y_proba,
        )
    else:
        metrics_dict = {}

    result = {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics_dict,
    }
    return result


# ---------------------------------------------------------------------------
# 3. FEATURE IMPORTANCE EXTRACTION (WITH REAL FEATURE NAMES)
# ---------------------------------------------------------------------------

def extract_feature_importance(
    trained_pipeline: Pipeline,
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract feature importances / coefficients from the final model.

    - Tries to pull the real expanded feature names from the
      preprocessing step (ColumnTransformer + Pipelines) via
      get_feature_names_out().
    - Falls back to generic names feature_0, feature_1, ... if
      that is not available.
    """
    model = None

    # Locate final estimator
    if isinstance(trained_pipeline, Pipeline):
        if "model" in trained_pipeline.named_steps:
            model = trained_pipeline.named_steps["model"]
        else:
            model = list(trained_pipeline.named_steps.values())[-1]
    else:
        model = trained_pipeline

    if model is None:
        return []

    # --- Try to get feature names from preprocessor if not provided ---
    if feature_names is None and isinstance(trained_pipeline, Pipeline):
        pre = trained_pipeline.named_steps.get("preprocessor")
        if pre is not None:
            try:
                names_out = pre.get_feature_names_out()
                # convert to plain Python list
                if hasattr(names_out, "tolist"):
                    names_out = names_out.tolist()
                else:
                    names_out = list(names_out)

                # Optional: clean names like "numeric__LotArea" -> "LotArea"
                cleaned = []
                for n in names_out:
                    n_str = str(n)
                    if "__" in n_str:
                        n_str = n_str.split("__", 1)[1]
                    cleaned.append(n_str)
                feature_names = cleaned
            except Exception:
                feature_names = None

    # --- Get raw importance values ---
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef_arr = np.array(model.coef_, dtype=float)
        if coef_arr.ndim > 1:
            coef_arr = np.mean(np.abs(coef_arr), axis=0)
        importances = np.abs(coef_arr)
    else:
        # Model doesn't expose any importance info
        return []

    n_features = importances.shape[0]

    # If we still don't have names, fall back to generic
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    else:
        if len(feature_names) != n_features:
            # length mismatch -> safest is generic
            feature_names = [f"feature_{i}" for i in range(n_features)]

    # Build list and sort
    records: List[Dict[str, Any]] = []
    for name, val in zip(feature_names, importances):
        records.append({"feature": name, "importance": float(val)})

    records.sort(key=lambda x: x["importance"], reverse=True)
    return records
