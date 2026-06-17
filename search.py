# search.py
"""
Exhaustive pipeline search via cross-validation.

Iterates over all combinations of preprocessing strategy × model family ×
hyperparameter grid, runs stratified or regular k-fold cross-validation for
each combination, collects results, and provides helpers to pick the best
configuration per model and the single best overall.
"""

from typing import Dict, List, Any, Tuple

import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    KFold,
    ParameterGrid,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

from data_io import DatasetBundle
from analysis import ColumnInfo
from preprocessing import (
    get_feature_groups,
    define_preprocessing_strategies,
    apply_feature_filtering,
    build_preprocessing_transformer,
)
from models import ModelConfig, instantiate_model
from metrics import get_metric_function, is_higher_better


# ---------------------------------------------------------------------------
# 1. BUILD PIPELINE (PREPROCESSOR + MODEL)
# ---------------------------------------------------------------------------

def build_pipeline(preprocessor, model) -> Pipeline:
    """
    Create a sklearn Pipeline that first applies preprocessing and then the model.
    """
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipe


# ---------------------------------------------------------------------------
# 2. CROSS-VALIDATION FOR A SINGLE PIPELINE
# ---------------------------------------------------------------------------

def cross_validate_pipeline(
    pipeline: Pipeline,
    X_train,
    y_train,
    task_type: str,
    metric_name: str,
    cv_folds: int,
    random_seed: int,
) -> Tuple[float, float]:
    """
    Perform cross-validation for a given pipeline and return mean/std of the score.
    """
    task_type = task_type.lower().strip()
    metric_name = metric_name.lower().strip()

    if task_type == "classification":
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_seed,
        )
    elif task_type == "regression":
        cv = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_seed,
        )
    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    metric_func = get_metric_function(metric_name, task_type)
    higher_better = is_higher_better(metric_name)

    if metric_name == "roc_auc":
        scorer = make_scorer(metric_func, greater_is_better=True, needs_proba=True)
    else:
        scorer = make_scorer(metric_func, greater_is_better=higher_better)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scorer,
        n_jobs=None,
    )

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    return mean_score, std_score


# ---------------------------------------------------------------------------
# 3. EVALUATE ALL PIPELINES (WITH LIVE LOGGING)
# ---------------------------------------------------------------------------

def evaluate_all_pipelines(
    bundle: DatasetBundle,
    schema: Dict[str, ColumnInfo],
    strategy_configs: List[Dict],
    model_configs: List[ModelConfig],
    task_type: str,
    metric_name: str,
    cv_folds: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    """
    For each strategy × model × hyperparam combination, run CV and collect results.
    Prints LIVE progress so the user can see what is happening.
    """
    X_train = bundle.X_train
    y_train = bundle.y_train

    if X_train is None or y_train is None:
        raise ValueError(
            "X_train and y_train must not be None in evaluate_all_pipelines. "
            "Ensure target_detection.apply_target_column has been called."
        )

    results: List[Dict[str, Any]] = []

    for model_config in model_configs:
        if model_config.problem_type != task_type:
            continue

        for strategy_config in strategy_configs:
            strategy_name = strategy_config["name"]
            print(f"\n[SEARCH] Model={model_config.name}, Strategy={strategy_name}")

            X_tr_filtered, _, schema_filtered = apply_feature_filtering(
                X_train, None, schema, strategy_config
            )

            feature_groups = get_feature_groups(schema_filtered)
            preprocessor = build_preprocessing_transformer(
                strategy_config, feature_groups
            )

            if model_config.hyperparam_grid:
                grid = ParameterGrid(model_config.hyperparam_grid)
            else:
                grid = [dict()]

            for hyperparams in grid:
                print(f"  - Trying hyperparams={hyperparams} ...")
                model = instantiate_model(model_config, hyperparams)
                pipeline = build_pipeline(preprocessor, model)

                try:
                    mean_score, std_score = cross_validate_pipeline(
                        pipeline,
                        X_tr_filtered,
                        y_train,
                        task_type=task_type,
                        metric_name=metric_name,
                        cv_folds=cv_folds,
                        random_seed=random_seed,
                    )
                    print(f"    -> CV mean={mean_score:.4f}, std={std_score:.4f}")
                except Exception as e:
                    print(
                        f"[WARN] Failed for model={model_config.name}, "
                        f"strategy={strategy_name}, hyperparams={hyperparams}: {e}"
                    )
                    mean_score = float("-inf") if is_higher_better(metric_name) else float("inf")
                    std_score = float("nan")

                result_row = {
                    "model_name": model_config.name,
                    "strategy_name": strategy_name,
                    "hyperparams": hyperparams,
                    "mean_score": mean_score,
                    "std_score": std_score,
                }
                results.append(result_row)

    return results


# ---------------------------------------------------------------------------
# 4. SELECT BEST PIPELINE PER MODEL
# ---------------------------------------------------------------------------

def select_best_by_model(
    all_results: List[Dict[str, Any]],
    metric_name: str,
) -> List[Dict[str, Any]]:
    """
    For each distinct model name in all_results, keep the row with the best CV score.

    Optimisation direction (higher or lower is better) is determined by
    is_higher_better(metric_name).

    Parameters
    ----------
    all_results : flat list of result dicts from evaluate_all_pipelines()
    metric_name : metric key used to compare rows

    Returns
    -------
    List[Dict] — one entry per unique model_name, the best row for that model
    """
    if not all_results:
        return []

    higher_better = is_higher_better(metric_name)

    best_per_model: Dict[str, Dict[str, Any]] = {}
    for row in all_results:
        model_name = row["model_name"]
        score = row["mean_score"]

        if model_name not in best_per_model:
            best_per_model[model_name] = row
            continue

        current_score = best_per_model[model_name]["mean_score"]
        if higher_better:
            if score > current_score:
                best_per_model[model_name] = row
        else:
            if score < current_score:
                best_per_model[model_name] = row

    return list(best_per_model.values())


# ---------------------------------------------------------------------------
# 5. SELECT BEST PIPELINE OVERALL
# ---------------------------------------------------------------------------

def select_best_overall(
    best_per_model: List[Dict[str, Any]],
    metric_name: str,
) -> Dict[str, Any]:
    """
    Return the single best result row across all per-model winners.

    Parameters
    ----------
    best_per_model : list returned by select_best_by_model()
    metric_name    : metric key used to compare rows

    Returns
    -------
    Dict — the result row with the best overall CV score
    """
    if not best_per_model:
        raise ValueError("best_per_model is empty in select_best_overall.")

    higher_better = is_higher_better(metric_name)

    best_row = best_per_model[0]
    for row in best_per_model[1:]:
        score = row["mean_score"]
        current_score = best_row["mean_score"]
        if higher_better:
            if score > current_score:
                best_row = row
        else:
            if score < current_score:
                best_row = row

    return best_row
