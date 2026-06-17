# main.py
"""
Entry point for the Goal-Aware AutoML system.

Provides two runtime modes via a simple CLI menu:
  - Training mode  : interactive wizard that loads data, detects the target
    column, infers the task type, searches across preprocessing strategies and
    model families via cross-validation, evaluates the winner on a held-out
    test set, saves the best pipeline to automl_outputs/, and produces plots
    and a text report.
  - Prediction mode: loads the serialised pipeline from automl_outputs/ and
    predicts a single example supplied interactively by the user.
"""

import os
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import joblib

from config import (
    create_config_from_args,
    print_config_summary,
    Config,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
)
from data_io import (
    load_data_from_config,
    build_initial_dataset_bundle,
    split_train_test_if_needed,
    DatasetBundle,
)
from analysis import (
    infer_column_types,
    compute_column_stats,
    build_schema,
    print_schema_summary,
)
from target_detection import (
    score_columns_for_goal,
    select_best_target_column,
    confirm_target_with_user,
    apply_target_column,
)
from preprocessing import (
    get_feature_groups,
    define_preprocessing_strategies,
)
from models import (
    get_classification_models,
    get_regression_models,
    ModelConfig,
)
from search import (
    evaluate_all_pipelines,
    select_best_by_model,
    select_best_overall,
)
from evaluation import (
    train_best_pipeline,
    evaluate_on_test,
    extract_feature_importance,
)
from reporting import (
    plot_classification_results,
    plot_regression_results,
    plot_feature_importance,
    generate_run_summary,
)


OUTPUT_DIR = "automl_outputs"



# ======================================================================
# SMALL INPUT HELPERS
# ======================================================================

def _ask_int(prompt: str, default: int, min_val: int, max_val: int) -> int:
    """
    Prompt for an integer in [min_val, max_val], re-asking until the input is
    valid. Returns default when the user presses Enter without typing.
    """
    while True:
        raw = input(f"{prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if min_val <= v <= max_val:
                return v
            print(f"Please enter an integer between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid integer.")


def _ask_float(prompt: str, default: float, min_val: float, max_val: float) -> float:
    """
    Prompt for a float in [min_val, max_val], re-asking until the input is
    valid. Returns default when the user presses Enter without typing.
    """
    while True:
        raw = input(f"{prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            v = float(raw)
            if min_val <= v <= max_val:
                return v
            print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")



# ======================================================================
# BASIC SETUP
# ======================================================================

def ask_basic_setup() -> Dict[str, Any]:
    """
    Interactively collect basic run configuration from the user.

    Prompts for: input mode (single CSV or separate train/test), file paths,
    a free-text goal description, CV fold count, test size, and random seed.

    Returns
    -------
    dict with keys: data_path, train_path, test_path, goal_text, cv_folds,
    test_size, random_seed.
    """
    print("=== Goal-Aware AutoML: Configuration ===")
    print("Choose input mode:")
    print("  1) Single CSV file (data.csv)")
    print("  2) Separate train.csv and test.csv")
    mode_choice = input("Enter 1 or 2 [default=1]: ").strip() or "1"

    data_path = None
    train_path = None
    test_path = None

    if mode_choice == "2":
        train_path = input("Path to train.csv: ").strip() or None
        test_path = input("Path to test.csv: ").strip() or None
    else:
        data_path = input("Path to data.csv: ").strip() or None

    goal_text = input("Describe your goal: ").strip()
    if goal_text == "":
        goal_text = "Run an AutoML experiment."

    cv_folds = _ask_int("Number of CV folds", 5, 2, 20)
    test_size = _ask_float("Test size for single-file mode", 0.2, 0.1, 0.5)
    random_seed = _ask_int("Random seed", 42, 0, 10000)

    return {
        "data_path": data_path,
        "train_path": train_path,
        "test_path": test_path,
        "goal_text": goal_text,
        "cv_folds": cv_folds,
        "test_size": test_size,
        "random_seed": random_seed,
    }



# ======================================================================
# PREDICTION MODE (UPDATED TO LOAD FROM automl_outputs/)
# ======================================================================

def run_prediction_mode():
    """
    Load the saved pipeline and feature list from automl_outputs/ and predict
    a single example entered interactively by the user.

    Prints the prediction result, or an informative error message if the
    serialised files cannot be loaded or prediction fails.
    """
    print("\n=== Prediction Mode ===")

    try:
        pipeline = joblib.load(os.path.join(OUTPUT_DIR, "best_pipeline.pkl"))
    except Exception as e:
        print("Error: Could not load automl_outputs/best_pipeline.pkl")
        print("You must run AutoML Training first.")
        print("Details:", e)
        return

    try:
        feature_names = joblib.load(os.path.join(OUTPUT_DIR, "best_features.pkl"))
    except Exception as e:
        print("Error: Could not load automl_outputs/best_features.pkl")
        print("Details:", e)
        return

    print("Model and feature list loaded.\n")
    print("Expected features:")
    print("  " + ", ".join(feature_names))
    print("Press Enter empty to cancel.\n")

    row_data = {}
    for feat in feature_names:
        val_str = input(f"{feat}: ").strip()
        if val_str == "":
            print("Cancelled.")
            return
        try:
            row_data[feat] = float(val_str)
        except ValueError:
            row_data[feat] = val_str

    try:
        pred = pipeline.predict(pd.DataFrame([row_data]))
        print("\nPrediction:", pred[0] if len(pred) == 1 else pred)
    except Exception as e:
        print("Error during prediction:", e)



def predict_with_pipeline_in_memory(pipeline, feature_names: List[str]):
    """
    Predict a single example using an already-loaded pipeline and feature list.

    Called immediately after training so the user can test the model without
    restarting. Collects one value per feature from stdin, casts numeric-looking
    strings to float, and prints the model's prediction.

    Parameters
    ----------
    pipeline      : fitted sklearn Pipeline
    feature_names : ordered list of feature names the pipeline expects
    """
    print("\n=== Prediction after Training ===")
    print("Expected features:")
    print("  " + ", ".join(feature_names))
    print("Press Enter empty to cancel.\n")

    row_data = {}
    for feat in feature_names:
        val_str = input(f"{feat}: ").strip()
        if val_str == "":
            print("Cancelled.")
            return
        try:
            row_data[feat] = float(val_str)
        except ValueError:
            row_data[feat] = val_str

    try:
        pred = pipeline.predict(pd.DataFrame([row_data]))
        print("\nPrediction:", pred[0] if len(pred) == 1 else pred)
    except Exception as e:
        print("Error during prediction:", e)



# ======================================================================
# TASK TYPE AUTO-DETECTION
# ======================================================================

def _auto_detect_task_type(y: pd.Series) -> str:
    """
    Heuristically decide whether the target series is a regression or
    classification problem.

    Returns "regression" only if the target is numeric AND has more than 20
    distinct values AND those values span more than 10 % of the total rows —
    otherwise returns "classification".
    """
    if pd.api.types.is_numeric_dtype(y):
        # Both thresholds required: > 20 guards against small integer codes
        # (e.g. 0–5 class labels); > 10 % of rows guards against columns like
        # "age bucket 1–30" that have many unique values but still low coverage.
        if y.nunique() > 20 and y.nunique() > len(y) * 0.1:
            return "regression"
    return "classification"



# ======================================================================
# TRAINING MODE (FULL AUTOML)
# ======================================================================

def run_training_mode():
    """
    Execute the full AutoML training workflow in 14 sequential steps:

      1.  Collect basic setup from the user (file paths, goal, CV params).
      2.  Load and clean the raw CSV data; write a cleaning report.
      3.  Infer column types and score each column as a candidate target.
      4.  Confirm or override the auto-detected target column.
      5.  Auto-detect (and let the user override) the task type.
      6.  Ask which evaluation metric to optimise.
      7.  Rebuild the final Config with the confirmed task type and metric.
      8.  Inspect the feature schema.
      9.  Define preprocessing strategies and load the model catalog.
      10. Run cross-validated search over all strategy × model × hyperparam combinations.
      11. Re-train the winning pipeline on the full training set.
      12. Evaluate on the held-out test set and print metrics.
      13. Extract and display feature importances.
      14. Save the pipeline, generate plots and a text report.

    Offers an optional single-row prediction before returning.
    """

    # ------------------------------------------------------------------
    # 1) Basic setup
    # ------------------------------------------------------------------
    setup = ask_basic_setup()

    try:
        load_config = create_config_from_args(
            data_path=setup["data_path"],
            train_path=setup["train_path"],
            test_path=setup["test_path"],
            task_type="regression",
            goal_text=setup["goal_text"],
            metric_name="rmse",
            cv_folds=setup["cv_folds"],
            random_seed=setup["random_seed"],
            test_size=setup["test_size"],
            verbose=True,
        )
    except Exception as e:
        print("Error creating configuration:", e)
        return

    print_config_summary(load_config)

    # Load + clean
    try:
        raw_train_df, raw_test_df = load_data_from_config(load_config)
    except Exception as e:
        print("Error loading data:", e)
        return

    print("\nData loaded and cleaned.")
    print(f"TRAIN: {raw_train_df.shape}")
    if raw_test_df is not None:
        print(f"TEST : {raw_test_df.shape}")
    print(f"Cleaning report: {os.path.join(OUTPUT_DIR, 'cleaning_report.txt')}")

    # ------------------------------------------------------------------
    # 2) Target detection
    # ------------------------------------------------------------------
    print("\n=== Target Detection ===")
    col_types_raw = infer_column_types(raw_train_df)
    col_stats_raw = compute_column_stats(raw_train_df)
    schema_raw = build_schema(col_types_raw, col_stats_raw)

    scores = score_columns_for_goal(
        goal_text=load_config.goal_text,
        df=raw_train_df,
        task_type="regression",
        schema=schema_raw,
    )
    best_guess = select_best_target_column(scores)
    final_target = confirm_target_with_user(best_guess, raw_train_df)

    # Apply target
    bundle = build_initial_dataset_bundle(raw_train_df, raw_test_df)
    bundle = apply_target_column(bundle, final_target)

    if load_config.mode == "single_file":
        bundle = split_train_test_if_needed(bundle, load_config)

    # ------------------------------------------------------------------
    # 3) Auto task detect
    # ------------------------------------------------------------------
    detected_task = _auto_detect_task_type(bundle.y_train)
    print(f"\nDetected task: {detected_task}")
    override = input("Override (classification/regression) or Enter to accept: ").strip().lower()
    task_type = override if override in {"classification", "regression"} else detected_task
    print(f"Using task type: {task_type}")

    # ------------------------------------------------------------------
    # 4) Ask metric
    # ------------------------------------------------------------------
    if task_type == "classification":
        valid_metrics = sorted(CLASSIFICATION_METRICS)
        default_metric = "accuracy"
    else:
        valid_metrics = sorted(REGRESSION_METRICS)
        default_metric = "rmse"

    print("\nAvailable metrics:")
    print("  " + ", ".join(valid_metrics))
    metric_name = input(f"Metric [Enter={default_metric}]: ").strip().lower() or default_metric

    while metric_name not in valid_metrics:
        print("Invalid metric. Choose from:", ", ".join(valid_metrics))
        metric_name = input(f"Metric [Enter={default_metric}]: ").strip().lower() or default_metric

    # ------------------------------------------------------------------
    # 5) Final config
    # ------------------------------------------------------------------
    config = create_config_from_args(
        data_path=load_config.data_path,
        train_path=load_config.train_path,
        test_path=load_config.test_path,
        task_type=task_type,
        goal_text=setup["goal_text"],
        metric_name=metric_name,
        cv_folds=setup["cv_folds"],
        random_seed=setup["random_seed"],
        test_size=setup["test_size"],
        verbose=True,
    )

    print("\n=== Final Configuration ===")
    print_config_summary(config)

    # ------------------------------------------------------------------
    # 6) Schema for features
    # ------------------------------------------------------------------
    print("\n=== Feature Schema ===")
    X_train = bundle.X_train
    feature_names = list(X_train.columns)

    schema = build_schema(
        infer_column_types(X_train),
        compute_column_stats(X_train),
    )
    print_schema_summary(schema)

    # ------------------------------------------------------------------
    # 7) Preprocessing + Models
    # ------------------------------------------------------------------
    print("\n=== Defining preprocessing strategies ===")
    feature_groups = get_feature_groups(schema)
    strategy_configs = define_preprocessing_strategies(feature_groups)
    print(f"{len(strategy_configs)} strategies defined.")

    print("\n=== Preparing models ===")
    model_configs = get_classification_models() if task_type == "classification" else get_regression_models()
    print(f"{len(model_configs)} models loaded.")

    # ------------------------------------------------------------------
    # 8) Evaluate all pipelines
    # ------------------------------------------------------------------
    print("\n=== Running AutoML search ===")
    all_results = evaluate_all_pipelines(
        bundle=bundle,
        schema=schema,
        strategy_configs=strategy_configs,
        model_configs=model_configs,
        task_type=task_type,
        metric_name=metric_name,
        cv_folds=config.cv_folds,
        random_seed=config.random_seed,
    )

    if not all_results:
        print("ERROR: No pipelines produced.")
        return

    print(f"\nTotal pipelines evaluated: {len(all_results)}")

    best_per_model = select_best_by_model(all_results, metric_name)
    best_overall = select_best_overall(best_per_model, metric_name)

    print("\n=== Best per model ===")
    for row in best_per_model:
        print(f"- {row['model_name']} | {row['strategy_name']} | {row['hyperparams']} | CV={row['mean_score']:.4f}")

    print("\n=== BEST OVERALL ===")
    print(best_overall)

    # ------------------------------------------------------------------
    # 9) Train best pipeline
    # ------------------------------------------------------------------
    print("\n=== Training best pipeline ===")
    trained_pipeline = train_best_pipeline(
        best_overall,
        bundle,
        schema,
        strategy_configs,
        model_configs,
        task_type,
    )

    # ------------------------------------------------------------------
    # 10) Evaluate
    # ------------------------------------------------------------------
    print("\n=== Evaluating on test set ===")
    test_result = evaluate_on_test(trained_pipeline, bundle, task_type)

    print("\nTest metrics:")
    for k, v in test_result["metrics"].items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 11) Feature importance
    # ------------------------------------------------------------------
    print("\n=== Feature Importance ===")
    feature_importances = extract_feature_importance(trained_pipeline)
    if feature_importances:
        for rec in feature_importances[:10]:
            print(f"  {rec['feature']}: {rec['importance']:.4f}")
    else:
        print("No feature importance available.")

    # ------------------------------------------------------------------
    # 12) SAVE MODEL + FEATURES (NOW IN automl_outputs/)
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline_path = os.path.join(OUTPUT_DIR, "best_pipeline.pkl")
    features_path = os.path.join(OUTPUT_DIR, "best_features.pkl")

    joblib.dump(trained_pipeline, pipeline_path)
    joblib.dump(feature_names, features_path)

    print(f"\nSaved best pipeline → {pipeline_path}")
    print(f"Saved feature list  → {features_path}")

    # ------------------------------------------------------------------
    # 13) Reporting
    # ------------------------------------------------------------------
    print("\n=== Generating plots and summary ===")

    plot_paths = {}

    if task_type == "classification":
        cls = plot_classification_results(
            test_result["y_true"], test_result["y_pred"], test_result["y_proba"],
            output_dir=OUTPUT_DIR, prefix="best_model"
        )
        plot_paths.update(cls)
    else:
        reg = plot_regression_results(
            test_result["y_true"], test_result["y_pred"],
            output_dir=OUTPUT_DIR, prefix="best_model"
        )
        plot_paths.update(reg)

    fi_path = plot_feature_importance(
        feature_importances,
        output_dir=OUTPUT_DIR,
        filename="best_model_feature_importance.png",
        top_k=20,
    )
    if fi_path:
        plot_paths["feature_importance"] = fi_path

    summary_text = generate_run_summary(
        config, bundle, schema, all_results, best_per_model, best_overall, test_result
    )

    summary_path = os.path.join(OUTPUT_DIR, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nSummary saved    → {summary_path}")
    print(f"Cleaning report  → {os.path.join(OUTPUT_DIR, 'cleaning_report.txt')}")
    print("\nPlots saved:")
    for k, v in plot_paths.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 14) Optional prediction
    # ------------------------------------------------------------------
    while True:
        ans = input("\nPredict new example now? (yes/no): ").strip().lower()
        if ans in {"yes", "y"}:
            predict_with_pipeline_in_memory(trained_pipeline, feature_names)
            break
        if ans in {"no", "n"}:
            print("Done.")
            break



# ======================================================================
# MAIN MENU
# ======================================================================

def main():
    """CLI entry point. Displays the main menu and dispatches to training or prediction mode."""
    print("=== Goal-Aware AutoML System ===")
    print("1) Run AutoML Training")
    print("2) Predict a New Example (using best_pipeline.pkl)")
    choice = input("Choose an option (1/2): ").strip()

    if choice == "2":
        run_prediction_mode()
    else:
        run_training_mode()



if __name__ == "__main__":
    main()
