# config.py
"""
Configuration dataclass and validation for an AutoML run.

Holds all user-specified settings (data paths, task type, evaluation metric,
cross-validation parameters) in a single Config object. create_config_from_args()
is the primary constructor; it delegates to determine_mode() to set the input
mode and validate_config() to catch inconsistencies before the pipeline starts.
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ---- Supported metrics by task type ----

CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_score",  # alias
    "roc_auc",
}

REGRESSION_METRICS = {
    "mae",
    "mse",
    "rmse",
    "r2",
    "r2_score",  # alias
}


@dataclass
class Config:
    """
    Central configuration object for an AutoML run.

    Populated by create_config_from_args(), validated by validate_config(),
    and threaded through the rest of the pipeline. The ``mode`` field is
    derived automatically from which data-path arguments are supplied;
    ``is_valid`` and ``errors`` are set during validation.
    """
    # Data paths
    data_path: Optional[str] = None      # for single-file mode
    train_path: Optional[str] = None     # for train/test mode
    test_path: Optional[str] = None      # for train/test mode

    # Task & goal
    task_type: str = "classification"    # "classification" or "regression"
    goal_text: str = ""                  # natural language goal

    # Metrics & evaluation
    metric_name: str = "accuracy"        # e.g. "accuracy", "f1", "mae", "rmse", "r2"
    cv_folds: int = 5
    random_seed: int = 42
    test_size: float = 0.2               # only used in single-file mode (data.csv)

    # Misc
    verbose: bool = True

    # Derived / internal fields
    mode: Optional[str] = None           # "single_file" or "train_test"
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def create_config_from_args(
    data_path: Optional[str] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    task_type: str = "classification",
    goal_text: str = "",
    metric_name: str = "accuracy",
    cv_folds: int = 5,
    random_seed: int = 42,
    test_size: float = 0.2,
    verbose: bool = True,
) -> Config:
    """
    Create a Config object from user-provided arguments.

    This function DOES NOT parse command-line args by itself.
    You can call it from a notebook, another script, or wrap it
    in an argparse-based main later.
    """
    config = Config(
        data_path=data_path,
        train_path=train_path,
        test_path=test_path,
        task_type=task_type.lower().strip(),
        goal_text=goal_text.strip(),
        metric_name=metric_name.lower().strip(),
        cv_folds=cv_folds,
        random_seed=random_seed,
        test_size=test_size,
        verbose=verbose,
    )

    determine_mode(config)
    validate_config(config)

    return config


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def determine_mode(config: Config) -> None:
    """
    Decide whether we are in 'single_file' mode or 'train_test' mode
    based on which paths are provided.

    - single_file: data_path is set, train_path and test_path are None
    - train_test: train_path and test_path are set, data_path is None
    """
    has_data = bool(config.data_path)
    has_train = bool(config.train_path)
    has_test = bool(config.test_path)

    if has_data and not has_train and not has_test:
        config.mode = "single_file"
    elif not has_data and has_train and has_test:
        config.mode = "train_test"
    else:
        config.mode = None
        config.errors.append(
            "You must provide either:\n"
            "  - data_path (single CSV), OR\n"
            "  - train_path AND test_path (separate train/test CSVs)."
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(config: Config) -> None:
    """
    Validate the configuration. If anything is inconsistent, errors are
    added to config.errors. If there are errors, this function raises
    a ValueError.

    After successful validation:
        - config.is_valid = True
    """
    # --- Mode must be determined and valid ---
    if config.mode is None:
        # determine_mode should have already added an error if needed
        pass

    # --- Task type ---
    if config.task_type not in {"classification", "regression"}:
        config.errors.append(
            f"Invalid task_type '{config.task_type}'. "
            "Must be 'classification' or 'regression'."
        )

    # --- Metric known? ---
    if (
        config.metric_name not in CLASSIFICATION_METRICS
        and config.metric_name not in REGRESSION_METRICS
    ):
        config.errors.append(
            f"Unknown metric '{config.metric_name}'. "
            f"Classification metrics: {sorted(CLASSIFICATION_METRICS)} | "
            f"Regression metrics: {sorted(REGRESSION_METRICS)}."
        )

    # --- Metric-task consistency ---
    if config.task_type == "classification":
        if config.metric_name in REGRESSION_METRICS:
            config.errors.append(
                f"Metric '{config.metric_name}' is a regression metric "
                f"but task_type is 'classification'."
            )
    elif config.task_type == "regression":
        if config.metric_name in CLASSIFICATION_METRICS:
            config.errors.append(
                f"Metric '{config.metric_name}' is a classification metric "
                f"but task_type is 'regression'."
            )

    # --- CV folds ---
    if config.cv_folds < 2:
        config.errors.append(
            f"cv_folds must be at least 2, got {config.cv_folds}."
        )

    # --- Test size (only really used for single_file mode) ---
    if not (0.0 < config.test_size < 1.0):
        config.errors.append(
            f"test_size must be between 0 and 1, got {config.test_size}."
        )

    # --- Final decision ---
    if config.errors:
        config.is_valid = False
        # Raise an error with all messages joined together
        message = "Invalid configuration:\n" + "\n".join(
            f"- {err}" for err in config.errors
        )
        raise ValueError(message)
    else:
        config.is_valid = True


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def print_config_summary(config: Config) -> None:
    """
    Print a human-readable summary of the configuration.
    Useful for debugging and for logging what setup was used.
    """
    print("=== AutoML Config Summary ===")
    print(f"  Mode:        {config.mode}")
    print(f"  Task type:   {config.task_type}")
    print(f"  Goal:        {config.goal_text!r}")
    print()
    print("  Data paths:")
    print(f"    data_path:  {config.data_path}")
    print(f"    train_path: {config.train_path}")
    print(f"    test_path:  {config.test_path}")
    print()
    print("  Metric & CV:")
    print(f"    metric_name: {config.metric_name}")
    print(f"    cv_folds:    {config.cv_folds}")
    print(f"    test_size:   {config.test_size}")
    print(f"    random_seed: {config.random_seed}")
    print()
    print(f"  Verbose:     {config.verbose}")
    print(f"  Is valid:    {config.is_valid}")
    print("=============================")
