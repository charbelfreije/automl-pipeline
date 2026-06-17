# data_io.py
"""
Data loading, generic cleaning, and dataset packaging.

Reads CSV files according to the Config (single-file or separate train/test),
applies a sequence of cleaning steps — replacing infinities, deduplicating rows,
dropping constant and mostly-missing columns, and trimming whitespace — then
wraps the cleaned DataFrames in a DatasetBundle for use by the rest of the
pipeline. A human-readable cleaning report is written to automl_outputs/.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config


DEFAULT_OUTPUT_DIR = "automl_outputs"


@dataclass
class DatasetBundle:
    """
    Container for all dataset-related pieces that the rest of the
    AutoML system will use.

    At the beginning, y_train, y_test, and target_name may be None,
    and X_train/X_test may still contain the target column.
    """
    X_train: pd.DataFrame
    y_train: Optional[pd.Series]
    X_test: Optional[pd.DataFrame]
    y_test: Optional[pd.Series]
    target_name: Optional[str] = None


# ---------------------------------------------------------------------------
# LOADING + GENERIC CLEANING
# ---------------------------------------------------------------------------

def load_data_from_config(config: Config) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load the raw data according to the config and apply a generic cleaning
    process to TRAIN (and consistent column drops to TEST).

    Also writes a human-readable cleaning report to:
        automl_outputs/cleaning_report.txt

    Returns
    -------
    train_df : pd.DataFrame
    test_df  : Optional[pd.DataFrame]
    """
    if config.mode not in {"single_file", "train_test"}:
        raise ValueError(
            f"config.mode must be 'single_file' or 'train_test', got {config.mode!r}."
        )

    # --- Load raw data ---
    if config.mode == "single_file":
        if not config.data_path:
            raise ValueError("config.data_path is empty in single_file mode.")
        raw_train_df = pd.read_csv(config.data_path)
        raw_test_df = None
    else:
        if not config.train_path or not config.test_path:
            raise ValueError(
                "config.train_path and config.test_path must both be set in train_test mode."
            )
        raw_train_df = pd.read_csv(config.train_path)
        raw_test_df = pd.read_csv(config.test_path)

    # Basic pre-checks
    _basic_dataframe_checks(raw_train_df, name="train")
    if raw_test_df is not None:
        _basic_dataframe_checks(raw_test_df, name="test")

    # --- Generic cleaning ---
    train_df, test_df, report_text = _clean_train_and_test(
        raw_train_df, raw_test_df, output_dir=DEFAULT_OUTPUT_DIR
    )

    # Post-clean checks + column compatibility
    _basic_dataframe_checks(train_df, name="train (clean)")
    if test_df is not None:
        _basic_dataframe_checks(test_df, name="test (clean)")
        _check_compatible_columns(train_df, test_df)

    return train_df, test_df


def _clean_train_and_test(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    output_dir: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """
    Apply generic cleaning steps to TRAIN and consistent column drops to TEST.

    Steps on TRAIN:
      - Replace ±inf with NaN (numeric columns)
      - Drop duplicate rows
      - Drop constant columns (n_unique <= 1)
      - Drop columns with > 95% missing
      - Trim whitespace in object columns

    TEST:
      - Apply same column drops (constant/high-missing from TRAIN)
      - Replace ±inf with NaN (numeric)
      - Drop duplicate rows
      - Trim whitespace in object columns
    """
    lines: List[str] = []
    lines.append("CLEANING REPORT")
    lines.append("===============")
    lines.append("")

    # --- Initial shape (TRAIN) ---
    initial_rows, initial_cols = train_df.shape
    lines.append(f"Initial TRAIN shape: {initial_rows} rows, {initial_cols} columns")
    lines.append("")

    df_train = train_df.copy()

    # 1) Replace ±inf with NaN
    numeric_cols = df_train.select_dtypes(include=["number"]).columns.tolist()
    n_inf = 0
    if numeric_cols:
        arr = df_train[numeric_cols].to_numpy()
        n_inf = int(np.isinf(arr).sum())
        if n_inf > 0:
            df_train[numeric_cols] = df_train[numeric_cols].replace([np.inf, -np.inf], np.nan)
    lines.append("--- Infinite values (TRAIN) ---")
    if n_inf > 0:
        lines.append(f"Replaced {n_inf} ±inf values with NaN in numeric columns.")
    else:
        lines.append("No infinite values found.")
    lines.append("")

    # 2) Drop duplicate rows
    before_dups = df_train.shape[0]
    df_train = df_train.drop_duplicates()
    dropped_dups = before_dups - df_train.shape[0]
    lines.append("--- Duplicate rows (TRAIN) ---")
    lines.append(f"Dropped {dropped_dups} duplicate rows.")
    lines.append("")

    # 3) Constant columns
    constant_cols = [c for c in df_train.columns if df_train[c].nunique(dropna=False) <= 1]
    df_train = df_train.drop(columns=constant_cols, errors="ignore")
    lines.append("--- Constant columns removed (TRAIN) ---")
    if constant_cols:
        for c in constant_cols:
            lines.append(f"  - {c}")
    else:
        lines.append("No constant columns removed.")
    lines.append("")

    # 4) High-missing columns
    missing_ratio = df_train.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.95].index.tolist()
    df_train = df_train.drop(columns=high_missing_cols, errors="ignore")
    lines.append("--- Columns with >95% missing (TRAIN, removed) ---")
    if high_missing_cols:
        for c in high_missing_cols:
            lines.append(f"  - {c}")
    else:
        lines.append("No columns exceeded 95% missingness.")
    lines.append("")

    # 5) Whitespace trimming (TRAIN)
    trimmed_info: Dict[str, int] = {}
    for col in df_train.select_dtypes(include=["object"]).columns:
        before = df_train[col].astype(str)
        after = before.str.strip()
        corrected = int((before != after).sum())
        if corrected > 0:
            trimmed_info[col] = corrected
        df_train[col] = after
    lines.append("--- Whitespace trimming (TRAIN, object columns) ---")
    if trimmed_info:
        for col, cnt in trimmed_info.items():
            lines.append(f"  - {col}: {cnt} values corrected")
    else:
        lines.append("No whitespace issues found.")
    lines.append("")

    # Final shape
    final_rows, final_cols = df_train.shape
    row_reduction = initial_rows - final_rows
    col_reduction = initial_cols - final_cols
    row_pct = (row_reduction / initial_rows * 100.0) if initial_rows > 0 else 0.0
    col_pct = (col_reduction / initial_cols * 100.0) if initial_cols > 0 else 0.0

    lines.append("Final TRAIN shape:")
    lines.append(f"  {final_rows} rows, {final_cols} columns")
    lines.append(f"Row reduction : {row_reduction} ({row_pct:.2f}%)")
    lines.append(f"Column reduction: {col_reduction} ({col_pct:.2f}%)")
    lines.append("")

    lines.append("Note: Further preprocessing (imputation, scaling, encoding)")
    lines.append("is handled inside the modeling pipelines.")
    lines.append("")

    report_text = "\n".join(lines)

    # Write report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "cleaning_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # --- Clean TEST consistently ---
    df_test_clean: Optional[pd.DataFrame] = None
    if test_df is not None:
        df_test_clean = test_df.copy()

        # Replace ±inf with NaN
        t_numeric = df_test_clean.select_dtypes(include=["number"]).columns.tolist()
        if t_numeric:
            df_test_clean[t_numeric] = df_test_clean[t_numeric].replace(
                [np.inf, -np.inf], np.nan
            )

        # Drop duplicates
        df_test_clean = df_test_clean.drop_duplicates()

        # Drop the SAME columns as TRAIN (constant + high-missing)
        drop_cols = list(set(constant_cols) | set(high_missing_cols))
        df_test_clean = df_test_clean.drop(columns=drop_cols, errors="ignore")

        # Whitespace trimming for object columns
        for col in df_test_clean.select_dtypes(include=["object"]).columns:
            df_test_clean[col] = df_test_clean[col].astype(str).str.strip()

    return df_train, df_test_clean, report_text


# ---------------------------------------------------------------------------
# BASIC CHECKS + COMPATIBILITY
# ---------------------------------------------------------------------------

def _basic_dataframe_checks(df: pd.DataFrame, name: str = "data") -> None:
    """
    Perform very basic sanity checks on a DataFrame.
    """
    if df is None:
        raise ValueError(f"{name} DataFrame is None.")

    if df.empty:
        raise ValueError(f"{name} DataFrame is empty (no rows).")

    if df.shape[1] < 2:
        raise ValueError(f"{name} DataFrame must have at least 2 columns, got {df.shape[1]}.")

    if df.columns.duplicated().any():
        raise ValueError(f"{name} DataFrame has duplicate column names, which is not allowed.")


def _check_compatible_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Check that train and test DataFrames have compatible columns.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    extra_in_test = test_cols - train_cols
    if extra_in_test:
        raise ValueError(
            "The test CSV has columns that are not present in the train CSV: "
            f"{sorted(extra_in_test)}"
        )


# ---------------------------------------------------------------------------
# DATASET BUNDLE CREATION + SPLIT
# ---------------------------------------------------------------------------

def build_initial_dataset_bundle(
    raw_train_df: pd.DataFrame,
    raw_test_df: Optional[pd.DataFrame],
) -> DatasetBundle:
    """
    Create the initial DatasetBundle from raw (cleaned) DataFrames.
    """
    bundle = DatasetBundle(
        X_train=raw_train_df.copy(),
        y_train=None,
        X_test=raw_test_df.copy() if raw_test_df is not None else None,
        y_test=None,
        target_name=None,
    )
    return bundle


def split_train_test_if_needed(bundle: DatasetBundle, config: Config) -> DatasetBundle:
    """
    For single_file mode, split X_train/y_train into train/test sets using config.test_size.
    """
    if config.mode == "train_test":
        return bundle

    if config.mode != "single_file":
        raise ValueError(
            f"split_train_test_if_needed called with unexpected mode: {config.mode!r}"
        )

    if bundle.target_name is None or bundle.y_train is None:
        raise ValueError(
            "Target must be detected and applied before splitting in single_file mode. "
            "bundle.target_name and bundle.y_train cannot be None."
        )

    X_full = bundle.X_train
    y_full = bundle.y_train

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full,
        y_full,
        test_size=config.test_size,
        random_state=config.random_seed,
        shuffle=True,
        stratify=y_full if config.task_type == "classification" else None,
    )

    bundle.X_train = X_tr
    bundle.y_train = y_tr
    bundle.X_test = X_te
    bundle.y_test = y_te

    return bundle
