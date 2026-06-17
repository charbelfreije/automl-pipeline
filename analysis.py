# analysis.py
"""
Column-type inference, statistics, and schema construction.

Classifies each DataFrame column as numeric, categorical, datetime, or text;
computes per-column statistics (missingness, cardinality, min/max/mean); and
assembles a unified Dict[str, ColumnInfo] schema used by preprocessing,
target detection, and the run summary. Also provides analyze_target() for
quick descriptive statistics of the target variable.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnInfo:
    """
    Summary information about a single feature column.
    This acts as the schema entry for that column.
    """
    name: str
    kind: str  # "numeric", "categorical", "datetime", "text"
    missing_fraction: float
    n_unique: int
    id_like: bool
    too_missing: bool
    # Optional numeric stats (None for non-numeric)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None


def infer_column_types(X_train: pd.DataFrame) -> Dict[str, str]:
    """
    Infer a coarse type for each column in X_train.

    Returns
    -------
    column_types : dict
        Mapping column_name -> kind
        where kind is one of: "numeric", "categorical", "datetime", "text".
    """
    column_types: Dict[str, str] = {}

    for col in X_train.columns:
        series = X_train[col]

        # Numeric?
        if pd.api.types.is_numeric_dtype(series):
            column_types[col] = "numeric"
            continue

        # Datetime? (either already datetime dtype or can be parsed)
        if pd.api.types.is_datetime64_any_dtype(series):
            column_types[col] = "datetime"
            continue

        # Try to detect datetime-like strings using a small sample
        if series.dtype == object:
            sample = series.dropna().astype(str).head(50)
            if not sample.empty:
                # If most of the sample can be parsed as a date, call it datetime
                parsed = 0
                for val in sample:
                    try:
                        pd.to_datetime(val)
                        parsed += 1
                    except Exception:
                        pass
                if parsed / len(sample) >= 0.8:
                    column_types[col] = "datetime"
                    continue

        # Otherwise: treat as categorical or text based on cardinality
        # (we'll refine using stats later, but this is fine for now)
        n_unique = series.nunique(dropna=True)
        n_rows = len(series)

        # Heuristic: if unique values are small relative to rows -> categorical
        if n_rows > 0 and n_unique / n_rows < 0.5:
            column_types[col] = "categorical"
        else:
            column_types[col] = "text"

    return column_types


def compute_column_stats(X_train: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute basic statistics for each column in X_train.

    Returns
    -------
    stats : dict
        Mapping column_name -> dict with keys:
            - missing_fraction
            - n_unique
            - min_value (for numeric)
            - max_value (for numeric)
            - mean_value (for numeric)
    """
    stats: Dict[str, Dict[str, Any]] = {}

    n_rows = len(X_train)
    if n_rows == 0:
        raise ValueError("X_train is empty; cannot compute column statistics.")

    for col in X_train.columns:
        series = X_train[col]
        missing_fraction = float(series.isna().sum()) / float(n_rows)
        n_unique = int(series.nunique(dropna=True))

        col_stats: Dict[str, Any] = {
            "missing_fraction": missing_fraction,
            "n_unique": n_unique,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
        }

        if pd.api.types.is_numeric_dtype(series):
            # Use nan-aware functions to ignore missing values
            if series.notna().any():
                col_stats["min_value"] = float(series.min(skipna=True))
                col_stats["max_value"] = float(series.max(skipna=True))
                col_stats["mean_value"] = float(series.mean(skipna=True))

        stats[col] = col_stats

    return stats


def build_schema(
    column_types: Dict[str, str],
    column_stats: Dict[str, Dict[str, Any]],
    too_missing_threshold: float = 0.7,
) -> Dict[str, ColumnInfo]:
    """
    Combine type and stats information into a unified schema.

    Parameters
    ----------
    column_types : dict
        column_name -> kind
    column_stats : dict
        column_name -> stats dict (from compute_column_stats)
    too_missing_threshold : float
        If missing_fraction >= this, column is flagged as too_missing.

    Returns
    -------
    schema : dict
        column_name -> ColumnInfo
    """
    if not column_stats:
        raise ValueError("column_stats is empty; did you call compute_column_stats?")

    # Build a distribution of n_unique to help detect ID-like columns
    all_n_unique = [s["n_unique"] for s in column_stats.values()]
    if all_n_unique:
        unique_sorted = sorted(all_n_unique)
        # 90th percentile of n_unique
        idx_90 = max(0, int(0.9 * (len(unique_sorted) - 1)))
        n_unique_90th = unique_sorted[idx_90]
    else:
        n_unique_90th = 0

    schema: Dict[str, ColumnInfo] = {}

    for col, ctype in column_types.items():
        stats = column_stats.get(col)
        if stats is None:
            raise ValueError(f"No stats found for column {col!r}.")

        missing_fraction = stats["missing_fraction"]
        n_unique = stats["n_unique"]

        # Heuristic for id_like using 90th percentile of n_unique
        id_like = (n_unique >= n_unique_90th) and (n_unique > 50)

        too_missing = missing_fraction >= too_missing_threshold

        info = ColumnInfo(
            name=col,
            kind=ctype,
            missing_fraction=missing_fraction,
            n_unique=n_unique,
            id_like=id_like,
            too_missing=too_missing,
            min_value=stats.get("min_value"),
            max_value=stats.get("max_value"),
            mean_value=stats.get("mean_value"),
        )
        schema[col] = info

    return schema


def analyze_target(y_train: pd.Series, task_type: str) -> Dict[str, Any]:
    """
    Produce a simple summary of the target variable.

    For classification:
        - class_counts
        - class_proportions
        - n_classes
    For regression:
        - min, max, mean, std
    """
    if y_train is None:
        raise ValueError("y_train is None in analyze_target.")

    summary: Dict[str, Any] = {}

    if task_type == "classification":
        value_counts = y_train.value_counts(dropna=False)
        total = float(len(y_train))
        proportions = (value_counts / total).to_dict()

        summary["type"] = "classification"
        summary["class_counts"] = value_counts.to_dict()
        summary["class_proportions"] = proportions
        summary["n_classes"] = int(value_counts.shape[0])

    elif task_type == "regression":
        summary["type"] = "regression"
        summary["min"] = float(y_train.min(skipna=True))
        summary["max"] = float(y_train.max(skipna=True))
        summary["mean"] = float(y_train.mean(skipna=True))
        summary["std"] = float(y_train.std(skipna=True))
    else:
        raise ValueError(
            f"Unknown task_type {task_type!r} in analyze_target. "
            "Expected 'classification' or 'regression'."
        )

    return summary


def print_schema_summary(schema: Dict[str, ColumnInfo]) -> None:
    """
    Print a concise summary of the schema for debugging / EDA.
    """
    print("=== Schema Summary ===")
    for col, info in schema.items():
        print(
            f"- {col}: kind={info.kind}, "
            f"missing={info.missing_fraction:.2%}, "
            f"n_unique={info.n_unique}, "
            f"id_like={info.id_like}, "
            f"too_missing={info.too_missing}"
        )
    print("======================")
