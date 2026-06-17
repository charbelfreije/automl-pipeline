# preprocessing.py
"""
Feature grouping and preprocessing pipeline construction.

Groups schema columns by type (numeric, categorical, datetime), defines three
named preprocessing strategies that differ in imputation and scaling choices,
and assembles a sklearn ColumnTransformer for each strategy so the search loop
can evaluate every strategy × model combination.
"""

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from analysis import ColumnInfo


# ---------------------------------------------------------------------------
# 1. GROUP FEATURES BY TYPE
# ---------------------------------------------------------------------------

def get_feature_groups(schema: Dict[str, ColumnInfo]) -> Dict[str, List[str]]:
    """
    From the schema, create lists of feature names grouped by type:
    - numeric
    - categorical
    - datetime (optional, not heavily used in basic pipelines)

    Excludes:
    - ID-like columns
    - Too-missing columns
    """
    numeric = []
    categorical = []
    datetime_cols = []

    for col, info in schema.items():

        if info.id_like:
            # This prevents things like "CustomerID" from being treated as features
            continue

        if info.too_missing:
            # Strategy may choose to drop very broken columns
            continue

        if info.kind == "numeric":
            numeric.append(col)

        elif info.kind == "categorical":
            categorical.append(col)

        elif info.kind == "datetime":
            datetime_cols.append(col)

        # text is ignored for now (course does not require NLP)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
    }


# ---------------------------------------------------------------------------
# 2. DEFINE MULTIPLE CLEANING STRATEGIES
# ---------------------------------------------------------------------------

def define_preprocessing_strategies(feature_groups: Dict[str, List[str]]) -> List[Dict]:
    """
    Returns a list of strategies, each describing a different combination of:
    - numeric imputation
    - numeric scaling
    - categorical imputation
    - categorical encoding
    - optional drops
    """
    numeric = feature_groups["numeric"]
    categorical = feature_groups["categorical"]

    strategies = [

        # ------------------------------
        # STRATEGY A
        # ------------------------------
        {
            "name": "strategy_A_median_scaler",
            "numeric_imputer": ("median",),
            "numeric_scaler": StandardScaler(),
            "categorical_imputer": ("most_frequent",),
            "categorical_encoder": OneHotEncoder(handle_unknown="ignore"),
        },

        # ------------------------------
        # STRATEGY B
        # ------------------------------
        {
            "name": "strategy_B_mean_no_scaler",
            "numeric_imputer": ("mean",),
            "numeric_scaler": None,
            "categorical_imputer": ("most_frequent",),
            "categorical_encoder": OneHotEncoder(handle_unknown="ignore"),
        },

        # ------------------------------
        # STRATEGY C
        # ------------------------------
        {
            "name": "strategy_C_median_minmax",
            "numeric_imputer": ("median",),
            "numeric_scaler": MinMaxScaler(),
            "categorical_imputer": ("constant", "Unknown"),
            "categorical_encoder": OneHotEncoder(handle_unknown="ignore"),
        },
    ]

    return strategies


# ---------------------------------------------------------------------------
# 3. APPLY FEATURE DROPPING BASED ON STRATEGY
# ---------------------------------------------------------------------------

def apply_feature_filtering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
    schema: Dict[str, ColumnInfo],
    strategy_config: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, Dict[str, ColumnInfo]]:
    """
    Allows strategies to drop columns if needed (ID-like and too-missing already removed in grouping).
    Currently minimal — but extendable later.

    Returns updated X_train, X_test, and updated schema.
    """

    # Currently this function does nothing extra, because ID-like and too-missing
    # columns are already filtered out in get_feature_groups().
    # But the structure is here for future extension.

    return X_train, X_test, schema


# ---------------------------------------------------------------------------
# 4. BUILD THE COLUMN TRANSFORMER FOR A GIVEN STRATEGY
# ---------------------------------------------------------------------------

def build_preprocessing_transformer(
    strategy_config: Dict,
    feature_groups: Dict[str, List[str]],
) -> ColumnTransformer:
    """
    Creates the sklearn ColumnTransformer for a given strategy.
    """

    numeric_features = feature_groups["numeric"]
    categorical_features = feature_groups["categorical"]

    transformers = []

    # --- Numeric pipeline ---
    if numeric_features:
        num_imputer_type = strategy_config["numeric_imputer"][0]
        num_imputer = SimpleImputer(strategy=num_imputer_type)

        if strategy_config["numeric_scaler"] is not None:
            num_scaler = strategy_config["numeric_scaler"]
            numeric_pipeline = [("imputer", num_imputer), ("scaler", num_scaler)]
        else:
            numeric_pipeline = [("imputer", num_imputer)]

        transformers.append(
            ("numeric", 
             make_pipeline_from_steps(numeric_pipeline),
             numeric_features)
        )

    # --- Categorical pipeline ---
    if categorical_features:
        cat_imputer_type = strategy_config["categorical_imputer"][0]
        if cat_imputer_type == "constant":
            fill_value = strategy_config["categorical_imputer"][1]
            cat_imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
        else:
            cat_imputer = SimpleImputer(strategy=cat_imputer_type)

        cat_encoder = strategy_config["categorical_encoder"]

        categorical_pipeline = [
            ("imputer", cat_imputer),
            ("encoder", cat_encoder),
        ]

        transformers.append(
            ("categorical",
             make_pipeline_from_steps(categorical_pipeline),
             categorical_features)
        )

    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# HELPER: construct a sklearn Pipeline from steps list
# ---------------------------------------------------------------------------

from sklearn.pipeline import Pipeline

def make_pipeline_from_steps(steps_list):
    """
    Utility to construct a sklearn Pipeline from a list of (name, transformer) pairs.
    """
    return Pipeline(steps_list)
