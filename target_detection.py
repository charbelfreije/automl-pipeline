# target_detection.py
"""
Heuristic target-column detection for the AutoML pipeline.

Scores each column in the dataset against the user's free-text goal and
schema metadata to propose the most likely target variable, then lets the
user confirm or manually override the choice. The confirmed target is
separated from the feature matrix and stored in the DatasetBundle.
"""

from typing import Dict

import pandas as pd

from data_io import DatasetBundle
from analysis import ColumnInfo


# ---------------------------------------------------------------------------
# Scoring columns for target detection
# ---------------------------------------------------------------------------

def score_columns_for_goal(
    goal_text: str,
    df: pd.DataFrame,
    task_type: str,
    schema: Dict[str, ColumnInfo],
) -> Dict[str, float]:
    """
    Assign a score to each column indicating how likely it is to be the target.

    The score is based on:
    - Name similarity between goal_text and column name.
    - Alignment between column type/cardinality and the task type.
    - Penalties for id_like and too_missing columns.
    - Extra heuristics for common words like price/sale/sales/revenue.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame in score_columns_for_goal.")

    task_type = task_type.lower().strip()
    if task_type not in {"classification", "regression"}:
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}."
        )

    goal_lower = goal_text.lower()
    goal_tokens = [tok for tok in goal_lower.split() if tok]

    scores: Dict[str, float] = {}

    # Flags for common goal concepts
    goal_mentions_price = any(
        w in goal_lower for w in ["price", "prices", "sale", "sales", "revenue", "income"]
    )

    for col in df.columns:
        col_lower = col.lower()
        info = schema.get(col)
        score = 0.0

        # --- 1) Name-based similarity ---
        # Direct token inclusion
        for tok in goal_tokens:
            if tok in col_lower:
                score += 2.0
            elif tok.endswith("s") and tok[:-1] in col_lower:
                score += 2.0  # "sales" -> "saleprice"
            elif col_lower in tok:
                score += 1.0

        if col_lower in goal_tokens:
            score += 3.0

        # Extra boost for price/sales-related names if goal talks about price/sales
        if goal_mentions_price:
            if any(
                key in col_lower
                for key in [
                    "price",
                    "saleprice",
                    "sale_price",
                    "amount",
                    "revenue",
                    "income",
                ]
            ):
                score += 6.0  # strong bonus to hit SalePrice in house prices data

        # --- 2) Type & cardinality alignment ---
        if info is not None:
            if task_type == "classification":
                if info.kind in {"categorical", "numeric"}:
                    if info.n_unique <= 20:
                        score += 2.0
                    if info.n_unique <= 10:
                        score += 1.0
            elif task_type == "regression":
                if info.kind == "numeric":
                    score += 2.0
                    if info.n_unique > 20:
                        score += 1.0

            # Penalize ID-like and too-missing columns
            if info.id_like:
                score -= 5.0
            if info.too_missing:
                score -= 3.0

        # Base score so columns are never exactly 0
        if score == 0.0:
            score = 0.1

        scores[col] = score

    return scores


def select_best_target_column(scores: Dict[str, float]) -> str:
    """
    Return the column name with the highest score from the scored-columns dict.

    Parameters
    ----------
    scores : dict mapping column name -> float score (from score_columns_for_goal)

    Returns
    -------
    str — the column name with the maximum score
    """
    if not scores:
        raise ValueError("No scores provided to select_best_target_column.")
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# User confirmation and manual override
# ---------------------------------------------------------------------------

def confirm_target_with_user(best_guess: str, df: pd.DataFrame) -> str:
    """
    Show the auto-detected target column to the user and ask for confirmation.

    If the user rejects the suggestion, lists all available columns and prompts
    for a selection by exact name or zero-based index, looping until a valid
    choice is made.

    Parameters
    ----------
    best_guess : column name proposed by the scoring heuristic
    df         : the DataFrame whose columns are presented to the user

    Returns
    -------
    str — the confirmed or manually chosen target column name
    """
    if best_guess not in df.columns:
        raise ValueError(
            f"best_guess column {best_guess!r} not found in DataFrame columns."
        )

    print(f"\nI think your target column is: {best_guess!r}.")
    while True:
        answer = input("Do you confirm? (yes/no): ").strip().lower()
        if answer in {"yes", "y"}:
            print(f"Target column confirmed: {best_guess!r}")
            return best_guess
        elif answer in {"no", "n"}:
            break
        else:
            print("Please answer 'yes' or 'no'.")

    print("\nAvailable columns:")
    cols = list(df.columns)
    for idx, name in enumerate(cols):
        print(f"  [{idx}] {name}")

    while True:
        user_choice = input(
            "Type the exact column NAME or its INDEX from the list above: "
        ).strip()

        if user_choice.isdigit():
            idx = int(user_choice)
            if 0 <= idx < len(cols):
                chosen = cols[idx]
                print(f"Target column set to: {chosen!r}")
                return chosen
            else:
                print(f"Index {idx} is out of range. Try again.")
                continue

        if user_choice in df.columns:
            print(f"Target column set to: {user_choice!r}")
            return user_choice

        print(
            f"'{user_choice}' is not a valid column name or index. "
            "Please try again."
        )


# ---------------------------------------------------------------------------
# Apply chosen target to DatasetBundle
# ---------------------------------------------------------------------------

def apply_target_column(bundle: DatasetBundle, target_name: str) -> DatasetBundle:
    """
    Separate the target column from the feature matrix in the DatasetBundle.

    Moves target_name from X_train into y_train (and from X_test into y_test
    if the test set contains that column), then sets bundle.target_name.

    Parameters
    ----------
    bundle      : DatasetBundle with X_train still containing the target column
    target_name : name of the column to use as the label

    Returns
    -------
    DatasetBundle — the same object, modified in place and returned
    """
    if target_name not in bundle.X_train.columns:
        raise ValueError(
            f"Target column {target_name!r} not found in X_train columns."
        )

    bundle.y_train = bundle.X_train[target_name].copy()
    bundle.X_train = bundle.X_train.drop(columns=[target_name])

    if bundle.X_test is not None and target_name in bundle.X_test.columns:
        bundle.y_test = bundle.X_test[target_name].copy()
        bundle.X_test = bundle.X_test.drop(columns=[target_name])

    bundle.target_name = target_name
    return bundle
