# models.py
"""
Model catalog and instantiation for the AutoML pipeline.

Defines ModelConfig, a dataclass that bundles a model class, its problem type,
and the hyperparameter grid to search. Provides two catalog functions —
get_classification_models() and get_regression_models() — and a single factory
function, instantiate_model(), that handles special cases: GPU-accelerated
XGBoost (identified by sentinel strings in estimator_cls) and PolynomialRegression
(assembled as an sklearn Pipeline of PolynomialFeatures + LinearRegression).
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import xgboost as xgb


@dataclass
class ModelConfig:
    """
    Configuration descriptor for a single model type.

    Attributes
    ----------
    name            : human-readable model identifier used in result tables
    estimator_cls   : the sklearn estimator class to instantiate, or the
                      sentinel string "XGB_GPU_CLS" / "XGB_GPU_REG" for
                      GPU XGBoost models; None for PolynomialRegression
    problem_type    : "classification" or "regression"
    hyperparam_grid : dict of hyperparameter name -> list of candidate values;
                      an empty dict means only default parameters are tried
    """
    name: str
    estimator_cls: Any
    problem_type: str  # "classification" or "regression"
    hyperparam_grid: Dict[str, List[Any]]


# ---------------------------------------------------------------------------
# GPU XGBOOST HELPERS  (XGBoost >= 2.0 uses tree_method="hist", device="cuda")
# ---------------------------------------------------------------------------

def _gpu_xgb_classifier(params: Dict[str, Any]):
    """
    Build an XGBClassifier configured for GPU acceleration (XGBoost >= 2.0).

    Uses tree_method="hist" and device="cuda". Caller-supplied params are
    merged on top of the GPU defaults and can override any of them.

    Parameters
    ----------
    params : hyperparameter dict (e.g. n_estimators, max_depth, learning_rate)

    Returns
    -------
    xgb.XGBClassifier
    """
    core = {
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "logloss",
        "random_state": 42,
    }
    core.update(params)
    return xgb.XGBClassifier(**core)


def _gpu_xgb_regressor(params: Dict[str, Any]):
    """
    Build an XGBRegressor configured for GPU acceleration (XGBoost >= 2.0).

    Uses tree_method="hist" and device="cuda". Caller-supplied params are
    merged on top of the GPU defaults and can override any of them.

    Parameters
    ----------
    params : hyperparameter dict (e.g. n_estimators, max_depth, learning_rate)

    Returns
    -------
    xgb.XGBRegressor
    """
    core = {
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "rmse",
        "random_state": 42,
    }
    core.update(params)
    return xgb.XGBRegressor(**core)


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION MODELS
# ---------------------------------------------------------------------------

def get_classification_models() -> List[ModelConfig]:
    """
    Return the catalog of classification ModelConfigs for the search loop.

    Includes LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    KNeighborsClassifier, GaussianNB, SVC, and GPU-accelerated XGBoostClassifier.

    Returns
    -------
    List[ModelConfig]
    """
    models: List[ModelConfig] = []

    # Logistic Regression
    models.append(
        ModelConfig(
            name="LogisticRegression",
            estimator_cls=LogisticRegression,
            problem_type="classification",
            hyperparam_grid={
                "C": [0.5, 1.0],
                "max_iter": [2000],
            },
        )
    )

    # Decision Tree
    models.append(
        ModelConfig(
            name="DecisionTreeClassifier",
            estimator_cls=DecisionTreeClassifier,
            problem_type="classification",
            hyperparam_grid={
                "max_depth": [None, 8, 12],
            },
        )
    )

    # Random Forest
    models.append(
        ModelConfig(
            name="RandomForestClassifier",
            estimator_cls=RandomForestClassifier,
            problem_type="classification",
            hyperparam_grid={
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
            },
        )
    )

    # KNN
    models.append(
        ModelConfig(
            name="KNeighborsClassifier",
            estimator_cls=KNeighborsClassifier,
            problem_type="classification",
            hyperparam_grid={
                "n_neighbors": [3, 5, 7],
            },
        )
    )

    # Naive Bayes
    models.append(
        ModelConfig(
            name="GaussianNB",
            estimator_cls=GaussianNB,
            problem_type="classification",
            hyperparam_grid={},
        )
    )

    # SVC
    models.append(
        ModelConfig(
            name="SVC",
            estimator_cls=SVC,
            problem_type="classification",
            hyperparam_grid={
                "C": [1.0],
                "kernel": ["rbf"],
                "probability": [True],
            },
        )
    )

    # GPU XGBoost Classifier
    models.append(
        ModelConfig(
            name="XGBoostClassifier_GPU",
            estimator_cls="XGB_GPU_CLS",
            problem_type="classification",
            hyperparam_grid={
                "n_estimators": [200, 300],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
        )
    )

    return models


# ---------------------------------------------------------------------------
# 2. REGRESSION MODELS
# ---------------------------------------------------------------------------

def get_regression_models() -> List[ModelConfig]:
    """
    Return the catalog of regression ModelConfigs for the search loop.

    Includes LinearRegression, PolynomialRegression (degree-2 pipeline),
    DecisionTreeRegressor, RandomForestRegressor, KNeighborsRegressor, and
    GPU-accelerated XGBoostRegressor.

    Returns
    -------
    List[ModelConfig]
    """
    models: List[ModelConfig] = []

    # Linear Regression
    models.append(
        ModelConfig(
            name="LinearRegression",
            estimator_cls=LinearRegression,
            problem_type="regression",
            hyperparam_grid={},
        )
    )

    # Polynomial Regression
    models.append(
        ModelConfig(
            name="PolynomialRegression",
            estimator_cls=None,  # sentinel: instantiate_model builds a Pipeline for this
            problem_type="regression",
            hyperparam_grid={
                "degree": [2],   # [2,3] for slower but higher accuracy
            },
        )
    )

    # Decision Tree
    models.append(
        ModelConfig(
            name="DecisionTreeRegressor",
            estimator_cls=DecisionTreeRegressor,
            problem_type="regression",
            hyperparam_grid={
                "max_depth": [None, 8, 12],
            },
        )
    )

    # Random Forest
    models.append(
        ModelConfig(
            name="RandomForestRegressor",
            estimator_cls=RandomForestRegressor,
            problem_type="regression",
            hyperparam_grid={
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
            },
        )
    )

    # KNN
    models.append(
        ModelConfig(
            name="KNeighborsRegressor",
            estimator_cls=KNeighborsRegressor,
            problem_type="regression",
            hyperparam_grid={
                "n_neighbors": [3, 5, 7],
            },
        )
    )

    # GPU XGBoost Regressor
    models.append(
        ModelConfig(
            name="XGBoostRegressor_GPU",
            estimator_cls="XGB_GPU_REG",
            problem_type="regression",
            hyperparam_grid={
                "n_estimators": [200, 400],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
        )
    )

    return models


# ---------------------------------------------------------------------------
# 3. MODEL INSTANTIATION
# ---------------------------------------------------------------------------

def instantiate_model(model_config: ModelConfig, hyperparams: Optional[Dict[str, Any]] = None):
    """
    Construct a model instance from a ModelConfig and a hyperparameter dict.

    Handles three special cases before falling back to a standard sklearn
    estimator constructor call:
      - "XGB_GPU_CLS" sentinel  → GPU XGBoostClassifier via _gpu_xgb_classifier()
      - "XGB_GPU_REG" sentinel  → GPU XGBoostRegressor via _gpu_xgb_regressor()
      - name "PolynomialRegression" → sklearn Pipeline(PolynomialFeatures + LinearRegression)

    Parameters
    ----------
    model_config : ModelConfig describing the model family
    hyperparams  : dict of parameter name -> value; defaults to {} when None

    Returns
    -------
    A fitted-able sklearn-compatible estimator or Pipeline.
    """
    if hyperparams is None:
        hyperparams = {}

    # GPU XGBoost Classifier
    if model_config.estimator_cls == "XGB_GPU_CLS":
        print(f"  [MODEL] Using GPU XGBoostClassifier with params={hyperparams}")
        return _gpu_xgb_classifier(hyperparams)

    # GPU XGBoost Regressor
    if model_config.estimator_cls == "XGB_GPU_REG":
        print(f"  [MODEL] Using GPU XGBoostRegressor with params={hyperparams}")
        return _gpu_xgb_regressor(hyperparams)

    # Polynomial Regression
    if model_config.name == "PolynomialRegression":
        degree = hyperparams.get("degree", 2)
        print(f"  [MODEL] Using PolynomialRegression (degree={degree})")
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression()),
        ])

    # Normal sklearn models
    print(f"  [MODEL] Using {model_config.name} with params={hyperparams}")
    return model_config.estimator_cls(**hyperparams)
