"""Standard ML model wrappers for use in meta-learnder and final-stage models.

Lightweight wrappers around scikit-learn, LightGBM, and XGBoost estimators
that expose a ``default_search_space``, ``_classification_class``, and ``_regression_class``
class attribute for integration with CaML's AutoML tuning pipeline.

These are **not** CATE estimators --- they are building blocks used inside CATE estimator wrappers.

```{python}
from caml.estimators.standard_ml import AVAILABLE_STANDARD_ML_ESTIMATORS

# List available model identifiers
list(AVAILABLE_STANDARD_ML_ESTIMATORS.keys())
```

See Also
--------
[`StandardMLSpec`](search_space.qmd#caml.automl.search_space.StandardMLSpec) : Search space spec referencing these models.
"""

from __future__ import annotations

import inspect

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
)
from xgboost import XGBClassifier, XGBRegressor

from caml.automl.search_space import (
    CategoricalSpec,
    ConstantSpec,
    FloatSpec,
    IntSpec,
    SearchSpace,
)


class BaseStandardMLEstimator:
    """Abstract base for standard ML wrappers with a tunable search space.

    Notes
    -----
    Subclasses must set three class-level attributes:

    * ``default_search_space`` -- a ``SearchSpace`` tuple of ``SearchSpaceSpec``
      objects describing the hyperparameters to tune.
    * ``_classifier_class`` -- the sklearn-compatible classifier class, or
      ``None`` if the model is regression-only.
    * ``_regressor_class`` -- the sklearn-compatible regressor class, or
      ``None`` if the model is classification-only.
    """

    # Subclasses override these
    _classifier_class: type | None = None
    _regressor_class: type | None = None
    default_search_space: SearchSpace

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce ``default_search_space`` on concrete subclasses."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and any(
            attr not in cls.__dict__
            for attr in [
                "default_search_space",
                "_classifier_class",
                "_regressor_class",
            ]
        ):
            raise TypeError(
                f"{cls.__name__} must define 'default_search_space', '_classifier_class', "
                f"and '_regressor_class' as a class attribute."
            )


class LGBMModel(BaseStandardMLEstimator):
    """LightGBM gradient-boosted tree wrapper."""

    default_search_space: SearchSpace = (
        IntSpec(name="n_estimators", lower=4, upper=2048, log=True),
        IntSpec(name="num_leaves", lower=4, upper=2048, log=True),
        IntSpec(name="min_child_samples", lower=2, upper=128, log=True),
        FloatSpec(name="learning_rate", lower=1 / 1024, upper=1.0, log=True),
        IntSpec(name="max_bin", lower=7, upper=1023),
        FloatSpec(name="colsample_bytree", lower=0.01, upper=1.0),
        FloatSpec(name="reg_alpha", lower=1 / 1024, upper=1024, log=True),
        FloatSpec(name="reg_lambda", lower=1 / 1024, upper=1024, log=True),
    )

    _regressor_class = LGBMRegressor
    _classifier_class = LGBMClassifier


class XGBoostModel(BaseStandardMLEstimator):
    """XGBoost gradient-boosted tree wrapper (unlimited depth / ``lossguide``)."""

    default_search_space: SearchSpace = (
        IntSpec(name="n_estimators", lower=4, upper=2048, log=True),
        IntSpec(name="max_leaves", lower=4, upper=2048, log=True),
        FloatSpec(name="min_child_weight", lower=0.001, upper=128, log=True),
        FloatSpec(name="learning_rate", lower=1 / 1024, upper=1.0, log=True),
        FloatSpec(name="subsample", lower=0.1, upper=1.0),
        FloatSpec(name="colsample_bylevel", lower=0.01, upper=1.0),
        FloatSpec(name="colsample_bytree", lower=0.01, upper=1.0),
        FloatSpec(name="reg_alpha", lower=1 / 1024, upper=1024, log=True),
        FloatSpec(name="reg_lambda", lower=1 / 1024, upper=1024, log=True),
    )

    _regressor_class = XGBRegressor
    _classifier_class = XGBClassifier


class XGBoostLimitDepthModel(BaseStandardMLEstimator):
    """XGBoost gradient-boosted tree wrapper with limited ``max_depth``."""

    default_search_space: SearchSpace = (
        IntSpec(name="n_estimators", lower=4, upper=2048, log=True),
        IntSpec(name="max_depth", lower=1, upper=15),
        FloatSpec(name="min_child_weight", lower=0.001, upper=128, log=True),
        FloatSpec(name="learning_rate", lower=1 / 1024, upper=1.0, log=True),
        FloatSpec(name="subsample", lower=0.1, upper=1.0),
        FloatSpec(name="colsample_bylevel", lower=0.01, upper=1.0),
        FloatSpec(name="colsample_bytree", lower=0.01, upper=1.0),
        FloatSpec(name="reg_alpha", lower=1 / 1024, upper=1024, log=True),
        FloatSpec(name="reg_lambda", lower=1 / 1024, upper=1024, log=True),
    )

    _regressor_class = XGBRegressor
    _classifier_class = XGBClassifier


class RandomForestModel(BaseStandardMLEstimator):
    """Random Forest wrapper."""

    default_search_space: SearchSpace = (
        IntSpec(name="n_estimators", lower=4, upper=2048, log=True),
        FloatSpec(name="max_features", lower=0.1, upper=1.0, log=True),
        IntSpec(name="max_leaf_nodes", lower=4, upper=2048, log=True),
        CategoricalSpec(name="criterion", choices=["gini", "entropy"]),
    )

    _regressor_class = RandomForestRegressor
    _classifier_class = RandomForestClassifier


class ExtraTreesModel(BaseStandardMLEstimator):
    """Extra Trees wrapper."""

    default_search_space: SearchSpace = (
        IntSpec(name="n_estimators", lower=4, upper=2048, log=True),
        FloatSpec(name="max_features", lower=0.1, upper=1.0, log=True),
        IntSpec(name="max_leaf_nodes", lower=4, upper=2048, log=True),
        CategoricalSpec(name="criterion", choices=["gini", "entropy"]),
    )

    _regressor_class = ExtraTreesRegressor
    _classifier_class = ExtraTreesClassifier


class LogisticRegressionModel(BaseStandardMLEstimator):
    """Logistic Regression with L1 (Lasso) regularisation."""

    default_search_space: SearchSpace = (
        FloatSpec(name="C", lower=0.03125, upper=32768.0, log=True),
        ConstantSpec(name="solver", value="saga"),
        CategoricalSpec(name="penalty", choices=["l1", "l2"]),
    )

    _regressor_class = None
    _classifier_class = LogisticRegression


class ElasticNetModel(BaseStandardMLEstimator):
    """Elastic Net linear regression wrapper."""

    default_search_space: SearchSpace = (
        FloatSpec(name="alpha", lower=0.0001, upper=1.0, log=True),
        FloatSpec(name="l1_ratio", lower=0.0, upper=1.0),
        CategoricalSpec(name="selection", choices=["cyclic", "random"]),
    )

    _regressor_class = ElasticNet
    _classifier_class = None


AVAILABLE_STANDARD_ML_ESTIMATORS: dict = {
    "lightgbm": LGBMModel,
    "xgboost": XGBoostModel,
    "xgboost_limitdepth": XGBoostLimitDepthModel,
    "random_forest": RandomForestModel,
    "extra_tress": ExtraTreesModel,
    "logistic": LogisticRegressionModel,
    "elastic_net": ElasticNetModel,
}
"""Dictionary of available traditional ML models for meta-learners and final stage models."""
