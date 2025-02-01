from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas
from flaml import AutoML
from sklearn.model_selection import train_test_split

from ..generics import cls_typechecked
from ..logging import setup_logging

# Optional dependencies
try:
    import polars  # noqa: F401

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

try:
    import pyspark  # noqa: F401
    # from pyspark.sql import SparkSession

    _HAS_PYSPARK = True
except ImportError:
    _HAS_PYSPARK = False

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


@cls_typechecked
class CamlBase(metaclass=abc.ABCMeta):
    """
    Base ABC class for core Caml classes.

    This class contains the shared methods and properties for the Caml classes.
    """

    def __init__(self, verbose: int = 1):
        setup_logging(verbose)

        self._data_backend = (
            "pandas"
            if isinstance(self.df, pandas.DataFrame)
            else "polars"
            if _HAS_POLARS and isinstance(self.df, polars.DataFrame)
            else "pyspark"
            if _HAS_PYSPARK
            and isinstance(self.df, (pyspark.sql.DataFrame, pyspark.pandas.DataFrame))
            else "unknown"
        )

    @property
    def validation_estimator(self):
        if self._validation_estimator is not None:
            return self._validation_estimator
        else:
            raise ValueError(
                "No validation estimator has been fit yet. Please run fit_validator() method first."
            )

    @property
    def final_estimator(self):
        if self._final_estimator is not None:
            return self._final_estimator
        else:
            raise ValueError(
                "No final estimator has been fit yet. Please run fit_final() method first."
            )

    @abc.abstractmethod
    def fit_validator(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def fit_final(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def summarize(self):
        pass

    def interpret(self):
        raise NotImplementedError

    def _split_data(
        self,
        *,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        sample_fraction: float = 1.0,
    ):
        """
        Splits the data into training, validation, and test sets.

        Sets the `_data_splits` internal attribute of the class.

        Parameters
        ----------
        validation_size
            The size of the validation set. Default is None.
        test_size
            The size of the test set. Default is 0.2.
        sample_fraction
            The size of the sample to use for training. Default is 1.0.

        """
        X = self._X
        W = self._W
        Y = self._Y
        T = self._T

        validation_size = int(validation_size * X.shape[0])
        test_size = int(test_size * X.shape[0])

        if sample_fraction != 1.0:
            X = X.sample(frac=sample_fraction, random_state=self.seed)
            W = W.loc[X.index]
            Y = Y.loc[X.index]
            T = T.loc[X.index]

        X_train, X_test, W_train, W_test, T_train, T_test, Y_train, Y_test = (
            train_test_split(X, W, T, Y, test_size=test_size, random_state=self.seed)
        )

        self._data_splits = {
            "X_train": X_train,
            "X_test": X_test,
            "W_train": W_train,
            "W_test": W_test,
            "T_train": T_train,
            "T_test": T_test,
            "Y_train": Y_train,
            "Y_test": Y_test,
        }

        if validation_size:
            X_train, X_val, W_train, W_val, T_train, T_val, Y_train, Y_val = (
                train_test_split(
                    X_train,
                    W_train,
                    T_train,
                    Y_train,
                    test_size=validation_size,
                    random_state=self.seed,
                )
            )

            self._data_splits["X_val"] = X_val
            self._data_splits["W_val"] = W_val
            self._data_splits["T_val"] = T_val
            self._data_splits["Y_val"] = Y_val
            self._data_splits["X_train"] = X_train
            self._data_splits["W_train"] = W_train
            self._data_splits["T_train"] = T_train
            self._data_splits["Y_train"] = Y_train

    def _run_auto_nuisance_functions(
        self,
        *,
        outcome: np.ndarray,
        features: np.ndarray | list[np.ndarray],
        discrete_outcome: bool,
        flaml_kwargs: dict | None,
        use_ray: bool,
        use_spark: bool,
    ):
        """
        AutoML utilizing FLAML to find the best nuisance models.

        Parameters
        ----------
        outcome
            The outcome variable.
        features
            The features matrix/matrices.
        discrete_outcome
            Whether the outcome is discrete or continuous.
        flaml_kwargs
            The keyword arguments to pass to FLAML.
        use_ray
            Whether to use Ray for parallel processing.
        use_spark
            Whether to use Spark for parallel processing.

        Returns
        -------
        sklearn.base.BaseEstimator
            The best nuisance model found by FLAML.
        """
        automl = AutoML()

        base_settings = {
            "n_jobs": -1,
            "log_file_name": "",
            "seed": self.seed,
            "time_budget": 300,
            "early_stop": "True",
            "eval_method": "cv",
            "n_splits": 3,
            "starting_points": "static",
            "estimator_list": ["rf", "lgbm", "extra_tree", "xgb_limitdepth"],
        }

        _flaml_kwargs = base_settings.copy()

        if discrete_outcome:
            _flaml_kwargs["task"] = "classification"
            _flaml_kwargs["metric"] = "log_loss"
        else:
            _flaml_kwargs["task"] = "regression"
            _flaml_kwargs["metric"] = "mse"

        if use_spark:
            _flaml_kwargs["use_spark"] = True
            _flaml_kwargs["n_concurrent_trials"] = 4
        elif use_ray:
            _flaml_kwargs["use_ray"] = True
            _flaml_kwargs["n_concurrent_trials"] = 4

        if flaml_kwargs is not None:
            _flaml_kwargs.update(flaml_kwargs)

        if isinstance(features, list):
            feature_matrix = features[0]
            for feature in features[1:]:
                try:
                    feature_matrix = np.hstack((feature_matrix, feature))
                except ValueError:
                    pass
        else:
            feature_matrix = features

        automl.fit(feature_matrix, outcome.ravel(), **_flaml_kwargs)

        model = automl.model.estimator

        return model

    def _dataframe_to_numpy(self):
        if self._data_backend == "pandas":
            _Y = self.df[self.Y].to_numpy()
            _T = self.df[self.T].to_numpy()
            _X = self.df[self.X].to_numpy()
            _W = self.df[self.W].to_numpy()
        elif self._data_backend == "polars":
            _Y = self.df.select(self.Y).to_numpy()
            _T = self.df.select(self.T).to_numpy()
            _X = self.df.select(self.X).to_numpy()
            _W = self.df.select(self.W).to_numpy()
        elif self._data_backend == "pyspark":
            _Y = self.df.select(self.Y).to_pandas().to_numpy()
            _T = self.df.select(self.T).to_pandas().to_numpy()
            _X = self.df.select(self.X).to_pandas().to_numpy()
            _W = self.df.select(self.W).to_pandas().to_numpy()
        else:
            raise ValueError(f"Unsupported DataFrame type: {type(self.df)}")

        return _Y, _T, _X, _W
