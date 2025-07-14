from __future__ import annotations

import abc
from typing import Any, Sequence

import pandas as pd
from flaml import AutoML
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ..generics.interfaces import PandasConvertibleDataFrame
from ..logging import DEBUG, ERROR


class BaseCamlEstimator(metaclass=abc.ABCMeta):
    """
    Base ABC class for core Caml classes.

    This class contains the shared methods and properties for the Caml classes.
    """

    X: list[str]
    W: list[str]
    T: list[str] | str
    Y: list[str]
    seed: int | None

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def estimate_ate(self):
        pass

    @abc.abstractmethod
    def estimate_cate(self):
        pass

    def interpret(self):
        raise NotImplementedError

    def dose_response(self):
        raise NotImplementedError

    def _split_data(
        self,
        *,
        df: pd.DataFrame,
        validation_fraction: float,
        test_fraction: float,
    ) -> dict[str, Any]:
        X = df[self.X]
        W = df[self.W]
        Y = df[self.Y]
        T = df[self.T]

        validation_size = int(validation_fraction * X.shape[0])
        test_size = int(test_fraction * X.shape[0])

        X_train, X_test, W_train, W_test, T_train, T_test, Y_train, Y_test = (
            train_test_split(X, W, T, Y, test_size=test_size, random_state=self.seed)
        )

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

        return {
            "X_train": X_train,
            "X_test": X_test,
            "X_val": X_val,
            "W_train": W_train,
            "W_test": W_test,
            "W_val": W_val,
            "T_train": T_train,
            "T_test": T_test,
            "T_val": T_val,
            "Y_train": Y_train,
            "Y_test": Y_test,
            "Y_val": Y_val,
        }

    @staticmethod
    def _run_automl(**flaml_kwargs) -> BaseEstimator:
        """
        AutoML utilizing FLAML.

        Parameters
        ----------
        **flaml_kwargs
            The keyword arguments to pass to FLAML.

        Returns
        -------
        sklearn.base.BaseEstimator
            The best nuisance model found by FLAML.
        """
        automl = AutoML()

        automl.fit(**flaml_kwargs)

        model = automl.model.estimator  # pyright: ignore[reportOptionalMemberAccess]

        print(
            f"Best estimator: {automl.best_estimator} with loss {automl.best_loss}"
            f" found on iteration {automl.best_iteration} in {automl.time_to_find_best_model} seconds."
        )

        return model

    @staticmethod
    def _convert_dataframe_to_pandas(
        df: PandasConvertibleDataFrame, groups: Sequence[str] | None = None
    ) -> pd.DataFrame:
        def convert_groups_to_categorical(df, groups):
            for col in groups or []:
                df[col] = df[col].astype("category")
            return df

        if isinstance(df, PandasConvertibleDataFrame):
            if isinstance(df, pd.DataFrame):
                return convert_groups_to_categorical(df, groups)

            DEBUG(f"Converting input dataframe of type {type(df)} to pandas")
            if hasattr(df, "toPandas"):
                return convert_groups_to_categorical(df.toPandas(), groups)
            if hasattr(df, "to_pandas"):
                return convert_groups_to_categorical(df.to_pandas(), groups)

        ERROR(f"Unsupported dataframe type: {type(df)}")
        raise ValueError(f"Pandas conversion not currently supported for {type(df)}.")
