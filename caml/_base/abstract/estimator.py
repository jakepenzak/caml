from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import pandas as pd
from flaml import AutoML
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from caml._generics.interfaces import (
    PandasConvertibleDataFrame,
    to_pandasConvertible,
    toPandasConvertible,
)
from caml._generics.logging import DEBUG, ERROR, INFO


class BaseCamlEstimator(ABC):
    """Base ABC class for CaML estimators."""

    X: list[str]
    W: list[str]
    T: list[str] | str
    Y: list[str]
    _seed: int | None

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, df: PandasConvertibleDataFrame, **kwargs) -> "BaseCamlEstimator":
        """Fit the estimator to the data.

        Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def predict(self, df: PandasConvertibleDataFrame, **kwargs) -> Any:
        """Predict method.

        Must be implemented by child classes. Should primarily be alias for cate estimation
        and optionally to predict outcome y.
        """
        pass

    def estimate(
        self,
        df: PandasConvertibleDataFrame,
        *,
        estimand: str,
        query: str | None = None,
        **kwargs,
    ) -> Any:  # TODO: Change return type
        """Base estimate method that handles estimand routing.

        Child classes may override this method to accept additional kwargs, but
        should call `super().estimate()` to leverage the base routing logic.
        """
        df = self._convert_dataframe_to_pandas(df)
        estimand = estimand.lower()
        if estimand == "ate":
            return self._estimate_ate(df, **kwargs)
        if estimand == "cate":
            return self._estimate_cate(df, **kwargs)
        if estimand == "gate":
            if query is None:
                raise ValueError("Query string is required for GATE estimation")
            return self._estimate_gate(df, query=query, **kwargs)
        if estimand == "gatt":
            return self._estimate_gatt(df, **kwargs)
        if estimand == "att":
            return self._estimate_att(df, **kwargs)
        if estimand == "atc":
            return self._estimate_atc(df, **kwargs)
        raise ValueError(f"Invalid estimand: {estimand}")

    def _estimate_gate(self, df: pd.DataFrame, query: str, **kwargs):
        raise NotImplementedError(
            f"GATE estimation is not supported for {self.__class__.__name__}"
        )

    def _estimate_ate(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError(
            f"ATE estimation is not supported for {self.__class__.__name__}"
        )

    def _estimate_cate(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError(
            f"CATE estimation is not supported for {self.__class__.__name__}"
        )

    def _estimate_att(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError(
            f"ATT estimation is not supported for {self.__class__.__name__}"
        )

    def _estimate_atc(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError(
            f"ATC estimation is not supported for {self.__class__.__name__}"
        )

    def _estimate_gatt(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError(
            f"GATT estimation is not supported for {self.__class__.__name__}"
        )

    def interpret(self):
        raise NotImplementedError(
            f"Interpretation is not supported for {self.__class__.__name__}"
        )

    def dose_response(self):
        raise NotImplementedError(
            f"Dose-response estimation is not supported for {self.__class__.__name__}"
        )

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
            train_test_split(X, W, T, Y, test_size=test_size, random_state=self._seed)
        )

        X_train, X_val, W_train, W_val, T_train, T_val, Y_train, Y_val = (
            train_test_split(
                X_train,
                W_train,
                T_train,
                Y_train,
                test_size=validation_size,
                random_state=self._seed,
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
    def _encode_categoricals(
        df: pd.DataFrame,
        *,
        is_training: bool = False,
        categorical_mappings: dict = dict(),
    ) -> tuple[pd.DataFrame, dict]:
        cat_columns = df.select_dtypes(include=["category"]).columns
        if not cat_columns.empty:
            df = df.copy()
            if is_training:
                categorical_mappings = {}
                for col in cat_columns:
                    categories = df[col].cat.categories
                    categorical_mappings[col] = {
                        cat: i for i, cat in enumerate(categories)
                    }
                    df[col] = df[col].map(categorical_mappings[col]).astype("int")
            else:
                if categorical_mappings is None:
                    raise ValueError(
                        "No mappings passed for categorical columns and is_training is False."
                    )
                for col in cat_columns:
                    if col in categorical_mappings:
                        mapping = categorical_mappings[col]
                        df[col] = df[col].map(mapping).astype("int")
                    else:
                        raise ValueError(
                            f"No stored mapping found for categorical column '{col}'"
                        )

        return df, categorical_mappings

    @staticmethod
    def _run_automl(**flaml_kwargs) -> BaseEstimator:
        automl = AutoML()

        automl.fit(**flaml_kwargs)

        model = automl.model.estimator  # pyright: ignore[reportOptionalMemberAccess]

        INFO(
            f"Best estimator: {automl.best_estimator} with loss {automl.best_loss}"
            f" found on iteration {automl.best_iteration} in {automl.time_to_find_best_model} seconds.\n"
        )

        return model

    @staticmethod
    def _convert_dataframe_to_pandas(
        df: PandasConvertibleDataFrame,
        groups: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        def _convert_groups_to_categorical(df, groups):
            for col in groups or []:
                df[col] = df[col].astype("category")
            return df

        if isinstance(df, PandasConvertibleDataFrame):
            if isinstance(df, pd.DataFrame):
                return _convert_groups_to_categorical(df, groups)

            DEBUG(f"Converting input dataframe of type {type(df)} to pandas")
            if isinstance(df, toPandasConvertible):
                return _convert_groups_to_categorical(df.toPandas(), groups)
            if isinstance(df, to_pandasConvertible):
                return _convert_groups_to_categorical(df.to_pandas(), groups)

        ERROR(f"Unsupported dataframe type: {type(df)}")
        raise ValueError(f"Pandas conversion not currently supported for {type(df)}.")
