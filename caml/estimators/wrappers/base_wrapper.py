"""Shared base class and functionality (mixin) for wrapped AutoCATE estimators."""

from __future__ import annotations

import inspect
from abc import abstractmethod

import numpy as np
import pandas as pd
from econml._cate_estimator import BaseCateEstimator

from caml.data.dataset import CausalDataset
from caml.inference.inference_enums import InferenceType
from caml.inference.results import InferenceResult

from ..base_estimator import AutoCateEstimator, BaseAutoCateEstimatorMixin


class BaseEconMLWrapperMixin(BaseAutoCateEstimatorMixin):
    """Mixin providing common functionality for EconML wrappers."""

    _estimator: BaseCateEstimator

    @abstractmethod
    def fit(self, data: CausalDataset, **kwargs) -> AutoCateEstimator: ...

    def effect(self, X: np.ndarray | pd.DataFrame, **effect_kwargs) -> np.ndarray:
        """Predict CATE for given features.

        Parameters
        ----------
        X
            Feature matrix.
        **effect_kwargs
            Additional arguments passed to EconML's effect().
            For discrete treatment: T0, T1 specify treatment comparison (default 0 vs 1).
            For continuous treatment: T0, T1 specify dose levels to compare.

        Returns
        -------
        np.ndarray
            Estimated CATE.
        """
        self.check_fitted()
        return self._estimator.effect(X, **effect_kwargs)

    def effect_inference(
        self,
        X: np.ndarray | pd.DataFrame,
        inference_type: InferenceType | None = None,
        bootstrapper: bool | None = None,
        **effect_inference_kwargs,
    ) -> InferenceResult:
        """Get complete inference results for CATE estimates.

        Returns results in a single `~~results.InferenceResult` object, which can be used for hypothesis testing and confidence interval generation.

        **TODO: Implement Bootstrapper & cache functionality**

        Parameters
        ----------
        X
            Feature matrix for inference.
        inference_type
            Inference method to use from `~~inference_enums.InferenceType`
            (`ANALYTIC`, `BOOTSTRAP`, or `None` for auto-selection).
        bootstrapper
            Bootstrap sampler to use when `inference_type` requests
            `BOOTSTRAP` from `~~inference_enums.InferenceType`. If `None`,
            uses the default bootstrapper.
        **effect_inference_kwargs
            Additional arguments (e.g., `n_bootstrap`, `random_state`).

        Returns
        -------
        InferenceResult
            Complete inference results with point estimates, CIs, stderr, and metadata.

        Raises
        ------
        ValueError
            If method not supported.
        """
        if inference_type == InferenceType.BOOTSTRAP:
            raise NotImplementedError("Bootstrap inference not yet implemented.")

        effect_inference = self._estimator.effect_inference(
            X, **effect_inference_kwargs
        )

        return InferenceResult(
            effect=effect_inference.point_estimate,
            stderr=effect_inference.stderr,
            method=inference_type,
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns all parameters of the underlying LinearDML estimator,
        including defaults that weren't explicitly passed.

        Parameters
        ----------
        deep
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        # Get the __init__ signature of LinearDML
        sig = inspect.signature(self._estimator.__init__)

        # Start with all default parameters
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.default != inspect.Parameter.empty:
                params[param_name] = param.default

        # Override with user-provided kwargs
        params.update(self._econml_kwargs)

        # If deep=True, try to get params from nested estimators
        if deep:
            for key, value in list(params.items()):
                if hasattr(value, "get_params"):
                    nested_params = value.get_params(deep=True)
                    params.update({f"{key}__{k}": v for k, v in nested_params.items()})

        return params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self
            Estimator instance.
        """
        # Handle nested parameters (e.g., model_y__alpha)
        nested_params = {}
        direct_params = {}

        for key, value in params.items():
            if "__" in key:
                # This is a nested parameter
                estimator_name, param_name = key.split("__", 1)
                if estimator_name not in nested_params:
                    nested_params[estimator_name] = {}
                nested_params[estimator_name][param_name] = value
            else:
                direct_params[key] = value

        # Update direct parameters
        self._econml_kwargs.update(direct_params)

        # Update nested estimators
        for estimator_name, nested_dict in nested_params.items():
            if estimator_name in self._econml_kwargs:
                estimator = self._econml_kwargs[estimator_name]
                if hasattr(estimator, "set_params"):
                    estimator.set_params(**nested_dict)

        # Recreate estimator with updated parameters
        self._estimator = self._estimator.__class__(**self._econml_kwargs)
        # Reset fitted state since parameters changed
        self._is_fitted = False

        return self

    def __getattr__(self, name: str):
        """Forward attribute access to underlying estimator if not found on Wrapper."""
        if self._estimator is not None and hasattr(self._estimator, name):
            return getattr(self._estimator, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
