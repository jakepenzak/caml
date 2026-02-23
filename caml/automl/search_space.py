"""Search space specifications for AutoML hyperparameter tuning.

Provides dataclass-based specifications that can be converted to Optuna,
or other AutoML library formats.

```{python}

from caml.automl import (
    IntSpec,
    FloatSpec,
    CategoricalSpec,
    BoolSpec,
    ConstantSpec,
    NuisanceModelSpec,
    SearchSpaceSpec,
    SearchSpace
)

search_space: SearchSpace = (
    IntSpec(name="cv", lower=2, upper=10),
    FloatSpec(name="alpha", lower=0.0, upper=1.0),
    CategoricalSpec(name="solver", choices=["auto", "svd", "cholesky", "lsqr"]),
    BoolSpec(name="fit_intercept"),
    ConstantSpec(name="random_state", value=42),
    NuisanceModelSpec(name="model_y", model_type="outcome"),
)

assert all(isinstance(spec, SearchSpaceSpec) for spec in search_space)
```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Sequence


@dataclass
class SearchSpaceSpec(ABC):
    """Base class for hyperparameter search space specifications.

    All search space specs must provide conversion methods to supported
    AutoML libraries (Optuna).
    """

    name: str

    # @abstractmethod
    # def to_flaml(self) -> dict[str, Any]:
    #     """Convert to FLAML search space format.

    #     Returns
    #     -------
    #     dict
    #         FLAML-compatible search space configuration.
    #     """
    #     ...

    @abstractmethod
    def to_optuna(self, trial, prefix: str = "") -> Any:
        """Sample value using Optuna trial.

        Parameters
        ----------
        trial
            Optuna trial object for sampling.
        prefix
            Optional prefix for parameter name (useful for nested search spaces), by default ""

        Returns
        -------
        Any
            Sampled hyperparameter value.
        """
        ...

    @abstractmethod
    def validate(self) -> None:
        """Validate spec configuration.

        Raises
        ------
        ValueError
            If spec configuration is invalid.
        """
        pass


@dataclass
class NumericSpec(SearchSpaceSpec, ABC):
    """Base class for numeric (int/float) search spaces."""

    name: str
    lower: float
    upper: float
    log: bool = False

    def __post_init__(self):
        """Validate numeric range."""
        self.validate()

    def validate(self) -> None:
        """Validate numeric bounds."""
        if self.lower >= self.upper:
            raise ValueError(
                f"Invalid range for '{self.name}': lower ({self.lower}) "
                f"must be < upper ({self.upper})"
            )
        if self.log and self.lower <= 0:
            raise ValueError(
                f"Log scale requires positive lower bound for '{self.name}', "
                f"got {self.lower}"
            )


@dataclass
class IntSpec(NumericSpec):
    """Integer hyperparameter search space.

    Parameters
    ----------
    name
        Parameter name.
    lower
        Minimum value (inclusive).
    upper
        Maximum value (inclusive).
    log
        If True, sample on log scale.
    step
        Step size for grid search (optional).

    Examples
    --------
    ```python
    from caml.automl import IntSpec

    # Linear scale
    cv_spec = IntSpec(name="cv", lower=2, upper=10)

    # Log scale for things like max_iter
    max_iter_spec = IntSpec(name="max_iter", lower=100, upper=10000, log=True)
    ```
    """

    name: str
    lower: int
    upper: int
    log: bool = False
    step: int = 1

    def to_optuna(self, trial, prefix: str = "") -> int:
        """Sample integer using Optuna."""
        return trial.suggest_int(
            f"{prefix}{self.name}",
            self.lower,
            self.upper,
            log=self.log,
            step=self.step,
        )


@dataclass
class FloatSpec(NumericSpec):
    """Float hyperparameter search space.

    Parameters
    ----------
    name
        Parameter name.
    lower
        Minimum value.
    upper
        Maximum value.
    log
        If True, sample on log scale.
    step
        Step size for discretization (optional).

    Examples
    --------
    ```python
    from caml.automl import FloatSpec

    # Linear scale
    alpha_spec = FloatSpec(name="alpha", lower=0.0, upper=1.0)

    # Log scale for regularization
    lambda_spec = FloatSpec(name="lambda", lower=1e-5, upper=1e-1, log=True)
    ```
    """

    name: str
    lower: float
    upper: float
    log: bool = False
    step: float | None = None

    def to_optuna(self, trial, prefix: str = "") -> float:
        """Sample float using Optuna."""
        return trial.suggest_float(
            f"{prefix}{self.name}",
            self.lower,
            self.upper,
            log=self.log,
            step=self.step if not self.log else None,
        )


@dataclass
class CategoricalSpec(SearchSpaceSpec):
    """Categorical hyperparameter search space.

    Parameters
    ----------
    name
        Parameter name.
    choices
        List of possible values.

    Examples
    --------
    ```python
    from caml.automl import CategoricalSpec

    solver_spec = CategoricalSpec(
        name="solver", choices=["auto", "svd", "cholesky", "lsqr"]
    )
    ```
    """

    name: str
    choices: list[Any]

    def __post_init__(self):
        """Validate choices."""
        self.validate()

    def validate(self) -> None:
        """Validate categorical choices."""
        if not self.choices:
            raise ValueError(f"Empty choices list for '{self.name}'")
        if len(self.choices) != len(set(str(c) for c in self.choices)):
            raise ValueError(f"Duplicate choices in '{self.name}'")

    def to_optuna(self, trial, prefix: str = "") -> Any:
        """Sample categorical using Optuna."""
        return trial.suggest_categorical(f"{prefix}{self.name}", self.choices)


@dataclass
class BoolSpec(SearchSpaceSpec):
    """Boolean hyperparameter search space.

    Parameters
    ----------
    name
        Parameter name.

    Examples
    --------
    ```python
    from caml.automl import BoolSpec

    fit_intercept_spec = BoolSpec(name="fit_intercept")
    ```
    """

    name: str

    def to_optuna(self, trial, prefix: str = "") -> bool:
        """Sample boolean using Optuna."""
        return trial.suggest_categorical(f"{prefix}{self.name}", [True, False])

    def validate(self) -> None:
        """No validation needed for boolean spec."""
        pass


@dataclass
class ConstantSpec(SearchSpaceSpec):
    """Constant (fixed) hyperparameter value.

    Useful for fixing certain parameters during search while keeping
    a unified search space interface.

    Parameters
    ----------
    name
        Parameter name.
    value
        Fixed value.

    Examples
    --------
    ```python
    from caml.automl import ConstantSpec

    # Fix random state during search
    random_state_spec = ConstantSpec(name="random_state", value=42)
    ```
    """

    name: str
    value: Any

    def to_optuna(self, trial, prefix: str = "") -> Any:
        """Return constant value (no sampling)."""
        return self.value

    def validate(self) -> None:
        """No validation needed for constant spec."""
        pass


@dataclass
class NuisanceModelSpec(SearchSpaceSpec):
    """Reference to a tuned nuisance model.

    Special spec type that indicates a hyperparameter should be set
    to a fitted nuisance model from NuisanceTuner.

    Parameters
    ----------
    name
        Parameter name (e.g., "model_y", "model_t").
    model_type
        Type of nuisance model to use.

    Examples
    --------
    ```python
    from caml.automl import NuisanceModelSpec

    # For DML estimators
    model_y_spec = NuisanceModelSpec(name="model_y", model_type="outcome")
    model_t_spec = NuisanceModelSpec(name="model_t", model_type="treatment")
    ```
    """

    name: str
    model_type: Literal["treatment", "outcome", "regression"]

    def __post_init__(self):
        """Validate model type."""
        self.validate()

    def validate(self) -> None:
        """Validate nuisance model type."""
        if self.model_type not in {"treatment", "outcome", "regression"}:
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                f"Must be 'treatment', 'outcome', or 'regression'."
            )

    def to_optuna(self, trial, prefix: str = "") -> Any:
        """Not applicable for nuisance models."""
        raise NotImplementedError(
            "NuisanceModelSpec is handled separately, not by Optuna"
        )


@dataclass
class StandardMLSpec(SearchSpaceSpec):
    """Reference to a traditional ML model for use in meta-learners and final stage models.

    Parameters
    ----------
    name
        Parameter name (e.g., "model_final", etc.).
    models
        Type of nuisance model to use.

    Examples
    --------
    ```python
    from caml.automl import StandardMLSpec

    # For DML estimators using traditional ML models for nuisances
    model_y_spec = StandardMLSpec(
        name="model_final", models=["lightgbm", "xgboost", "random_forest"]
    )
    ```
    """

    name: str
    models: Sequence[str] | None = None

    def __post_init__(self):
        """Validate model type.

        If `models` is not provided, default to the full set of currently
        registered standard ML estimators (resolved lazily).
        """
        from caml.estimators.standard_ml import AVAILABLE_STANDARD_ML_ESTIMATORS

        self._VALID_KEYS = list(AVAILABLE_STANDARD_ML_ESTIMATORS.keys())
        if self.models is None:
            # Resolve available models lazily to avoid circular import issues.
            self.models = self._VALID_KEYS
        self.validate()

    def validate(self) -> None:
        """Validate traditional ML model type."""
        if self.models is None:
            raise ValueError("models cannot be None after __post_init__")
        if any(mt not in self._VALID_KEYS for mt in self.models):
            raise ValueError(
                f"Invalid model_types: {self.models}. Must be subset of {self._VALID_KEYS}."
            )

    def to_optuna(self, trial, prefix: str = "") -> Any:
        """Return a fitted traditional ML model based on the trial's choice.

        All nested hyperparameters for the chosen model are also sampled and set on the estimator.
        """
        from caml.estimators.standard_ml import AVAILABLE_STANDARD_ML_ESTIMATORS

        estimator_name = trial.suggest_categorical(f"{prefix}{self.name}", self.models)
        standard_ml_estimator = AVAILABLE_STANDARD_ML_ESTIMATORS[estimator_name]
        # TODO: Add logic to handle regressor/classifier distinction (eg, for metalearners).
        estimator = standard_ml_estimator._regressor_class()
        for param in standard_ml_estimator.default_search_space:
            estimator.set_params(
                **{param.name: param.to_optuna(trial, prefix=f"{prefix}{self.name}__")}
            )
        return estimator


SearchSpace = Sequence[SearchSpaceSpec]
"""Convenience alias for a sequence of search space specifications. That is, `Sequence[SearchSpaceSpec]`."""
