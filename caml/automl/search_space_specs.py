"""Search space specifications for AutoML hyperparameter tuning.

Provides dataclass-based specifications that can be converted to Optuna,
or other AutoML library formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


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
    def to_optuna(self, trial) -> Any:
        """Sample value using Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for sampling.

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

    lower: int
    upper: int
    step: int | None = None

    def to_optuna(self, trial) -> int:
        """Sample integer using Optuna."""
        return trial.suggest_int(
            self.name,
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

    step: float | None = None

    def to_optuna(self, trial) -> float:
        """Sample float using Optuna."""
        return trial.suggest_float(
            self.name,
            self.lower,
            self.upper,
            log=self.log,
            step=self.step,
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

    def to_optuna(self, trial) -> Any:
        """Sample categorical using Optuna."""
        return trial.suggest_categorical(self.name, self.choices)


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

    def to_optuna(self, trial) -> bool:
        """Sample boolean using Optuna."""
        return trial.suggest_categorical(self.name, [True, False])


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

    value: Any

    def to_optuna(self, trial) -> Any:
        """Return constant value (no sampling)."""
        return self.value


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

    def to_optuna(self, trial) -> Any:
        """Not applicable for nuisance models."""
        raise NotImplementedError(
            "NuisanceModelSpec is handled separately, not by Optuna"
        )


SearchSpace = list[SearchSpaceSpec]
"""Convenience type alias for a list of search space specifications."""
