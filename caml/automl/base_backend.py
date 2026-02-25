"""Protocol and base class for AutoML tuner backends.

CaML delegates CATE model-selection optimization to pluggable backends.
This module defines `~~base_backend.TunerBackend` (structural-typing protocol) and
`~~base_backend.BaseTunerBackend` (ABC for concrete implementations). The default
backend is `~~optuna.OptunaBackend`, and is currently the only supported backend.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol

from typing_extensions import runtime_checkable


@runtime_checkable
class TunerBackend(Protocol):
    """Protocol defining the AutoML tuner backend interface (structural subtyping).

    Any object exposing `~~base_backend.TunerBackend.direction`, `~~base_backend.TunerBackend.optimize()`,
    and `~~base_backend.TunerBackend.create_objective()` satisfies this protocol.
    Runtime-checkable via `isinstance(obj, TunerBackend)`.

    See Also
    --------
    `~~base_backend.BaseTunerBackend` : ABC with shared utilities.

    `~~optuna.OptunaBackend` : Default Optuna-based implementation.
    """

    direction: str
    """Optimization direction (`"minimize"` or `"maximize"`)."""

    def optimize(self, objective: Callable, n_trials: int, n_jobs: int) -> Any:
        """Run the optimization loop."""
        ...

    def create_objective(self, *args, **kwargs) -> Callable:
        """Build the objective function consumed by `~~base_backend.TunerBackend.optimize()`."""
        ...


class BaseTunerBackend(ABC):
    """Abstract base class for AutoML tuner backends.

    Provides the contract that concrete backends (e.g., `~~optuna.OptunaBackend`) must
    fulfill. Subclasses implement `~~base_backend.BaseTunerBackend.optimize()` and `~~base_backend.BaseTunerBackend.create_objective()`.

    Notes
    -----
    **Abstract methods (must be implemented by subclasses):**

    - `~~base_backend.BaseTunerBackend.optimize()` — Execute the optimization loop.
    - `~~base_backend.BaseTunerBackend.create_objective()` — Build the trial-level objective function.

    **Required attributes:**

    - `~~base_backend.BaseTunerBackend.direction` — `"minimize"` or `"maximize"`.

    See Also
    --------
    `~~base_backend.TunerBackend` : Protocol defining the interface.

    `~~optuna.OptunaBackend` : Default Optuna-based implementation.

    Examples
    --------
    ```{python}
    from caml.automl import OptunaBackend, TunerBackend, BaseTunerBackend

    backend = OptunaBackend(direction="minimize")

    # Protocol conformance
    assert isinstance(backend, TunerBackend)
    assert isinstance(backend, BaseTunerBackend)
    print(f"direction: {backend.direction}")
    ```
    """

    direction: str
    """Optimization direction (`"minimize"` or `"maximize"`)."""

    @abstractmethod
    def optimize(
        self, objective: Callable, n_trials: int, n_jobs: int, **kwargs
    ) -> Any:
        """Run the optimization loop (**ABSTRACT**).

        Parameters
        ----------
        objective
            Callable scoring function created by `~~base_backend.BaseTunerBackend.create_objective()`.
        n_trials
            Number of optimization trials.
        n_jobs
            Number of parallel workers.
        **kwargs
            Backend-specific options (e.g., `timeout`).

        Returns
        -------
        Any
            Backend-specific study or result object.
        """
        pass

    @abstractmethod
    def create_objective(self, *args, **kwargs) -> Callable:
        """Build the objective function consumed by `~~base_backend.BaseTunerBackend.optimize()` (**ABSTRACT**).

        Returns
        -------
        Callable
            Callable that accepts a backend-specific trial and returns a score.
        """
        pass
