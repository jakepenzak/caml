"""Base tuner backend protocol."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from typing_extensions import runtime_checkable


@runtime_checkable
class TunerBackend(Protocol):
    """Protocol for AutoML backends."""

    direction: str

    def optimize(self, objective, n_trials: int, n_jobs: int) -> Any:
        """Run optimization."""
        ...

    def create_objective(self, *args, **kwargs) -> Any:
        """Create objective function."""
        ...


class BaseTunerBackend(ABC):
    """Base class for AutoML backends."""

    direction: str

    @abstractmethod
    def optimize(self, objective, n_trials: int, n_jobs: int, **kwargs):
        """Run optimization."""
        pass

    @abstractmethod
    def create_objective(self, *args, **kwargs):
        """Create objective function."""
        pass
