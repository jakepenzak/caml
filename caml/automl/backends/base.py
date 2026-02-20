"""Base tuner backend protocol."""

from abc import ABC, abstractmethod
from typing import Protocol


class TunerBackend(Protocol):
    """Protocol for AutoML backends."""

    def optimize(self, objective, n_trials: int, **kwargs):
        """Run optimization."""
        ...


class BaseTunerBackend(ABC):
    """Base class for AutoML backends."""

    @abstractmethod
    def optimize(self, objective, n_trials: int, **kwargs):
        """Run optimization."""
        pass
