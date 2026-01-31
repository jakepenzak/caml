"""Base scorer class."""

from abc import ABC, abstractmethod

import numpy as np

from caml.data.dataset import CausalDataset


class BaseScorer(ABC):
    """Base class for CATE scorers."""

    @abstractmethod
    def __call__(self, estimator, data: CausalDataset) -> float:
        """Score the estimator on data.

        Parameters
        ----------
        estimator : CATEEstimator
            Estimator to score
        data : CausalDataset
            Data to score on

        Returns
        -------
        float
            Score (higher is better for Optuna)
        """
        pass


def clip(arr: np.ndarray, lb: float = 0.01, ub: float = np.inf) -> np.ndarray:
    """Clip numpy array between lb and ub.

    Used for trimming propensity scores, when used in inverse propensity scores (e.g., IPW, DR, etc.)

    Parameters
    ----------
    arr
        Array to clip
    lb
        Lower bound
    ub
        Upper bound

    Returns
    -------
    np.ndarray
        Clipped array
    """
    return np.clip(arr, lb, np.inf)
