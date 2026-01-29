"""Base scorer class."""

from abc import ABC, abstractmethod

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
