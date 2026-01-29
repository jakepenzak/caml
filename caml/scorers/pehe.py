import numpy as np

from caml.data import CausalDataset
from caml.scorers.base_scorer import BaseScorer


## Add supported outcome and treatment types if needed
class PEHE(BaseScorer):
    """Precision in Estimation of Heterogeneous Effect (PEHE) metric.

    Requires true CATEs for computation. Useful for simulation and benchmarking.
    """

    def __init__(self, true_cates: np.ndarray | None = None):
        self.true_cates = true_cates

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Compute PEHE metric."""
        tau_hat = estimator.effect(data.X)

        if self.true_cates is not None:
            true_cates = self.true_cates
        else:
            true_cates = data.true_cates

        squared_error = (tau_hat - true_cates) ** 2
        pehe = np.mean(squared_error)

        return pehe
