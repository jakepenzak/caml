import numpy as np

from caml.data import CausalDataset
from caml.samplers import CrossFitter
from caml.scorers.base_scorer import BaseScorer, clip


## Add supported outcome and treatment types!
class DRLoss(BaseScorer):
    """R-learner loss for CATE model selection."""

    def __init__(
        self,
        treatment_model,
        regression_model,
        cv: int = 3,
        random_state: int | None = None,
        normalized: bool = False,
    ):
        self.treatment_model = treatment_model
        self.regression_model = regression_model
        self.cv = cv
        self.random_state = random_state
        self.normalized = normalized
        self._cross_fitter = CrossFitter(cv=cv, random_state=random_state)

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Compute out-of-fold DR-loss."""
        # Get out-of-fold nuisance predictions
        mu_0, mu_1, e_hat = self._cross_fitter.fit_predict_nuisances_dr(
            data=data,
            regression_model=self.regression_model,
            treatment_model=self.treatment_model,
        )

        # Compute DR pseudo-outcome
        dr = mu_1 + ((data.Y - mu_1) / clip(e_hat)) * data.T
        dr -= mu_0 + ((data.Y - mu_0) / clip(1 - e_hat)) * (1 - data.T)

        # Get estimator CATE predictions tau_hat
        tau_hat = estimator.effect(data.X)

        # Compute DR-Loss
        dr_loss = np.mean((dr - tau_hat) ** 2)

        # Optionally, normalize for interpretability; [-inf, 0] = bad, [0, 1] = good
        if self.normalized:
            baseline_loss = np.mean((dr - np.mean(dr)) ** 2)
            dr_loss = 1 - dr_loss / baseline_loss
        return dr_loss
