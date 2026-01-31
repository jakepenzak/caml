import numpy as np

from caml.data import CausalDataset
from caml.samplers import CrossFitter
from caml.scorers.base_scorer import BaseScorer, clip


## Add supported outcome and treatment types if needed
class QLoss(BaseScorer):
    r"""$\hat{Q}$ loss for CATE model selection."""

    def __init__(
        self,
        treatment_model,
        cv: int = 3,
        random_state: int | None = None,
    ):
        self.treatment_model = treatment_model
        self.cv = cv
        self.random_state = random_state
        self._cross_fitter = CrossFitter(cv=cv, random_state=random_state)

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Compute out-of-fold Q-Loss"""
        e_hat = self._cross_fitter.fit_predict_treatment_model(
            data=data,
            treatment_model=self.treatment_model,
        )

        ipw = (data.T * data.Y) / clip(e_hat)
        ipw -= ((1 - data.T) * data.Y) / clip(1 - e_hat)
        tau_hat = estimator.effect(data.X)

        Q_loss = np.mean(tau_hat**2 - 2 * tau_hat * ipw)

        return Q_loss
