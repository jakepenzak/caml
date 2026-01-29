import numpy as np
import statsmodels.api as sm

from caml.data import CausalDataset
from caml.samplers import CrossFitter
from caml.scorers.base_scorer import BaseScorer


## Add supported outcome and treatment types if needed
class RLoss(BaseScorer):
    """R-learner loss for CATE model selection."""

    def __init__(
        self,
        treatment_model,
        outcome_model,
        cv: int = 3,
        random_state: int | None = None,
        normalized: bool = False,
    ):
        self.treatment_model = treatment_model
        self.outcome_model = outcome_model
        self.cv = cv
        self.random_state = random_state
        self.normalized = normalized  # Normalize R-loss to [0, 1] range, relative to constant effect model
        self._cross_fitter = CrossFitter(cv=cv, random_state=random_state)

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Compute out-of-fold R-loss."""
        # Step 1: Get out-of-fold nuisance predictions
        m_hat, e_hat = self._cross_fitter.fit_predict_nuisances(
            data=data,
            outcome_model=self.outcome_model,
            treatment_model=self.treatment_model,
        )

        # Step 2: Compute residuals
        Y_res = data.Y - m_hat  # Outcome residual
        T_res = data.T - e_hat  # Treatment residual

        # Step 3: Predict CATE for data
        tau_hat = estimator.effect(data.X)

        # Step 4: Compute R-loss
        squared_error = (Y_res - tau_hat * T_res) ** 2
        r_loss = np.mean(squared_error)

        if self.normalized:
            baseline_loss = sm.OLS(Y_res, T_res).fit().mse_resid
            r_loss = 1 - r_loss / baseline_loss
        return r_loss
