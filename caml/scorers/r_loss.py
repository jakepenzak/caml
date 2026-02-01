"""R-loss (R-learner based) scorer for CATE model selection.

R-loss [@nie2021quasi] evaluates a CATE estimator using residualized outcome and treatment
signals computed from cross-fitted nuisance models. See the
[Scorer Details](../02_Concepts/scorers.qmd#sec-r-loss) for the full
mathematical derivation and interpretation guide.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator

from caml.data import CausalDataset
from caml.registry import ScorerFamily, auto_register
from caml.samplers import CrossFitter

from ._validation import _validate_cate_array
from .base_scorer import BaseCateScorerMixin


@auto_register(name="RLoss", family=ScorerFamily.PSUEDO_OUTCOME, is_estimator=False)
class RLoss(BaseCateScorerMixin):
    r"""R-loss for CATE model evaluation & selection via orthogonal residualization.

    Parameters
    ----------
    treatment_model
        Model to estimate treatment $m(X) = \mathbb{E} \big[T \mid X,W \big]$.
    outcome_model
        Model to estimate outcome $\ell(X) = \mathbb{E} \big[Y \mid X, W \big]$.
    cv
        Number of cross-fitting folds.
    random_state
        Random state for cross-fitting.
    normalized
        If ``True``, returns an $R^2$-like score in $(-\infty, 1]$.

    Notes
    -----
    Define residuals $\tilde{Y} = Y - m(X)$ and $\tilde{T} = T - \ell(X)$. R-loss is:

    $$
    \mathcal{L}_R(\hat{\tau}) = \mathbb{E}_X\big[(\tilde{Y} - \hat{\tau}(X)\,\tilde{T})^2\big]
    $$

    R-loss satisfies Neyman orthogonality: nuisance estimation errors have only
    second-order effects, enabling quasi-oracle model selection.

    See [Scorer Details](../02_Concepts/scorers.qmd#sec-r-loss) for the full
    derivation, interpretation, and self-serving bias considerations.
    """

    def __init__(
        self,
        treatment_model: BaseEstimator,
        outcome_model: BaseEstimator,
        cv: int = 3,
        random_state: int | None = None,
        normalized: bool = False,
    ):
        self.treatment_model = treatment_model
        self.outcome_model = outcome_model
        self.cv = cv
        self.random_state = random_state
        self.normalized = normalized
        self._cross_fitter = CrossFitter(cv=cv, random_state=random_state)

    def __call__(self, estimator, data: CausalDataset) -> float:
        r"""Compute R-loss metric.

        Parameters
        ----------
        estimator
            Fitted CATE estimator implementing ``effect(X)``.
        data
            Causal dataset.

        Returns
        -------
        float
            R-loss or, if ``normalized=True``, an $R^2$-like score in $(-\infty, 1]$.

        Examples
        --------
        ```{python}
        from sklearn.linear_model import LinearRegression, LogisticRegression
        import numpy as np

        from caml.estimators.dml import WrappedLinearDML
        from caml.data import CausalDataset, OutcomeType, TreatmentType
        from caml.extensions.synthetic_data import SyntheticDataGenerator
        from caml.scorers import RLoss

        gen = SyntheticDataGenerator(n_cont_modifiers=3, seed=10)
        df = gen.df
        true_cates = np.array(gen.cates)

        data = CausalDataset.from_dataframe(
            df=df,
            X=["X1_continuous", "X2_continuous", "X3_continuous"],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
            true_cates=true_cates,
        )

        estimator = WrappedLinearDML(model_y=LinearRegression(), model_t=LogisticRegression(), cv=3)
        estimator.fit(data)

        scorer = RLoss(treatment_model=LogisticRegression(), outcome_model=LinearRegression())
        print(f"R-loss: {scorer(estimator, data):.2f}")

        nrm_scorer = RLoss(treatment_model=LogisticRegression(), outcome_model=LinearRegression(), normalized=True)
        print(f"Normalized R-loss: {nrm_scorer(estimator, data):.2f}")
        ```
        """
        # Step 1: Get out-of-fold nuisance predictions
        m_hat, l_hat = self._cross_fitter.fit_predict_nuisances_dml(
            data=data,
            outcome_model=self.outcome_model,
            treatment_model=self.treatment_model,
        )

        # Step 2: Compute residuals
        Y_res = data.Y - m_hat  # Outcome residual
        T_res = data.T - l_hat  # Treatment residual

        # Step 3: Predict CATE for data and validate shape
        tau_hat = estimator.effect(data.X)
        n_samples = len(data.Y)
        tau_hat = _validate_cate_array(tau_hat, n_samples, "CATE predictions (tau_hat)")

        # Flatten residuals to 1D for consistent computation
        Y_res = _validate_cate_array(Y_res, n_samples, "outcome residuals (Y_res)")
        T_res = _validate_cate_array(T_res, n_samples, "treatment residuals (T_res)")

        # Step 4: Compute R-loss
        squared_error = (Y_res - tau_hat * T_res) ** 2
        r_loss = np.mean(squared_error)

        # Optionally, normalize for interpretability; [-inf, 0] = bad, [0, 1] = good
        if self.normalized:
            baseline_loss = sm.OLS(Y_res, T_res).fit().mse_resid
            r_loss = 1 - r_loss / baseline_loss
        return float(r_loss)
