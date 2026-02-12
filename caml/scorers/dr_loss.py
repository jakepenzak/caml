"""Doubly-Robust loss (DR-Loss) scorer for CATE model selection.

DR-loss [@kennedy2023towards] uses a doubly-robust pseudo-outcome and scores a CATE estimator by mean
squared error against that pseudo-outcome. See the
[Scorer Details](../02_Concepts/scorers.qmd#sec-dr-loss) for the full
mathematical derivation and interpretation guide.
"""

import numpy as np

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.registry import ScorerFamily, auto_register
from caml.samplers import CrossFitter

from ._validation import _clip, _validate_scorer_inputs
from .base_scorer import BaseCateScorerMixin, ScorerCapabilities


@auto_register(name="DRLoss", family=ScorerFamily.PSEUDO_OUTCOME, is_estimator=False)
class DRLoss(BaseCateScorerMixin):
    r"""Doubly-robust loss for CATE model selection.

    Parameters
    ----------
    treatment_model
        Model to estimate propensity $e(X) = P(T=1 \mid X)$.
    regression_model
        Model to estimate outcome regressions $\mu_t(X) = \mathbb{E}[Y \mid X,T=t]$.
    cv
        Number of cross-fitting folds.
    random_state
        Random state for cross-fitting.
    normalized
        If ``True``, returns an $R^2$-like score in $(-\infty, 1]$.

    Notes
    -----
    Define the DR pseudo-outcome:

    $$
    \mathcal{Y}^{\mathrm{DR}} = \mu_1(X) - \mu_0(X)
        + \frac{T}{e(X)}(Y-\mu_1(X)) - \frac{1-T}{1-e(X)}(Y-\mu_0(X))
    $$

    DR-loss is calculated as the squared error against this pseudo-outcome:

    $$
    \mathcal{L}_{\mathrm{DR}}(\hat{\tau}) = \mathbb{E}_X\big[(\mathcal{Y}^{\mathrm{DR}}-\hat{\tau}(X))^2\big]
    $$

    DR-loss is consistent if either the propensity or outcome models are correct.

    See [Scorer Details](../02_Concepts/scorers.qmd#sec-dr-loss) for the full
    derivation, double robustness property, and interpretation guide.
    """

    capabilities = ScorerCapabilities(
        treatment_types={TreatmentType.BINARY},
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        requires_treatment_model=True,
        requires_outcome_model=False,
        requires_regression_model=True,
        requires_oracle_cates=False,
        higher_is_better=False,
        supports_weights=False,
    )

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
        r"""Compute out-of-fold DR-loss.

        Parameters
        ----------
        estimator
            Fitted CATE estimator implementing ``effect(X)``.
        data
            Causal dataset.

        Returns
        -------
        float
            DR-loss or, if ``normalized=True``, an $R^2$-like score in $(-\infty, 1]$.

        Examples
        --------
        ```{python}
        from sklearn.linear_model import LinearRegression, LogisticRegression
        import numpy as np

        from caml.estimators.dml import WrappedLinearDML
        from caml.data import CausalDataset, OutcomeType, TreatmentType
        from caml.extensions.synthetic_data import SyntheticDataGenerator
        from caml.scorers import DRLoss

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

        scorer = DRLoss(treatment_model=LogisticRegression(), regression_model=LinearRegression())
        print(f"DR-loss: {scorer(estimator, data):.2f}")

        nrm_scorer = DRLoss(treatment_model=LogisticRegression(), regression_model=LinearRegression(), normalized=True)
        print(f"Normalized DR-loss: {nrm_scorer(estimator, data):.2f}")
        ```
        """
        # Get out-of-fold nuisance predictions
        mu_0, mu_1, e_hat = self._cross_fitter.fit_predict_nuisances_dr(
            data=data,
            regression_model=self.regression_model,
            treatment_model=self.treatment_model,
        )

        # Compute DR pseudo-outcome
        dr = mu_1 + ((data.Y - mu_1) / _clip(e_hat)) * data.T
        dr -= mu_0 + ((data.Y - mu_0) / _clip(1 - e_hat)) * (1 - data.T)

        # Get estimator CATE predictions tau_hat and validate shapes
        tau_hat = estimator.effect(data.X)
        tau_hat, dr = _validate_scorer_inputs(
            tau_hat, dr, "CATE predictions (tau_hat)", "DR pseudo-outcome"
        )

        # Compute DR-Loss
        dr_loss = np.mean((dr - tau_hat) ** 2)

        # Optionally, normalize for interpretability; [-inf, 0] = bad, [0, 1] = good
        if self.normalized:
            baseline_loss = np.mean((dr - np.mean(dr)) ** 2)
            dr_loss = 1 - dr_loss / baseline_loss
        return float(dr_loss)
