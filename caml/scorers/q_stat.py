"""Q-statistic scorer for CATE model selection.

Q-statistic [@yu2024qstatistic] uses only a propensity model to build an IPW pseudo-outcome and
scores a CATE estimator by its MSE ranking (lower is better). See the
[Scorer Details](../02_Concepts/scorers.qmd#sec-q-statistic) for the full
mathematical derivation and interpretation guide.
"""

import numpy as np

from caml.data import CausalDataset
from caml.registry import ScorerFamily, auto_register
from caml.samplers import CrossFitter

from ._validation import _clip, _validate_cate_array
from .base_scorer import BaseCateScorerMixin


@auto_register(
    name="Q-Statistic", family=ScorerFamily.RANKING_RELATIVE_PROXY, is_estimator=False
)
class QStat(BaseCateScorerMixin):
    r"""Q-statistic for CATE model selection via IPW pseudo-outcomes.

    Parameters
    ----------
    treatment_model
        Model to estimate propensity scores $e(X) = P(T=1 \mid X)$.
    cv
        Number of cross-fitting folds.
    random_state
        Random state for cross-fitting.

    Notes
    -----
    Let $e(X) = P(T=1 \mid X)$ be the propensity score. Define the IPW pseudo-outcome:

    $$
    \Gamma(x,t,y) = \frac{t \cdot y}{e(x)} - \frac{(1-t) \cdot y}{1-e(x)}
    $$

    The Q-statistic is computed as:

    $$
    \hat{Q}(\hat{\tau}) = \frac{1}{N} \sum_{n=1}^{N}
        \left[ \hat{\tau}^2(x_n) - 2\hat{\tau}(x_n) \cdot \Gamma(x_n, t_n, y_n) \right]
    $$

    $\hat{Q}$ equals PEHE minus a constant, so ranking by $\hat{Q}$ is equivalent to
    ranking by MSE. A score $\hat{Q} \geq 0$ indicates degeneracy (worse than zero-effect).

    See [Scorer Details](../docs/02_Concepts/scorers.qmd#sec-q-statistic) for the full
    derivation, interpretation, and degeneracy indicators.
    """

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
        r"""Compute out-of-fold Q-statistic.

        Parameters
        ----------
        estimator
            Fitted CATE estimator implementing ``effect(X)``.
        data
            Causal dataset.

        Returns
        -------
        float
            Q-statistic (lower is better). $\hat{Q} \geq 0$ indicates degeneracy.

        Examples
        --------
        ```{python}
        from sklearn.linear_model import LinearRegression, LogisticRegression
        import numpy as np

        from caml.estimators.dml import WrappedLinearDML
        from caml.data import CausalDataset, OutcomeType, TreatmentType
        from caml.extensions.synthetic_data import SyntheticDataGenerator
        from caml.scorers import QStat

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

        scorer = QStat(treatment_model=LogisticRegression())
        print(f"Q-statistic: {scorer(estimator, data):.2f}")
        ```
        """
        e_hat = self._cross_fitter.fit_predict_treatment_model(
            data=data,
            treatment_model=self.treatment_model,
        )

        # Compute IPW pseudo-outcome
        ipw = (data.T * data.Y) / _clip(e_hat)
        ipw -= ((1 - data.T) * data.Y) / _clip(1 - e_hat)

        # Get CATE predictions and validate shape
        tau_hat = estimator.effect(data.X)
        n_samples = len(data.Y)
        tau_hat = _validate_cate_array(tau_hat, n_samples, "CATE predictions (tau_hat)")

        # Flatten IPW to 1D for consistent computation
        ipw = _validate_cate_array(ipw, n_samples, "IPW pseudo-outcome")

        q_stat = np.mean(tau_hat**2 - 2 * tau_hat * ipw)

        return float(q_stat)
