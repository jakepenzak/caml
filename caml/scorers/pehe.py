"""PEHE (Precision in Estimation of Heterogeneous Effects) oracle metric.

PEHE [@hill2011bayesian] requires true CATE values (e.g., from simulation) and scores an estimator
by mean squared error against ground truth. See the
[Scorer Details](../02_Concepts/scorers.qmd#sec-pehe) for background and
comparison with proxy metrics.
"""

import numpy as np

from caml.data.data_enums import OutcomeType, TreatmentType
from caml.data.dataset import CausalDataset
from caml.registry.registry import auto_register
from caml.registry.registry_enums import ScorerFamily

from ._validation import _validate_scorer_inputs
from .base_scorer import BaseCateScorerMixin, ScorerCapabilities


@auto_register(name="Pehe", family=ScorerFamily.ORACLE, is_estimator=False)
class Pehe(BaseCateScorerMixin):
    r"""Precision in Estimation of Heterogeneous Effects (PEHE) oracle metric.

    Parameters
    ----------
    true_cates
        True CATEs for scoring. If `None`, uses ``data.true_cates`` from
        `~~dataset.CausalDataset`.
    normalized
        If `True`, returns an $R^2$-like score in $(-\infty, 1]$.

    Notes
    -----
    PEHE is the gold standard for CATE evaluation, requiring ground truth $\tau(X)$:

    $$
    \text{PEHE}(\hat{\tau}) = \mathbb{E}_X\left[(\tau(X) - \hat{\tau}(X))^2\right]
    $$

    PEHE is only computable when both potential outcomes are observed (e.g., in
    simulations). For real-world data, use proxy metrics like `~~q_stat.QStat`,
    `~~r_loss.RLoss`, or `~~dr_loss.DRLoss`.

    See [Scorer Details](../02_Concepts/scorers.qmd#sec-pehe) for relationship to
    proxy metrics and interpretation guide.
    """

    capabilities = ScorerCapabilities(
        treatment_types={TreatmentType.BINARY, TreatmentType.CONTINUOUS},
        outcome_types={OutcomeType.CONTINUOUS, OutcomeType.BINARY},
        requires_treatment_model=False,
        requires_outcome_model=False,
        requires_regression_model=False,
        requires_oracle_cates=True,
        greater_is_better=False,
        supports_weights=False,
    )

    def __init__(self, true_cates: np.ndarray | None = None, normalized: bool = False):
        self.true_cates = true_cates
        self.normalized = normalized

    def __call__(self, estimator, data: CausalDataset) -> float:
        r"""Compute PEHE oracle metric.

        Parameters
        ----------
        estimator
            Fitted CATE estimator implementing
            `~~base_estimator.AutoCateEstimator.effect()`.
        data
            `~~dataset.CausalDataset`, with true CATEs available via
            ``data.true_cates`` if not instantiated with ``true_cates``.

        Returns
        -------
        float
            PEHE or, if `normalized=True`, an $R^2$-like score in $(-\infty, 1]$.

        Raises
        ------
        ValueError
            If true CATEs are not provided.

        Examples
        --------
        ```{python}
        from sklearn.linear_model import LinearRegression, LogisticRegression
        import numpy as np

        from caml.estimators.dml import WrappedLinearDML
        from caml.data import CausalDataset, OutcomeType, TreatmentType
        from caml.utilities.synthetic_data import SyntheticDataGenerator
        from caml.scorers import Pehe

        gen = SyntheticDataGenerator(n_cont_modifiers=3, n_cont_confounders=3, seed=10)
        df = gen.df
        true_cates = np.array(gen.cates)

        data = CausalDataset.from_dataframe(
            df=df,
            X=["X1_continuous", "X2_continuous"],
            W=["W1_continuous", "W2_continuous", "W3_continuous"],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
            true_cates=true_cates,
        )

        estimator = WrappedLinearDML(model_y=LinearRegression(), model_t=LogisticRegression(), cv=3)
        estimator.fit(data)

        scorer = Pehe()
        print(f"PEHE: {scorer(estimator, data):.2f}")

        nrm_scorer = Pehe(normalized=True)
        print(f"Normalized PEHE: {nrm_scorer(estimator, data):.2f}")
        ```
        """
        tau_hat = estimator.effect(data.X)

        if self.true_cates is not None:
            true_cates = self.true_cates
        else:
            true_cates = data.true_cates

        if true_cates is None:
            raise ValueError(
                "Pehe requires true CATEs. Provide `true_cates=` to the scorer or set "
                "`data.true_cates`."
            )

        # Validate and align shapes
        tau_hat, true_cates = _validate_scorer_inputs(
            tau_hat, true_cates, "CATE predictions (tau_hat)", "true CATEs"
        )

        pehe = np.mean((true_cates - tau_hat) ** 2)

        # Optionally, normalize for interpretability; [-inf, 0] = bad, [0, 1] = good
        if self.normalized:
            baseline_loss = np.mean((true_cates - np.mean(tau_hat)) ** 2)
            pehe = 1 - pehe / baseline_loss

        return float(pehe)
