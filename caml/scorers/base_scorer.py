"""Shared base functionality, protocols, and interfaces for CATE scorers.

CaML scorers evaluate fitted CATE estimators (minimum requirement is estimators implement ``effect(X)``)
on a `CausalDataset`. They are primarily intended for model selection (e.g., Optuna),
where scores are compared across candidate estimators. These scores can also be used
for general evaluation outside of CaML's tuning framework.

Most causal scores depend on nuisance quantities (e.g., propensity scores,
outcome regressions). In CaML these are computed out-of-fold using
`CrossFitter`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from caml.data import CausalDataset, OutcomeType, TreatmentType


@dataclass(frozen=True)
class ScorerCapabilities:
    r""""""

    treatment_types: set[TreatmentType]
    outcome_types: set[OutcomeType]
    supports_weights: bool
    requires_treatment_model: bool
    requires_outcome_model: bool
    requires_regression_model: bool
    requires_oracle_cates: bool = False

    # TODO: Refine compatibility logic!!
    def is_compatible(self, data: CausalDataset) -> bool:
        """Check if scorer can handle the given dataset.

        Parameters
        ----------
        data
            Dataset to check compatibility with.

        Returns
        -------
        bool
            True if estimator supports the dataset's treatment and outcome types.

        Examples
        --------
        ```{python}
        from caml.data import TreatmentType, OutcomeType, Estimand, CausalDataset
        from caml.estimators import EstimatorCapabilities
        import numpy as np

        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=True,
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            supports_inference=False
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS
        )

        print(capabilities.is_compatible(data))  # True
        ```
        """
        return (
            data.treatment_type in self.treatment_types
            and data.outcome_type in self.outcome_types
        )


@runtime_checkable
class CateScorer(Protocol):
    capabilities: ScorerCapabilities

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool: ...

    def __call__(self, estimator, data: CausalDataset) -> float: ...


class BaseCateScorerMixin(ABC):
    """Base class for CATE scorers.

    Notes
    -----
    Some scorers naturally return a *loss* (lower is better). If using a
    maximization-based tuner, negate the loss or use a normalized score.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.scorers import BaseCateScorerMixin

    class NegMAEOnOracleCATE(BaseCateScorerMixin):
        def __call__(self, estimator, data):
            tau_hat = estimator.effect(data.X)
            mae = np.mean(np.abs(tau_hat - data.true_cates))
            return -mae
    ```
    """

    capabilities: ScorerCapabilities

    @abstractmethod
    def __call__(self, estimator, data: CausalDataset) -> float:
        """Score the estimator on data.

        Parameters
        ----------
        estimator
            Fitted CATE estimator implementing ``effect(X)``.
        data
            Causal Dataset to score on

        Returns
        -------
        float
            Loss or score
        """

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool: ...
