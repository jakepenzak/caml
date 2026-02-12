"""Shared base functionality, protocols, and interfaces for CATE scorers.

CaML scorers evaluate fitted CATE estimators (minimum requirement is estimators implement ``effect(X)``)
on a `CausalDataset`. They are primarily intended for model selection (e.g., Optuna),
where scores are compared across candidate estimators. These scores can also be used
for general evaluation outside of CaML's tuning framework.

Most causal scores depend on nuisance quantities (e.g., propensity scores,
outcome regressions). In CaML these are computed out-of-fold using
`CrossFitter`.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from caml.data import CausalDataset, OutcomeType, TreatmentType


@dataclass(frozen=True)
class ScorerCapabilities:
    r"""Metadata describing the capabilities and requirements of a CATE scorer.

    Defines what types of datasets a scorer can handle and what nuisance models it requires.
    Enables automatic compatibility checking and scorer filtering in AutoML pipelines.

    Parameters
    ----------
    treatment_types
        Treatment variable types the estimator supports (e.g., ``{TreatmentType.BINARY}``).
    outcome_types
        Outcome variable types the estimator supports (e.g., ``{OutcomeType.CONTINUOUS}``).
    requires_treatment_model
        If True, estimator needs a treatment model - $\mathbb{E}[T \mid X,W]$.
    requires_outcome_model
        If True, estimator needs an outcome model - $\mathbb{E}[Y \mid X,W]$.
    requires_regression_model
        If True, estimator needs a regression model - $\mathbb{E}[Y \mid T,X,W]$.
    requires_oracle_cates
        Whether the scorer requires oracle CATEs to be available.

    Examples
    --------
    ```{python}
    from caml.data import TreatmentType, OutcomeType
    from caml.scorers import ScorerCapabilities

    capabilities = ScorerCapabilities(
        treatment_types={TreatmentType.BINARY},
        outcome_types={OutcomeType.CONTINUOUS},
        requires_treatment_model=True,
        requires_outcome_model=True,
        requires_regression_model=False,
        requires_oracle_cates=False
    )
    ```
    """

    treatment_types: set[TreatmentType]
    outcome_types: set[OutcomeType]
    requires_treatment_model: bool
    requires_outcome_model: bool
    requires_regression_model: bool
    requires_oracle_cates: bool = False
    supports_weights: bool = (
        False  # Not Supported Yet - for future use with weighted scores like R-loss
    )

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
            True if scorer supports the dataset's treatment and outcome types.

        Examples
        --------
        ```{python}
        from caml.data import TreatmentType, OutcomeType, Estimand, CausalDataset
        from caml.scorers import ScorerCapabilities
        import numpy as np

        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            requires_oracle_cates=False
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS
        )

        print(capabilities.is_compatible(data))
        ```
        """
        return (
            data.treatment_type in self.treatment_types
            and data.outcome_type in self.outcome_types
        )


@runtime_checkable
class CateScorer(Protocol):
    """Core protocol defining the inferface for CATE scorers.

    All CATE scorers in CaML must implement this protocol.

    Notes
    -----
    - This is a Protocol (structural subtyping), not a base class
    - Runtime-checkable via ``isinstance(obj, CateScorer)``
    - ``capabilities`` is class attributes, not properties

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.scorers import CateScorer, ScorerCapabilities, BaseCateScorerMixin
    from caml.data import TreatmentType, OutcomeType

    class NegMAEOnOracleCATE(BaseCateScorerMixin):

        capabilities: ScorerCapabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            requires_oracle_cates=True
        )

        def __call__(self, estimator, data):
            tau_hat = estimator.effect(data.X)
            mae = np.mean(np.abs(tau_hat - data.true_cates))
            return -mae

    assert isinstance(NegMAEOnOracleCATE(), CateScorer)
    ```
    """

    capabilities: ScorerCapabilities

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check if scorer can handle the given dataset (class method).

        This is a class method, so you can check compatibility without
        instantiating the estimator. Useful for filtering candidate
        estimators in AutoML workflows.

        Parameters
        ----------
        data
            Dataset to check compatibility with.

        Returns
        -------
        bool
            True if scorer supports the dataset's treatment and outcome types.
        """
        ...

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
        ...


class BaseCateScorerMixin(ABC):
    """Base class and mixin for `CateScorer`.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.scorers import BaseCateScorerMixin, ScorerCapabilities
    from caml.data import TreatmentType, OutcomeType

    class NegMAEOnOracleCATE(BaseCateScorerMixin):

        capabilities: ScorerCapabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            requires_oracle_cates=False
        )

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
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check if scorer can handle the given dataset (class method).

        This is a class method, so you can check compatibility without
        instantiating the estimator. Useful for filtering candidate
        estimators in AutoML workflows.

        Parameters
        ----------
        data
            Dataset to check compatibility with.

        Returns
        -------
        bool
            True if scorer supports the dataset's treatment and outcome types.

        Examples
        --------
        ```{python}
        from caml.scorers import DRLoss
        from caml.data import CausalDataset, TreatmentType, OutcomeType
        from caml.extensions.synthetic_data import SyntheticDataGenerator

        # Generate data
        gen = SyntheticDataGenerator(seed=42)
        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS
        )

        # Check compatibility WITHOUT instantiating
        if DRLoss.is_compatible_with(data):
            print("DRLoss can handle this data!")
        ```

        ```{python}
        # Check multiple estimators efficiently
        from caml.scorers import (
            DRLoss,
            RLoss
        )

        candidates = [DRLoss, RLoss]
        compatible = [
            scr_class for scr_class in candidates
            if scr_class.is_compatible_with(data)
        ]
        print(f"Compatible scorers: {[c.__name__ for c in compatible]}")
        ```
        """
        return cls.capabilities.is_compatible(data)

    def __init_subclass__(cls, **kwargs) -> None:
        """Strictly enforce that subclasses define required class attributes (capabilities)."""
        super().__init_subclass__(**kwargs)
        if "capabilities" not in cls.__dict__ and not inspect.isabstract(cls):
            raise TypeError(f"{cls.__name__} must define capabilities")
