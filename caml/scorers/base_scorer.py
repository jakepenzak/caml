"""Shared base functionality, protocols, and interfaces for CATE scorers.

CaML scorers evaluate fitted CATE estimators implementing
`~~base_estimator.AutoCateEstimator.effect()` on a
`~~dataset.CausalDataset`. They are primarily intended for model selection
and can also be used for general evaluation outside of CaML's tuning
framework.

Most causal scores depend on nuisance quantities (e.g., propensity scores,
outcome regressions). In CaML these are computed out-of-fold using
`~~cross_fit.CrossFitter`.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from caml.data.data_enums import OutcomeType, TreatmentType
from caml.data.dataset import CausalDataset


@dataclass(frozen=True)
class ScorerCapabilities:
    r"""Metadata describing the capabilities and requirements of a CATE scorer.

    Defines what types of datasets a scorer can handle and what nuisance models it requires.
    Enables automatic compatibility checking and scorer filtering in AutoML pipelines.

    Parameters
    ----------
    treatment_types
        Treatment variable types the scorer supports, represented by members of
        `~~data_enums.TreatmentType`.
    outcome_types
        Outcome variable types the scorer supports, represented by members of
        `~~data_enums.OutcomeType`.
    requires_treatment_model
        If True, scorer needs a treatment model - $\mathbb{E}[T \mid X,W]$.
    requires_outcome_model
        If True, scorer needs an outcome model - $\mathbb{E}[Y \mid X,W]$.
    requires_regression_model
        If True, scorer needs a regression model - $\mathbb{E}[Y \mid T,X,W]$.
    requires_oracle_cates
        If True, scorer requires oracle CATEs to be available (for simulation studies).
    greater_is_better
        If True, higher scores indicate better performance (e.g., R^2). If False, lower scores are better (e.g., MSE).
    supports_weights
        If True, scorer handles sample weights (not yet supported in CaML).

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
        requires_oracle_cates=False,
        greater_is_better=False,
        supports_weights=False
    )
    ```
    """

    treatment_types: set[TreatmentType]
    outcome_types: set[OutcomeType]
    requires_treatment_model: bool
    requires_outcome_model: bool
    requires_regression_model: bool
    requires_oracle_cates: bool = False
    greater_is_better: bool = False
    supports_weights: bool = False

    def is_compatible(self, data: CausalDataset) -> bool:
        """Check if scorer can handle the given dataset.

        Parameters
        ----------
        data
            `~~dataset.CausalDataset` to check compatibility with.

        Returns
        -------
        bool
            True if scorer supports the dataset's treatment and outcome types.

        Examples
        --------
        ```{python}
        from caml.data import TreatmentType, OutcomeType, CausalDataset
        from caml.scorers import ScorerCapabilities
        import numpy as np

        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            requires_oracle_cates=False,
            greater_is_better=False,
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS
        )

        capabilities.is_compatible(data)
        ```
        """
        return (
            data.treatment_type in self.treatment_types
            and data.outcome_type in self.outcome_types
        )


@runtime_checkable
class CateScorer(Protocol):
    """Protocol defining the core CATE scorer interface (structural subtyping).

    Specifies the minimal interface all CATE scorers must implement. Scorers evaluate
    fitted CATE estimators on a dataset and return a score. Runtime-checkable via
    `isinstance` against `~~base_scorer.CateScorer`.

    Notes
    -----
    This is a Protocol using structural typing - any class implementing these methods
    will satisfy this interface. For a base implementation with compatibility checking
    and validation utilities, see `~~base_scorer.BaseCateScorerMixin`.

    The `~~base_scorer.CateScorer.capabilities` attribute must be a class
    attribute, not an instance attribute.

    Scorers are callable objects that take a fitted estimator and dataset, returning
    a score.

    See Also
    --------
    `~~base_scorer.BaseCateScorerMixin` : ABC base class with concrete implementations.

    `~~base_scorer.ScorerCapabilities` : Metadata for scorer features.
    """

    capabilities: ScorerCapabilities
    """Class attribute defining the scorer's capabilities and requirements. Must be defined by all implementations."""

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check scorer-data compatibility (class method)."""
        ...

    def __call__(self, estimator, data: CausalDataset) -> float:
        """Score the estimator on data."""
        ...


class BaseCateScorerMixin(ABC):
    """Abstract base class for `~~base_scorer.CateScorer` with validation and utilities.

    Provides concrete implementation of compatibility checking. Subclasses must
    implement `~~base_scorer.BaseCateScorerMixin.__call__()` to define the
    scoring logic.

    This class serves as the recommended base for all CATE scorers in CaML,
    providing a consistent interface and common utilities.

    Notes
    -----
    **Abstract Methods (must be implemented by subclasses):**

    - `~~base_scorer.BaseCateScorerMixin.__init__()` - Initialize scorer with any required nuisance models or parameters.
    - `~~base_scorer.BaseCateScorerMixin.__call__()` - Compute score for estimator on dataset

    **Concrete Methods (provided by this base class):**

    - `~~base_scorer.BaseCateScorerMixin.is_compatible_with()` - Class method for compatibility checking

    **Required Class Attributes:**

    - `~~base_scorer.BaseCateScorerMixin.capabilities` - `~~base_scorer.ScorerCapabilities` instance defining supported features

    Subclasses must define `~~base_scorer.BaseCateScorerMixin.capabilities` as
    a class attribute. Failure to do so will raise a `TypeError` on class
    definition (enforced by `__init_subclass__`).

    See Also
    --------
    `~~base_scorer.CateScorer` : Protocol defining the interface.

    `~~base_scorer.ScorerCapabilities` : Metadata for scorer features.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.scorers import BaseCateScorerMixin, ScorerCapabilities
    from caml.data import TreatmentType, OutcomeType

    class MAEOnOracleCATE(BaseCateScorerMixin):
        # Required class attribute
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            requires_oracle_cates=True,
            greater_is_better=False,
        )

        def __init__(self, true_cates: np.ndarray | None = None):
            self.true_cates = true_cates

        def __call__(self, estimator, data):

            if self.true_cates is None:
                true_cates = data.true_cates
            else:
                true_cates = self.true_cates
            # Abstract method implementation
            tau_hat = estimator.effect(data.X)
            mae = np.mean(np.abs(tau_hat - true_cates))
            return mae

    # Verify protocol conformance
    from caml.scorers import CateScorer
    scorer = MAEOnOracleCATE()
    assert isinstance(scorer, CateScorer)
    assert isinstance(scorer, BaseCateScorerMixin)
    ```

    ```{python}
    # Use is_compatible_with before instantiation
    from caml.utilities.synthetic_data import SyntheticDataGenerator
    from caml.data import CausalDataset

    gen = SyntheticDataGenerator(seed=42)
    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
        true_cates=gen.cates
    )

    # Check compatibility before instantiating
    if MAEOnOracleCATE.is_compatible_with(data):
        print("MAEOnOracleCATE can handle this data!")
    ```
    """

    # Class attribute that must be overridden by subclasses
    capabilities: ScorerCapabilities
    """Class attribute defining the scorer's capabilities and requirements. Must be defined by all implementations."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize scorer with any required nuisance models or parameters.

        Subclasses can define their own signature based on their requirements (e.g., if they require a treatment model, outcome model, etc.).
        The presence of required parameters will be validated in __init_subclass__.

        If a scorer has `~~base_scorer.ScorerCapabilities.requires_treatment_model`
        set to True, then `__init__` must have a `treatment_model` parameter.
        Similar rules apply for `~~base_scorer.ScorerCapabilities.requires_outcome_model`
        and `~~base_scorer.ScorerCapabilities.requires_regression_model`.

        Examples
        --------
        ```python
        from caml.scorers import BaseCateScorerMixin, ScorerCapabilities


        class ExampleScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=True,
                requires_outcome_model=False,
                requires_regression_model=False,
                requires_oracle_cates=False,
                greater_is_better=False,
            )

            def __init__(self, treatment_model):
                self.treatment_model = treatment_model

            def __call__(self, estimator, data):
                # Scoring logic here
                pass


        ExampleScorer(treatment_model="some_model")
        ```
        """
        ...

    @abstractmethod
    def __call__(self, estimator, data: CausalDataset) -> float:
        """Score the estimator on data (**ABSTRACT**).

        Notes
        -----
        **This is an abstract method.** Subclasses must provide a complete
        implementation that computes a score for the estimator on the dataset.

        Implementations should:

        1. Extract CATE predictions via `~~base_estimator.AutoCateEstimator.effect()`
        2. Compute the score using the predictions and any necessary nuisance quantities (e.g., true CATEs, pseudo-outcomes)
        3. Return a scalar score

        Parameters
        ----------
        estimator
            Fitted CATE estimator implementing
            `~~base_estimator.AutoCateEstimator.effect()`.
        data
            `~~dataset.CausalDataset` to score on.

        Returns
        -------
        float
            Score value.

        Examples
        --------
        ```python
        # Typical implementation pattern for oracle-based scorer
        import numpy as np
        from caml.scorers import BaseCateScorerMixin, ScorerCapabilities
        from caml.data import TreatmentType, OutcomeType


        class RMSEScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                requires_oracle_cates=True,
                greater_is_better=False,
            )

            def __init__(self, true_cates: np.ndarray | None = None):
                self.true_cates = true_cates

            def __call__(self, estimator, data):
                # 1. Get CATE predictions
                tau_hat = estimator.effect(data.X)

                # 2. Compute score (requires oracle CATEs)
                rmse = np.sqrt(np.mean((tau_hat - self.true_cates) ** 2))

                # 3. Return score
                return rmse


        RMSEScorer()
        ```

        ```python
        # Pattern for scorer using nuisance models (pseudo-outcome)
        from caml.scorers import BaseCateScorerMixin, ScorerCapabilities


        class PseudoOutcomeScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=False,
                requires_oracle_cates=False,
                greater_is_better=False,
            )

            def __init__(self, treatment_model, outcome_model):
                self.treatment_model = treatment_model
                self.outcome_model = outcome_model

            def __call__(self, estimator, data):
                # 1. Get predictions
                tau_hat = estimator.effect(data.X)

                # 2. Extract nuisance predictions (computed out-of-fold)
                pseudo_outcome = compute_pseudo_outcome(
                    data, self.treatment_model, self.outcome_model
                )

                # 3. Compute score
                score = np.mean((tau_hat - pseudo_outcome) ** 2)

                # 4. Return score
                return score


        PseudoOutcomeScorer()
        ```
        """
        ...

    @classmethod
    def is_compatible_with(cls, data: CausalDataset) -> bool:
        """Check scorer-data compatibility (**CONCRETE**).

        This is a class method enabling compatibility checking without instantiation.
        Useful for filtering candidate scorers before evaluation.

        Parameters
        ----------
        data
            `~~dataset.CausalDataset` to check compatibility with.

        Returns
        -------
        bool
            True if scorer supports the dataset's treatment and outcome types.

        Examples
        --------
        ```{python}
        from caml.scorers import RLoss
        from caml.data import CausalDataset, TreatmentType, OutcomeType
        from caml.utilities.synthetic_data import SyntheticDataGenerator

        # Generate test data
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
        if RLoss.is_compatible_with(data):
            print("RLoss can handle this data!")
        ```

        ```{python}
        # Filter multiple scorers efficiently
        from caml.scorers import RLoss, DRLoss

        candidates = [RLoss, DRLoss]
        compatible = [
            scorer_class for scorer_class in candidates
            if scorer_class.is_compatible_with(data)
        ]
        print(f"Compatible: {[c.__name__ for c in compatible]}")
        ```
        """
        return cls.capabilities.is_compatible(data)

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce that subclasses define required class attributes and follow additional expected patterns - (**CONCRETE**).

        Raises
        ------
        TypeError
            If non-abstract subclass doesn't define
            `~~base_scorer.BaseCateScorerMixin.capabilities`.
        """
        super().__init_subclass__(**kwargs)
        if "capabilities" not in cls.__dict__ and not inspect.isabstract(cls):
            raise TypeError(
                f"{cls.__name__} must define 'capabilities' as a class attribute. "
                f"See ScorerCapabilities for details."
            )
        # If requires_regression_model, requires_treatment_model, or requires_outcome_model, ensure __init__ signature has
        # regression_model, treatment_model, or outcome_model parameters. Only checked for concrete classes that have
        # already passed the capabilities check above (abstract classes may not define capabilities yet).
        if not inspect.isabstract(cls):
            sig = inspect.signature(cls.__init__)
            if cls.capabilities.requires_regression_model:
                if "regression_model" not in sig.parameters:
                    raise TypeError(
                        f"{cls.__name__} requires a regression model but __init_() is missing 'regression_model' parameter."
                    )
            if cls.capabilities.requires_treatment_model:
                if "treatment_model" not in sig.parameters:
                    raise TypeError(
                        f"{cls.__name__} requires a treatment model but __init__() is missing 'treatment_model' parameter."
                    )
            if cls.capabilities.requires_outcome_model:
                if "outcome_model" not in sig.parameters:
                    raise TypeError(
                        f"{cls.__name__} requires an outcome model but __init__() is missing 'outcome_model' parameter."
                    )
