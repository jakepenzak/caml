"""Core data type definitions for Causal datasets in CaML.

Defines enumerations for treatment variables, outcome variables, and target estimands
used throughout CaML. These enable automatic estimator compatibility checking and validation.
"""

from enum import Enum


class TreatmentType(Enum):
    """Categories of treatment variables supported by CATE estimators.

    Attributes
    ----------
    BINARY : str
        Binary treatment (0/1). Supported by all estimators.
    MULTI : str
        Multi-valued discrete treatment (3+ categories). Supported by meta-learners.
    CONTINUOUS : str
        Continuous treatment (e.g., dosage). Supported by DML and DR learners.

    See Also
    --------
    [`OutcomeType`](data_schema.qmd#caml.data.data_schema.OutcomeType) : Outcome variable categories.

    [`CausalDataset`](dataset.qmd#caml.data.dataset.CausalDataset) : Data container using treatment types.

    [`EstimatorCapabilities`](base.qmd#caml.estimators.base.EstimatorCapabilities) : Estimator capability metadata.

    Examples
    --------
    ```{python}
    from caml.data import TreatmentType

    assert TreatmentType.BINARY.is_discrete()
    assert not TreatmentType.CONTINUOUS.is_discrete()
    ```
    """

    BINARY = "binary"
    MULTI = "multi"
    CONTINUOUS = "continuous"

    def is_discrete(self) -> bool:
        """Check if treatment is discrete (binary or multi-valued).

        Returns
        -------
        bool
            True for binary/multi, False for continuous.
        """
        return self in {TreatmentType.BINARY, TreatmentType.MULTI}


class OutcomeType(Enum):
    """Categories of outcome variables supported by CATE estimators.

    Attributes
    ----------
    BINARY : str
        Binary outcome (0/1). Modeled with classification algorithms.
    CONTINUOUS : str
        Continuous outcome. Modeled with regression algorithms.

    Notes
    -----
    Outcome type affects nuisance model selection (e.g., ``LGBMClassifier`` vs
    ``LGBMRegressor``) rather than core CATE estimation strategy.

    See Also
    --------
    [`TreatmentType`](data_schema.qmd#caml.data.data_schema.TreatmentType) : Treatment variable categories.

    [`CausalDataset`](dataset.qmd#caml.data.dataset.CausalDataset) : Data container using outcome types.

    Examples
    --------
    ```{python}
    from caml.data import OutcomeType

    assert OutcomeType.BINARY.is_discrete()
    assert not OutcomeType.CONTINUOUS.is_discrete()
    ```
    """

    BINARY = "binary"
    CONTINUOUS = "continuous"

    def is_discrete(self) -> bool:
        """Check if outcome is discrete (binary).

        Returns
        -------
        bool
            True for binary, False for continuous.
        """
        return self == OutcomeType.BINARY


class Estimand(Enum):
    """Target causal quantities estimable by CaML methods.

    Attributes
    ----------
    ATE : str
        Average Treatment Effect across population.
    ATT : str
        Average Treatment Effect on the Treated.
    ATC : str
        Average Treatment Effect on Control.
    CATE : str
        Conditional Average Treatment Effect given covariates.
    GATE : str
        Group Average Treatment Effect within subgroups.

    Notes
    -----
    Most CATE estimators naturally estimate CATE, from which ATE/ATT/ATC can be
    derived via aggregation.

    See Also
    --------
    [`EstimatorCapabilities`](base.qmd#caml.estimators.base.EstimatorCapabilities) : Metadata including supported estimands.

    [`AutoCateEstimator`](base.qmd#caml.estimators.base.AutoCateEstimator) : Protocol for CATE estimators.

    Examples
    --------
    ```{python}
    from caml.data import Estimand
    import numpy as np

    # Derive ATE from CATE estimates
    np.random.seed(42)
    cate = np.random.randn(1000)
    ate = np.mean(cate)
    print(f"{Estimand.ATE.value.upper()}: {ate:.3f}")
    ```
    """

    ATE = "ate"  # Average Treatment Effect
    ATT = "att"  # Average Treatment Effect on Treated
    ATC = "atc"  # Average Treatment Effect on Control
    CATE = "cate"  # Conditional ATE
    GATE = "gate"  # Group ATE
