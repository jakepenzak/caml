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
        Binary treatment (0/1).
    MULTI : str
        Multi-valued discrete treatment (3+ categories).
    CONTINUOUS : str
        Continuous treatment (e.g., dosage).

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
        Binary outcome (0/1).
    CONTINUOUS : str
        Continuous outcome.

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
