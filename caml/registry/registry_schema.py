"""Registry schema definitions for estimator and scorer families."""

from enum import Enum


class EstimatorFamily(Enum):
    """Enum for estimator families. Used for global registry categorization.

    Attributes
    ----------
    DML : str
        Double Machine Learning estimators.
    DR : str
        Doubly Robust estimators.
    META : str
        Meta-learners.
    ORF : str
        Orthogonal Random Forest estimators.
    CUSTOM : str
        Custom estimators registered by users.

    Examples
    --------
    ```{python}
    from caml.registry import EstimatorFamily

    print(EstimatorFamily.DML)
    print(EstimatorFamily("dml"))
    ```
    """

    DML = "dml"
    DR = "dr"
    META = "meta"
    ORF = "orf"
    CUSTOM = "custom"
