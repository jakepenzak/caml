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
        User-defined estimators that do not fall into the predefined families.

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


class ScorerFamily(Enum):
    """Enum for scorer families. Used for global registry categorization.

    Provides high-level taxonomy for CATE model scoring metrics.

    Attributes
    ----------
    ORACLE : str
        Oracle metrics that compare predictions to the true CATE (eg, PEHE).
    PLUG_IN : str
        Plug-in surrogate metrics based on a learned CATE reference. (eg, T-Loss)
    PSEUDO_OUTCOME : str
        Pseudo-outcome (transformed outcome) based metrics (eg, DR-Loss).
    RANKING_RELATIVE_PROXY : str
        Relative performance proxy metrics based on ranking or loss reformulation (eg, Q-Statistic).
    RANKING_CURVE : str
        Ranking-based cumulative gain / uplift curve metrics (eg, Qini, Uplift, AUUC).
    POLICY : str
        Policy or decision-value evaluation metrics (eg, policy value).
    CUSTOM : str
        User-defined scorers that do not fall into the predefined families.

    Examples
    --------
    ```{python}
    from caml.registry import ScorerFamily
    print(ScorerFamily.ORACLE)
    print(ScorerFamily("oracle"))
    ```
    """

    ORACLE = "oracle"
    PLUG_IN = "plug_in"
    PSEUDO_OUTCOME = "pseudo_outcome"
    RANKING_RELATIVE_PROXY = "ranking_relative_proxy"
    RANKING_CURVE = "ranking_curve"
    POLICY = "policy"
    CUSTOM = "custom"
