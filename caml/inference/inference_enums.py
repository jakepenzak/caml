"""Inference method definitions for uncertainty quantification.

Defines statistical inference methods for computing confidence intervals and
standard errors for CATE estimates.
"""

from enum import Enum


class InferenceType(Enum):
    """Categories of statistical inference methods for CATE uncertainty quantification.

    Attributes
    ----------
    ANALYTIC : str
        Analytic inference using closed-form variance formulas. Fast but requires
        regularity conditions (e.g., correct specification, large samples).
    BOOTSTRAP : str
        Bootstrap inference via resampling. Computationally expensive but makes
        fewer distributional assumptions.

    Notes
    -----
    - Not all estimators support all inference types (check ``capabilities.inference_types``)
    - Bootstrap is more robust but slower; analytic is faster but requires stronger assumptions
    - DML estimators typically use analytic; tree-based methods use bootstrap

    Examples
    --------
    ```{python}
    from caml.data import TreatmentType, OutcomeType, Estimand
    from caml.inference import InferenceType
    from caml.estimators import EstimatorCapabilities

    capabilities = EstimatorCapabilities(
        treatment_types={TreatmentType.BINARY},
        outcome_types={OutcomeType.CONTINUOUS},
        inference_types={InferenceType.ANALYTIC, InferenceType.BOOTSTRAP},
        estimands={Estimand.CATE},
        supports_controls_in_first_stage_only=True,
        supports_weights=True,
        requires_treatment_model=True,
        requires_outcome_model=True,
        requires_regression_model=False,
        supports_inference=True,
    )

    print(InferenceType.ANALYTIC in capabilities.inference_types)  # True
    ```
    """

    ANALYTIC = "analytic"
    BOOTSTRAP = "bootstrap"
