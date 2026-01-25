"""Result containers for inference outputs.

Provides ``InferenceResult`` dataclass for packaging CATE estimates with uncertainty
quantification (standard errors, confidence intervals) and inference metadata.
"""

from dataclasses import dataclass

import numpy as np

from caml.inference.inference_schema import InferenceType


@dataclass
class InferenceResult:
    """Container for CATE estimates with uncertainty quantification.

    Packages point estimates, standard errors, confidence intervals, and inference
    metadata. Supports both scalar (ATE) and vector (individual CATE) estimates.

    Parameters
    ----------
    point_estimate
        Treatment effect estimate(s). Scalar for ATE/ATT/ATC, array for CATE.
    stderr
        Standard error(s) matching shape of ``point_estimate``.
    ci_lower
        Lower confidence bound(s) matching shape of ``point_estimate``.
    ci_upper
        Upper confidence bound(s) matching shape of ``point_estimate``.
    alpha
        Significance level for confidence intervals (e.g., 0.05 for 95% CI).
    method
        Inference method used (``InferenceType.ANALYTIC`` or ``InferenceType.BOOTSTRAP``).
    n_bootstrap
        Number of bootstrap iterations if ``method`` is ``InferenceType.BOOTSTRAP``.

    See Also
    --------
    [`InferenceType`](inference_schema.qmd#caml.inference.inference_schema.InferenceType) : Inference method categories.

    [`InferenceProvider`](inference.qmd#caml.protocols.inference.InferenceProvider) : Protocol for inference-capable estimators.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.inference import InferenceResult, InferenceType

    # Scalar ATE result
    result = InferenceResult(
        point_estimate=2.5,
        stderr=0.3,
        ci_lower=1.9,
        ci_upper=3.1,
        alpha=0.05,
        method=InferenceType.ANALYTIC
    )
    print(f"ATE: {result.point_estimate:.2f} [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    ```
    """

    point_estimate: float | np.ndarray
    stderr: float | np.ndarray | None = None
    ci_lower: float | np.ndarray | None = None
    ci_upper: float | np.ndarray | None = None
    alpha: float = 0.05
    method: InferenceType | None = None
    n_bootstrap: int | None = None

    def __len__(self):
        """Return number of estimates (1 for scalar, n for array)."""
        return (
            len(self.point_estimate)
            if isinstance(self.point_estimate, np.ndarray)
            else 1
        )

    def __repr__(self):
        """Return concise string representation."""
        n = len(self)
        if self.stderr is not None:
            return f"InferenceResult(n={n}, method={self.method.value if self.method else 'unknown'})"
        return f"InferenceResult(n={n})"
