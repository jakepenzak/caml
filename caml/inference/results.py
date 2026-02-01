"""Result containers for inference outputs.

Provides ``InferenceResult`` dataclass for packaging CATE estimates with uncertainty
quantification (standard errors, confidence intervals) and inference metadata.
"""

from dataclasses import dataclass

import numpy as np

from .inference_enums import InferenceType


# TODO: Implement helpers for generating summaries, CIs, hypothesis tests, etc.
@dataclass
class InferenceResult:
    """Container for CATE estimates with uncertainty quantification.

    Packages point estimates, standard errors, confidence intervals, and inference
    metadata. Supports both scalar (ATE) and vector (individual CATE) estimates.

    **TODO: Implement helpers for generating summaries, CIs, hypothesis tests, etc.**

    Parameters
    ----------
    effect
        Treatment effect estimate(s). Scalar for ATE/ATT/ATC, array for CATE.
    stderr
        Standard error(s) matching shape of ``effect``.
    method
        Inference method used (``InferenceType.ANALYTIC`` or ``InferenceType.BOOTSTRAP``).

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.inference import InferenceResult, InferenceType

    # Scalar ATE result
    result = InferenceResult(
        effect=2.5,
        stderr=0.3,
        method=InferenceType.ANALYTIC
    )
    print(f"ATE: {result.effect:.2f}")
    ```
    """

    effect: float | np.ndarray
    stderr: float | np.ndarray | None = None
    method: InferenceType | None = None

    def __len__(self):
        """Return number of estimates (1 for scalar, n for array)."""
        return len(self.effect) if isinstance(self.effect, np.ndarray) else 1

    def __repr__(self):
        """Return concise string representation."""
        n = len(self)
        if self.stderr is not None:
            return f"InferenceResult(n={n}, method={self.method.value if self.method else 'auto'})"
        return f"InferenceResult(n={n})"
