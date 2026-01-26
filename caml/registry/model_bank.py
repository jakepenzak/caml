"""Module defining a dictionary of available causal estimators with their corresponding classes and families.

```{python}
from caml.registry import available_estimators

available_estimators
```
"""

from caml.estimators.wrappers.dml import (
    WrappedCausalForestDML,
    WrappedKernelDML,
    WrappedLinearDML,
    WrappedNonParamDML,
    WrappedSparseLinearDML,
)
from caml.estimators.wrappers.dr import (
    WrappedDRLearner,
    WrappedForestDRLearner,
    WrappedLinearDRLearner,
    WrappedSparseLinearDRLearner,
)
from caml.estimators.wrappers.meta import (
    WrappedSLearner,
    WrappedTLearner,
    WrappedXLearner,
)
from caml.estimators.wrappers.orf import WrappedDMLOrthoForest, WrappedDROrthoForest

available_estimators: dict = {
    "CausalForestDML": {
        "estimator": WrappedCausalForestDML,
        "family": "dml",
    },
    "KernelDML": {"estimator": WrappedKernelDML, "family": "dml"},
    "LinearDML": {"estimator": WrappedLinearDML, "family": "dml"},
    "NonParamDML": {"estimator": WrappedNonParamDML, "family": "dml"},
    "SparseLinearDML": {
        "estimator": WrappedSparseLinearDML,
        "family": "dml",
    },
    "DRLearner": {"estimator": WrappedDRLearner, "family": "dr"},
    "ForestDRLearner": {
        "estimator": WrappedForestDRLearner,
        "family": "dr",
    },
    "LinearDRLearner": {
        "estimator": WrappedLinearDRLearner,
        "family": "dr",
    },
    "SparseLinearDRLearner": {
        "estimator": WrappedSparseLinearDRLearner,
        "family": "dr",
    },
    "SLearner": {"estimator": WrappedSLearner, "family": "meta"},
    "TLearner": {"estimator": WrappedTLearner, "family": "meta"},
    "XLearner": {"estimator": WrappedXLearner, "family": "meta"},
    "DMLOrthoForest": {
        "estimator": WrappedDMLOrthoForest,
        "family": "orf",
    },
    "DROrthoForest": {
        "estimator": WrappedDROrthoForest,
        "family": "orf",
    },
}
"""Dictionary of available estimators with their corresponding classes and families."""
