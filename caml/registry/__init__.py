from .registry import (
    AVAILABLE_CATE_ESTIMATORS,
    AVAILABLE_CATE_SCORERS,
    auto_register,
    get_compatible_estimators,
    get_compatible_scorers,
    register_estimator,
    register_scorer,
)
from .registry_enums import EstimatorFamily, ScorerFamily

__all__ = [
    "AVAILABLE_CATE_ESTIMATORS",
    "AVAILABLE_CATE_SCORERS",
    "register_scorer",
    "EstimatorFamily",
    "ScorerFamily",
    "get_compatible_estimators",
    "get_compatible_scorers",
    "register_estimator",
    "auto_register",
]


# Import estimators & scorers to trigger auto-registration (must be after registry imports)
from caml import (
    estimators,  # noqa: F401
    scorers,  # noqa: F401
)
