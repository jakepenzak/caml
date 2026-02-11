from .registry import (
    auto_register,
    available_estimators,
    available_scorers,
    get_compatible_estimators,
    get_compatible_scorers,
    register_estimator,
    register_scorer,
)
from .registry_enums import EstimatorFamily, ScorerFamily

__all__ = [
    "available_estimators",
    "available_scorers",
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
