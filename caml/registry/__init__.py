from .registry import (
    auto_register,
    available_estimators,
    get_compatible_estimators,
    register_estimator,
)
from .registry_schema import EstimatorFamily

__all__ = [
    "available_estimators",
    "EstimatorFamily",
    "get_compatible_estimators",
    "register_estimator",
    "auto_register",
]


# Import estimators to trigger auto-registration (must be after registry imports)
from caml import estimators  # noqa: F401
