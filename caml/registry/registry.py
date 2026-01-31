"""Global registry utilities for CaML estimators and scorers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

from caml.data import CausalDataset
from caml.registry.registry_schema import EstimatorFamily

if TYPE_CHECKING:
    from caml.estimators.base import AutoCateEstimator


available_estimators: dict = dict()
"""Dictionary of available estimators with their corresponding classes and families."""


def get_compatible_estimators(
    data: CausalDataset, families: list[EstimatorFamily | str] | None = None
) -> dict:
    """Get estimators compatible with dataset.

    For custom estimators, ensure they are registered using `register_estimator`.

    Parameters
    ----------
    data
        Dataset to check compatibility.
    families
        Estimator families to include: ["dml", "dr", "meta", "orf"] or any custom ones created using
        `register_estimator`. Defaults to None, which includes all available estimators.

    Returns
    -------
    dict
        List of compatible estimator INSTANCES (ready to use).

    Examples
    --------
    ```{python}
    from caml.registry import get_compatible_estimators, EstimatorFamily
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS
    )

    # Automatically filter to compatible estimators
    compatible = get_compatible_estimators(data, families=[EstimatorFamily.DML, "dr"])
    print(f"Found {len(compatible)} compatible estimators")
    compatible
    ```
    """
    if families is None:
        candidate_estimators = available_estimators
    else:
        # Normalize families to EstimatorFamily enum
        normalized_families = []
        for family in families:
            if isinstance(family, str):
                normalized_families.append(EstimatorFamily(family))
            else:
                normalized_families.append(family)

        candidate_estimators = {}
        for family in normalized_families:
            candidate_estimators.update(
                {
                    name: est
                    for name, est in available_estimators.items()
                    if est["family"] == family
                }
            )

    # Filter to compatible classes
    compatible_estimators = {
        **{
            name: est
            for name, est in candidate_estimators.items()
            if est["estimator"].is_compatible_with(data)
        }
    }

    return compatible_estimators


def register_estimator(
    name: str,
    estimator: AutoCateEstimator,
    family: EstimatorFamily | str = EstimatorFamily.CUSTOM,
) -> None:
    """Register a new estimator in the global registry.

    Parameters
    ----------
    name
        Name of the estimator to register.
    estimator
        Estimator class to register.
    family
        Family for the estimator

    Examples
    --------
    ```{python}
    from caml.registry import register_estimator, available_estimators, EstimatorFamily
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.estimators import EstimatorCapabilities

    class SimpleEstimator:

        clean_name = "SimpleEstimator"

        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False
            )

        def __init__(self):
            self.effect_value = None

        def is_compatible_with(cls, data: CausalDataset) -> bool:
            temp_instance = cls()
            return temp_instance.capabilities.is_compatible(data)

        def check_compatibility(
            self, data: CausalDataset, raise_error: bool = True
        ) -> bool:
            is_compatible = self.capabilities.is_compatible(data)
            return is_compatible

        def fit(self, data, **kwargs):
            T = np.asarray(data.T)
            Y = np.asarray(data.Y)
            self.effect_value = Y[T == 1].mean() - Y[T == 0].mean()
            return self

        def effect(self, X, **kwargs):
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, self.effect_value)

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self


    # Current available estimators
    print(available_estimators.keys())

    # Updated available estimators
    register_estimator(name="SimpleEstimator", estimator=SimpleEstimator, family=EstimatorFamily.CUSTOM)
    print(available_estimators.keys())
    ```
    """
    available_estimators[name] = {
        "estimator": estimator,
        "family": family
        if isinstance(family, EstimatorFamily)
        else EstimatorFamily(family),
    }


def auto_register(
    name: str | None = None,
    family: EstimatorFamily | str = EstimatorFamily.CUSTOM,
    type: Literal["estimator"] = "estimator",
) -> Callable:
    """Decorator to register estimators and scorers in the global registry.

    This decorator registers the class at import time, making it immediately
    available in the global registry.

    Parameters
    ----------
    name
        The name to register the object under. If None, uses the class's
        ``clean_name`` attribute or ``__name__``.
    family
        The family of the estimator or scorer (e.g., "dml", "dr", "meta", "orf", "custom").
    type
        The type of the object to register: "estimator" or "scorer".

    Returns
    -------
    Callable
        The decorated class (unmodified).

    Examples
    --------
    ```{python}
    from caml.registry import auto_register, available_estimators, EstimatorFamily
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.estimators import EstimatorCapabilities
    import numpy as np

    @auto_register(name="MyCustomEstimator", family=EstimatorFamily.CUSTOM)
    class MyCustomEstimator:

        clean_name = "MyCustomEstimator"

        capabilities = EstimatorCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            inference_types=set(),
            estimands={Estimand.CATE},
            supports_controls_in_first_stage_only=False,
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            supports_inference=False
        )

        def __init__(self):
            self.effect_value = None

        @classmethod
        def is_compatible_with(cls, data: CausalDataset) -> bool:
            temp_instance = cls()
            return temp_instance.capabilities.is_compatible(data)

        def check_compatibility(
            self, data: CausalDataset, raise_error: bool = True
        ) -> bool:
            return self.capabilities.is_compatible(data)

        def fit(self, data, **kwargs):
            T = np.asarray(data.T)
            Y = np.asarray(data.Y)
            self.effect_value = Y[T == 1].mean() - Y[T == 0].mean()
            return self

        def effect(self, X, **kwargs):
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, self.effect_value)

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    # Check that it was registered
    print("MyCustomEstimator" in available_estimators)
    ```
    """

    def decorator(obj):
        """Inner decorator that performs the registration."""
        registry_name = name
        if registry_name is None:
            registry_name = getattr(obj, "clean_name", obj.__name__)
        if type == "estimator":
            register_estimator(name=registry_name, estimator=obj, family=family)
        # Future extension: add scorer registration here
        # elif type == "scorer":
        #     register_scorer(name=registry_name, scorer=obj, family=family)

        return obj

    return decorator
