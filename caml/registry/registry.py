"""Global registry utilities for CaML estimators and scorers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from caml.data.dataset import CausalDataset

from .registry_enums import EstimatorFamily, ScorerFamily

if TYPE_CHECKING:
    from caml.estimators import AutoCateEstimator


AVAILABLE_CATE_ESTIMATORS: dict = dict()
"""Dictionary of available estimators with their corresponding classes and families."""

AVAILABLE_CATE_SCORERS: dict = dict()
"""Dictionary of available scorers with their corresponding classes and families."""


def get_compatible_estimators(
    data: CausalDataset, families: list[EstimatorFamily | str] | None = None
) -> dict:
    """Get estimators compatible with dataset.

    For custom estimators, ensure they are registered using
    `~~registry.register_estimator`.

    Parameters
    ----------
    data
        Dataset to check compatibility.
    families
        Estimator families to include, expressed as
        `~~registry_enums.EstimatorFamily` values or their string forms.
        Defaults to None, which includes all available estimators.

    Returns
    -------
    dict
        List of compatible estimator INSTANCES (ready to use).

    Examples
    --------
    ```{python}
    from caml.registry import get_compatible_estimators, EstimatorFamily
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.utilities.synthetic_data import SyntheticDataGenerator

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
        candidate_estimators = AVAILABLE_CATE_ESTIMATORS
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
                    for name, est in AVAILABLE_CATE_ESTIMATORS.items()
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


def get_compatible_scorers(
    data: CausalDataset, families: list[ScorerFamily | str] | None = None
) -> dict:
    """Get scorers compatible with dataset.

    For custom scorers, ensure they are registered using
    `~~registry.register_scorer`.

    Parameters
    ----------
    data
        Dataset to check compatibility.
    families
        Scorer families to include, expressed as
        `~~registry_enums.ScorerFamily` values or their string forms.
        Defaults to None, which includes all available scorers.

    Returns
    -------
    dict
        List of compatible scorer CLASSES (not instantiated).

    Examples
    --------
    ```{python}
    from caml.registry import get_compatible_scorers, ScorerFamily
    from caml.data import CausalDataset, TreatmentType, OutcomeType
    from caml.utilities.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    data = CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
        true_cates=gen.cates
    )

    # Automatically filter to compatible scorers
    compatible = get_compatible_scorers(data, families=[ScorerFamily.ORACLE, "pseudo_outcome"])
    print(f"Found {len(compatible)} compatible scorers")
    compatible
    ```
    """
    if families is None:
        candidate_scorers = AVAILABLE_CATE_SCORERS
    else:
        # Normalize families to ScorerFamily enum
        normalized_families = []
        for family in families:
            if isinstance(family, str):
                normalized_families.append(ScorerFamily(family))
            else:
                normalized_families.append(family)

        candidate_scorers = {}
        for family in normalized_families:
            candidate_scorers.update(
                {
                    name: scorer
                    for name, scorer in AVAILABLE_CATE_SCORERS.items()
                    if scorer["family"] == family
                }
            )

    # Filter to compatible classes
    compatible_scorers = {
        **{
            name: scorer
            for name, scorer in candidate_scorers.items()
            if scorer["scorer"].is_compatible_with(data)
        }
    }

    return compatible_scorers


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
        Estimator class to register implementing
        `~~base_estimator.AutoCateEstimator`.
    family
        Family for the estimator as `~~registry_enums.EstimatorFamily` or a
        matching string.

    Examples
    --------
    ```{python}
    from caml.registry import register_estimator, AVAILABLE_CATE_ESTIMATORS, EstimatorFamily
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.estimators import EstimatorCapabilities, BaseAutoCateEstimatorMixin
    from caml.automl import SearchSpace, IntSpec

    class SimpleEstimator(BaseAutoCateEstimatorMixin):

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

        default_search_space = (
            IntSpec(name="x", lower=1, upper=10),
        )

        def __init__(self, x: int = 1):
            self.x = 1

        def fit(self, data, **kwargs):
            T = np.asarray(data.T)
            Y = np.asarray(data.Y)
            self.effect_value = Y[T == 1].mean() - Y[T == 0].mean()
            return self

        def effect(self, X, **kwargs):
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, self.effect_value)

    # Current available estimators
    print(AVAILABLE_CATE_ESTIMATORS.keys())

    # Updated available estimators
    register_estimator(name="SimpleEstimator", estimator=SimpleEstimator, family=EstimatorFamily.CUSTOM)
    print(AVAILABLE_CATE_ESTIMATORS.keys())
    ```
    """
    AVAILABLE_CATE_ESTIMATORS[name] = {
        "estimator": estimator,
        "family": family
        if isinstance(family, EstimatorFamily)
        else EstimatorFamily(family),
    }


def register_scorer(
    name: str,
    scorer: Callable,
    family: ScorerFamily | str = ScorerFamily.CUSTOM,
) -> None:
    """Register a new scorer in the global registry.

    Parameters
    ----------
    name
        Name of the scorer to register.
    scorer
        Scorer callable to register, often a
        `~~base_scorer.BaseCateScorerMixin` subclass.
    family
        Family for the scorer as `~~registry_enums.ScorerFamily` or a matching
        string.

    Examples
    --------
    ```{python}
    import numpy as np
    from caml.scorers import BaseCateScorerMixin, ScorerCapabilities
    from caml.registry import register_scorer, AVAILABLE_CATE_SCORERS, ScorerFamily

    class NegMAEOnOracleCATE(BaseCateScorerMixin):
        capabilities: ScorerCapabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            requires_oracle_cates=False
        )

        def __call__(self, estimator, data):
            tau_hat = estimator.effect(data.X)
            mae = np.mean(np.abs(tau_hat - data.true_cates))
            return -mae

    # Current available estimators
    print(AVAILABLE_CATE_SCORERS.keys())

    # Updated available estimators
    register_scorer(name="NegMAEOnOracleCATE", scorer=NegMAEOnOracleCATE, family=ScorerFamily.CUSTOM)
    print(AVAILABLE_CATE_SCORERS.keys())
    ```
    """
    AVAILABLE_CATE_SCORERS[name] = {
        "scorer": scorer,
        "family": family if isinstance(family, ScorerFamily) else ScorerFamily(family),
    }


def auto_register(
    name: str, family: EstimatorFamily | ScorerFamily | str, is_estimator: bool = True
) -> Callable:
    """Decorator to register estimators and scorers in the global registry.

    This decorator registers the class at import time, making it immediately
    available in the global registry.

    Parameters
    ----------
    name
        The name to register the object under.
    family
        The family of the estimator or scorer as
        `~~registry_enums.EstimatorFamily`, `~~registry_enums.ScorerFamily`, or
        a matching string.
    is_estimator
        If True, registers as an estimator; if False, registers as a scorer.

    Returns
    -------
    Callable
        The decorated object (unmodified).

    Examples
    --------
    ```{python}
    import numpy as np

    from caml.registry import auto_register, AVAILABLE_CATE_ESTIMATORS, EstimatorFamily
    from caml.data import CausalDataset, TreatmentType, OutcomeType, Estimand
    from caml.estimators import EstimatorCapabilities, BaseAutoCateEstimatorMixin
    from caml.automl import SearchSpace, IntSpec

    @auto_register(name="MyCustomEstimator", family=EstimatorFamily.CUSTOM)
    class MyCustomEstimator(BaseAutoCateEstimatorMixin):

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

        default_search_space = (
            IntSpec(name="x", lower=1, upper=10),
        )

        def __init__(self, x: int = 1):
            self.x = x

        @classmethod
        def is_compatible_with(cls, data: CausalDataset) -> bool:
            temp_instance = cls()
            return temp_instance.capabilities.is_compatible(data)

        def fit(self, data, **kwargs):
            T = np.asarray(data.T)
            Y = np.asarray(data.Y)
            self.effect_value = Y[T == 1].mean() - Y[T == 0].mean()
            return self

        def effect(self, X, **kwargs):
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, self.effect_value)

    # Check that it was registered
    print("MyCustomEstimator" in AVAILABLE_CATE_ESTIMATORS)
    ```

    ```{python}
    from caml.registry import auto_register, AVAILABLE_CATE_SCORERS, ScorerFamily

    @auto_register(name="NegMAEOnOracleCATE", family=ScorerFamily.ORACLE, is_estimator=False)
    class NegMAEOnOracleCATE(BaseCateScorerMixin):

        capabilities: ScorerCapabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            supports_weights=False,
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            requires_oracle_cates=False
        )

        def __call__(self, estimator, data):
            tau_hat = estimator.effect(data.X)
            mae = np.mean(np.abs(tau_hat - data.true_cates))
            return -mae

    # Check that it was registered
    print("NegMAEOnOracleCATE" in AVAILABLE_CATE_SCORERS)
    ```
    """

    def decorator(obj):
        """Inner decorator that performs the registration."""
        if is_estimator:
            if not (isinstance(family, str) or isinstance(family, EstimatorFamily)):
                raise ValueError(
                    "Estimator family must be a string or EstimatorFamily enum."
                )
            register_estimator(name=name, estimator=obj, family=family)
        else:
            if not (isinstance(family, str) or isinstance(family, ScorerFamily)):
                raise ValueError("Scorer family must be a string or ScorerFamily enum.")
            register_scorer(name=name, scorer=obj, family=family)

        return obj

    return decorator
