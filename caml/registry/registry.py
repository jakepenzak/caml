"""Registry functions for estimators."""

from caml.data import CausalDataset
from caml.estimators.base import AutoCateEstimator
from caml.registry.model_bank import available_estimators


def get_compatible_estimators(
    data: CausalDataset, families: list[str] | None = None
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
    from caml.registry import get_compatible_estimators
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
    compatible = get_compatible_estimators(data, families=["dml", "dr"])
    print(f"Found {len(compatible)} compatible estimators")
    compatible
    ```
    """
    if families is None:
        candidate_estimators = available_estimators
    else:
        candidate_estimators = {}
        for family in families:
            candidate_estimators = {
                **{
                    name: est
                    for name, est in available_estimators.items()
                    if est["family"] == family
                }
            }

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
    name: str, estimator: AutoCateEstimator, family: str = "custom"
) -> None:
    """Register a new estimator in the global registry.

    Parameters
    ----------
    name
        Name of the estimator to register.
    estimator
        Estimator class to register.
    family
        Family name for the estimator (e.g., "dml", "dr", "meta", "orf", or "custom").

    Examples
    --------
    ```{python}
    from caml.registry import register_estimator, available_estimators
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
    register_estimator(name="SimpleEstimator", estimator=SimpleEstimator, family="custom")
    print(available_estimators.keys())
    ```
    """
    if not isinstance(estimator, AutoCateEstimator):
        raise ValueError("Estimator must be a subclass of AutoCateEstimator.")

    available_estimators[name] = {
        "estimator": estimator,
        "family": family,
    }
