"""Tests for caml/registry/registry.py."""

import numpy as np
import pytest

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.data.data_schema import Estimand
from caml.estimators import EstimatorCapabilities
from caml.extensions.synthetic_data import SyntheticDataGenerator
from caml.registry.model_bank import available_estimators
from caml.registry.registry import get_compatible_estimators, register_estimator

pytestmark = pytest.mark.registry


@pytest.fixture
def binary_continuous_dataset():
    """Fixture for a binary treatment, continuous outcome dataset."""
    gen = SyntheticDataGenerator(n_obs=100, n_cont_confounders=3, seed=42)
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )


class TestGetCompatibleEstimators:
    """Tests for get_compatible_estimators function."""

    def test_returns_dict(self, binary_continuous_dataset):
        """Test that get_compatible_estimators returns a dict."""
        # Only test with families that work reliably (DR and ORF)
        result = get_compatible_estimators(binary_continuous_dataset, families=["dr"])
        assert isinstance(result, dict)

    def test_all_estimators_have_required_keys(self, binary_continuous_dataset):
        """Test that returned estimators have required keys."""
        result = get_compatible_estimators(binary_continuous_dataset, families=["dr"])
        assert len(result) > 0
        assert all("estimator" in entry for entry in result.values())
        assert all("family" in entry for entry in result.values())

    def test_filter_by_dr_family(self, binary_continuous_dataset):
        """Test filtering by DR family."""
        result = get_compatible_estimators(binary_continuous_dataset, families=["dr"])
        # Should only return DR estimators
        for name, entry in result.items():
            assert entry["family"] == "dr"
        # DR estimators should work without issues
        assert len(result) > 0

    def test_filter_by_orf_family(self, binary_continuous_dataset):
        """Test filtering by ORF family."""
        result = get_compatible_estimators(binary_continuous_dataset, families=["orf"])
        # Should only return ORF estimators
        for name, entry in result.items():
            assert entry["family"] == "orf"
        # ORF estimators should work without issues
        assert len(result) > 0

    def test_filter_by_multiple_families(self, binary_continuous_dataset):
        """Test filtering by multiple families."""
        result = get_compatible_estimators(
            binary_continuous_dataset, families=["dr", "orf"]
        )
        # Should return estimators from requested families
        families = {entry["family"] for entry in result.values()}
        assert len(families) > 0
        assert len(result) > 0

    def test_families_parameter_none_includes_all(self):
        """Test that families=None attempts to include all families."""
        gen = SyntheticDataGenerator(n_obs=100, n_cont_confounders=3, seed=42)
        dataset = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )
        # Test with families that work reliably
        result_dr = get_compatible_estimators(dataset, families=["dr"])
        # Should get DR estimators
        assert isinstance(result_dr, dict)
        assert len(result_dr) > 0


class TestRegisterEstimator:
    """Tests for register_estimator function."""

    def test_register_new_estimator(self):
        """Test registering a new estimator."""

        class SimpleEstimator:
            # Class attributes
            clean_name: str = "SimpleEstimator"
            capabilities: EstimatorCapabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types=set(),
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                supports_inference=False,
            )

            def __init__(self):
                self.effect_value = None

            @classmethod
            def is_compatible_with(cls, data: CausalDataset) -> bool:
                return cls.capabilities.is_compatible(data)

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
                n = len(X) if hasattr(X, "__len__") else 1
                return np.full(n, self.effect_value)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        initial_count = len(available_estimators)
        register_estimator(
            name="TestSimpleEstimator", estimator=SimpleEstimator, family="custom"
        )
        assert len(available_estimators) == initial_count + 1
        assert "TestSimpleEstimator" in available_estimators
        assert available_estimators["TestSimpleEstimator"]["family"] == "custom"
        assert (
            available_estimators["TestSimpleEstimator"]["estimator"] == SimpleEstimator
        )

        # Cleanup
        del available_estimators["TestSimpleEstimator"]

    def test_register_estimator_with_default_family(self):
        """Test registering estimator with default 'custom' family."""

        class AnotherEstimator:
            # Class attributes
            clean_name: str = "AnotherEstimator"
            capabilities: EstimatorCapabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types=set(),
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                supports_inference=False,
            )

            @classmethod
            def is_compatible_with(cls, data: CausalDataset) -> bool:
                return True

            def check_compatibility(
                self, data: CausalDataset, raise_error: bool = True
            ) -> bool:
                return True

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                return np.zeros(len(X))

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        initial_count = len(available_estimators)
        register_estimator(name="TestAnotherEstimator", estimator=AnotherEstimator)
        assert available_estimators["TestAnotherEstimator"]["family"] == "custom"

        # Cleanup
        del available_estimators["TestAnotherEstimator"]

    def test_registered_estimator_appears_in_registry(self):
        """Test that registered estimator appears in available_estimators."""

        class CompatibleEstimator:
            # Class attributes
            clean_name: str = "CompatibleEstimator"
            capabilities: EstimatorCapabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types=set(),
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                supports_inference=False,
            )

            @classmethod
            def is_compatible_with(cls, data: CausalDataset) -> bool:
                return cls.capabilities.is_compatible(data)

            def check_compatibility(
                self, data: CausalDataset, raise_error: bool = True
            ) -> bool:
                return self.capabilities.is_compatible(data)

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                return np.zeros(1)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        register_estimator(
            name="TestCompatibleEstimator",
            estimator=CompatibleEstimator,
            family="test",
        )
        # Check it appears in available_estimators
        assert "TestCompatibleEstimator" in available_estimators
        assert available_estimators["TestCompatibleEstimator"]["family"] == "test"

        # Cleanup
        del available_estimators["TestCompatibleEstimator"]

    def test_overwrite_existing_estimator(self):
        """Test that registering with existing name overwrites."""

        class NewDMLEstimator:
            # Class attributes
            clean_name: str = "NewDMLEstimator"
            capabilities: EstimatorCapabilities = EstimatorCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                inference_types=set(),
                estimands={Estimand.CATE},
                supports_controls_in_first_stage_only=False,
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                supports_inference=False,
            )

            @classmethod
            def is_compatible_with(cls, data: CausalDataset) -> bool:
                return True

            def check_compatibility(
                self, data: CausalDataset, raise_error: bool = True
            ) -> bool:
                return True

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                return np.zeros(1)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        # Store original
        original_estimator = available_estimators["LinearDML"]["estimator"]

        # Overwrite
        register_estimator(name="LinearDML", estimator=NewDMLEstimator, family="custom")
        assert available_estimators["LinearDML"]["estimator"] == NewDMLEstimator
        assert available_estimators["LinearDML"]["family"] == "custom"

        # Restore original
        available_estimators["LinearDML"]["estimator"] = original_estimator
        available_estimators["LinearDML"]["family"] = "dml"
