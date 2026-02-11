"""Tests for caml/registry/registry.py."""

import numpy as np
import pytest

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators import EstimatorCapabilities
from caml.extensions.synthetic_data import SyntheticDataGenerator
from caml.registry import (
    EstimatorFamily,
    auto_register,
    available_estimators,
    get_compatible_estimators,
    register_estimator,
)

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


@pytest.fixture
def simple_estimator_class():
    """Fixture that creates a simple test estimator class."""

    class SimpleEstimator:
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
            supports_inference=False,
        )

        def __init__(self):
            self.effect_value = None

        @classmethod
        def is_compatible_with(cls, data: CausalDataset) -> bool:
            return cls.capabilities.is_compatible(data)

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

    return SimpleEstimator


class TestAvailableEstimators:
    """Tests for available_estimators dict."""

    def test_available_estimators_populated(self):
        """Test that available_estimators is populated on import."""
        # Should have DML estimators from auto-registration
        assert len(available_estimators) > 0
        assert "LinearDML" in available_estimators

    def test_estimator_structure(self):
        """Test that estimators have correct structure."""
        for name, entry in available_estimators.items():
            assert "estimator" in entry
            assert "family" in entry
            assert isinstance(entry["family"], EstimatorFamily)


class TestGetCompatibleEstimators:
    """Tests for get_compatible_estimators function."""

    def test_returns_dict(self, binary_continuous_dataset):
        """Test that get_compatible_estimators returns a dict."""
        result = get_compatible_estimators(binary_continuous_dataset, families=["dml"])
        assert isinstance(result, dict)

    def test_all_estimators_have_required_keys(self, binary_continuous_dataset):
        """Test that returned estimators have required keys."""
        result = get_compatible_estimators(
            binary_continuous_dataset, families=[EstimatorFamily.DML]
        )
        assert len(result) > 0
        for entry in result.values():
            assert "estimator" in entry
            assert "family" in entry

    def test_filter_by_dml_family(self, binary_continuous_dataset):
        """Test filtering by DML family."""
        result = get_compatible_estimators(
            binary_continuous_dataset, families=[EstimatorFamily.DML]
        )
        assert len(result) > 0
        for name, entry in result.items():
            assert entry["family"] == EstimatorFamily.DML

    def test_filter_by_multiple_families(self, binary_continuous_dataset):
        """Test filtering by multiple families."""
        result = get_compatible_estimators(
            binary_continuous_dataset,
            families=[EstimatorFamily.DML, EstimatorFamily.DR],
        )
        families = {entry["family"] for entry in result.values()}
        assert len(families) >= 1  # At least DML should be present
        assert len(result) > 0

    def test_families_none_includes_all(self, binary_continuous_dataset):
        """Test that families=None includes all families."""
        result_all = get_compatible_estimators(binary_continuous_dataset, families=None)
        result_dml = get_compatible_estimators(
            binary_continuous_dataset, families=[EstimatorFamily.DML]
        )
        # All should include at least the DML estimators
        assert len(result_all) >= len(result_dml)

    def test_families_accepts_strings(self, binary_continuous_dataset):
        """Test that families parameter accepts string values."""
        result = get_compatible_estimators(binary_continuous_dataset, families=["dml"])
        assert len(result) > 0
        for entry in result.values():
            assert entry["family"] == EstimatorFamily.DML

    def test_families_accepts_mixed_enum_and_strings(self, binary_continuous_dataset):
        """Test that families parameter accepts mixed enum and string values."""
        result = get_compatible_estimators(
            binary_continuous_dataset, families=[EstimatorFamily.DML, "dr"]
        )
        assert len(result) > 0


class TestRegisterEstimator:
    """Tests for register_estimator function."""

    def test_register_new_estimator(self, simple_estimator_class):
        """Test registering a new estimator."""
        initial_count = len(available_estimators)
        register_estimator(
            name="TestSimpleEstimator",
            estimator=simple_estimator_class,
            family=EstimatorFamily.CUSTOM,
        )

        assert len(available_estimators) == initial_count + 1
        assert "TestSimpleEstimator" in available_estimators
        assert (
            available_estimators["TestSimpleEstimator"]["family"]
            == EstimatorFamily.CUSTOM
        )
        assert (
            available_estimators["TestSimpleEstimator"]["estimator"]
            == simple_estimator_class
        )

        # Cleanup
        del available_estimators["TestSimpleEstimator"]

    def test_register_with_default_family(self, simple_estimator_class):
        """Test registering estimator with default 'custom' family."""
        register_estimator(name="TestDefaultFamily", estimator=simple_estimator_class)
        assert (
            available_estimators["TestDefaultFamily"]["family"]
            == EstimatorFamily.CUSTOM
        )

        # Cleanup
        del available_estimators["TestDefaultFamily"]

    def test_register_with_string_family(self, simple_estimator_class):
        """Test registering estimator with string family value."""
        register_estimator(
            name="TestStringFamily", estimator=simple_estimator_class, family="custom"
        )
        assert (
            available_estimators["TestStringFamily"]["family"] == EstimatorFamily.CUSTOM
        )

        # Cleanup
        del available_estimators["TestStringFamily"]

    def test_registered_estimator_appears_in_get_compatible(
        self, binary_continuous_dataset, simple_estimator_class
    ):
        """Test that registered estimator appears in get_compatible_estimators."""
        register_estimator(
            name="TestCompatible",
            estimator=simple_estimator_class,
            family=EstimatorFamily.CUSTOM,
        )

        result = get_compatible_estimators(
            binary_continuous_dataset, families=[EstimatorFamily.CUSTOM]
        )
        assert "TestCompatible" in result
        assert result["TestCompatible"]["family"] == EstimatorFamily.CUSTOM

        # Cleanup
        del available_estimators["TestCompatible"]

    def test_overwrite_existing_estimator(self, simple_estimator_class):
        """Test that registering with existing name overwrites."""
        # Store original
        original_entry = available_estimators["LinearDML"].copy()

        # Overwrite
        register_estimator(
            name="LinearDML",
            estimator=simple_estimator_class,
            family=EstimatorFamily.CUSTOM,
        )
        assert available_estimators["LinearDML"]["estimator"] == simple_estimator_class
        assert available_estimators["LinearDML"]["family"] == EstimatorFamily.CUSTOM

        # Restore original
        available_estimators["LinearDML"] = original_entry


class TestAutoRegister:
    """Tests for auto_register decorator."""

    def test_auto_register_with_explicit_name(self):
        """Test auto_register with explicit name."""

        @auto_register(name="ExplicitName", family=EstimatorFamily.CUSTOM)
        class TestEstimator:
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
                supports_inference=False,
            )

            @classmethod
            def is_compatible_with(cls, data):
                return True

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                return np.zeros(len(X))

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        assert "ExplicitName" in available_estimators
        assert available_estimators["ExplicitName"]["estimator"] == TestEstimator
        assert available_estimators["ExplicitName"]["family"] == EstimatorFamily.CUSTOM

        # Cleanup
        del available_estimators["ExplicitName"]

    def test_auto_register_returns_unmodified_class(self):
        """Test that decorator returns the class unmodified."""

        @auto_register(name="UnmodifiedTest", family=EstimatorFamily.CUSTOM)
        class OriginalEstimator:
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
                supports_inference=False,
            )

            @classmethod
            def is_compatible_with(cls, data):
                return True

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                return np.zeros(len(X))

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        # Class should be instantiable normally
        OriginalEstimator()

        # Cleanup
        del available_estimators["UnmodifiedTest"]

    def test_auto_register_with_string_family(self):
        """Test that auto_register accepts string family values."""

        @auto_register(name="StringFamilyTest", family="custom")
        class StringFamilyEstimator:
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
                supports_inference=False,
            )

            @classmethod
            def is_compatible_with(cls, data):
                return True

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                return np.zeros(len(X))

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        assert (
            available_estimators["StringFamilyTest"]["family"] == EstimatorFamily.CUSTOM
        )

        # Cleanup
        del available_estimators["StringFamilyTest"]
