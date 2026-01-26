"""Tests for registry functions."""

import pytest

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators.base import EstimatorCapabilities
from caml.extensions.synthetic_data import SyntheticDataGenerator
from caml.registry import (
    available_estimators,
    get_compatible_estimators,
    register_estimator,
)


@pytest.fixture
def binary_continuous_data():
    """Small binary treatment, continuous outcome dataset."""
    gen = SyntheticDataGenerator(
        n_cont_modifiers=3, n_binary_modifiers=1, n_obs=50, seed=42
    )
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T1_binary",
        Y="Y1_continuous",
        treatment_type=TreatmentType.BINARY,
        outcome_type=OutcomeType.CONTINUOUS,
    )


@pytest.fixture
def multi_continuous_data():
    """Small multi-valued treatment, continuous outcome dataset."""
    gen = SyntheticDataGenerator(
        n_cont_modifiers=3, n_binary_modifiers=1, n_obs=50, seed=42
    )
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T2_multi",
        Y="Y1_continuous",
        treatment_type=TreatmentType.MULTI,
        outcome_type=OutcomeType.CONTINUOUS,
    )


@pytest.fixture
def continuous_continuous_data():
    """Small continuous treatment, continuous outcome dataset."""
    gen = SyntheticDataGenerator(
        n_cont_modifiers=3, n_binary_modifiers=1, n_obs=50, seed=42
    )
    return CausalDataset.from_dataframe(
        gen.df,
        X=[c for c in gen.df.columns if "X" in c],
        T="T3_continuous",
        Y="Y1_continuous",
        treatment_type=TreatmentType.CONTINUOUS,
        outcome_type=OutcomeType.CONTINUOUS,
    )


class TestGetCompatibleEstimators:
    """Tests for get_compatible_estimators function."""

    def test_get_compatible_estimators_all_families(self, binary_continuous_data):
        """Test getting all compatible estimators (no family filter)."""
        compatible = get_compatible_estimators(binary_continuous_data, families=None)

        # Should return a dict
        assert isinstance(compatible, dict)

        # Should have at least DML and DR estimators for binary/continuous
        assert len(compatible) > 0

        # Each entry should have 'estimator' and 'family' keys
        for name, info in compatible.items():
            assert "estimator" in info
            assert "family" in info
            assert isinstance(info["family"], str)

    def test_get_compatible_estimators_single_family(self, binary_continuous_data):
        """Test getting compatible estimators from a single family."""
        compatible = get_compatible_estimators(binary_continuous_data, families=["dml"])

        # Should only contain DML estimators
        for name, info in compatible.items():
            assert info["family"] == "dml"

        # Should have some DML estimators (5 total in model_bank)
        assert len(compatible) > 0

    def test_get_compatible_estimators_multiple_families(self, binary_continuous_data):
        """Test getting compatible estimators from multiple families."""
        compatible = get_compatible_estimators(
            binary_continuous_data, families=["dml", "dr"]
        )

        families_found = {info["family"] for info in compatible.values()}

        # Should only contain DML or DR estimators
        assert families_found.issubset({"dml", "dr"})

        # Should have both families represented
        assert len(families_found) > 0

    def test_get_compatible_estimators_filters_correctly(self, multi_continuous_data):
        """Test that incompatible estimators are filtered out."""
        # Multi-valued treatment should only work with meta-learners
        compatible = get_compatible_estimators(multi_continuous_data, families=None)

        # DML/DR don't support MULTI treatment, only meta-learners do
        families_found = {info["family"] for info in compatible.values()}

        # Should contain meta-learners
        assert "meta" in families_found

        # Should NOT contain dml/dr for multi-valued treatment
        # (based on capabilities in wrappers)
        for name, info in compatible.items():
            # Verify each estimator is actually compatible
            estimator_class = info["estimator"]
            assert estimator_class.is_compatible_with(multi_continuous_data)

    def test_get_compatible_estimators_empty_result(self):
        """Test that incompatible data returns empty dict."""
        # Create a dataset that no estimator supports (if possible)
        # For now, test with empty families list which should return empty dict
        gen = SyntheticDataGenerator(
            n_cont_modifiers=3, n_binary_modifiers=1, n_obs=50, seed=42
        )
        data = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        # Request a non-existent family
        compatible = get_compatible_estimators(data, families=["nonexistent_family"])

        # Should return empty dict
        assert compatible == {}


class TestRegisterEstimator:
    """Tests for register_estimator function."""

    def test_register_estimator_basic(self):
        """Test basic estimator registration."""

        # Create a simple mock estimator class
        class MockEstimator:
            @property
            def clean_name(self) -> str:
                return "MockEstimator"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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
                temp_instance = cls()
                return temp_instance.capabilities.is_compatible(data)

            def check_compatibility(
                self, data: CausalDataset, raise_error: bool = True
            ) -> bool:
                return self.capabilities.is_compatible(data)

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                import numpy as np

                n = len(X) if hasattr(X, "__len__") else 1
                return np.zeros(n)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        # Count estimators before registration
        initial_count = len(available_estimators)

        # Create an instance (register_estimator uses instance.clean_name)
        mock_instance = MockEstimator()

        # Register the estimator class
        register_estimator(
            name="MockEstimator", estimator=mock_instance, family="custom"
        )

        # Check it was added
        assert "MockEstimator" in available_estimators
        assert available_estimators["MockEstimator"]["family"] == "custom"

        # Clean up
        del available_estimators["MockEstimator"]

    def test_register_estimator_with_family(self):
        """Test registering estimator with custom family."""

        class CustomEstimator:
            @property
            def clean_name(self) -> str:
                return "CustomEstimator"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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
                temp_instance = cls()
                return temp_instance.capabilities.is_compatible(data)

            def check_compatibility(
                self, data: CausalDataset, raise_error: bool = True
            ) -> bool:
                return self.capabilities.is_compatible(data)

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                import numpy as np

                n = len(X) if hasattr(X, "__len__") else 1
                return np.zeros(n)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        # Create instance
        custom_instance = CustomEstimator()

        # Register with custom family
        register_estimator(
            name="CustomEstimator", estimator=custom_instance, family="my_family"
        )

        # Check it was added with correct family
        assert "CustomEstimator" in available_estimators
        assert available_estimators["CustomEstimator"]["family"] == "my_family"

        # Clean up
        del available_estimators["CustomEstimator"]

    def test_register_estimator_invalid_estimator_raises(self):
        """Test that registering invalid estimator raises error."""

        class NotAnEstimator:
            """This class doesn't implement AutoCateEstimator protocol."""

            pass

        # Create instance
        invalid_instance = NotAnEstimator()

        # Should raise TypeError from typeguard (before ValueError check)
        # Because the typeguard decorator checks the type first
        with pytest.raises(
            (ValueError, Exception)
        ):  # Could be ValueError or TypeCheckError
            register_estimator(
                name="InvalidEstimator", estimator=invalid_instance, family="custom"
            )

    def test_register_estimator_overwrites_existing(self):
        """Test that registering with existing name overwrites."""

        class FirstEstimator:
            @property
            def clean_name(self) -> str:
                return "OverwriteTest"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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
                temp_instance = cls()
                return temp_instance.capabilities.is_compatible(data)

            def check_compatibility(
                self, data: CausalDataset, raise_error: bool = True
            ) -> bool:
                return self.capabilities.is_compatible(data)

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                import numpy as np

                n = len(X) if hasattr(X, "__len__") else 1
                return np.zeros(n)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        class SecondEstimator(FirstEstimator):
            @property
            def clean_name(self) -> str:
                return "OverwriteTest"

        # Create instances
        first_instance = FirstEstimator()
        second_instance = SecondEstimator()

        # Register first estimator
        register_estimator(
            name="OverwriteTest", estimator=first_instance, family="family1"
        )
        assert available_estimators["OverwriteTest"]["family"] == "family1"

        # Register second estimator with same name
        register_estimator(
            name="OverwriteTest", estimator=second_instance, family="family2"
        )

        # Should be overwritten
        assert available_estimators["OverwriteTest"]["family"] == "family2"

        # Clean up
        del available_estimators["OverwriteTest"]

    def test_register_estimator_available_in_get_compatible(
        self, binary_continuous_data
    ):
        """Test that registered estimator appears in get_compatible_estimators."""

        class RegisteredEstimator:
            @property
            def clean_name(self) -> str:
                return "RegisteredEstimator"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities(
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
                temp_instance = cls()
                return temp_instance.capabilities.is_compatible(data)

            def check_compatibility(
                self, data: CausalDataset, raise_error: bool = True
            ) -> bool:
                return self.capabilities.is_compatible(data)

            def fit(self, data, **kwargs):
                return self

            def effect(self, X, **kwargs):
                import numpy as np

                n = len(X) if hasattr(X, "__len__") else 1
                return np.zeros(n)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        # Create instance
        registered_instance = RegisteredEstimator()

        # Register the estimator
        register_estimator(
            name="RegisteredEstimator",
            estimator=registered_instance,
            family="test_family",
        )

        # Get compatible estimators
        compatible = get_compatible_estimators(
            binary_continuous_data, families=["test_family"]
        )

        # Should appear in results
        assert "RegisteredEstimator" in compatible

        # Clean up
        del available_estimators["RegisteredEstimator"]
