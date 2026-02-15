"""Tests for caml/registry/registry.py."""

import numpy as np
import pytest

from caml.data import CausalDataset, Estimand, OutcomeType, TreatmentType
from caml.estimators import EstimatorCapabilities
from caml.extensions.synthetic_data import SyntheticDataGenerator
from caml.registry import (
    AVAILABLE_CATE_ESTIMATORS,
    AVAILABLE_CATE_SCORERS,
    EstimatorFamily,
    ScorerFamily,
    auto_register,
    get_compatible_estimators,
    get_compatible_scorers,
    register_estimator,
    register_scorer,
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
    """Tests for AVAILABLE_CATE_ESTIMATORS dict."""

    def test_AVAILABLE_CATE_ESTIMATORS_populated(self):
        """Test that AVAILABLE_CATE_ESTIMATORS is populated on import."""
        # Should have DML estimators from auto-registration
        assert len(AVAILABLE_CATE_ESTIMATORS) > 0
        assert "LinearDML" in AVAILABLE_CATE_ESTIMATORS

    def test_estimator_structure(self):
        """Test that estimators have correct structure."""
        for name, entry in AVAILABLE_CATE_ESTIMATORS.items():
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
        initial_count = len(AVAILABLE_CATE_ESTIMATORS)
        register_estimator(
            name="TestSimpleEstimator",
            estimator=simple_estimator_class,
            family=EstimatorFamily.CUSTOM,
        )

        assert len(AVAILABLE_CATE_ESTIMATORS) == initial_count + 1
        assert "TestSimpleEstimator" in AVAILABLE_CATE_ESTIMATORS
        assert (
            AVAILABLE_CATE_ESTIMATORS["TestSimpleEstimator"]["family"]
            == EstimatorFamily.CUSTOM
        )
        assert (
            AVAILABLE_CATE_ESTIMATORS["TestSimpleEstimator"]["estimator"]
            == simple_estimator_class
        )

        # Cleanup
        del AVAILABLE_CATE_ESTIMATORS["TestSimpleEstimator"]

    def test_register_with_default_family(self, simple_estimator_class):
        """Test registering estimator with default 'custom' family."""
        register_estimator(name="TestDefaultFamily", estimator=simple_estimator_class)
        assert (
            AVAILABLE_CATE_ESTIMATORS["TestDefaultFamily"]["family"]
            == EstimatorFamily.CUSTOM
        )

        # Cleanup
        del AVAILABLE_CATE_ESTIMATORS["TestDefaultFamily"]

    def test_register_with_string_family(self, simple_estimator_class):
        """Test registering estimator with string family value."""
        register_estimator(
            name="TestStringFamily", estimator=simple_estimator_class, family="custom"
        )
        assert (
            AVAILABLE_CATE_ESTIMATORS["TestStringFamily"]["family"]
            == EstimatorFamily.CUSTOM
        )

        # Cleanup
        del AVAILABLE_CATE_ESTIMATORS["TestStringFamily"]

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
        del AVAILABLE_CATE_ESTIMATORS["TestCompatible"]

    def test_overwrite_existing_estimator(self, simple_estimator_class):
        """Test that registering with existing name overwrites."""
        # Store original
        original_entry = AVAILABLE_CATE_ESTIMATORS["LinearDML"].copy()

        # Overwrite
        register_estimator(
            name="LinearDML",
            estimator=simple_estimator_class,
            family=EstimatorFamily.CUSTOM,
        )
        assert (
            AVAILABLE_CATE_ESTIMATORS["LinearDML"]["estimator"]
            == simple_estimator_class
        )
        assert (
            AVAILABLE_CATE_ESTIMATORS["LinearDML"]["family"] == EstimatorFamily.CUSTOM
        )

        # Restore original
        AVAILABLE_CATE_ESTIMATORS["LinearDML"] = original_entry


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

        assert "ExplicitName" in AVAILABLE_CATE_ESTIMATORS
        assert AVAILABLE_CATE_ESTIMATORS["ExplicitName"]["estimator"] == TestEstimator
        assert (
            AVAILABLE_CATE_ESTIMATORS["ExplicitName"]["family"]
            == EstimatorFamily.CUSTOM
        )

        # Cleanup
        del AVAILABLE_CATE_ESTIMATORS["ExplicitName"]

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
        del AVAILABLE_CATE_ESTIMATORS["UnmodifiedTest"]

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
            AVAILABLE_CATE_ESTIMATORS["StringFamilyTest"]["family"]
            == EstimatorFamily.CUSTOM
        )

        # Cleanup
        del AVAILABLE_CATE_ESTIMATORS["StringFamilyTest"]


# ==============================================================================
# SCORER REGISTRY TESTS
# ==============================================================================


class TestRegisterScorer:
    """Tests for register_scorer function."""

    def test_register_new_scorer(self):
        """Test registering a new scorer."""
        from caml.scorers import BaseCateScorerMixin, ScorerCapabilities

        class TestScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                requires_oracle_cates=False,
            )

            def __call__(self, estimator, data):
                return 0.0

        initial_count = len(AVAILABLE_CATE_SCORERS)
        register_scorer(
            name="TestScorer", scorer=TestScorer, family=ScorerFamily.CUSTOM
        )

        assert len(AVAILABLE_CATE_SCORERS) == initial_count + 1
        assert "TestScorer" in AVAILABLE_CATE_SCORERS
        assert AVAILABLE_CATE_SCORERS["TestScorer"]["family"] == ScorerFamily.CUSTOM
        assert AVAILABLE_CATE_SCORERS["TestScorer"]["scorer"] == TestScorer

        # Cleanup
        del AVAILABLE_CATE_SCORERS["TestScorer"]

    def test_register_scorer_with_string_family(self):
        """Test registering scorer with string family value."""
        from caml.scorers import BaseCateScorerMixin, ScorerCapabilities

        class TestScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                requires_oracle_cates=False,
            )

            def __call__(self, estimator, data):
                return 0.0

        register_scorer(name="TestStringScorer", scorer=TestScorer, family="custom")
        assert (
            AVAILABLE_CATE_SCORERS["TestStringScorer"]["family"] == ScorerFamily.CUSTOM
        )

        # Cleanup
        del AVAILABLE_CATE_SCORERS["TestStringScorer"]


class TestGetCompatibleScorers:
    """Tests for get_compatible_scorers function."""

    def test_returns_dict(self, binary_continuous_dataset):
        """Test that get_compatible_scorers returns a dict."""
        result = get_compatible_scorers(binary_continuous_dataset)
        assert isinstance(result, dict)

    def test_filter_by_family(self, binary_continuous_dataset):
        """Test filtering scorers by family."""
        # Register a test scorer
        from caml.scorers import BaseCateScorerMixin, ScorerCapabilities

        class TestScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                requires_oracle_cates=False,
            )

            def __call__(self, estimator, data):
                return 0.0

        register_scorer(
            name="TestFamilyScorer", scorer=TestScorer, family=ScorerFamily.CUSTOM
        )

        result = get_compatible_scorers(
            binary_continuous_dataset, families=[ScorerFamily.CUSTOM]
        )
        # Should at least include our test scorer if compatible
        assert isinstance(result, dict)

        # Cleanup
        del AVAILABLE_CATE_SCORERS["TestFamilyScorer"]

    def test_families_accepts_strings(self, binary_continuous_dataset):
        """Test that families parameter accepts string values."""
        result = get_compatible_scorers(
            binary_continuous_dataset, families=["pseudo_outcome"]
        )
        assert isinstance(result, dict)


class TestAutoRegisterScorer:
    """Tests for auto_register decorator with scorers."""

    def test_auto_register_scorer(self):
        """Test auto_register for scorers with is_estimator=False."""
        from caml.scorers import BaseCateScorerMixin, ScorerCapabilities

        @auto_register(
            name="AutoRegisteredScorer", family=ScorerFamily.CUSTOM, is_estimator=False
        )
        class TestAutoScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                requires_oracle_cates=False,
            )

            def __call__(self, estimator, data):
                return 0.0

        assert "AutoRegisteredScorer" in AVAILABLE_CATE_SCORERS
        assert (
            AVAILABLE_CATE_SCORERS["AutoRegisteredScorer"]["scorer"] == TestAutoScorer
        )
        assert (
            AVAILABLE_CATE_SCORERS["AutoRegisteredScorer"]["family"]
            == ScorerFamily.CUSTOM
        )

        # Cleanup
        del AVAILABLE_CATE_SCORERS["AutoRegisteredScorer"]

    def test_auto_register_scorer_with_string_family(self):
        """Test auto_register for scorers with string family."""
        from caml.scorers import BaseCateScorerMixin, ScorerCapabilities

        @auto_register(name="StringFamilyScorer", family="custom", is_estimator=False)
        class TestStringScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                supports_weights=False,
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                requires_oracle_cates=False,
            )

            def __call__(self, estimator, data):
                return 0.0

        assert (
            AVAILABLE_CATE_SCORERS["StringFamilyScorer"]["family"]
            == ScorerFamily.CUSTOM
        )

        # Cleanup
        del AVAILABLE_CATE_SCORERS["StringFamilyScorer"]


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


class TestAutoRegisterErrorHandling:
    """Test error handling in auto_register decorator."""

    def test_estimator_with_invalid_family_type_raises(self):
        """Test that invalid family type for estimator raises error."""
        with pytest.raises(ValueError, match="Estimator family must be"):

            @auto_register(name="BadEstimator", family=ScorerFamily.CUSTOM)
            class BadEstimator:
                pass

    def test_scorer_with_invalid_family_type_raises(self):
        """Test that invalid family type for scorer raises error."""
        with pytest.raises(ValueError, match="Scorer family must be"):

            @auto_register(
                name="BadScorer", family=EstimatorFamily.DML, is_estimator=False
            )
            class BadScorer:
                pass
