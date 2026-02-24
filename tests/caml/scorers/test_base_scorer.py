"""Tests for caml.scorers.base_scorer module."""

import numpy as np
import pytest

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.scorers import BaseCateScorerMixin, CateScorer, ScorerCapabilities

pytestmark = [pytest.mark.scorers]


# ==============================================================================
# SCORER CAPABILITIES TESTS
# ==============================================================================


class TestScorerCapabilities:
    """Test ScorerCapabilities dataclass."""

    def test_create_capabilities(self):
        """Test creating ScorerCapabilities with all fields."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
            requires_oracle_cates=False,
            greater_is_better=False,
            supports_weights=False,
        )

        assert TreatmentType.BINARY in capabilities.treatment_types
        assert OutcomeType.CONTINUOUS in capabilities.outcome_types
        assert capabilities.requires_treatment_model is True
        assert capabilities.requires_outcome_model is True
        assert capabilities.requires_regression_model is False
        assert capabilities.requires_oracle_cates is False
        assert capabilities.greater_is_better is False
        assert capabilities.supports_weights is False

    def test_capabilities_are_immutable(self):
        """Test that ScorerCapabilities is frozen/immutable."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            capabilities.requires_treatment_model = True

    def test_default_values(self):
        """Test that optional fields default correctly."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
        )

        assert capabilities.requires_oracle_cates is False
        assert capabilities.greater_is_better is False
        assert capabilities.supports_weights is False

    def test_multiple_treatment_types(self):
        """Test capabilities with multiple treatment types."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
        )

        assert len(capabilities.treatment_types) == 2
        assert TreatmentType.BINARY in capabilities.treatment_types
        assert TreatmentType.CONTINUOUS in capabilities.treatment_types

    def test_multiple_outcome_types(self):
        """Test capabilities with multiple outcome types."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.BINARY, OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
        )

        assert len(capabilities.outcome_types) == 2
        assert OutcomeType.BINARY in capabilities.outcome_types
        assert OutcomeType.CONTINUOUS in capabilities.outcome_types

    def test_greater_is_better_true(self):
        """Test capabilities with greater_is_better=True (e.g., R²-style metric)."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            greater_is_better=True,
        )

        assert capabilities.greater_is_better is True

    def test_requires_oracle_cates(self):
        """Test capabilities that require oracle CATEs (simulation studies)."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
            requires_oracle_cates=True,
        )

        assert capabilities.requires_oracle_cates is True

    def test_all_nuisance_model_flags(self):
        """Test capabilities requiring all nuisance model types."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=True,
        )

        assert capabilities.requires_treatment_model is True
        assert capabilities.requires_outcome_model is True
        assert capabilities.requires_regression_model is True


# ==============================================================================
# COMPATIBILITY CHECKING TESTS
# ==============================================================================


class TestScorerCompatibilityChecking:
    """Test ScorerCapabilities.is_compatible() method."""

    def test_compatible_binary_continuous(self):
        """Test compatibility with binary treatment and continuous outcome."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=True,
            requires_outcome_model=True,
            requires_regression_model=False,
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert capabilities.is_compatible(data) is True

    def test_incompatible_treatment_type(self):
        """Test incompatibility due to treatment type mismatch."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.randn(100),  # Continuous treatment
            Y=np.random.randn(100),
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert capabilities.is_compatible(data) is False

    def test_incompatible_outcome_type(self):
        """Test incompatibility due to outcome type mismatch."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.binomial(1, 0.5, 100),  # Binary outcome
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.BINARY,
        )

        assert capabilities.is_compatible(data) is False

    def test_compatible_multiple_types(self):
        """Test compatibility when scorer supports multiple treatment/outcome types."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY, TreatmentType.CONTINUOUS},
            outcome_types={OutcomeType.BINARY, OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
        )

        np.random.seed(42)
        # Binary treatment, continuous outcome
        data1 = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )
        assert capabilities.is_compatible(data1) is True

        # Continuous treatment, binary outcome
        data2 = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.randn(100),
            Y=np.random.binomial(1, 0.5, 100),
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.BINARY,
        )
        assert capabilities.is_compatible(data2) is True

    def test_incompatible_both_types(self):
        """Test incompatibility when both treatment and outcome types are unsupported."""
        capabilities = ScorerCapabilities(
            treatment_types={TreatmentType.BINARY},
            outcome_types={OutcomeType.CONTINUOUS},
            requires_treatment_model=False,
            requires_outcome_model=False,
            requires_regression_model=False,
        )

        np.random.seed(42)
        data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.randn(100),
            Y=np.random.binomial(1, 0.5, 100),
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.BINARY,
        )

        assert capabilities.is_compatible(data) is False


# ==============================================================================
# PROTOCOL TESTS
# ==============================================================================


class TestCateScorerProtocol:
    """Test CateScorer protocol."""

    def test_simple_scorer_implements_protocol(self):
        """Test that a simple scorer satisfying the interface implements the protocol."""

        class SimpleScorer:
            capabilities: ScorerCapabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            @classmethod
            def is_compatible_with(cls, data: CausalDataset) -> bool:
                return cls.capabilities.is_compatible(data)

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        scorer = SimpleScorer()
        assert isinstance(scorer, CateScorer)

    def test_incomplete_scorer_does_not_implement_protocol(self):
        """Test that an object missing required attributes doesn't satisfy CateScorer."""

        class IncompleteScorer:
            pass

        scorer = IncompleteScorer()
        assert not isinstance(scorer, CateScorer)

    def test_scorer_missing_call_does_not_implement_protocol(self):
        """Test that an object with capabilities but no __call__ doesn't satisfy CateScorer."""

        class NoCallScorer:
            capabilities: ScorerCapabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            @classmethod
            def is_compatible_with(cls, data: CausalDataset) -> bool:
                return cls.capabilities.is_compatible(data)

        scorer = NoCallScorer()
        assert not isinstance(scorer, CateScorer)

    def test_base_cate_scorer_mixin_subclass_implements_protocol(self):
        """Test that a valid BaseCateScorerMixin subclass satisfies CateScorer protocol."""

        class ConcreteScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __init__(self):
                pass

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        scorer = ConcreteScorer()
        assert isinstance(scorer, CateScorer)
        assert isinstance(scorer, BaseCateScorerMixin)


# ==============================================================================
# BASE CATE SCORER MIXIN ABC TESTS
# ==============================================================================


class TestBaseCateScorerMixinABC:
    """Test BaseCateScorerMixin abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseCateScorerMixin cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseCateScorerMixin()

    def test_subclass_must_implement_call(self):
        """Test that subclasses must implement __call__."""

        class IncompleteScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __init__(self):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteScorer()

    def test_subclass_must_implement_init(self):
        """Test that subclasses must implement __init__."""

        class IncompleteScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        with pytest.raises(TypeError, match="abstract"):
            IncompleteScorer()

    def test_subclass_must_have_capabilities(self):
        """Test that non-abstract subclasses must define capabilities."""
        with pytest.raises(TypeError, match="must define"):

            class NoCapabilitiesScorer(BaseCateScorerMixin):
                def __init__(self):
                    pass

                def __call__(self, estimator, data: CausalDataset) -> float:
                    return 0.0

    def test_valid_subclass_can_be_created(self):
        """Test that a properly implemented subclass can be instantiated."""

        class ConcreteScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __init__(self):
                pass

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        scorer = ConcreteScorer()
        assert hasattr(scorer, "capabilities")
        assert callable(scorer)

    def test_valid_subclass_call_returns_float(self):
        """Test that a valid subclass __call__ returns the expected value."""

        class ConcreteScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __init__(self):
                pass

            def __call__(self, estimator, data: CausalDataset) -> float:
                return -1.23

        scorer = ConcreteScorer()
        result = scorer(estimator=None, data=None)
        assert result == -1.23

    def test_is_compatible_with_uses_capabilities(self):
        """Test that is_compatible_with delegates to capabilities.is_compatible."""

        class TestScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __init__(self):
                pass

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        np.random.seed(42)
        compatible_data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.binomial(1, 0.5, 100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        incompatible_data = CausalDataset(
            X=np.random.randn(100, 3),
            T=np.random.randn(100),
            Y=np.random.randn(100),
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        assert TestScorer.is_compatible_with(compatible_data) is True
        assert TestScorer.is_compatible_with(incompatible_data) is False

    def test_requires_treatment_model_param_in_init(self):
        """Test that requires_treatment_model=True enforces treatment_model init param."""
        with pytest.raises(TypeError, match="treatment_model"):

            class MissingTreatmentModel(BaseCateScorerMixin):
                capabilities = ScorerCapabilities(
                    treatment_types={TreatmentType.BINARY},
                    outcome_types={OutcomeType.CONTINUOUS},
                    requires_treatment_model=True,
                    requires_outcome_model=False,
                    requires_regression_model=False,
                )

                def __init__(self):  # Missing treatment_model parameter
                    pass

                def __call__(self, estimator, data: CausalDataset) -> float:
                    return 0.0

    def test_requires_outcome_model_param_in_init(self):
        """Test that requires_outcome_model=True enforces outcome_model init param."""
        with pytest.raises(TypeError, match="outcome_model"):

            class MissingOutcomeModel(BaseCateScorerMixin):
                capabilities = ScorerCapabilities(
                    treatment_types={TreatmentType.BINARY},
                    outcome_types={OutcomeType.CONTINUOUS},
                    requires_treatment_model=False,
                    requires_outcome_model=True,
                    requires_regression_model=False,
                )

                def __init__(self):  # Missing outcome_model parameter
                    pass

                def __call__(self, estimator, data: CausalDataset) -> float:
                    return 0.0

    def test_requires_regression_model_param_in_init(self):
        """Test that requires_regression_model=True enforces regression_model init param."""
        with pytest.raises(TypeError, match="regression_model"):

            class MissingRegressionModel(BaseCateScorerMixin):
                capabilities = ScorerCapabilities(
                    treatment_types={TreatmentType.BINARY},
                    outcome_types={OutcomeType.CONTINUOUS},
                    requires_treatment_model=False,
                    requires_outcome_model=False,
                    requires_regression_model=True,
                )

                def __init__(self):  # Missing regression_model parameter
                    pass

                def __call__(self, estimator, data: CausalDataset) -> float:
                    return 0.0

    def test_scorer_with_all_nuisance_params_works(self):
        """Test that a scorer requiring all nuisance models is valid when params present."""

        class FullNuisanceScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=True,
                requires_outcome_model=True,
                requires_regression_model=True,
            )

            def __init__(self, treatment_model, outcome_model, regression_model):
                self.treatment_model = treatment_model
                self.outcome_model = outcome_model
                self.regression_model = regression_model

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        scorer = FullNuisanceScorer(
            treatment_model="tm", outcome_model="om", regression_model="rm"
        )
        assert scorer.treatment_model == "tm"
        assert scorer.outcome_model == "om"
        assert scorer.regression_model == "rm"

    def test_abstract_intermediate_class_skips_capabilities_check(self):
        """Test that abstract intermediate classes need not define capabilities."""
        from abc import abstractmethod

        # Abstract intermediate without capabilities — should not raise
        class IntermediateScorer(BaseCateScorerMixin):
            @abstractmethod
            def score_type(self) -> str: ...

        # Concrete child must still define capabilities explicitly
        with pytest.raises(TypeError, match="must define"):

            class MissingCaps(IntermediateScorer):
                def __init__(self):
                    pass

                def __call__(self, estimator, data: CausalDataset) -> float:
                    return 0.0

                def score_type(self) -> str:
                    return "test"

    def test_abstract_intermediate_class_with_capabilities_works(self):
        """Test that an abstract intermediate class that defines capabilities is valid,
        and that a concrete child that inherits (but does not redeclare) capabilities
        still must redeclare them in its own __dict__.
        """
        from abc import abstractmethod

        # Abstract intermediate with capabilities defined — should not raise
        class IntermediateScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            @abstractmethod
            def score_type(self) -> str: ...

        # Concrete child that redeclares capabilities works fine
        class ConcreteChild(IntermediateScorer):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __init__(self):
                pass

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

            def score_type(self) -> str:
                return "test"

        scorer = ConcreteChild()
        assert isinstance(scorer, BaseCateScorerMixin)
        assert scorer.score_type() == "test"

    def test_concrete_subclass_cannot_inherit_capabilities(self):
        """Test that a concrete subclass cannot satisfy the capabilities requirement
        by inheriting it — each concrete class must redeclare in its own __dict__.
        """
        from abc import abstractmethod

        class IntermediateScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            @abstractmethod
            def score_type(self) -> str: ...

        with pytest.raises(TypeError, match="must define"):

            class ConcreteChildMissingCaps(IntermediateScorer):
                def __init__(self):
                    pass

                def __call__(self, estimator, data: CausalDataset) -> float:
                    return 0.0

                def score_type(self) -> str:
                    return "test"  # capabilities not in __dict__ → TypeError

    def test_is_compatible_with_available_as_class_method(self):
        """Test that is_compatible_with can be called on the class without instantiation."""

        class TestScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __init__(self):
                pass

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        np.random.seed(0)
        data = CausalDataset(
            X=np.random.randn(50, 2),
            T=np.random.binomial(1, 0.5, 50),
            Y=np.random.randn(50),
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
        )

        # Called on the class, not an instance
        assert TestScorer.is_compatible_with(data) is True

    def test_scorer_with_treatment_model_param_works(self):
        """Test valid scorer requiring only a treatment model."""

        class TreatmentOnlyScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=True,
                requires_outcome_model=False,
                requires_regression_model=False,
            )

            def __init__(self, treatment_model):
                self.treatment_model = treatment_model

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        scorer = TreatmentOnlyScorer(treatment_model="some_model")
        assert scorer.treatment_model == "some_model"
        assert isinstance(scorer, BaseCateScorerMixin)

    def test_scorer_with_outcome_model_param_works(self):
        """Test valid scorer requiring only an outcome model."""

        class OutcomeOnlyScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=True,
                requires_regression_model=False,
            )

            def __init__(self, outcome_model):
                self.outcome_model = outcome_model

            def __call__(self, estimator, data: CausalDataset) -> float:
                return 0.0

        scorer = OutcomeOnlyScorer(outcome_model="some_model")
        assert scorer.outcome_model == "some_model"

    def test_scorer_with_oracle_cates_no_nuisance(self):
        """Test oracle-based scorer that requires no nuisance models."""

        class OracleScorer(BaseCateScorerMixin):
            capabilities = ScorerCapabilities(
                treatment_types={TreatmentType.BINARY},
                outcome_types={OutcomeType.CONTINUOUS},
                requires_treatment_model=False,
                requires_outcome_model=False,
                requires_regression_model=False,
                requires_oracle_cates=True,
                greater_is_better=False,
            )

            def __init__(self):
                pass

            def __call__(self, estimator, data: CausalDataset) -> float:
                tau_hat = estimator.effect(data.X)
                return float(np.mean(np.abs(tau_hat - data.true_cates)))

        scorer = OracleScorer()
        assert scorer.capabilities.requires_oracle_cates is True
        assert scorer.capabilities.greater_is_better is False
        assert isinstance(scorer, BaseCateScorerMixin)
        assert isinstance(scorer, CateScorer)
