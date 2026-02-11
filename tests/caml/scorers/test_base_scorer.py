"""Tests for caml.scorers.base_scorer module."""

import pytest

from caml.scorers import BaseCateScorerMixin

pytestmark = [pytest.mark.scorers]


class TestBaseCateScorerMixinABC:
    """Test BaseCateScorerMixin abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseCateScorerMixin cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseCateScorerMixin()

    def test_subclass_must_implement_call(self):
        """Test that subclasses must implement __call__."""

        class IncompleteScorer(BaseCateScorerMixin):
            capabilities = True  # dummy value to avoid ABCMeta complaining about missing capabilities
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteScorer()

    def test_subclass_must_have_capabilities(self):
        """Test that subclasses must have capabilities attribute."""
        with pytest.raises(TypeError, match="must define"):

            class NoCapabilitiesScorer(BaseCateScorerMixin):
                def __call__(self, estimator, data) -> float:
                    return 0.0

    def test_valid_subclass_works(self):
        """Test that a properly implemented subclass works."""

        class SimpleScorer(BaseCateScorerMixin):
            capabilities = True  # dummy value to avoid ABCMeta complaining about missing capabilities

            def __call__(self, estimator, data) -> float:
                return 0.0

        scorer = SimpleScorer()
        assert callable(scorer)
        assert scorer(estimator=None, data=None) == 0.0
