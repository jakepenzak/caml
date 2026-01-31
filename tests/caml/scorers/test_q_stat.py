"""Tests for caml.scorers.q_stat module."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from caml.scorers import QStat

pytestmark = [pytest.mark.scorers]


class TestQStatInit:
    """Test QStat initialization."""

    def test_default_init(self):
        """Test default initialization."""
        scorer = QStat(treatment_model=LogisticRegression())

        assert scorer.cv == 3
        assert scorer.random_state is None

    def test_custom_init(self):
        """Test custom initialization."""
        scorer = QStat(
            treatment_model=LogisticRegression(),
            cv=5,
            random_state=42,
        )

        assert scorer.cv == 5
        assert scorer.random_state == 42


class TestQStatComputation:
    """Test QStat computation."""

    def test_returns_float(self, fitted_estimator, causal_dataset):
        """Test that scorer returns a float."""
        scorer = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )

        result = scorer(fitted_estimator, causal_dataset)

        assert isinstance(result, float)

    def test_lower_is_better(self, fitted_estimator, causal_dataset):
        """Test that Q-stat can be negative (good) or positive (degenerate)."""
        scorer = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )

        result = scorer(fitted_estimator, causal_dataset)

        # Q-stat equals PEHE minus a constant, so finite is expected
        assert np.isfinite(result)

    def test_deterministic_with_random_state(self, fitted_estimator, causal_dataset):
        """Test that same random_state produces same results."""
        scorer1 = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )
        scorer2 = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )

        result1 = scorer1(fitted_estimator, causal_dataset)
        result2 = scorer2(fitted_estimator, causal_dataset)

        np.testing.assert_almost_equal(result1, result2)


class TestQStatPerformance:
    """Test QStat produces expected negative values for well-fitted estimators."""

    def test_negative_for_good_estimator(self, fitted_estimator, causal_dataset):
        """Test that Q-stat is negative for a well-fitted estimator.

        Q-stat equals PEHE minus a non-negative constant, so a good estimator
        with low PEHE should produce a negative Q-stat. This indicates the
        estimator is capturing heterogeneity rather than overfitting.
        """
        scorer = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )

        result = scorer(fitted_estimator, causal_dataset)

        # Well-fitted estimator should have negative Q-stat
        assert result < 0.0, f"Expected negative Q-stat, got {result}"

    def test_substantially_negative(self, fitted_estimator, causal_dataset):
        """Test that Q-stat is substantially negative for good heterogeneity detection.

        For a well-specified LinearDML on linear synthetic data with true
        heterogeneous effects, Q-stat should be clearly negative (< -1.0),
        indicating strong evidence of captured treatment heterogeneity.
        """
        scorer = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )

        result = scorer(fitted_estimator, causal_dataset)

        # Should be substantially negative for well-fitted estimator
        assert result < -1.0, f"Expected Q-stat < -1.0, got {result}"


class TestQStatShapeHandling:
    """Test QStat shape validation and error handling."""

    def test_handles_2d_estimator_output(self, causal_dataset):
        """Test that scorer handles 2D (n, 1) estimator output."""

        class MockEstimator2D:
            """Mock estimator returning 2D array."""

            def effect(self, X):
                return np.zeros((len(X), 1))

        scorer = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )
        result = scorer(MockEstimator2D(), causal_dataset)

        assert isinstance(result, float)

    def test_raises_on_wrong_n_samples(self, causal_dataset):
        """Test that scorer raises error for wrong sample count."""

        class MockEstimatorWrongSamples:
            """Mock estimator returning wrong number of samples."""

            def effect(self, X):
                return np.zeros(10)  # Wrong number

        scorer = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )

        with pytest.raises(ValueError, match="samples"):
            scorer(MockEstimatorWrongSamples(), causal_dataset)

    def test_raises_on_multi_column_output(self, causal_dataset):
        """Test that scorer raises error for multi-column estimator output."""

        class MockEstimatorMultiColumn:
            """Mock estimator returning multiple columns."""

            def effect(self, X):
                return np.zeros((len(X), 3))

        scorer = QStat(
            treatment_model=LogisticRegression(),
            cv=3,
            random_state=42,
        )

        with pytest.raises(ValueError, match="columns"):
            scorer(MockEstimatorMultiColumn(), causal_dataset)
