"""Tests for caml.scorers.dr_loss module."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from caml.scorers import DRLoss

pytestmark = [pytest.mark.scorers]


class TestDRLossInit:
    """Test DRLoss initialization."""

    def test_default_init(self):
        """Test default initialization."""
        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
        )

        assert scorer.cv == 3
        assert scorer.random_state is None
        assert scorer.normalized is False

    def test_custom_init(self):
        """Test custom initialization."""
        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=5,
            random_state=42,
            normalized=True,
        )

        assert scorer.cv == 5
        assert scorer.random_state == 42
        assert scorer.normalized is True


class TestDRLossComputation:
    """Test DRLoss computation."""

    def test_returns_float(self, fitted_estimator, causal_dataset):
        """Test that scorer returns a float."""
        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=3,
            random_state=42,
        )

        result = scorer(fitted_estimator, causal_dataset)

        assert isinstance(result, float)

    def test_unnormalized_is_positive(self, fitted_estimator, causal_dataset):
        """Test that unnormalized DR-loss is non-negative (MSE)."""
        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=3,
            random_state=42,
            normalized=False,
        )

        result = scorer(fitted_estimator, causal_dataset)

        assert result >= 0

    def test_normalized_bounded_above(self, fitted_estimator, causal_dataset):
        """Test that normalized DR-loss is bounded above by 1."""
        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=3,
            random_state=42,
            normalized=True,
        )

        result = scorer(fitted_estimator, causal_dataset)

        assert result <= 1.0

    def test_deterministic_with_random_state(self, fitted_estimator, causal_dataset):
        """Test that same random_state produces same results."""
        scorer1 = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=3,
            random_state=42,
        )
        scorer2 = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=3,
            random_state=42,
        )

        result1 = scorer1(fitted_estimator, causal_dataset)
        result2 = scorer2(fitted_estimator, causal_dataset)

        np.testing.assert_almost_equal(result1, result2)


class TestDRLossPerformance:
    """Test DRLoss produces reasonable scores for well-fitted estimators."""

    def test_normalized_positive_for_good_estimator(
        self, fitted_estimator, causal_dataset
    ):
        """Test that normalized DR-loss is positive for a well-fitted estimator.

        A positive normalized DR-loss indicates the estimator explains more variance
        than a naive baseline, which should hold for a correctly specified DML
        estimator on linear synthetic data.
        """
        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=3,
            random_state=42,
            normalized=True,
        )

        result = scorer(fitted_estimator, causal_dataset)

        # Well-fitted estimator on linear data should achieve positive R^2-like score
        assert result > 0.0, f"Expected positive normalized DR-loss, got {result}"

    def test_normalized_exceeds_threshold(self, fitted_estimator, causal_dataset):
        """Test that normalized DR-loss exceeds a reasonable threshold.

        For well-specified linear data with LinearDML, we expect the normalized
        DR-loss to be substantially positive (>0.3), indicating good CATE recovery.
        """
        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=3,
            random_state=42,
            normalized=True,
        )

        result = scorer(fitted_estimator, causal_dataset)

        # Conservative threshold - well-fitted estimator should exceed this
        assert result > 0.3, f"Expected normalized DR-loss > 0.3, got {result}"


class TestDRLossShapeHandling:
    """Test DRLoss shape validation and error handling."""

    def test_handles_2d_estimator_output(self, causal_dataset):
        """Test that scorer handles 2D (n, 1) estimator output."""

        class MockEstimator2D:
            """Mock estimator returning 2D array."""

            def effect(self, X):
                return np.zeros((len(X), 1))

        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
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

        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
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

        scorer = DRLoss(
            treatment_model=LogisticRegression(),
            regression_model=LinearRegression(),
            cv=3,
            random_state=42,
        )

        with pytest.raises(ValueError, match="columns"):
            scorer(MockEstimatorMultiColumn(), causal_dataset)
