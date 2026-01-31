"""Tests for caml.scorers.pehe module."""

import numpy as np
import pytest

from caml.data import CausalDataset, OutcomeType, TreatmentType
from caml.extensions.synthetic_data import SyntheticDataGenerator
from caml.scorers import PEHE

pytestmark = [pytest.mark.scorers]


class TestPEHEInit:
    """Test PEHE initialization."""

    def test_default_init(self):
        """Test default initialization."""
        scorer = PEHE()

        assert scorer.true_cates is None
        assert scorer.normalized is False

    def test_custom_init(self):
        """Test custom initialization with true_cates."""
        true_cates = np.array([1.0, 2.0, 3.0])
        scorer = PEHE(true_cates=true_cates, normalized=True)

        np.testing.assert_array_equal(scorer.true_cates, true_cates)
        assert scorer.normalized is True


class TestPEHEComputation:
    """Test PEHE computation."""

    def test_returns_float(self, fitted_estimator, causal_dataset):
        """Test that scorer returns a float."""
        scorer = PEHE()

        result = scorer(fitted_estimator, causal_dataset)

        assert isinstance(result, float)

    def test_unnormalized_is_positive(self, fitted_estimator, causal_dataset):
        """Test that unnormalized PEHE is non-negative (MSE)."""
        scorer = PEHE(normalized=False)

        result = scorer(fitted_estimator, causal_dataset)

        assert result >= 0

    def test_normalized_bounded_above(self, fitted_estimator, causal_dataset):
        """Test that normalized PEHE is bounded above by 1."""
        scorer = PEHE(normalized=True)

        result = scorer(fitted_estimator, causal_dataset)

        assert result <= 1.0

    def test_uses_data_true_cates_by_default(self, fitted_estimator, causal_dataset):
        """Test that scorer uses data.true_cates when not provided."""
        scorer = PEHE()

        # Should not raise since causal_dataset has true_cates
        result = scorer(fitted_estimator, causal_dataset)

        assert isinstance(result, float)

    def test_uses_provided_true_cates(self, fitted_estimator, causal_dataset):
        """Test that scorer uses provided true_cates."""
        # Provide different true_cates
        custom_cates = np.zeros_like(causal_dataset.true_cates)
        scorer = PEHE(true_cates=custom_cates)

        result = scorer(fitted_estimator, causal_dataset)

        assert isinstance(result, float)


class TestPEHEValidation:
    """Test PEHE validation and error handling."""

    def test_raises_without_true_cates(self, fitted_estimator):
        """Test that scorer raises error when true_cates unavailable."""
        # Create dataset without true_cates
        gen = SyntheticDataGenerator(
            n_obs=100,
            n_cont_modifiers=3,
            n_binary_treatments=1,
            n_cont_outcomes=1,
            seed=42,
        )
        data_no_cates = CausalDataset.from_dataframe(
            gen.df,
            X=[c for c in gen.df.columns if "X" in c],
            T="T1_binary",
            Y="Y1_continuous",
            treatment_type=TreatmentType.BINARY,
            outcome_type=OutcomeType.CONTINUOUS,
            # No true_cates provided
        )

        scorer = PEHE()

        with pytest.raises(ValueError, match="requires true CATEs"):
            scorer(fitted_estimator, data_no_cates)


class TestPEHEPerformance:
    """Test PEHE produces reasonable scores for well-fitted estimators."""

    def test_normalized_positive_for_good_estimator(
        self, fitted_estimator, causal_dataset
    ):
        """Test that normalized PEHE is positive for a well-fitted estimator.

        A positive normalized PEHE (R^2-like) indicates the estimator explains more
        variance in true CATEs than a naive baseline, which should hold for a
        correctly specified DML estimator on linear synthetic data.
        """
        scorer = PEHE(normalized=True)

        result = scorer(fitted_estimator, causal_dataset)

        # Well-fitted estimator on linear data should achieve positive R^2-like score
        assert result > 0.0, f"Expected positive normalized PEHE, got {result}"

    def test_normalized_exceeds_threshold(self, fitted_estimator, causal_dataset):
        """Test that normalized PEHE exceeds a reasonable threshold.

        For well-specified linear data with LinearDML, we expect the normalized
        PEHE to be substantially positive (>0.5), indicating good CATE recovery
        against ground truth.
        """
        scorer = PEHE(normalized=True)

        result = scorer(fitted_estimator, causal_dataset)

        # Higher threshold for PEHE since we compare against ground truth directly
        assert result > 0.5, f"Expected normalized PEHE > 0.5, got {result}"

    def test_unnormalized_low_for_good_estimator(
        self, fitted_estimator, causal_dataset
    ):
        """Test that unnormalized PEHE (MSE) is reasonably low.

        The raw PEHE should be small relative to the variance of true CATEs
        for a well-fitted estimator.
        """
        scorer = PEHE(normalized=False)

        result = scorer(fitted_estimator, causal_dataset)
        true_cate_var = np.var(causal_dataset.true_cates)

        # PEHE should be less than the variance of true CATEs (i.e., better than predicting mean)
        assert (
            result < true_cate_var
        ), f"Expected PEHE ({result}) < true CATE variance ({true_cate_var})"


class TestPEHEShapeHandling:
    """Test PEHE shape validation and error handling."""

    def test_handles_2d_estimator_output(self, causal_dataset):
        """Test that scorer handles 2D (n, 1) estimator output."""

        class MockEstimator2D:
            """Mock estimator returning 2D array."""

            def effect(self, X):
                return np.zeros((len(X), 1))

        scorer = PEHE()
        result = scorer(MockEstimator2D(), causal_dataset)

        assert isinstance(result, float)

    def test_raises_on_wrong_n_samples(self, causal_dataset):
        """Test that scorer raises error for wrong sample count."""

        class MockEstimatorWrongSamples:
            """Mock estimator returning wrong number of samples."""

            def effect(self, X):
                return np.zeros(10)  # Wrong number

        scorer = PEHE()

        with pytest.raises(ValueError, match="samples"):
            scorer(MockEstimatorWrongSamples(), causal_dataset)

    def test_raises_on_multi_column_output(self, causal_dataset):
        """Test that scorer raises error for multi-column estimator output."""

        class MockEstimatorMultiColumn:
            """Mock estimator returning multiple columns."""

            def effect(self, X):
                return np.zeros((len(X), 3))

        scorer = PEHE()

        with pytest.raises(ValueError, match="columns"):
            scorer(MockEstimatorMultiColumn(), causal_dataset)
