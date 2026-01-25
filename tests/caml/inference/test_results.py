"""Tests for caml.inference.results."""

import numpy as np

from caml.inference.inference_schema import InferenceType
from caml.inference.results import InferenceResult


class TestInferenceResultCreation:
    """Tests for creating InferenceResult instances."""

    def test_minimal_creation_scalar(self):
        """Test creating InferenceResult with minimal scalar data."""
        result = InferenceResult(point_estimate=1.5)

        assert result.point_estimate == 1.5
        assert result.stderr is None
        assert result.ci_lower is None
        assert result.ci_upper is None
        assert result.alpha == 0.05
        assert result.method is None
        assert result.n_bootstrap is None

    def test_minimal_creation_array(self):
        """Test creating InferenceResult with minimal array data."""
        point_estimate = np.array([1.0, 2.0, 3.0])
        result = InferenceResult(point_estimate=point_estimate)

        np.testing.assert_array_equal(result.point_estimate, point_estimate)
        assert result.stderr is None

    def test_creation_with_stderr_scalar(self):
        """Test creating InferenceResult with scalar stderr."""
        result = InferenceResult(point_estimate=1.5, stderr=0.3)

        assert result.point_estimate == 1.5
        assert result.stderr == 0.3

    def test_creation_with_stderr_array(self):
        """Test creating InferenceResult with array stderr."""
        point_estimate = np.array([1.0, 2.0, 3.0])
        stderr = np.array([0.1, 0.2, 0.3])
        result = InferenceResult(point_estimate=point_estimate, stderr=stderr)

        np.testing.assert_array_equal(result.stderr, stderr)

    def test_creation_with_confidence_intervals_scalar(self):
        """Test creating InferenceResult with scalar confidence intervals."""
        result = InferenceResult(
            point_estimate=1.5, ci_lower=1.0, ci_upper=2.0, alpha=0.05
        )

        assert result.ci_lower == 1.0
        assert result.ci_upper == 2.0
        assert result.alpha == 0.05

    def test_creation_with_confidence_intervals_array(self):
        """Test creating InferenceResult with array confidence intervals."""
        point_estimate = np.array([1.0, 2.0, 3.0])
        ci_lower = np.array([0.5, 1.5, 2.5])
        ci_upper = np.array([1.5, 2.5, 3.5])
        result = InferenceResult(
            point_estimate=point_estimate, ci_lower=ci_lower, ci_upper=ci_upper
        )

        np.testing.assert_array_equal(result.ci_lower, ci_lower)
        np.testing.assert_array_equal(result.ci_upper, ci_upper)

    def test_creation_with_method_analytic(self):
        """Test creating InferenceResult with analytic method."""
        result = InferenceResult(
            point_estimate=1.5, stderr=0.3, method=InferenceType.ANALYTIC
        )

        assert result.method == InferenceType.ANALYTIC

    def test_creation_with_method_bootstrap(self):
        """Test creating InferenceResult with bootstrap method."""
        result = InferenceResult(
            point_estimate=1.5,
            stderr=0.3,
            method=InferenceType.BOOTSTRAP,
            n_bootstrap=1000,
        )

        assert result.method == InferenceType.BOOTSTRAP
        assert result.n_bootstrap == 1000

    def test_creation_with_custom_alpha(self):
        """Test creating InferenceResult with custom alpha level."""
        result = InferenceResult(point_estimate=1.5, alpha=0.01)

        assert result.alpha == 0.01

    def test_full_creation_scalar(self):
        """Test creating InferenceResult with all parameters (scalar)."""
        result = InferenceResult(
            point_estimate=1.5,
            stderr=0.3,
            ci_lower=1.0,
            ci_upper=2.0,
            alpha=0.05,
            method=InferenceType.BOOTSTRAP,
            n_bootstrap=1000,
        )

        assert result.point_estimate == 1.5
        assert result.stderr == 0.3
        assert result.ci_lower == 1.0
        assert result.ci_upper == 2.0
        assert result.alpha == 0.05
        assert result.method == InferenceType.BOOTSTRAP
        assert result.n_bootstrap == 1000

    def test_full_creation_array(self):
        """Test creating InferenceResult with all parameters (array)."""
        point_estimate = np.array([1.0, 2.0, 3.0])
        stderr = np.array([0.1, 0.2, 0.3])
        ci_lower = np.array([0.8, 1.6, 2.4])
        ci_upper = np.array([1.2, 2.4, 3.6])

        result = InferenceResult(
            point_estimate=point_estimate,
            stderr=stderr,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alpha=0.05,
            method=InferenceType.ANALYTIC,
        )

        np.testing.assert_array_equal(result.point_estimate, point_estimate)
        np.testing.assert_array_equal(result.stderr, stderr)
        np.testing.assert_array_equal(result.ci_lower, ci_lower)
        np.testing.assert_array_equal(result.ci_upper, ci_upper)


class TestInferenceResultLen:
    """Tests for InferenceResult.__len__() method."""

    def test_len_scalar(self):
        """Test __len__ with scalar point estimate."""
        result = InferenceResult(point_estimate=1.5)
        assert len(result) == 1

    def test_len_array(self):
        """Test __len__ with array point estimate."""
        point_estimate = np.array([1.0, 2.0, 3.0, 4.0])
        result = InferenceResult(point_estimate=point_estimate)
        assert len(result) == 4

    def test_len_empty_array(self):
        """Test __len__ with empty array."""
        point_estimate = np.array([])
        result = InferenceResult(point_estimate=point_estimate)
        assert len(result) == 0

    def test_len_single_element_array(self):
        """Test __len__ with single-element array."""
        point_estimate = np.array([1.5])
        result = InferenceResult(point_estimate=point_estimate)
        assert len(result) == 1


class TestInferenceResultRepr:
    """Tests for InferenceResult.__repr__() method."""

    def test_repr_minimal_scalar(self):
        """Test __repr__ with minimal scalar data."""
        result = InferenceResult(point_estimate=1.5)
        repr_str = repr(result)

        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str
        assert "method" not in repr_str or "unknown" in repr_str

    def test_repr_minimal_array(self):
        """Test __repr__ with minimal array data."""
        point_estimate = np.array([1.0, 2.0, 3.0])
        result = InferenceResult(point_estimate=point_estimate)
        repr_str = repr(result)

        assert "InferenceResult" in repr_str
        assert "n=3" in repr_str

    def test_repr_with_stderr(self):
        """Test __repr__ with stderr provided."""
        result = InferenceResult(
            point_estimate=1.5, stderr=0.3, method=InferenceType.ANALYTIC
        )
        repr_str = repr(result)

        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str
        assert "method=analytic" in repr_str

    def test_repr_with_bootstrap(self):
        """Test __repr__ with bootstrap method."""
        result = InferenceResult(
            point_estimate=1.5, stderr=0.3, method=InferenceType.BOOTSTRAP
        )
        repr_str = repr(result)

        assert "method=bootstrap" in repr_str

    def test_repr_without_method(self):
        """Test __repr__ without method specified but with stderr."""
        result = InferenceResult(point_estimate=1.5, stderr=0.3)
        repr_str = repr(result)

        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str


class TestInferenceResultEdgeCases:
    """Tests for edge cases in InferenceResult."""

    def test_negative_point_estimate(self):
        """Test InferenceResult with negative point estimate."""
        result = InferenceResult(point_estimate=-1.5)
        assert result.point_estimate == -1.5

    def test_zero_point_estimate(self):
        """Test InferenceResult with zero point estimate."""
        result = InferenceResult(point_estimate=0.0)
        assert result.point_estimate == 0.0

    def test_large_array(self):
        """Test InferenceResult with large array."""
        point_estimate = np.random.randn(10000)
        result = InferenceResult(point_estimate=point_estimate)
        assert len(result) == 10000

    def test_ci_lower_greater_than_upper(self):
        """Test InferenceResult allows invalid CI bounds (no validation)."""
        result = InferenceResult(
            point_estimate=1.5,
            ci_lower=2.0,
            ci_upper=1.0,  # Invalid order
        )
        # No validation in the dataclass, so this should work
        assert result.ci_lower == 2.0
        assert result.ci_upper == 1.0

    def test_alpha_out_of_range(self):
        """Test InferenceResult allows alpha outside [0, 1] (no validation)."""
        result = InferenceResult(point_estimate=1.5, alpha=1.5)
        assert result.alpha == 1.5

    def test_negative_n_bootstrap(self):
        """Test InferenceResult allows negative n_bootstrap (no validation)."""
        result = InferenceResult(point_estimate=1.5, n_bootstrap=-100)
        assert result.n_bootstrap == -100


class TestInferenceResultDataclass:
    """Tests for InferenceResult as a dataclass."""

    def test_is_dataclass(self):
        """Test InferenceResult is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(InferenceResult)

    def test_equality(self):
        """Test InferenceResult equality."""
        result1 = InferenceResult(point_estimate=1.5, stderr=0.3)
        result2 = InferenceResult(point_estimate=1.5, stderr=0.3)

        assert result1 == result2

    def test_inequality(self):
        """Test InferenceResult inequality."""
        result1 = InferenceResult(point_estimate=1.5, stderr=0.3)
        result2 = InferenceResult(point_estimate=1.5, stderr=0.4)

        assert result1 != result2

    def test_mutable(self):
        """Test InferenceResult is mutable (not frozen)."""
        result = InferenceResult(point_estimate=1.5)
        result.stderr = 0.3  # Should work since not frozen
        assert result.stderr == 0.3
