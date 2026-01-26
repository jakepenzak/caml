"""Tests for caml.inference.results module."""

import numpy as np
import pytest

from caml.inference import InferenceResult, InferenceType

pytestmark = pytest.mark.inference


# ==============================================================================
# CREATION TESTS
# ==============================================================================


class TestInferenceResultCreation:
    """Test InferenceResult creation and initialization."""

    def test_create_scalar_result(self):
        """Test creation with scalar effect."""
        result = InferenceResult(effect=2.5, stderr=0.3, method=InferenceType.ANALYTIC)

        assert result.effect == 2.5
        assert result.stderr == 0.3
        assert result.method == InferenceType.ANALYTIC

    def test_create_array_result(self):
        """Test creation with array effect."""
        np.random.seed(42)
        effects = np.random.randn(100)
        stderrs = np.random.uniform(0.1, 0.5, 100)

        result = InferenceResult(
            effect=effects, stderr=stderrs, method=InferenceType.BOOTSTRAP
        )

        assert isinstance(result.effect, np.ndarray)
        assert len(result.effect) == 100
        assert isinstance(result.stderr, np.ndarray)
        assert len(result.stderr) == 100
        assert result.method == InferenceType.BOOTSTRAP

    def test_create_without_stderr(self):
        """Test creation without standard errors."""
        result = InferenceResult(effect=3.0, method=InferenceType.ANALYTIC)

        assert result.effect == 3.0
        assert result.stderr is None
        assert result.method == InferenceType.ANALYTIC

    def test_create_without_method(self):
        """Test creation without inference method."""
        result = InferenceResult(effect=2.0, stderr=0.5)

        assert result.effect == 2.0
        assert result.stderr == 0.5
        assert result.method is None

    def test_create_minimal(self):
        """Test creation with only effect."""
        result = InferenceResult(effect=1.5)

        assert result.effect == 1.5
        assert result.stderr is None
        assert result.method is None


# ==============================================================================
# LENGTH TESTS
# ==============================================================================


class TestInferenceResultLength:
    """Test InferenceResult __len__ method."""

    def test_len_scalar_effect(self):
        """Test length of scalar result."""
        result = InferenceResult(effect=2.5)
        assert len(result) == 1

    def test_len_array_effect(self):
        """Test length of array result."""
        result = InferenceResult(effect=np.random.randn(50))
        assert len(result) == 50

    def test_len_single_element_array(self):
        """Test length of single-element array."""
        result = InferenceResult(effect=np.array([2.5]))
        assert len(result) == 1

    def test_len_empty_array(self):
        """Test length of empty array."""
        result = InferenceResult(effect=np.array([]))
        assert len(result) == 0

    def test_len_large_array(self):
        """Test length of large array."""
        result = InferenceResult(effect=np.random.randn(10000))
        assert len(result) == 10000


# ==============================================================================
# REPR TESTS
# ==============================================================================


class TestInferenceResultRepr:
    """Test InferenceResult __repr__ method."""

    def test_repr_scalar_with_stderr_and_method(self):
        """Test repr with all components."""
        result = InferenceResult(effect=2.5, stderr=0.3, method=InferenceType.ANALYTIC)

        repr_str = repr(result)
        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str
        assert "method=analytic" in repr_str

    def test_repr_array_with_stderr_and_method(self):
        """Test repr with array and all components."""
        result = InferenceResult(
            effect=np.random.randn(100),
            stderr=np.random.randn(100),
            method=InferenceType.BOOTSTRAP,
        )

        repr_str = repr(result)
        assert "InferenceResult" in repr_str
        assert "n=100" in repr_str
        assert "method=bootstrap" in repr_str

    def test_repr_without_stderr(self):
        """Test repr without standard errors."""
        result = InferenceResult(effect=2.5)

        repr_str = repr(result)
        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str
        assert "method" not in repr_str

    def test_repr_without_method(self):
        """Test repr without inference method."""
        result = InferenceResult(effect=np.random.randn(50), stderr=np.random.randn(50))

        repr_str = repr(result)
        assert "InferenceResult" in repr_str
        assert "n=50" in repr_str
        assert "method=auto" in repr_str

    def test_repr_minimal(self):
        """Test repr with minimal result."""
        result = InferenceResult(effect=1.0)

        repr_str = repr(result)
        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str


# ==============================================================================
# INFERENCE TYPE TESTS
# ==============================================================================


class TestInferenceTypes:
    """Test different inference types."""

    def test_analytic_inference(self):
        """Test result with analytic inference."""
        result = InferenceResult(effect=2.0, stderr=0.4, method=InferenceType.ANALYTIC)

        assert result.method == InferenceType.ANALYTIC
        assert result.method.value == "analytic"

    def test_bootstrap_inference(self):
        """Test result with bootstrap inference."""
        result = InferenceResult(
            effect=np.random.randn(100),
            stderr=np.random.uniform(0.1, 0.5, 100),
            method=InferenceType.BOOTSTRAP,
        )

        assert result.method == InferenceType.BOOTSTRAP
        assert result.method.value == "bootstrap"


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_effect(self):
        """Test result with zero effect."""
        result = InferenceResult(effect=0.0, stderr=0.1)
        assert result.effect == 0.0

    def test_negative_effect(self):
        """Test result with negative effect."""
        result = InferenceResult(effect=-2.5, stderr=0.3)
        assert result.effect == -2.5

    def test_zero_stderr(self):
        """Test result with zero standard error."""
        result = InferenceResult(effect=2.0, stderr=0.0)
        assert result.stderr == 0.0

    def test_large_stderr(self):
        """Test result with large standard error."""
        result = InferenceResult(effect=1.0, stderr=100.0)
        assert result.stderr == 100.0

    def test_array_all_zeros(self):
        """Test result with all-zero effects."""
        result = InferenceResult(effect=np.zeros(50))
        assert np.all(result.effect == 0)

    def test_array_with_nan(self):
        """Test result with NaN values."""
        effects = np.array([1.0, np.nan, 3.0])
        result = InferenceResult(effect=effects)
        assert np.isnan(result.effect[1])

    def test_array_with_inf(self):
        """Test result with infinite values."""
        effects = np.array([1.0, np.inf, 3.0])
        result = InferenceResult(effect=effects)
        assert np.isinf(result.effect[1])

    def test_mismatched_effect_stderr_shapes(self):
        """Test that mismatched shapes are allowed (no validation)."""
        # Note: The class doesn't validate shape matching
        result = InferenceResult(
            effect=np.random.randn(100),
            stderr=np.random.randn(50),  # Different size
        )
        # Should not raise, but would be user error
        assert len(result.effect) == 100
        assert len(result.stderr) == 50


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Test integration with typical use cases."""

    def test_ate_result(self):
        """Test typical ATE result."""
        result = InferenceResult(effect=2.5, stderr=0.3, method=InferenceType.ANALYTIC)

        assert len(result) == 1
        assert isinstance(result.effect, float)
        assert isinstance(result.stderr, float)

    def test_cate_result(self):
        """Test typical CATE result."""
        np.random.seed(42)
        n_obs = 1000
        result = InferenceResult(
            effect=np.random.randn(n_obs) * 2 + 3,
            stderr=np.random.uniform(0.1, 0.5, n_obs),
            method=InferenceType.BOOTSTRAP,
        )

        assert len(result) == n_obs
        assert isinstance(result.effect, np.ndarray)
        assert isinstance(result.stderr, np.ndarray)

    def test_result_without_inference(self):
        """Test result without uncertainty quantification."""
        result = InferenceResult(effect=np.random.randn(100))

        assert len(result) == 100
        assert result.stderr is None
        assert result.method is None
        assert "InferenceResult" in repr(result)

    def test_converting_scalar_to_array(self):
        """Test behavior when converting scalar to array."""
        result = InferenceResult(effect=2.5)
        # Can wrap in array manually
        effect_array = np.array([result.effect])
        assert len(effect_array) == 1
        assert effect_array[0] == 2.5

    def test_multiple_results_comparison(self):
        """Test comparing multiple results."""
        result1 = InferenceResult(effect=2.5, stderr=0.3, method=InferenceType.ANALYTIC)
        result2 = InferenceResult(
            effect=3.0, stderr=0.4, method=InferenceType.BOOTSTRAP
        )

        assert result1.effect < result2.effect
        assert result1.stderr < result2.stderr
        assert result1.method != result2.method
