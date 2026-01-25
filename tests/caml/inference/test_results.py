"""Tests for caml.inference.results."""

import numpy as np

from caml.inference.inference_schema import InferenceType
from caml.inference.results import InferenceResult


class TestInferenceResultCreation:
    """Tests for creating InferenceResult instances."""

    def test_minimal_creation_scalar(self):
        """Test creating InferenceResult with minimal scalar data."""
        result = InferenceResult(effect=1.5)

        assert result.effect == 1.5
        assert result.stderr is None
        assert result.method is None

    def test_minimal_creation_array(self):
        """Test creating InferenceResult with minimal array data."""
        effect = np.array([1.0, 2.0, 3.0])
        result = InferenceResult(effect=effect)

        np.testing.assert_array_equal(result.effect, effect)
        assert result.stderr is None

    def test_creation_with_stderr_scalar(self):
        """Test creating InferenceResult with scalar stderr."""
        result = InferenceResult(effect=1.5, stderr=0.3)

        assert result.effect == 1.5
        assert result.stderr == 0.3

    def test_creation_with_stderr_array(self):
        """Test creating InferenceResult with array stderr."""
        effect = np.array([1.0, 2.0, 3.0])
        stderr = np.array([0.1, 0.2, 0.3])
        result = InferenceResult(effect=effect, stderr=stderr)

        np.testing.assert_array_equal(result.stderr, stderr)

    def test_creation_with_method_analytic(self):
        """Test creating InferenceResult with analytic method."""
        result = InferenceResult(effect=1.5, stderr=0.3, method=InferenceType.ANALYTIC)

        assert result.method == InferenceType.ANALYTIC

    def test_creation_with_method_bootstrap(self):
        """Test creating InferenceResult with bootstrap method."""
        result = InferenceResult(
            effect=1.5,
            stderr=0.3,
            method=InferenceType.BOOTSTRAP,
        )

        assert result.method == InferenceType.BOOTSTRAP

    def test_full_creation_scalar(self):
        """Test creating InferenceResult with all parameters (scalar)."""
        result = InferenceResult(
            effect=1.5,
            stderr=0.3,
            method=InferenceType.BOOTSTRAP,
        )

        assert result.effect == 1.5
        assert result.stderr == 0.3
        assert result.method == InferenceType.BOOTSTRAP

    def test_full_creation_array(self):
        """Test creating InferenceResult with all parameters (array)."""
        effect = np.array([1.0, 2.0, 3.0])
        stderr = np.array([0.1, 0.2, 0.3])

        result = InferenceResult(
            effect=effect,
            stderr=stderr,
            method=InferenceType.ANALYTIC,
        )

        np.testing.assert_array_equal(result.effect, effect)
        np.testing.assert_array_equal(result.stderr, stderr)


class TestInferenceResultLen:
    """Tests for InferenceResult.__len__() method."""

    def test_len_scalar(self):
        """Test __len__ with scalar point estimate."""
        result = InferenceResult(effect=1.5)
        assert len(result) == 1

    def test_len_array(self):
        """Test __len__ with array point estimate."""
        effect = np.array([1.0, 2.0, 3.0, 4.0])
        result = InferenceResult(effect=effect)
        assert len(result) == 4

    def test_len_empty_array(self):
        """Test __len__ with empty array."""
        effect = np.array([])
        result = InferenceResult(effect=effect)
        assert len(result) == 0

    def test_len_single_element_array(self):
        """Test __len__ with single-element array."""
        effect = np.array([1.5])
        result = InferenceResult(effect=effect)
        assert len(result) == 1


class TestInferenceResultRepr:
    """Tests for InferenceResult.__repr__() method."""

    def test_repr_minimal_scalar(self):
        """Test __repr__ with minimal scalar data."""
        result = InferenceResult(effect=1.5)
        repr_str = repr(result)

        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str
        assert "method" not in repr_str or "auto" in repr_str

    def test_repr_minimal_array(self):
        """Test __repr__ with minimal array data."""
        effect = np.array([1.0, 2.0, 3.0])
        result = InferenceResult(effect=effect)
        repr_str = repr(result)

        assert "InferenceResult" in repr_str
        assert "n=3" in repr_str

    def test_repr_with_stderr(self):
        """Test __repr__ with stderr provided."""
        result = InferenceResult(effect=1.5, stderr=0.3, method=InferenceType.ANALYTIC)
        repr_str = repr(result)

        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str
        assert "method=analytic" in repr_str

    def test_repr_with_bootstrap(self):
        """Test __repr__ with bootstrap method."""
        result = InferenceResult(effect=1.5, stderr=0.3, method=InferenceType.BOOTSTRAP)
        repr_str = repr(result)

        assert "method=bootstrap" in repr_str

    def test_repr_without_method(self):
        """Test __repr__ without method specified but with stderr."""
        result = InferenceResult(effect=1.5, stderr=0.3)
        repr_str = repr(result)

        assert "InferenceResult" in repr_str
        assert "n=1" in repr_str
        assert "method=auto" in repr_str


class TestInferenceResultEdgeCases:
    """Tests for edge cases in InferenceResult."""

    def test_negative_effect(self):
        """Test InferenceResult with negative point estimate."""
        result = InferenceResult(effect=-1.5)
        assert result.effect == -1.5

    def test_zero_effect(self):
        """Test InferenceResult with zero point estimate."""
        result = InferenceResult(effect=0.0)
        assert result.effect == 0.0

    def test_large_array(self):
        """Test InferenceResult with large array."""
        effect = np.random.randn(10000)
        result = InferenceResult(effect=effect)
        assert len(result) == 10000

    def test_alpha_out_of_range(self):
        """Test InferenceResult allows alpha outside [0, 1] (no validation)."""

    def test_negative_n_bootstrap(self):
        """Test InferenceResult allows negative n_bootstrap (no validation)."""


class TestInferenceResultDataclass:
    """Tests for InferenceResult as a dataclass."""

    def test_is_dataclass(self):
        """Test InferenceResult is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(InferenceResult)

    def test_equality(self):
        """Test InferenceResult equality."""
        result1 = InferenceResult(effect=1.5, stderr=0.3)
        result2 = InferenceResult(effect=1.5, stderr=0.3)

        assert result1 == result2

    def test_inequality(self):
        """Test InferenceResult inequality."""
        result1 = InferenceResult(effect=1.5, stderr=0.3)
        result2 = InferenceResult(effect=1.5, stderr=0.4)

        assert result1 != result2

    def test_mutable(self):
        """Test InferenceResult is mutable (not frozen)."""
        result = InferenceResult(effect=1.5)
        result.stderr = 0.3  # Should work since not frozen
        assert result.stderr == 0.3
