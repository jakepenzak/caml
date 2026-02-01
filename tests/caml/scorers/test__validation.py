"""Tests for caml.scorers._validation module."""

import numpy as np
import pytest

from caml.scorers._validation import (
    _clip,
    _validate_cate_array,
    _validate_scorer_inputs,
)

pytestmark = [pytest.mark.scorers]


class TestClipFunction:
    """Test the clip utility function."""

    def test_clip_lower_bound(self):
        """Test that values below lower bound are clipped."""
        arr = np.array([0.001, 0.005, 0.01, 0.5])
        result = _clip(arr, lb=0.01)

        assert result[0] == 0.01
        assert result[1] == 0.01
        assert result[2] == 0.01
        assert result[3] == 0.5

    def test_clip_upper_bound(self):
        """Test that values above upper bound are clipped."""
        arr = np.array([0.5, 0.9, 0.99, 1.0])
        result = _clip(arr, lb=0.01, ub=0.95)

        assert result[0] == 0.5
        assert result[1] == 0.9
        assert result[2] == 0.95
        assert result[3] == 0.95

    def test_clip_default_bounds(self):
        """Test default bounds (lb=0.01, ub=inf)."""
        arr = np.array([0.0, 0.01, 0.5, 100.0])
        result = _clip(arr)

        assert result[0] == 0.01
        assert result[1] == 0.01
        assert result[2] == 0.5
        assert result[3] == 100.0  # No upper bound by default


class TestValidateCateArray:
    """Test validate_cate_array utility function."""

    def test_1d_array_passes_through(self):
        """Test that 1D array with correct samples passes through."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _validate_cate_array(arr, n_samples=5, name="test")

        np.testing.assert_array_equal(result, arr)
        assert result.shape == (5,)

    def test_2d_single_column_flattened(self):
        """Test that 2D array with single column is flattened."""
        arr = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = _validate_cate_array(arr, n_samples=5, name="test")

        assert result.shape == (5,)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_wrong_n_samples_raises(self):
        """Test that wrong number of samples raises error."""
        arr = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="has 3 samples, expected 5"):
            _validate_cate_array(arr, n_samples=5, name="test_array")

    def test_2d_multiple_columns_raises(self):
        """Test that 2D array with multiple columns raises error."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        with pytest.raises(ValueError, match="2 columns"):
            _validate_cate_array(arr, n_samples=3, name="test_array")

    def test_3d_array_raises(self):
        """Test that 3D array raises error."""
        arr = np.ones((5, 1, 1))

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            _validate_cate_array(arr, n_samples=5, name="test_array")

    def test_custom_name_in_error_message(self):
        """Test that custom name appears in error messages."""
        arr = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="tau_hat"):
            _validate_cate_array(arr, n_samples=10, name="tau_hat")


class TestValidateScorerInputs:
    """Test validate_scorer_inputs utility function."""

    def test_both_1d_same_shape(self):
        """Test that two 1D arrays with same shape pass through."""
        tau_hat = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.1, 2.1, 3.1])

        tau_result, ref_result = _validate_scorer_inputs(tau_hat, reference)

        np.testing.assert_array_equal(tau_result, tau_hat)
        np.testing.assert_array_equal(ref_result, reference)

    def test_mixed_shapes_aligned(self):
        """Test that 2D (n,1) and 1D (n,) arrays are aligned."""
        tau_hat = np.array([[1.0], [2.0], [3.0]])  # Shape (3, 1)
        reference = np.array([1.1, 2.1, 3.1])  # Shape (3,)

        tau_result, ref_result = _validate_scorer_inputs(tau_hat, reference)

        assert tau_result.shape == (3,)
        assert ref_result.shape == (3,)

    def test_both_2d_single_column_flattened(self):
        """Test that both 2D (n,1) arrays are flattened."""
        tau_hat = np.array([[1.0], [2.0], [3.0]])
        reference = np.array([[1.1], [2.1], [3.1]])

        tau_result, ref_result = _validate_scorer_inputs(tau_hat, reference)

        assert tau_result.shape == (3,)
        assert ref_result.shape == (3,)

    def test_different_n_samples_raises(self):
        """Test that different sample counts raise error."""
        tau_hat = np.array([1.0, 2.0, 3.0])  # 3 samples
        reference = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # 5 samples

        with pytest.raises(ValueError, match="has 3 samples, expected 5"):
            _validate_scorer_inputs(tau_hat, reference)

    def test_custom_names_in_error_messages(self):
        """Test that custom names appear in error messages."""
        tau_hat = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        with pytest.raises(ValueError, match="my_tau"):
            _validate_scorer_inputs(tau_hat, reference, tau_name="my_tau")

    def test_reference_multiple_columns_raises(self):
        """Test that reference with multiple columns raises error."""
        tau_hat = np.array([1.0, 2.0, 3.0])
        reference = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        with pytest.raises(ValueError, match="invalid shape"):
            _validate_scorer_inputs(tau_hat, reference)
