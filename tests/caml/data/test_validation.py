"""Tests for caml.data._validation functions."""

import numpy as np
import pandas as pd
import pytest

from caml.data._validation import (
    _check_1d_like,
    _check_length,
    _count_missing,
    _is_numeric,
    _n_obs,
    _unique_non_null,
    check_1d_targets,
    check_missing_data,
    check_outcome_type_matches_data,
    check_shapes_match,
    check_treatment_type_matches_data,
)
from caml.data.data_schema import OutcomeType, TreatmentType


class TestNObs:
    """Tests for _n_obs helper function."""

    def test_numpy_array(self):
        """Test _n_obs with numpy array."""
        arr = np.array([1, 2, 3])
        assert _n_obs(arr) == 3

    def test_pandas_series(self):
        """Test _n_obs with pandas Series."""
        s = pd.Series([1, 2, 3, 4])
        assert _n_obs(s) == 4

    def test_pandas_dataframe(self):
        """Test _n_obs with pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert _n_obs(df) == 2

    def test_list(self):
        """Test _n_obs with list."""
        lst = [1, 2, 3, 4, 5]
        assert _n_obs(lst) == 5

    def test_none_raises(self):
        """Test _n_obs with None raises ValueError."""
        with pytest.raises(ValueError, match="Expected array-like input, got None"):
            _n_obs(None)


class TestCheckLength:
    """Tests for _check_length helper function."""

    def test_matching_length(self):
        """Test _check_length with matching length."""
        arr = np.array([1, 2, 3])
        _check_length("test", arr, 3)  # Should not raise

    def test_mismatched_length_raises(self):
        """Test _check_length with mismatched length raises ValueError."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="test must have length 5, got 3"):
            _check_length("test", arr, 5)

    def test_none_passes(self):
        """Test _check_length with None passes."""
        _check_length("test", None, 10)  # Should not raise


class TestCheck1dLike:
    """Tests for _check_1d_like helper function."""

    def test_pandas_series(self):
        """Test _check_1d_like with pandas Series."""
        s = pd.Series([1, 2, 3])
        _check_1d_like("test", s)  # Should not raise

    def test_pandas_dataframe_single_column(self):
        """Test _check_1d_like with single-column DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        _check_1d_like("test", df)  # Should not raise

    def test_pandas_dataframe_multiple_columns_raises(self):
        """Test _check_1d_like with multi-column DataFrame raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(
            ValueError, match="test must be 1-dimensional, got 2 columns"
        ):
            _check_1d_like("test", df)

    def test_numpy_1d_array(self):
        """Test _check_1d_like with 1D numpy array."""
        arr = np.array([1, 2, 3])
        _check_1d_like("test", arr)  # Should not raise

    def test_numpy_2d_array_single_column(self):
        """Test _check_1d_like with 2D numpy array with single column."""
        arr = np.array([[1], [2], [3]])
        _check_1d_like("test", arr)  # Should not raise

    def test_numpy_2d_array_multiple_columns_raises(self):
        """Test _check_1d_like with 2D numpy array with multiple columns raises."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(ValueError, match="test must be 1-dimensional, got shape"):
            _check_1d_like("test", arr)

    def test_list_raises(self):
        """Test _check_1d_like with list raises TypeError."""
        with pytest.raises(
            TypeError, match="test must be a pandas or numpy array-like"
        ):
            _check_1d_like("test", [1, 2, 3])


class TestCountMissing:
    """Tests for _count_missing helper function."""

    def test_numpy_no_missing(self):
        """Test _count_missing with numpy array without missing values."""
        arr = np.array([1, 2, 3, 4])
        assert _count_missing(arr) == 0

    def test_numpy_with_missing(self):
        """Test _count_missing with numpy array with NaN values."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        assert _count_missing(arr) == 2

    def test_pandas_series_no_missing(self):
        """Test _count_missing with pandas Series without missing values."""
        s = pd.Series([1, 2, 3])
        assert _count_missing(s) == 0

    def test_pandas_series_with_missing(self):
        """Test _count_missing with pandas Series with NaN values."""
        s = pd.Series([1.0, np.nan, 3.0, None, 5.0])
        assert _count_missing(s) == 2

    def test_pandas_dataframe_with_missing(self):
        """Test _count_missing with pandas DataFrame with missing values."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
        assert _count_missing(df) == 2

    def test_none(self):
        """Test _count_missing with None returns 0."""
        assert _count_missing(None) == 0


class TestUniqueNonNull:
    """Tests for _unique_non_null helper function."""

    def test_numpy_array(self):
        """Test _unique_non_null with numpy array."""
        arr = np.array([1, 2, 2, 3, 3, 3])
        result = _unique_non_null(arr)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_numpy_with_nan(self):
        """Test _unique_non_null excludes NaN values."""
        arr = np.array([1.0, np.nan, 2.0, 2.0, np.nan, 3.0])
        result = _unique_non_null(arr)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_pandas_series(self):
        """Test _unique_non_null with pandas Series."""
        s = pd.Series([1, 2, 2, 3, 3, 3])
        result = _unique_non_null(s)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_pandas_series_with_nan(self):
        """Test _unique_non_null with pandas Series with NaN."""
        s = pd.Series([1.0, np.nan, 2.0, None, 3.0])
        result = _unique_non_null(s)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_pandas_dataframe(self):
        """Test _unique_non_null with pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [2, 3]})
        result = _unique_non_null(df)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_all_nan_returns_empty(self):
        """Test _unique_non_null with all NaN returns empty array."""
        arr = np.array([np.nan, np.nan, np.nan])
        result = _unique_non_null(arr)
        assert result.size == 0


class TestIsNumeric:
    """Tests for _is_numeric helper function."""

    def test_numpy_numeric_array(self):
        """Test _is_numeric with numeric numpy array."""
        arr = np.array([1, 2, 3])
        assert _is_numeric(arr) is True

    def test_numpy_float_array(self):
        """Test _is_numeric with float numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        assert _is_numeric(arr) is True

    def test_numpy_string_array(self):
        """Test _is_numeric with string numpy array."""
        arr = np.array(["a", "b", "c"])
        assert _is_numeric(arr) is False

    def test_pandas_numeric_series(self):
        """Test _is_numeric with numeric pandas Series."""
        s = pd.Series([1, 2, 3])
        assert _is_numeric(s) is True

    def test_pandas_float_series(self):
        """Test _is_numeric with float pandas Series."""
        s = pd.Series([1.0, 2.0, 3.0])
        assert _is_numeric(s) is True

    def test_pandas_string_series(self):
        """Test _is_numeric with string pandas Series."""
        s = pd.Series(["a", "b", "c"])
        assert _is_numeric(s) is False

    def test_pandas_numeric_dataframe(self):
        """Test _is_numeric with numeric pandas DataFrame.

        Note: For DataFrames, df.dtypes returns a Series, not a single dtype,
        so is_numeric_dtype returns False. This is expected behavior.
        """
        df = pd.DataFrame({"a": [1, 2, 3]})
        # DataFrame dtypes is a Series, so is_numeric_dtype returns False
        assert _is_numeric(df) is False


class TestCheckShapesMatch:
    """Tests for check_shapes_match function."""

    def test_matching_shapes(self):
        """Test check_shapes_match with matching shapes."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([0, 1, 0])
        Y = np.array([1.0, 2.0, 3.0])
        check_shapes_match(X, T, Y)  # Should not raise

    def test_matching_shapes_with_w(self):
        """Test check_shapes_match with W provided."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        W = np.array([[0.1, 0.2], [0.3, 0.4]])
        check_shapes_match(X, T, Y, W)  # Should not raise

    def test_mismatched_y_length_raises(self):
        """Test check_shapes_match with mismatched Y length."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([0, 1, 0])
        Y = np.array([1.0, 2.0])  # Wrong length
        with pytest.raises(ValueError, match="Y must have length 3, got 2"):
            check_shapes_match(X, T, Y)

    def test_mismatched_x_length_raises(self):
        """Test check_shapes_match with mismatched X length."""
        X = np.array([[1, 2], [3, 4]])  # Wrong length
        T = np.array([0, 1, 0])
        Y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="X must have length 3, got 2"):
            check_shapes_match(X, T, Y)

    def test_mismatched_w_length_raises(self):
        """Test check_shapes_match with mismatched W length."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([0, 1, 0])
        Y = np.array([1.0, 2.0, 3.0])
        W = np.array([[0.1, 0.2]])  # Wrong length
        with pytest.raises(ValueError, match="W must have length 3, got 1"):
            check_shapes_match(X, T, Y, W)


class TestCheck1dTargets:
    """Tests for check_1d_targets function."""

    def test_valid_1d_arrays(self):
        """Test check_1d_targets with valid 1D arrays."""
        T = np.array([0, 1, 0, 1])
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        check_1d_targets(T, Y)  # Should not raise

    def test_valid_pandas_series(self):
        """Test check_1d_targets with pandas Series."""
        T = pd.Series([0, 1, 0, 1])
        Y = pd.Series([1.0, 2.0, 3.0, 4.0])
        check_1d_targets(T, Y)  # Should not raise

    def test_2d_treatment_raises(self):
        """Test check_1d_targets with 2D treatment array raises."""
        T = np.array([[0, 1], [1, 0]])
        Y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="T must be 1-dimensional"):
            check_1d_targets(T, Y)

    def test_2d_outcome_raises(self):
        """Test check_1d_targets with 2D outcome array raises."""
        T = np.array([0, 1])
        Y = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Y must be 1-dimensional"):
            check_1d_targets(T, Y)


class TestCheckMissingData:
    """Tests for check_missing_data function."""

    def test_no_missing_data(self):
        """Test check_missing_data with no missing values."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        result = check_missing_data(X, T, Y)
        assert result == {"X": 0, "T": 0, "Y": 0, "W": 0}

    def test_with_missing_data(self):
        """Test check_missing_data with missing values."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        T = np.array([0.0, np.nan])
        Y = np.array([1.0, 2.0])
        result = check_missing_data(X, T, Y)
        assert result["X"] == 1
        assert result["T"] == 1
        assert result["Y"] == 0

    def test_with_w_no_missing(self):
        """Test check_missing_data with W provided, no missing values."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        W = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = check_missing_data(X, T, Y, W)
        assert result == {"X": 0, "T": 0, "Y": 0, "W": 0}

    def test_with_w_with_missing(self):
        """Test check_missing_data with W provided, with missing values."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        W = np.array([[0.1, np.nan], [0.3, 0.4]])
        result = check_missing_data(X, T, Y, W)
        assert result["W"] == 1


class TestCheckTreatmentTypeMatchesData:
    """Tests for check_treatment_type_matches_data function."""

    def test_binary_with_two_values(self):
        """Test BINARY with 2 unique values."""
        T = np.array([0, 1, 0, 1, 0])
        check_treatment_type_matches_data(T, TreatmentType.BINARY)  # Should not raise

    def test_binary_with_one_value(self):
        """Test BINARY with 1 unique value."""
        T = np.array([0, 0, 0, 0])
        check_treatment_type_matches_data(T, TreatmentType.BINARY)  # Should not raise

    def test_binary_with_three_values_raises(self):
        """Test BINARY with 3 unique values raises ValueError."""
        T = np.array([0, 1, 2, 0, 1])
        with pytest.raises(ValueError, match="BINARY expects at most 2 unique values"):
            check_treatment_type_matches_data(T, TreatmentType.BINARY)

    def test_multi_with_three_values(self):
        """Test MULTI with 3 unique values."""
        T = np.array([0, 1, 2, 0, 1, 2])
        check_treatment_type_matches_data(T, TreatmentType.MULTI)  # Should not raise

    def test_multi_with_two_values_raises(self):
        """Test MULTI with 2 unique values raises ValueError."""
        T = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="MULTI expects more than 2 unique values"):
            check_treatment_type_matches_data(T, TreatmentType.MULTI)

    def test_multi_with_one_value_raises(self):
        """Test MULTI with 1 unique value raises ValueError."""
        T = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="MULTI expects more than 2 unique values"):
            check_treatment_type_matches_data(T, TreatmentType.MULTI)

    def test_continuous_with_numeric(self):
        """Test CONTINUOUS with numeric values."""
        T = np.array([0.1, 0.5, 0.9, 1.2])
        check_treatment_type_matches_data(
            T, TreatmentType.CONTINUOUS
        )  # Should not raise

    def test_continuous_with_non_numeric_raises(self):
        """Test CONTINUOUS with non-numeric values raises ValueError."""
        T = np.array(["a", "b", "c"])
        with pytest.raises(
            ValueError, match="CONTINUOUS expects numeric treatment values"
        ):
            check_treatment_type_matches_data(T, TreatmentType.CONTINUOUS)


class TestCheckOutcomeTypeMatchesData:
    """Tests for check_outcome_type_matches_data function."""

    def test_binary_with_two_values(self):
        """Test BINARY with 2 unique values."""
        Y = np.array([0, 1, 0, 1, 0])
        check_outcome_type_matches_data(Y, OutcomeType.BINARY)  # Should not raise

    def test_binary_with_one_value(self):
        """Test BINARY with 1 unique value."""
        Y = np.array([0, 0, 0, 0])
        check_outcome_type_matches_data(Y, OutcomeType.BINARY)  # Should not raise

    def test_binary_with_three_values_raises(self):
        """Test BINARY with 3 unique values raises ValueError."""
        Y = np.array([0, 1, 2, 0, 1])
        with pytest.raises(ValueError, match="BINARY expects at most 2 unique values"):
            check_outcome_type_matches_data(Y, OutcomeType.BINARY)

    def test_continuous_with_numeric(self):
        """Test CONTINUOUS with numeric values."""
        Y = np.array([0.1, 0.5, 0.9, 1.2])
        check_outcome_type_matches_data(Y, OutcomeType.CONTINUOUS)  # Should not raise

    def test_continuous_with_non_numeric_raises(self):
        """Test CONTINUOUS with non-numeric values raises ValueError."""
        Y = np.array(["a", "b", "c"])
        with pytest.raises(
            ValueError, match="CONTINUOUS expects numeric treatment values"
        ):
            check_outcome_type_matches_data(Y, OutcomeType.CONTINUOUS)
