"""Tests for caml.data._validation module."""

import numpy as np
import pandas as pd
import pytest

from caml.data._validation import (
    check_1d_targets,
    check_missing_data,
    check_outcome_type_matches_data,
    check_shapes_match,
    check_treatment_type_matches_data,
)
from caml.data.data_schema import OutcomeType, TreatmentType

pytestmark = [pytest.mark.data]


# ==============================================================================
# SHAPE VALIDATION TESTS
# ==============================================================================


class TestShapeValidation:
    """Test shape matching validation."""

    def test_shapes_match_with_arrays(self):
        """Test valid shapes with numpy arrays."""
        X = np.random.randn(100, 5)
        T = np.random.randn(100)
        Y = np.random.randn(100)
        W = np.random.randn(100, 3)

        # Should not raise
        check_shapes_match(X, T, Y, W)

    def test_shapes_match_with_dataframes(self):
        """Test valid shapes with pandas DataFrames."""
        X = pd.DataFrame(np.random.randn(100, 5))
        T = pd.Series(np.random.randn(100))
        Y = pd.Series(np.random.randn(100))
        W = pd.DataFrame(np.random.randn(100, 3))

        # Should not raise
        check_shapes_match(X, T, Y, W)

    def test_shapes_match_without_W(self):
        """Test valid shapes without confounders."""
        X = np.random.randn(50, 3)
        T = np.random.randn(50)
        Y = np.random.randn(50)

        # Should not raise
        check_shapes_match(X, T, Y, W=None)

    def test_mismatched_X_length(self):
        """Test error when X has wrong length."""
        X = np.random.randn(50, 3)
        T = np.random.randn(100)
        Y = np.random.randn(100)

        with pytest.raises(ValueError, match="X must have length 100"):
            check_shapes_match(X, T, Y)

    def test_mismatched_Y_length(self):
        """Test error when Y has wrong length."""
        X = np.random.randn(100, 3)
        T = np.random.randn(100)
        Y = np.random.randn(50)

        with pytest.raises(ValueError, match="Y must have length 100"):
            check_shapes_match(X, T, Y)

    def test_mismatched_W_length(self):
        """Test error when W has wrong length."""
        X = np.random.randn(100, 3)
        T = np.random.randn(100)
        Y = np.random.randn(100)
        W = np.random.randn(50, 2)

        with pytest.raises(ValueError, match="W must have length 100"):
            check_shapes_match(X, T, Y, W)

    def test_none_treatment_raises_error(self):
        """Test that None treatment raises error."""
        X = np.random.randn(100, 3)
        Y = np.random.randn(100)

        with pytest.raises(ValueError, match="Expected array-like input"):
            check_shapes_match(X, None, Y)


# ==============================================================================
# 1D TARGET VALIDATION TESTS
# ==============================================================================


class Test1DValidation:
    """Test 1-dimensional target validation."""

    def test_1d_array_is_valid(self):
        """Test that 1D numpy arrays are valid."""
        T = np.random.randn(100)
        Y = np.random.randn(100)

        # Should not raise
        check_1d_targets(T, Y)

    def test_series_is_valid(self):
        """Test that pandas Series are valid."""
        T = pd.Series(np.random.randn(100))
        Y = pd.Series(np.random.randn(100))

        # Should not raise
        check_1d_targets(T, Y)

    def test_2d_array_single_column_is_valid(self):
        """Test that 2D arrays with 1 column are valid."""
        T = np.random.randn(100, 1)
        Y = np.random.randn(100, 1)

        # Should not raise
        check_1d_targets(T, Y)

    def test_dataframe_single_column_is_valid(self):
        """Test that DataFrames with 1 column are valid."""
        T = pd.DataFrame(np.random.randn(100, 1))
        Y = pd.DataFrame(np.random.randn(100, 1))

        # Should not raise
        check_1d_targets(T, Y)

    def test_2d_array_multiple_columns_raises_error(self):
        """Test that 2D arrays with >1 column raise error."""
        T = np.random.randn(100, 2)
        Y = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="T must be 1-dimensional"):
            check_1d_targets(T, Y)

    def test_dataframe_multiple_columns_raises_error(self):
        """Test that DataFrames with >1 column raise error."""
        T = pd.DataFrame(np.random.randn(100, 1))
        Y = pd.DataFrame(np.random.randn(100, 3))

        with pytest.raises(ValueError, match="Y must be 1-dimensional"):
            check_1d_targets(T, Y)


# ==============================================================================
# MISSING DATA TESTS
# ==============================================================================


class TestMissingData:
    """Test missing data detection."""

    def test_no_missing_data(self):
        """Test that no missing values are reported correctly."""
        X = np.random.randn(100, 5)
        T = np.random.randn(100)
        Y = np.random.randn(100)
        W = np.random.randn(100, 3)

        summary = check_missing_data(X, T, Y, W)

        assert summary["X"] == 0
        assert summary["T"] == 0
        assert summary["Y"] == 0
        assert summary["W"] == 0

    def test_missing_data_in_arrays(self):
        """Test missing data detection in numpy arrays."""
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan
        X[5, 2] = np.nan

        T = np.random.randn(100)
        T[10] = np.nan

        Y = np.random.randn(100)

        summary = check_missing_data(X, T, Y, W=None)

        assert summary["X"] == 2
        assert summary["T"] == 1
        assert summary["Y"] == 0
        assert summary["W"] == 0

    def test_missing_data_in_dataframes(self):
        """Test missing data detection in pandas DataFrames."""
        X = pd.DataFrame(np.random.randn(100, 5))
        X.iloc[0, 0] = np.nan
        X.iloc[1, 1] = np.nan
        X.iloc[2, 2] = np.nan

        T = pd.Series(np.random.randn(100))
        Y = pd.Series(np.random.randn(100))

        summary = check_missing_data(X, T, Y, W=None)

        assert summary["X"] == 3
        assert summary["T"] == 0
        assert summary["Y"] == 0

    def test_missing_data_without_W(self):
        """Test that W=None returns 0 missing."""
        X = np.random.randn(100, 3)
        T = np.random.randn(100)
        Y = np.random.randn(100)

        summary = check_missing_data(X, T, Y, W=None)

        assert summary["W"] == 0


# ==============================================================================
# TREATMENT TYPE VALIDATION TESTS
# ==============================================================================


class TestTreatmentTypeValidation:
    """Test treatment type validation."""

    def test_binary_treatment_with_two_values(self):
        """Test binary treatment with exactly 2 unique values."""
        T = np.array([0, 1, 0, 1, 0, 1])

        # Should not raise
        check_treatment_type_matches_data(T, TreatmentType.BINARY)

    def test_binary_treatment_with_one_value(self):
        """Test binary treatment with 1 unique value."""
        T = np.array([1, 1, 1, 1])

        # Should not raise (<=2 values)
        check_treatment_type_matches_data(T, TreatmentType.BINARY)

    def test_binary_treatment_with_three_values_raises_error(self):
        """Test binary treatment with >2 unique values raises error."""
        T = np.array([0, 1, 2, 0, 1, 2])

        with pytest.raises(ValueError, match="BINARY expects at most 2 unique values"):
            check_treatment_type_matches_data(T, TreatmentType.BINARY)

    def test_multi_treatment_with_three_values(self):
        """Test multi treatment with 3+ unique values."""
        T = np.array([0, 1, 2, 0, 1, 2])

        # Should not raise
        check_treatment_type_matches_data(T, TreatmentType.MULTI)

    def test_multi_treatment_with_two_values_raises_error(self):
        """Test multi treatment with <=2 values raises error."""
        T = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="MULTI expects more than 2 unique values"):
            check_treatment_type_matches_data(T, TreatmentType.MULTI)

    def test_continuous_treatment_with_numeric_values(self):
        """Test continuous treatment with numeric values."""
        T = np.random.randn(100)

        # Should not raise
        check_treatment_type_matches_data(T, TreatmentType.CONTINUOUS)

    def test_continuous_treatment_with_non_numeric_raises_error(self):
        """Test continuous treatment with non-numeric values raises error."""
        T = pd.Series(["a", "b", "c"])

        with pytest.raises(ValueError, match="CONTINUOUS expects numeric"):
            check_treatment_type_matches_data(T, TreatmentType.CONTINUOUS)

    def test_treatment_validation_ignores_nan(self):
        """Test that NaN values are ignored in uniqueness check."""
        T = np.array([0.0, 1.0, np.nan, 0.0, 1.0, np.nan])

        # Should not raise (only 2 non-null unique values)
        check_treatment_type_matches_data(T, TreatmentType.BINARY)


# ==============================================================================
# OUTCOME TYPE VALIDATION TESTS
# ==============================================================================


class TestOutcomeTypeValidation:
    """Test outcome type validation."""

    def test_binary_outcome_with_two_values(self):
        """Test binary outcome with exactly 2 unique values."""
        Y = np.array([0, 1, 0, 1, 0, 1])

        # Should not raise
        check_outcome_type_matches_data(Y, OutcomeType.BINARY)

    def test_binary_outcome_with_three_values_raises_error(self):
        """Test binary outcome with >2 unique values raises error."""
        Y = np.array([0, 1, 2, 0, 1, 2])

        with pytest.raises(ValueError, match="BINARY expects at most 2 unique values"):
            check_outcome_type_matches_data(Y, OutcomeType.BINARY)

    def test_continuous_outcome_with_numeric_values(self):
        """Test continuous outcome with numeric values."""
        Y = np.random.randn(100)

        # Should not raise
        check_outcome_type_matches_data(Y, OutcomeType.CONTINUOUS)

    def test_continuous_outcome_with_non_numeric_raises_error(self):
        """Test continuous outcome with non-numeric values raises error."""
        Y = pd.Series(["a", "b", "c"])

        with pytest.raises(ValueError, match="CONTINUOUS expects numeric"):
            check_outcome_type_matches_data(Y, OutcomeType.CONTINUOUS)

    def test_outcome_validation_ignores_nan(self):
        """Test that NaN values are ignored in uniqueness check."""
        Y = np.array([0.0, 1.0, np.nan, 0.0, 1.0, np.nan])

        # Should not raise (only 2 non-null unique values)
        check_outcome_type_matches_data(Y, OutcomeType.BINARY)
