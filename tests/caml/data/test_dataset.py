"""Tests for caml.data.dataset.CausalDataset."""

import numpy as np
import pandas as pd
import pytest

from caml.data.data_schema import OutcomeType, TreatmentType
from caml.data.dataset import CausalDataset


class TestCausalDatasetCreation:
    """Tests for creating CausalDataset instances."""

    def test_minimal_creation_with_numpy(self):
        """Test creating CausalDataset with minimal numpy arrays."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([0, 1, 0])
        Y = np.array([1.0, 2.0, 3.0])

        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert dataset.X.shape == (3, 2)
        assert dataset.T.shape == (3,)
        assert dataset.Y.shape == (3,)
        assert dataset.treatment_type == TreatmentType.BINARY
        assert dataset.outcome_type == OutcomeType.CONTINUOUS

    def test_minimal_creation_with_pandas(self):
        """Test creating CausalDataset with pandas objects."""
        X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
        T = pd.Series([0, 1, 0])
        Y = pd.Series([1.0, 2.0, 3.0])

        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert isinstance(dataset.X, pd.DataFrame)
        assert isinstance(dataset.T, pd.Series)
        assert isinstance(dataset.Y, pd.Series)

    def test_creation_with_w(self):
        """Test creating CausalDataset with instrumental variables W."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        W = np.array([[0.1, 0.2], [0.3, 0.4]])

        dataset = CausalDataset(X=X, T=T, Y=Y, W=W)

        assert dataset.W is not None
        assert dataset.W.shape == (2, 2)

    def test_creation_with_weights(self):
        """Test creating CausalDataset with sample weights."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])
        weights = np.array([0.5, 1.5])

        dataset = CausalDataset(X=X, T=T, Y=Y, weights=weights)

        assert dataset.weights is not None
        np.testing.assert_array_equal(dataset.weights, weights)

    def test_creation_with_custom_types(self):
        """Test creating CausalDataset with custom treatment and outcome types."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([0.1, 0.5, 0.9])
        Y = np.array([0, 1, 0])

        dataset = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.BINARY,
        )

        assert dataset.treatment_type == TreatmentType.CONTINUOUS
        assert dataset.outcome_type == OutcomeType.BINARY

    def test_creation_with_feature_names(self):
        """Test creating CausalDataset with feature names."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])

        dataset = CausalDataset(
            X=X,
            T=T,
            Y=Y,
            X_names=["feature1", "feature2"],
            T_name="treatment",
            Y_name="outcome",
        )

        assert dataset.X_names == ["feature1", "feature2"]
        assert dataset.T_name == "treatment"
        assert dataset.Y_name == "outcome"


class TestCausalDatasetValidation:
    """Tests for CausalDataset validation."""

    def test_mismatched_shapes_raises(self):
        """Test creating CausalDataset with mismatched shapes raises ValueError."""
        X = np.array([[1, 2], [3, 4]])  # 2 obs
        T = np.array([0, 1, 0])  # 3 obs
        Y = np.array([1.0, 2.0])  # 2 obs

        with pytest.raises(ValueError, match="Y must have length 3"):
            CausalDataset(X=X, T=T, Y=Y)

    def test_2d_treatment_raises(self):
        """Test creating CausalDataset with 2D treatment raises ValueError."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([[0, 1], [1, 0]])
        Y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="T must be 1-dimensional"):
            CausalDataset(X=X, T=T, Y=Y)

    def test_2d_outcome_raises(self):
        """Test creating CausalDataset with 2D outcome raises ValueError."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Y must be 1-dimensional"):
            CausalDataset(X=X, T=T, Y=Y)

    def test_binary_treatment_with_three_values_raises(self):
        """Test binary treatment type with 3 unique values raises ValueError."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([0, 1, 2])
        Y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="BINARY expects at most 2 unique values"):
            CausalDataset(X=X, T=T, Y=Y, treatment_type=TreatmentType.BINARY)

    def test_multi_treatment_with_two_values_raises(self):
        """Test multi treatment type with 2 unique values raises ValueError."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="MULTI expects more than 2 unique values"):
            CausalDataset(X=X, T=T, Y=Y, treatment_type=TreatmentType.MULTI)

    def test_continuous_treatment_with_non_numeric_raises(self):
        """Test continuous treatment with non-numeric values raises ValueError."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array(["a", "b"])
        Y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="CONTINUOUS expects numeric"):
            CausalDataset(X=X, T=T, Y=Y, treatment_type=TreatmentType.CONTINUOUS)

    def test_binary_outcome_with_three_values_raises(self):
        """Test binary outcome type with 3 unique values raises ValueError."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([0, 1, 0])
        Y = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="BINARY expects at most 2 unique values"):
            CausalDataset(X=X, T=T, Y=Y, outcome_type=OutcomeType.BINARY)

    def test_continuous_outcome_with_non_numeric_raises(self):
        """Test continuous outcome with non-numeric values raises ValueError."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array(["a", "b"])

        with pytest.raises(ValueError, match="CONTINUOUS expects numeric"):
            CausalDataset(X=X, T=T, Y=Y, outcome_type=OutcomeType.CONTINUOUS)


class TestCausalDatasetFromDataFrame:
    """Tests for CausalDataset.from_dataframe() factory method."""

    def test_from_dataframe_basic(self):
        """Test creating CausalDataset from DataFrame."""
        df = pd.DataFrame(
            {
                "x1": [1, 2, 3],
                "x2": [4, 5, 6],
                "treatment": [0, 1, 0],
                "outcome": [1.0, 2.0, 3.0],
            }
        )

        dataset = CausalDataset.from_dataframe(
            df=df, X=["x1", "x2"], T="treatment", Y="outcome"
        )

        assert isinstance(dataset.X, pd.DataFrame)
        assert dataset.X.shape == (3, 2)
        assert isinstance(dataset.T, pd.Series)
        assert isinstance(dataset.Y, pd.Series)
        assert dataset.X_names == ["x1", "x2"]
        assert dataset.T_name == "treatment"
        assert dataset.Y_name == "outcome"

    def test_from_dataframe_with_w(self):
        """Test creating CausalDataset from DataFrame with W."""
        df = pd.DataFrame(
            {
                "x1": [1, 2],
                "w1": [0.1, 0.2],
                "w2": [0.3, 0.4],
                "treatment": [0, 1],
                "outcome": [1.0, 2.0],
            }
        )

        dataset = CausalDataset.from_dataframe(
            df=df, X=["x1"], T="treatment", Y="outcome", W=["w1", "w2"]
        )

        assert isinstance(dataset.W, pd.DataFrame)
        assert dataset.W.shape == (2, 2)
        assert dataset.W_names == ["w1", "w2"]

    def test_from_dataframe_with_custom_types(self):
        """Test creating CausalDataset from DataFrame with custom types."""
        df = pd.DataFrame(
            {
                "x1": [1, 2, 3],
                "treatment": [0.1, 0.5, 0.9],
                "outcome": [0, 1, 0],
            }
        )

        dataset = CausalDataset.from_dataframe(
            df=df,
            X=["x1"],
            T="treatment",
            Y="outcome",
            treatment_type=TreatmentType.CONTINUOUS,
            outcome_type=OutcomeType.BINARY,
        )

        assert dataset.treatment_type == TreatmentType.CONTINUOUS
        assert dataset.outcome_type == OutcomeType.BINARY

    def test_from_dataframe_with_kwargs(self):
        """Test creating CausalDataset from DataFrame with extra kwargs."""
        df = pd.DataFrame(
            {
                "x1": [1, 2, 3],
                "treatment": [0, 1, 0],
                "outcome": [1.0, 2.0, 3.0],
            }
        )

        weights = np.array([0.5, 1.0, 1.5])

        dataset = CausalDataset.from_dataframe(
            df=df,
            X=["x1"],
            T="treatment",
            Y="outcome",
            weights=weights,
        )

        assert dataset.weights is not None

    def test_from_dataframe_missing_column_raises(self):
        """Test creating CausalDataset from DataFrame with missing column raises."""
        df = pd.DataFrame({"x1": [1, 2], "treatment": [0, 1]})

        with pytest.raises(KeyError):
            CausalDataset.from_dataframe(
                df=df,
                X=["x1"],
                T="treatment",
                Y="outcome",  # Missing 'outcome'
            )


class TestCausalDatasetValidateMethod:
    """Tests for CausalDataset.validate() method."""

    def test_validate_does_not_raise_on_valid_data(self):
        """Test validate() does not raise on valid data."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])

        dataset = CausalDataset.__new__(CausalDataset)
        dataset.X = X
        dataset.T = T
        dataset.Y = Y
        dataset.W = None
        dataset.weights = None
        dataset.treatment_type = TreatmentType.BINARY
        dataset.outcome_type = OutcomeType.CONTINUOUS
        dataset.X_names = None
        dataset.W_names = None
        dataset.T_name = "treatment"
        dataset.Y_name = "outcome"

        dataset.validate()  # Should not raise

    def test_validate_called_in_post_init(self):
        """Test validate() is called in __post_init__."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1, 0])  # Wrong length
        Y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Y must have length 3"):
            CausalDataset(X=X, T=T, Y=Y)


class TestCausalDatasetEdgeCases:
    """Tests for edge cases in CausalDataset."""

    def test_single_observation(self):
        """Test CausalDataset with single observation."""
        X = np.array([[1, 2]])
        T = np.array([0])
        Y = np.array([1.0])

        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert dataset.X.shape == (1, 2)
        assert dataset.T.shape == (1,)
        assert dataset.Y.shape == (1,)

    def test_single_feature(self):
        """Test CausalDataset with single feature."""
        X = np.array([[1], [2], [3]])
        T = np.array([0, 1, 0])
        Y = np.array([1.0, 2.0, 3.0])

        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert dataset.X.shape == (3, 1)

    def test_1d_x_as_series(self):
        """Test CausalDataset with 1D X as pandas Series."""
        X = pd.Series([1, 2, 3])
        T = pd.Series([0, 1, 0])
        Y = pd.Series([1.0, 2.0, 3.0])

        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert isinstance(dataset.X, pd.Series)

    def test_constant_treatment(self):
        """Test CausalDataset with constant treatment (1 unique value)."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 0])  # All same value
        Y = np.array([1.0, 2.0])

        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert dataset.treatment_type == TreatmentType.BINARY

    def test_constant_outcome(self):
        """Test CausalDataset with constant outcome."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, 1.0])  # All same value

        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert dataset.outcome_type == OutcomeType.CONTINUOUS


class TestCausalDatasetWithMissingData:
    """Tests for CausalDataset with missing data."""

    def test_missing_data_in_x_allowed(self):
        """Test CausalDataset allows missing data in X."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        T = np.array([0, 1])
        Y = np.array([1.0, 2.0])

        dataset = CausalDataset(X=X, T=T, Y=Y)

        # Should create successfully - validation checks but doesn't reject
        assert dataset.X.shape == (2, 2)

    def test_missing_data_in_t_allowed(self):
        """Test CausalDataset allows missing data in T."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0.0, np.nan])
        Y = np.array([1.0, 2.0])

        # Note: This may raise depending on treatment type validation
        # with NaN values excluded from unique check
        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert dataset.T.shape == (2,)

    def test_missing_data_in_y_allowed(self):
        """Test CausalDataset allows missing data in Y."""
        X = np.array([[1, 2], [3, 4]])
        T = np.array([0, 1])
        Y = np.array([1.0, np.nan])

        dataset = CausalDataset(X=X, T=T, Y=Y)

        assert dataset.Y.shape == (2,)
