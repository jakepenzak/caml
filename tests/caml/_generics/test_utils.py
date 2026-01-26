"""Tests for caml._generics.utils module."""

import pytest

from caml._generics.utils import FittedAttr, is_module_available

pytestmark = pytest.mark.generics


# ==============================================================================
# MODULE AVAILABILITY TESTS
# ==============================================================================


class TestIsModuleAvailable:
    """Test is_module_available function."""

    def test_available_module_numpy(self):
        """Test with available module numpy."""
        assert is_module_available("numpy") is True

    def test_available_module_pandas(self):
        """Test with available module pandas."""
        assert is_module_available("pandas") is True

    def test_unavailable_module(self):
        """Test with unavailable module."""
        assert is_module_available("nonexistent_module_xyz") is False

    def test_available_submodule(self):
        """Test with available submodule."""
        assert is_module_available("caml.data") is True

    def test_unavailable_submodule(self):
        """Test with unavailable submodule."""
        assert is_module_available("caml.nonexistent") is False


# ==============================================================================
# FITTED ATTR TESTS
# ==============================================================================


class TestFittedAttr:
    """Test FittedAttr descriptor."""

    def test_create_fitted_attr(self):
        """Test creating a FittedAttr descriptor."""
        attr = FittedAttr("_my_attr")
        assert attr.name == "_my_attr"

    def test_access_on_unfitted_object_raises_error(self):
        """Test accessing FittedAttr on unfitted object raises RuntimeError."""

        class DummyModel:
            result_ = FittedAttr("_result")

            def __init__(self):
                self._fitted = False
                self._result = None

        model = DummyModel()

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            _ = model.result_

    def test_access_on_fitted_object_succeeds(self):
        """Test accessing FittedAttr on fitted object succeeds."""

        class DummyModel:
            result_ = FittedAttr("_result")

            def __init__(self):
                self._fitted = False
                self._result = None

            def fit(self):
                self._fitted = True
                self._result = 42
                return self

        model = DummyModel()
        model.fit()

        assert model.result_ == 42

    def test_access_on_class_returns_descriptor(self):
        """Test accessing FittedAttr on class returns the descriptor."""

        class DummyModel:
            result_ = FittedAttr("_result")

        assert isinstance(DummyModel.result_, FittedAttr)

    def test_multiple_fitted_attrs(self):
        """Test using multiple FittedAttr descriptors."""

        class DummyModel:
            result_ = FittedAttr("_result")
            params_ = FittedAttr("_params")

            def __init__(self):
                self._fitted = False
                self._result = None
                self._params = None

            def fit(self):
                self._fitted = True
                self._result = 42
                self._params = {"a": 1, "b": 2}
                return self

        model = DummyModel()

        # Both should raise before fitting
        with pytest.raises(RuntimeError):
            _ = model.result_
        with pytest.raises(RuntimeError):
            _ = model.params_

        # Fit and verify both work
        model.fit()
        assert model.result_ == 42
        assert model.params_ == {"a": 1, "b": 2}

    def test_fitted_attr_with_none_value(self):
        """Test FittedAttr can return None if fitted."""

        class DummyModel:
            result_ = FittedAttr("_result")

            def __init__(self):
                self._fitted = False
                self._result = None

            def fit(self):
                self._fitted = True
                # _result remains None
                return self

        model = DummyModel()
        model.fit()

        assert model.result_ is None
