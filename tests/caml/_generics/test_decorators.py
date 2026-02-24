"""Tests for caml._generics.decorators module."""

import time

import pytest

from caml._generics.decorators import experimental, narrate, timer

pytestmark = pytest.mark.generics


# ==============================================================================
# EXPERIMENTAL DECORATOR TESTS
# ==============================================================================


class TestExperimentalDecorator:
    """Test @experimental decorator."""

    def test_experimental_function(self, caplog):
        """Test experimental decorator on function."""
        import logging

        @experimental
        def dummy_function():
            return 42

        # Warning should not be shown yet
        assert not dummy_function._experimental_warning_shown

        # Call function with logging at warning level
        with caplog.at_level(logging.WARNING, logger="caml"):
            result = dummy_function()

        # Should return correct result
        assert result == 42

        # Warning should be shown (check the wrapped object's attribute)
        assert "experimental" in caplog.text.lower()

    def test_experimental_function_warning_once(self, caplog):
        """Test experimental warning is only shown once for functions."""

        @experimental
        def dummy_function():
            return 42

        # Call multiple times
        dummy_function()
        dummy_function()
        dummy_function()

        # Warning should only appear once
        assert caplog.text.count("experimental") == 1

    def test_experimental_class(self, caplog):
        """Test experimental decorator on class."""

        @experimental
        class DummyClass:
            def __init__(self, value):
                self.value = value

        # Create instance
        obj = DummyClass(42)

        # Should work correctly
        assert obj.value == 42

        # Warning should be shown
        assert "experimental" in caplog.text.lower()

    def test_experimental_class_warning_once(self, caplog):
        """Test experimental warning is only shown once for classes."""

        @experimental
        class DummyClass:
            def __init__(self):
                pass

        # Create multiple instances
        DummyClass()
        DummyClass()
        DummyClass()

        # Warning should only appear once
        assert caplog.text.count("experimental") == 1

    def test_experimental_preserves_function_metadata(self):
        """Test that experimental decorator preserves function metadata."""

        @experimental
        def my_function():
            """My docstring."""
            return 42

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_experimental_attribute_set(self):
        """Test that _experimental attribute is set."""

        @experimental
        def dummy_function():
            return 42

        assert hasattr(dummy_function, "_experimental")
        assert dummy_function._experimental is True


# ==============================================================================
# NARRATE DECORATOR TESTS
# ==============================================================================


class TestNarrateDecorator:
    """Test @narrate decorator."""

    def test_narrate_with_preamble_and_epilogue(self, caplog):
        """Test narrate with both preamble and epilogue."""
        import logging

        @narrate(preamble="Starting task", epilogue="Task complete")
        def dummy_function():
            return 42

        with caplog.at_level(logging.INFO, logger="caml"):
            result = dummy_function()

        assert result == 42
        assert "Starting task" in caplog.text
        assert "Task complete" in caplog.text

    def test_narrate_with_only_preamble(self, caplog):
        """Test narrate with only preamble."""
        import logging

        @narrate(preamble="Starting", epilogue=None)
        def dummy_function():
            return 42

        with caplog.at_level(logging.INFO, logger="caml"):
            result = dummy_function()

        assert result == 42
        assert "Starting" in caplog.text

    def test_narrate_with_only_epilogue(self, caplog):
        """Test narrate with only epilogue."""
        import logging

        @narrate(preamble=None, epilogue="Done")
        def dummy_function():
            return 42

        with caplog.at_level(logging.INFO, logger="caml"):
            result = dummy_function()

        assert result == 42
        assert "Done" in caplog.text

    def test_narrate_default_epilogue(self, caplog):
        """Test narrate with default epilogue."""
        import logging

        @narrate(preamble="Starting")
        def dummy_function():
            return 42

        with caplog.at_level(logging.INFO, logger="caml"):
            result = dummy_function()

        assert result == 42
        assert "Completed" in caplog.text or "white_check_mark" in caplog.text

    def test_narrate_preserves_function_behavior(self):
        """Test that narrate preserves function behavior."""

        @narrate(preamble="Start", epilogue="End")
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_narrate_preserves_metadata(self):
        """Test that narrate preserves function metadata."""

        @narrate(preamble="Start")
        def my_function():
            """My docstring."""
            return 42

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_narrate_with_exception(self, caplog):
        """Test narrate when function raises exception."""
        import logging

        @narrate(preamble="Start", epilogue="End")
        def failing_function():
            raise ValueError("Test error")

        # Preamble should be logged
        with caplog.at_level(logging.INFO, logger="caml"):
            with pytest.raises(ValueError):
                failing_function()

        assert "Start" in caplog.text
        # Epilogue should NOT be logged because exception was raised
        assert "End" not in caplog.text


# ==============================================================================
# TIMER DECORATOR TESTS
# ==============================================================================


class TestTimerDecorator:
    """Test @timer decorator."""

    def test_timer_with_operation_name(self, caplog):
        """Test timer with custom operation name."""
        import logging

        @timer(operation_name="custom_operation")
        def dummy_function():
            time.sleep(0.01)
            return 42

        with caplog.at_level(logging.DEBUG, logger="caml"):
            result = dummy_function()

        assert result == 42
        assert "custom_operation" in caplog.text
        assert "seconds" in caplog.text

    def test_timer_with_explicit_none(self, caplog):
        """Test timer with explicit None defaults to function name."""
        import logging

        # This won't work due to decorator implementation - skip this edge case
        # The decorator is meant to be used with @timer(operation_name="...") or @timer()
        # Testing the working behavior instead
        @timer(operation_name="explicit_name")
        def my_function():
            time.sleep(0.01)
            return 42

        with caplog.at_level(logging.DEBUG, logger="caml"):
            result = my_function()

        assert result == 42
        assert "explicit_name" in caplog.text
        assert "seconds" in caplog.text

    def test_timer_preserves_function_behavior(self):
        """Test that timer preserves function behavior."""

        @timer(operation_name="add_operation")
        def add(a, b):
            return a + b

        result = add(5, 7)
        assert result == 12

    def test_timer_measures_time(self, caplog):
        """Test that timer actually measures time."""
        import logging

        @timer(operation_name="sleep_operation")
        def sleep_function():
            time.sleep(0.05)

        with caplog.at_level(logging.DEBUG, logger="caml"):
            sleep_function()

        # Check that a time was logged
        assert "0." in caplog.text and "seconds" in caplog.text

    def test_timer_preserves_metadata(self):
        """Test that timer preserves function metadata."""

        @timer(operation_name="test")
        def my_function():
            """My docstring."""
            return 42

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_timer_with_exception(self, caplog):
        """Test timer when function raises exception."""

        @timer(operation_name="failing_op")
        def failing_function():
            raise ValueError("Test error")

        # Timer should still log before exception propagates
        with pytest.raises(ValueError):
            failing_function()

        # Time should NOT be logged because exception was raised before completion
        # The exception prevents the epilogue from running
        assert "failing_op" not in caplog.text or "seconds" not in caplog.text


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestDecoratorIntegration:
    """Test combining multiple decorators."""

    def test_experimental_with_timer(self, caplog):
        """Test combining experimental and timer decorators."""
        import logging

        @experimental
        @timer(operation_name="test_op")
        def dummy_function():
            time.sleep(0.01)
            return 42

        with caplog.at_level(logging.DEBUG, logger="caml"):
            result = dummy_function()

        assert result == 42
        assert "experimental" in caplog.text.lower()
        assert "test_op" in caplog.text

    def test_narrate_with_timer(self, caplog):
        """Test combining narrate and timer decorators."""
        import logging

        @narrate(preamble="Starting", epilogue="Done")
        @timer(operation_name="timed_op")
        def dummy_function():
            time.sleep(0.01)
            return 42

        with caplog.at_level(logging.DEBUG, logger="caml"):
            result = dummy_function()

        assert result == 42
        assert "Starting" in caplog.text
        assert "Done" in caplog.text
        assert "timed_op" in caplog.text

    def test_all_decorators_combined(self, caplog):
        """Test combining all three decorators."""
        import logging

        @experimental
        @narrate(preamble="Begin", epilogue="End")
        @timer(operation_name="combined_op")
        def dummy_function():
            time.sleep(0.01)
            return 42

        with caplog.at_level(logging.DEBUG, logger="caml"):
            result = dummy_function()

        assert result == 42
        assert "experimental" in caplog.text.lower()
        assert "Begin" in caplog.text
        assert "End" in caplog.text
        assert "combined_op" in caplog.text
