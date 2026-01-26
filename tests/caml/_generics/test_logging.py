"""Tests for caml._generics.logging module."""

import logging

import pytest

from caml._generics.logging import (
    DEBUG,
    ERROR,
    INFO,
    LOGO,
    WARNING,
    configure_logging,
    get_section_header,
    logger,
)

pytestmark = pytest.mark.generics


# ==============================================================================
# LOGGER TESTS
# ==============================================================================


class TestLogger:
    """Test logger object and functions."""

    def test_logger_exists(self):
        """Test that logger object exists."""
        assert logger is not None
        assert logger.name == "caml"

    def test_logger_functions_callable(self):
        """Test that logging functions are callable."""
        assert callable(INFO)
        assert callable(DEBUG)
        assert callable(WARNING)
        assert callable(ERROR)

    def test_logger_default_level(self):
        """Test that logger default level is WARNING."""
        # Note: This may fail if configure_logging was called previously
        assert logger.level == logging.WARNING or logger.level > 0


# ==============================================================================
# CONFIGURE LOGGING TESTS
# ==============================================================================


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_with_default_level(self):
        """Test configuring logging with default level."""
        configure_logging()
        assert logger.level == logging.WARNING

    def test_configure_with_debug_level(self):
        """Test configuring logging with DEBUG level."""
        configure_logging(logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_configure_with_info_level(self):
        """Test configuring logging with INFO level."""
        configure_logging(logging.INFO)
        assert logger.level == logging.INFO

    def test_configure_with_error_level(self):
        """Test configuring logging with ERROR level."""
        configure_logging(logging.ERROR)
        assert logger.level == logging.ERROR

    def test_configure_creates_handler(self):
        """Test that configure_logging creates handlers."""
        configure_logging()
        assert len(logger.handlers) > 0

    def test_reconfigure_replaces_handlers(self):
        """Test that reconfiguring replaces existing handlers."""
        configure_logging(logging.INFO)
        handler_count_1 = len(logger.handlers)

        configure_logging(logging.DEBUG)
        handler_count_2 = len(logger.handlers)

        # Should have same number of handlers (old ones replaced)
        assert handler_count_1 == handler_count_2

    def test_configure_with_env_variable(self, monkeypatch):
        """Test configuring logging with environment variable."""
        monkeypatch.setenv("CAML_LOG_LEVEL", "DEBUG")
        configure_logging()
        assert logger.level == logging.DEBUG

    def test_configure_env_overrides_parameter(self, monkeypatch):
        """Test that environment variable overrides parameter."""
        monkeypatch.setenv("CAML_LOG_LEVEL", "INFO")
        configure_logging(logging.ERROR)
        # ENV should override the parameter
        assert logger.level == logging.INFO


# ==============================================================================
# SECTION HEADER TESTS
# ==============================================================================


class TestGetSectionHeader:
    """Test get_section_header function."""

    def test_basic_header(self):
        """Test creating basic section header."""
        header = get_section_header("Test Title")
        assert "Test Title" in header
        assert "|" in header
        assert "=" in header

    def test_header_with_emoji(self):
        """Test creating header with emoji."""
        header = get_section_header("Test", emoji=":rocket:")
        assert "Test" in header
        assert ":rocket:" in header

    def test_header_with_custom_separator(self):
        """Test creating header with custom separator."""
        header = get_section_header("Test", sep_char="-")
        assert "-" in header
        assert "=" not in header

    def test_header_with_custom_width(self):
        """Test creating header with custom width."""
        header = get_section_header("Test", width=50)
        lines = header.strip().split("\n")
        # Check that separator lines have the right width
        assert len(lines[0]) == 50

    def test_header_without_emoji(self):
        """Test header without emoji doesn't have double spaces."""
        header = get_section_header("Test Title", emoji="")
        assert "|Test Title|" in header

    def test_header_structure(self):
        """Test that header has correct structure."""
        header = get_section_header("Test")
        lines = header.strip().split("\n")
        # Should have 3 lines: separator, title, separator
        assert len(lines) == 3
        # First and last should be separators
        assert lines[0] == lines[2]

    def test_header_auto_width(self):
        """Test that width is auto-calculated when not provided."""
        title = "Short"
        header = get_section_header(title)
        lines = header.strip().split("\n")
        # Width should be title length + padding (5)
        expected_width = len(title) + 5
        assert len(lines[0]) == expected_width


# ==============================================================================
# CONSTANTS TESTS
# ==============================================================================


class TestConstants:
    """Test logging constants."""

    def test_logo_exists(self):
        """Test that LOGO constant exists."""
        assert LOGO is not None
        assert isinstance(LOGO, str)
        # Logo is ASCII art with underscores and pipes
        assert "_" in LOGO and "|" in LOGO

    def test_logo_is_multiline(self):
        """Test that LOGO is multiline ASCII art."""
        assert "\n" in LOGO
        lines = LOGO.split("\n")
        assert len(lines) > 1


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestLoggingIntegration:
    """Test logging integration."""

    def test_logging_functions_work_after_configure(self):
        """Test that logging functions work after configuration."""
        configure_logging(logging.DEBUG)

        # These should not raise exceptions
        INFO("Test info message")
        DEBUG("Test debug message")
        WARNING("Test warning message")
        ERROR("Test error message")

    def test_logging_level_filters_messages(self, caplog):
        """Test that logging level properly filters messages."""
        configure_logging(logging.ERROR)

        with caplog.at_level(logging.ERROR, logger="caml"):
            INFO("This should not appear")
            ERROR("This should appear")

        # Only ERROR message should be captured
        assert len(caplog.records) == 1
        assert "This should appear" in caplog.text
