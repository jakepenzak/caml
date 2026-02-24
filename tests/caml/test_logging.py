"""Tests for caml.logging module."""

import logging
import warnings

import pytest

from caml.logging import (
    _THIRD_PARTY_LOGGERS,
    _WARNING_FILTERS,
    LOGO,
    VERBOSITY_MAP,
    _verbosity_to_level,
    configure_logging,
    debug,
    error,
    get_section_header,
    info,
    logger,
    suppress_third_party_warnings,
    warning,
)

pytestmark = pytest.mark.generics


class TestLogger:
    def test_logger_name(self):
        assert logger.name == "caml"

    def test_aliases_callable(self):
        assert all(callable(fn) for fn in (info, debug, warning, error))


class TestVerbosityMap:
    def test_map_values(self):
        assert VERBOSITY_MAP == {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def test_clamp_low(self):
        assert _verbosity_to_level(-1) == logging.WARNING

    def test_clamp_high(self):
        assert _verbosity_to_level(3) == logging.DEBUG

    def test_exact(self):
        assert _verbosity_to_level(0) == logging.WARNING
        assert _verbosity_to_level(1) == logging.INFO
        assert _verbosity_to_level(2) == logging.DEBUG


class TestConfigureLogging:
    def test_default_is_info(self):
        configure_logging()
        assert logger.level == logging.INFO

    def test_verbose_0(self):
        configure_logging(verbose=0)
        assert logger.level == logging.WARNING

    def test_verbose_1(self):
        configure_logging(verbose=1)
        assert logger.level == logging.INFO

    def test_verbose_2(self):
        configure_logging(verbose=2)
        assert logger.level == logging.DEBUG

    def test_attaches_handler(self):
        configure_logging()
        assert len(logger.handlers) > 0

    def test_reconfigure_replaces_not_accumulates(self):
        configure_logging(verbose=1)
        n = len(logger.handlers)
        configure_logging(verbose=2)
        assert len(logger.handlers) == n

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CAML_LOG_LEVEL", "DEBUG")
        configure_logging(verbose=0)
        assert logger.level == logging.DEBUG


class TestThirdPartyPropagation:
    def test_third_party_share_handler(self):
        configure_logging(verbose=1)
        handler = logger.handlers[0]
        for name in _THIRD_PARTY_LOGGERS:
            assert handler in logging.getLogger(name).handlers

    def test_third_party_level_matches(self):
        for verbose in (0, 1, 2):
            configure_logging(verbose=verbose)
            for name in _THIRD_PARTY_LOGGERS:
                assert logging.getLogger(name).level == logger.level

    def test_third_party_no_propagation(self):
        configure_logging(verbose=1)
        for name in _THIRD_PARTY_LOGGERS:
            assert logging.getLogger(name).propagate is False

    def test_no_duplicate_handlers_on_reconfigure(self):
        configure_logging(verbose=1)
        configure_logging(verbose=2)
        for name in _THIRD_PARTY_LOGGERS:
            assert len(logging.getLogger(name).handlers) == 1

    def test_warnings_captured(self):
        configure_logging(verbose=1)
        pw = logging.getLogger("py.warnings")
        assert len(pw.handlers) > 0
        assert pw.propagate is False


class TestSuppressWarnings:
    def test_callable(self):
        suppress_third_party_warnings()  # must not raise

    def test_adds_ignore_filters(self):
        warnings.resetwarnings()
        suppress_third_party_warnings()
        assert len([f for f in warnings.filters if f[0] == "ignore"]) == len(
            _WARNING_FILTERS
        )

    def test_auto_at_verbose_0(self):
        warnings.resetwarnings()
        configure_logging(verbose=0)
        assert len([f for f in warnings.filters if f[0] == "ignore"]) >= len(
            _WARNING_FILTERS
        )

    def test_not_auto_at_verbose_1(self):
        warnings.resetwarnings()
        configure_logging(verbose=1)
        assert len([f for f in warnings.filters if f[0] == "ignore"]) == 0


class TestGetSectionHeader:
    def test_contains_title(self):
        assert "Title" in get_section_header("Title")

    def test_separator_char(self):
        assert "-" in get_section_header("T", sep_char="-")
        assert "=" not in get_section_header("T", sep_char="-")

    def test_custom_width(self):
        lines = get_section_header("T", width=20).strip().split("\n")
        assert len(lines[0]) == 20

    def test_auto_width(self):
        title = "Hello"
        lines = get_section_header(title).strip().split("\n")
        assert len(lines[0]) == len(title) + 5

    def test_structure(self):
        lines = get_section_header("T").strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == lines[2]


class TestConstants:
    def test_logo_is_ascii_art(self):
        assert "_" in LOGO and "|" in LOGO and "\n" in LOGO


class TestIntegration:
    def test_level_filters_messages(self, caplog):
        configure_logging(verbose=0)
        with caplog.at_level(logging.WARNING, logger="caml"):
            info("should be hidden")
            warning("should appear")
        assert any("should appear" in r.message for r in caplog.records)
        assert not any("should be hidden" in r.message for r in caplog.records)
