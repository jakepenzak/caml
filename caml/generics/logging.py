"""Logging utilities for CaML."""

import logging
import warnings

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Get the package logger
logger = logging.getLogger("caml")

INFO = logger.info
DEBUG = logger.debug
WARNING = logger.warning
ERROR = logger.error

# Add null handler by default
logger.addHandler(logging.NullHandler())

# Default to WARNING for the library
logger.setLevel(logging.WARNING)

custom_theme = Theme(
    {
        "logging.level.debug": "cyan",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold magenta",
        "logging.message": "white",
        "logging.time": "dim cyan",
    }
)


def configure_logging(level: int = logging.WARNING):
    """
    Configure logging for the entire application.

    Parameters
    ----------
    level
        The logging level to use. Defaults to WARNING.
        Can be overridden by environment variable CAML_LOG_LEVEL.
    """
    import os

    # Allow environment variable to override log level
    env_level = os.getenv("CAML_LOG_LEVEL", "").upper()
    if env_level and hasattr(logging, env_level):
        level = getattr(logging, env_level)

    # Remove existing handlers to allow reconfiguration
    logger.handlers = []

    # Create and add rich handler
    console = Console(theme=custom_theme)
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
    )
    logger.addHandler(handler)

    # Set levels
    logger.setLevel(level)

    # Configure library loggers
    logging.getLogger("patsy").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    logger.debug(f"Logging configured with level: {logging.getLevelName(level)}")


def get_section_header(
    title: str, emoji: str = "", sep_char: str = "=", width: int | None = None
) -> str:
    """
    Generate a formatted section header with separators.

    Parameters
    ----------
    title : str
        The title text to display
    emoji : str
        Optional emoji to include in the title
    width : int | None
        Width for the separators. If None, calculated from title length.

    Returns
    -------
    str
        Formatted header with top and bottom separators
    """
    if width is None:
        # Calculate width based on title length + padding
        width = len(title) + 5  # 2.5 chars padding on each side

    separator = sep_char * width
    formatted_title = f"|{emoji} {title}|" if emoji else f"|{title}|"

    return f"\n{separator}\n{formatted_title}\n{separator}\n"


LOGO = r"""
  ____      __  __ _
 / ___|__ _|  \/  | |
| |   / _` | |\/| | |
| |__| (_| | |  | | |___
 \____\__,_|_|  |_|_____|

"""

## AutoCATE Narrations
AUTOML_NUISANCE_PREAMBLE = get_section_header("AutoML Nuisance Functions", ":dart:")
AUTOML_CATE_PREAMBLE = get_section_header("AutoML CATE Functions", ":dart:")
CATE_TESTING_PREAMBLE = get_section_header("Testing Results", ":test_tube:")
REFIT_FINAL_PREAMBLE = get_section_header("Refitting Final Estimator", ":battery:")
