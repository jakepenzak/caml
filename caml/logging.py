import logging

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

    logger.debug(f"Logging configured with level: {logging.getLevelName(level)}")


def set_log_level(level: int):
    """
    Change the logging level after initial configuration.

    Parameters
    ----------
    level
        The new logging level to use.
    """
    logger.setLevel(level)
