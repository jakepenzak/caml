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


def set_log_level(level: int):
    """
    Change the logging level after initial configuration.

    Parameters
    ----------
    level
        The new logging level to use.
    """
    logger.setLevel(level)


def get_terminal_width(default_width: int = 80) -> int:
    """
    Get the terminal width for formatting output.

    Parameters
    ----------
    default_width
        Default width to use if terminal width cannot be detected.

    Returns
    -------
    int
        The terminal width in characters.
    """
    try:
        # Try to get width from existing Rich handlers
        for handler in logger.handlers:
            if isinstance(handler, RichHandler):
                console = handler.console
                if console and hasattr(console, "size"):
                    return console.size.width

        # Fallback: create a temporary console to get width
        console = Console()
        return console.size.width
    except Exception:
        # Fallback to default if anything goes wrong
        return default_width


def get_separator(char: str = "=", width: int | None = None, min_width: int = 20, max_width: int = 120) -> str:
    """
    Get a separator line that adapts to terminal width.

    Parameters
    ----------
    char
        Character to use for the separator.
    width
        Specific width to use. If None, auto-detects terminal width.
    min_width
        Minimum width for the separator.
    max_width
        Maximum width for the separator.

    Returns
    -------
    str
        A separator string of appropriate length.
    """
    if width is None:
        width = get_terminal_width()

    # Constrain width to reasonable bounds
    width = max(min_width, min(width, max_width))

    return char * width
