import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

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

console = Console(theme=custom_theme)

rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    markup=True,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[rich_handler]
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging has been set up.")


setup_logging()
