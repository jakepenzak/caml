"""CaML logging utilities.

By default CaML is silent — it attaches only a ``NullHandler`` to the ``"caml"``
logger and lets the host application decide how to route and display logs.

Call `configure_logging` once at the start of your script or notebook:

```{python}
from caml import configure_logging

configure_logging()          # INFO — sensible default
configure_logging(verbose=0) # warnings/errors only; third-party noise suppressed
configure_logging(verbose=2) # full debug output
```
"""

import logging
import warnings

import optuna
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

logger = logging.getLogger("caml")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

# Mapping from verbosity levels to logging levels
_VERBOSITY_MAP: dict[int, int] = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

# Third-party loggers routed through CaML's handler after configure_logging().
_THIRD_PARTY_LOGGERS: tuple[str, ...] = ("optuna", "flaml.automl.logger")

# Known-benign warnings suppressed
_WARNING_FILTERS: tuple[tuple[str, type[Warning]], ...] = (
    (r".*force_all_finite.*", FutureWarning),
    (r".*too small for a sparse model.*", UserWarning),
    (r".*A column-vector y was passed.*", UserWarning),
    (r".*X does not have valid feature names.*", UserWarning),
    (
        r".*Setting a suboptimal alpha can lead to miscalibrated confidence intervals.*",
        UserWarning,
    ),
)

# Custom Rich theme for log levels and message components
_custom_theme = Theme(
    {
        "logging.level.debug": "cyan",
        "logging.level.info": "green",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold magenta",
        "logging.message": "white",
        "logging.time": "dim cyan",
    }
)


def _verbosity_to_level(verbose: int) -> int:
    if verbose <= 0:
        return logging.WARNING
    if verbose >= 2:
        return logging.DEBUG
    return logging.INFO


def configure_logging(verbose: int = 1) -> None:
    """Configure CaML's logger for Rich-formatted terminal output.

    If Non-Null handlers are already attached to the root logger, this function will not add a new handler to avoid duplicate logs.
    However, it will still set the logging level for CaML and the specified verbosity.

    If no handlers are attached, it adds a RichHandler to the root logger, ensuring that all logs (including those from third-party libraries) are formatted uniformly.

    This function also suppresses known-benign warnings from third-party libraries to reduce noise in the logs.

    Verbosity levels:

    * ``0`` — warnings/errors only; third-party warnings suppressed.
    * ``1`` — INFO. *Default.*
    * ``2`` — DEBUG.

    Parameters
    ----------
    verbose
        Verbosity level. Higher values yield more detailed logs. Options: ``0``, ``1``, ``2``.

    Examples
    --------
    ```{python}
    from caml import configure_logging

    configure_logging()           # INFO — sensible default
    configure_logging(verbose=0)  # silent
    configure_logging(verbose=2)  # debug
    ```
    """
    level = _verbosity_to_level(verbose)

    # Suppress known-benign warnings from third-party libraries (e.g. scikit-learn)
    for pattern, category in _WARNING_FILTERS:
        warnings.filterwarnings("ignore", message=pattern, category=category)  # type: ignore[arg-type]

    handler = RichHandler(
        console=Console(
            theme=_custom_theme, force_terminal=True, markup=True, record=True
        ),
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )
    handler.setLevel(level)

    # Remove only handlers we own (NullHandlers and previously-installed RichHandlers).
    # User-added handlers (FileHandler, StreamHandler, etc.) are intentionally preserved.
    logger.handlers = [
        h
        for h in logger.handlers
        if not isinstance(h, (logging.NullHandler, RichHandler))
    ]
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = True  # allow any user root-handlers to also receive caml logs

    try:
        optuna.logging.disable_default_handler()
    except AssertionError:
        pass

    # Third-party loggers — share the same handler for uniform output
    for name in _THIRD_PARTY_LOGGERS:
        tp = logging.getLogger(name)
        tp.handlers = [
            h
            for h in tp.handlers
            if not isinstance(h, (logging.NullHandler, RichHandler))
        ]
        tp.addHandler(handler)
        tp.setLevel(level)
        tp.propagate = False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _get_section_header(
    title: str, emoji: str = "", sep_char: str = "=", width: int | None = None
) -> str:
    """Generate a formatted section header string with separators."""
    if width is None:
        width = len(title) + 5
    separator = sep_char * width
    formatted_title = f"|{emoji} {title}|" if emoji else f"|{title}|"
    return f"\n{separator}\n{formatted_title}\n{separator}\n"


_LOGO = r"""
  ____      __  __ _
 / ___|__ _|  \/  | |
| |   / _` | |\/| | |
| |__| (_| | |  | | |___
 \____\__,_|_|  |_|_____|

"""

_AUTO_NUISANCE_PREAMBLE = _get_section_header("Auto Nuisance Tuning", ":dart:")
_AUTO_CATE_PREAMBLE = _get_section_header("Auto CATE Tuning", ":dart:")
_OOS_TESTING_PREAMBLE = _get_section_header("Scoring Test Set", ":test_tube:")
_REFIT_FINAL_PREAMBLE = _get_section_header("Refitting Final Estimator", ":battery:")
