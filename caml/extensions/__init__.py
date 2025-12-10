"""Internal extensions for CaML including plotting utilities and synthetic data generation.

Note: Users should import from caml.plots, caml.synthetic_data, etc.
This internal structure is for code organization only.
"""

from . import plots, synthetic_data

# Internal use only - not in __all__ to discourage direct usage
# Users should use caml.synthetic_data instead of caml.extensions.synthetic_data
