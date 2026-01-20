"""Internal estimator organization by data type.

Note: Users should import from caml.cross_section, caml.panel, etc.
This internal structure is for code organization only.
"""

from . import unconfoundedness

# Internal use only - not in __all__ to discourage direct usage
# Users should use caml.cross_section instead of caml.estimators.cross_section
