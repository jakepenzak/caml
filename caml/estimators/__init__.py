import sys

from .base import BaseWrapperMixin
from .wrappers import dml, dr, meta, orf

# Create aliases so `from caml.estimators.dml import X` works
sys.modules["caml.estimators.dml"] = dml
sys.modules["caml.estimators.dr"] = dr
sys.modules["caml.estimators.meta"] = meta
sys.modules["caml.estimators.orf"] = orf

__all__ = ["BaseWrapperMixin", "dml", "dr", "meta", "orf"]
