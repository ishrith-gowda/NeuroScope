"""preprocessing module for neuroscope.

this module provides comprehensive preprocessing capabilities for medical imaging data,
including normalization, bias correction, registration, and skull stripping.
"""

from . import normalization

__all__ = [
    "normalization"
]