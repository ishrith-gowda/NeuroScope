"""data handling module for neuroscope.

this module provides comprehensive data handling capabilities including
loaders, transforms, datasets, and data splitting utilities.
"""

from . import datasets, loaders, splits, transforms

__all__ = ["datasets", "loaders", "splits", "transforms"]
