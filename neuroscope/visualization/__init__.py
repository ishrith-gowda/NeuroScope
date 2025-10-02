"""Visualization module for NeuroScope.

This module provides comprehensive visualization tools for medical imaging data,
including plotting utilities, montages, and interactive dashboards.
"""

from . import plotters
from . import montages
from . import dashboards

__all__ = [
    "plotters",
    "montages",
    "dashboards"
]