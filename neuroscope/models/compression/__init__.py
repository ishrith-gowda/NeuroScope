"""
neural compression modules for harmonize-and-compress architecture.

integrates learned image compression into the cyclegan harmonization pipeline,
enabling simultaneous domain adaptation and bitrate-efficient encoding.
"""

from neuroscope.models.compression.entropy_model import FactorizedPrior, HyperpriorModel
from neuroscope.models.compression.quantization import UniformQuantize, NoiseQuantize
from neuroscope.models.compression.compressed_generator import CompressedSAGenerator25D

__all__ = [
    "FactorizedPrior",
    "HyperpriorModel",
    "UniformQuantize",
    "NoiseQuantize",
    "CompressedSAGenerator25D",
]
