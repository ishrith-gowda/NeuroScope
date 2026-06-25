"""
neural compression modules for harmonize-and-compress architecture.

integrates learned image compression into the cyclegan harmonization pipeline,
enabling simultaneous domain adaptation and bitrate-efficient encoding.
"""

from neuroscope.models.compression.compressed_generator import CompressedSAGenerator25D
from neuroscope.models.compression.entropy_model import FactorizedPrior, HyperpriorModel
from neuroscope.models.compression.quantization import NoiseQuantize, UniformQuantize

__all__ = [
    "CompressedSAGenerator25D",
    "FactorizedPrior",
    "HyperpriorModel",
    "NoiseQuantize",
    "UniformQuantize",
]
