"""
federated learning simulation for cyclegan harmonization.

provides fedavg and fedprox implementations for simulating multi-site
federated training of sa-cyclegan-2.5d without centralizing patient data.
"""

from neuroscope.training.federated.fedavg import FedAvgAggregator
from neuroscope.training.federated.strategies import FedProxAggregator

__all__ = ["FedAvgAggregator", "FedProxAggregator"]
