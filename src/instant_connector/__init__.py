"""Instant Data Connector - ML-optimized data aggregation and serialization."""

from .aggregator import InstantDataConnector
from .pickle_manager import PickleManager, load_data_connector, save_data_connector
from .ml_optimizer import MLOptimizer

__version__ = "0.1.0"
__all__ = ["InstantDataConnector", "PickleManager", "MLOptimizer", "load_data_connector", "save_data_connector"]