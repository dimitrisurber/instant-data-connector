"""Instant Data Connector - Fast data aggregation and serialization."""

import warnings

# New FDW-based classes (recommended)
from .postgresql_fdw_manager import PostgreSQLFDWConnector
from .fdw_manager import FDWManager
from .lazy_query_builder import LazyQueryBuilder
from .virtual_table_manager import VirtualTableManager
from .config_parser import ConfigParser

# Secure serialization (recommended)
from .secure_serializer import SecureSerializer, load_data_connector, save_data_connector

# Legacy classes (deprecated but maintained for compatibility)
from .aggregator import InstantDataConnector as LegacyInstantDataConnector
from .pickle_manager import PickleManager
from .pickle_manager import load_data_connector as unsafe_load_data_connector
from .pickle_manager import save_data_connector as unsafe_save_data_connector

# New main connector class
from .connector import InstantDataConnector

# Credential management
from .secure_credentials import SecureCredentialManager

__version__ = "0.2.0"  # Updated version for FDW support

# Main exports (new FDW-based system)
__all__ = [
    # Primary connector class
    "InstantDataConnector",
    
    # FDW management classes
    "PostgreSQLFDWConnector",
    "FDWManager", 
    "LazyQueryBuilder",
    "VirtualTableManager",
    "ConfigParser",
    
    # Security and credentials
    "SecureCredentialManager",
    "SecureSerializer",
    
    # Secure serialization functions (replacing unsafe pickle)
    "load_data_connector", 
    "save_data_connector",
    
    # Legacy compatibility (deprecated)
    "LegacyInstantDataConnector",
    "PickleManager",
    "unsafe_load_data_connector", 
    "unsafe_save_data_connector"
]

# Security and backward compatibility warnings
def _show_deprecation_warning():
    """Show deprecation warning for legacy usage."""
    warnings.warn(
        "Direct data extraction methods are deprecated. "
        "Use the new FDW-based InstantDataConnector for better performance and scalability. "
        "See migration guide in documentation.",
        DeprecationWarning,
        stacklevel=3
    )

def _show_pickle_security_warning():
    """Show critical security warning for unsafe pickle usage."""
    warnings.warn(
        "CRITICAL SECURITY WARNING: You are using unsafe pickle operations that pose "
        "a REMOTE CODE EXECUTION risk (CVSS 9.8). Use 'load_data_connector' and "
        "'save_data_connector' from instant_connector (secure) instead of "
        "'unsafe_load_data_connector' or 'PickleManager' (insecure). "
        "See security documentation for migration guide.",
        SecurityWarning,
        stacklevel=3
    )

# Custom security warning class
class SecurityWarning(UserWarning):
    """Warning class for security-related issues."""
    pass

# Store original unsafe functions before wrapping
_original_unsafe_load = unsafe_load_data_connector
_original_unsafe_save = unsafe_save_data_connector

# Secure wrapper functions that warn about unsafe usage
def _secure_load_wrapper(*args, **kwargs):
    """Secure wrapper that warns and redirects to unsafe function."""
    _show_pickle_security_warning()
    return _original_unsafe_load(*args, **kwargs)

def _secure_save_wrapper(*args, **kwargs):
    """Secure wrapper that warns and redirects to unsafe function."""
    _show_pickle_security_warning()
    return _original_unsafe_save(*args, **kwargs)

# Replace unsafe functions with secure wrappers
unsafe_load_data_connector = _secure_load_wrapper
unsafe_save_data_connector = _secure_save_wrapper

# Maintain backward compatibility by providing legacy imports
try:
    # If someone imports the old way, show deprecation warning
    import sys
    if any('aggregator' in str(frame.filename) for frame in sys._current_frames().values()):
        _show_deprecation_warning()
except:
    pass