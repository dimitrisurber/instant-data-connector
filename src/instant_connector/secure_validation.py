"""Comprehensive input validation and secure error handling utilities."""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import ipaddress

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecureValidator:
    """Comprehensive input validation with security controls."""
    
    # Regex patterns for validation
    PATTERNS = {
        'alphanumeric': re.compile(r'^[a-zA-Z0-9_-]+$'),
        'safe_string': re.compile(r'^[a-zA-Z0-9_\-\.\s]+$'),
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'sql_identifier': re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$'),
        'file_name': re.compile(r'^[a-zA-Z0-9_\-\.]+$'),
        'url_safe': re.compile(r'^[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=%]+$')
    }
    
    # Dangerous patterns that should be rejected
    DANGEROUS_PATTERNS = [
        re.compile(r'[<>"\';\\&|`$]'),  # Shell/SQL injection characters
        re.compile(r'\.\./'),            # Path traversal
        re.compile(r'__[a-zA-Z_]+__'),   # Python magic methods
        re.compile(r'eval|exec|import|__import__|open|file'),  # Dangerous Python functions
        re.compile(r'(union|select|insert|update|delete|drop|create|alter)\s+', re.IGNORECASE),  # SQL keywords
        re.compile(r'<script|javascript:|vbscript:|onload=|onerror=', re.IGNORECASE)  # XSS patterns
    ]
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize secure validator.
        
        Args:
            strict_mode: Whether to use strict validation rules
        """
        self.strict_mode = strict_mode
    
    def validate_string(
        self,
        value: Any,
        name: str,
        pattern: Optional[str] = None,
        min_length: int = 0,
        max_length: int = 1000,
        allow_empty: bool = True
    ) -> str:
        """
        Validate string input with security checks.
        
        Args:
            value: Value to validate
            name: Name of the field for error messages
            pattern: Pattern name from PATTERNS dict
            min_length: Minimum string length
            max_length: Maximum string length
            allow_empty: Whether to allow empty strings
            
        Returns:
            Validated string
        """
        if value is None:
            if allow_empty:
                return ""
            raise ValidationError(f"{name} cannot be None")
        
        if not isinstance(value, (str, int, float)):
            raise ValidationError(f"{name} must be a string, int, or float, got {type(value)}")
        
        str_value = str(value).strip()
        
        if not allow_empty and not str_value:
            raise ValidationError(f"{name} cannot be empty")
        
        if len(str_value) < min_length:
            raise ValidationError(f"{name} must be at least {min_length} characters")
        
        if len(str_value) > max_length:
            raise ValidationError(f"{name} must be at most {max_length} characters")
        
        # Check for dangerous patterns
        for dangerous_pattern in self.DANGEROUS_PATTERNS:
            if dangerous_pattern.search(str_value):
                raise ValidationError(f"{name} contains potentially dangerous characters or patterns")
        
        # Validate against specific pattern if provided
        if pattern and pattern in self.PATTERNS:
            if not self.PATTERNS[pattern].match(str_value):
                raise ValidationError(f"{name} does not match required pattern: {pattern}")
        
        return str_value
    
    def validate_integer(
        self,
        value: Any,
        name: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """
        Validate integer input.
        
        Args:
            value: Value to validate
            name: Name of the field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated integer
        """
        if value is None:
            raise ValidationError(f"{name} cannot be None")
        
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{name} must be an integer, got {type(value)}")
        
        if min_value is not None and int_value < min_value:
            raise ValidationError(f"{name} must be at least {min_value}")
        
        if max_value is not None and int_value > max_value:
            raise ValidationError(f"{name} must be at most {max_value}")
        
        return int_value
    
    def validate_float(
        self,
        value: Any,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """
        Validate float input.
        
        Args:
            value: Value to validate
            name: Name of the field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated float
        """
        if value is None:
            raise ValidationError(f"{name} cannot be None")
        
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{name} must be a number, got {type(value)}")
        
        if not np.isfinite(float_value):
            raise ValidationError(f"{name} must be a finite number")
        
        if min_value is not None and float_value < min_value:
            raise ValidationError(f"{name} must be at least {min_value}")
        
        if max_value is not None and float_value > max_value:
            raise ValidationError(f"{name} must be at most {max_value}")
        
        return float_value
    
    def validate_path(
        self,
        value: Any,
        name: str,
        must_exist: bool = False,
        allowed_extensions: Optional[List[str]] = None,
        allowed_directories: Optional[List[Path]] = None
    ) -> Path:
        """
        Validate file path with security checks.
        
        Args:
            value: Path value to validate
            name: Name of the field for error messages
            must_exist: Whether the path must exist
            allowed_extensions: List of allowed file extensions
            allowed_directories: List of allowed parent directories
            
        Returns:
            Validated Path object
        """
        if value is None:
            raise ValidationError(f"{name} cannot be None")
        
        try:
            path = Path(value)
        except (TypeError, ValueError) as e:
            raise ValidationError(f"{name} is not a valid path: {e}")
        
        # Resolve to absolute path
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValidationError(f"{name} path resolution failed: {e}")
        
        # Check for suspicious path elements
        path_str = str(resolved_path)
        if '..' in path_str or '~' in path_str:
            raise ValidationError(f"{name} contains potentially dangerous path elements")
        
        # Check if path must exist
        if must_exist and not resolved_path.exists():
            raise ValidationError(f"{name} path does not exist: {resolved_path}")
        
        # Check allowed extensions
        if allowed_extensions and resolved_path.suffix.lower() not in allowed_extensions:
            raise ValidationError(f"{name} must have one of these extensions: {allowed_extensions}")
        
        # Check allowed directories
        if allowed_directories:
            path_allowed = False
            for allowed_dir in allowed_directories:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    path_allowed = True
                    break
                except ValueError:
                    continue
            
            if not path_allowed:
                raise ValidationError(f"{name} is not in an allowed directory")
        
        return resolved_path
    
    def validate_url(
        self,
        value: Any,
        name: str,
        allowed_schemes: Optional[List[str]] = None,
        allow_localhost: bool = False
    ) -> str:
        """
        Validate URL with security checks.
        
        Args:
            value: URL to validate
            name: Name of the field for error messages
            allowed_schemes: List of allowed URL schemes
            allow_localhost: Whether to allow localhost URLs
            
        Returns:
            Validated URL string
        """
        if value is None:
            raise ValidationError(f"{name} cannot be None")
        
        url_str = self.validate_string(value, name, pattern='url_safe', max_length=2000)
        
        try:
            parsed = urlparse(url_str)
        except Exception as e:
            raise ValidationError(f"{name} is not a valid URL: {e}")
        
        if not parsed.scheme:
            raise ValidationError(f"{name} must include a scheme (http, https, etc.)")
        
        if not parsed.netloc:
            raise ValidationError(f"{name} must include a host")
        
        # Check allowed schemes
        if allowed_schemes and parsed.scheme.lower() not in [s.lower() for s in allowed_schemes]:
            raise ValidationError(f"{name} scheme must be one of: {allowed_schemes}")
        
        # Check for localhost/private IPs if not allowed
        if not allow_localhost:
            hostname = parsed.hostname
            if hostname:
                if hostname.lower() in ['localhost', '127.0.0.1', '::1']:
                    raise ValidationError(f"{name} cannot use localhost")
                
                # Check for private IP ranges
                try:
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        raise ValidationError(f"{name} cannot use private/local IP addresses")
                except ValueError:
                    # Not an IP address, which is fine
                    pass
        
        return url_str
    
    def validate_dict(
        self,
        value: Any,
        name: str,
        required_keys: Optional[List[str]] = None,
        allowed_keys: Optional[List[str]] = None,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Validate dictionary input.
        
        Args:
            value: Dictionary to validate
            name: Name of the field for error messages
            required_keys: List of required keys
            allowed_keys: List of allowed keys
            max_depth: Maximum nesting depth
            
        Returns:
            Validated dictionary
        """
        if value is None:
            raise ValidationError(f"{name} cannot be None")
        
        if not isinstance(value, dict):
            raise ValidationError(f"{name} must be a dictionary, got {type(value)}")
        
        # Check dictionary depth
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                raise ValidationError(f"{name} dictionary nesting too deep (max: {max_depth})")
            
            if isinstance(obj, dict):
                for v in obj.values():
                    check_depth(v, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1)
        
        check_depth(value)
        
        # Check required keys
        if required_keys:
            missing_keys = [key for key in required_keys if key not in value]
            if missing_keys:
                raise ValidationError(f"{name} missing required keys: {missing_keys}")
        
        # Check allowed keys
        if allowed_keys:
            extra_keys = [key for key in value.keys() if key not in allowed_keys]
            if extra_keys:
                raise ValidationError(f"{name} contains disallowed keys: {extra_keys}")
        
        # Validate all string keys and values
        validated_dict = {}
        for key, val in value.items():
            # Validate key
            validated_key = self.validate_string(key, f"{name} key", max_length=100)
            
            # Recursively validate nested dictionaries
            if isinstance(val, dict):
                validated_dict[validated_key] = self.validate_dict(val, f"{name}[{key}]", max_depth=max_depth-1)
            elif isinstance(val, str):
                validated_dict[validated_key] = self.validate_string(val, f"{name}[{key}]")
            else:
                validated_dict[validated_key] = val
        
        return validated_dict
    
    def validate_dataframe(
        self,
        value: Any,
        name: str,
        max_rows: int = 1_000_000,
        max_cols: int = 1000,
        max_memory_mb: int = 500
    ) -> pd.DataFrame:
        """
        Validate DataFrame input.
        
        Args:
            value: DataFrame to validate
            name: Name of the field for error messages
            max_rows: Maximum number of rows
            max_cols: Maximum number of columns
            max_memory_mb: Maximum memory usage in MB
            
        Returns:
            Validated DataFrame
        """
        if value is None:
            raise ValidationError(f"{name} cannot be None")
        
        if not isinstance(value, pd.DataFrame):
            raise ValidationError(f"{name} must be a pandas DataFrame, got {type(value)}")
        
        if value.shape[0] > max_rows:
            raise ValidationError(f"{name} has too many rows: {value.shape[0]} > {max_rows}")
        
        if value.shape[1] > max_cols:
            raise ValidationError(f"{name} has too many columns: {value.shape[1]} > {max_cols}")
        
        memory_usage_mb = value.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_usage_mb > max_memory_mb:
            raise ValidationError(f"{name} uses too much memory: {memory_usage_mb:.1f} MB > {max_memory_mb} MB")
        
        # Validate column names
        for col in value.columns:
            self.validate_string(col, f"{name} column name", pattern='safe_string', max_length=100)
        
        return value


class SecureErrorHandler:
    """Secure error handling that doesn't leak sensitive information."""
    
    def __init__(self, log_sensitive_errors: bool = False):
        """
        Initialize secure error handler.
        
        Args:
            log_sensitive_errors: Whether to log full error details (development only)
        """
        self.log_sensitive_errors = log_sensitive_errors
    
    def sanitize_error_message(self, error: Exception, context: str = "") -> str:
        """
        Sanitize error message to remove sensitive information.
        
        Args:
            error: Original exception
            context: Context information for the error
            
        Returns:
            Sanitized error message
        """
        error_msg = str(error)
        
        # Remove potential sensitive information
        sensitive_patterns = [
            r'password[=:]\s*\S+',
            r'key[=:]\s*\S+',
            r'token[=:]\s*\S+',
            r'secret[=:]\s*\S+',
            r'/home/[^/\s]+',
            r'C:\\Users\\[^\\s]+',
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # Email addresses
        ]
        
        sanitized_msg = error_msg
        for pattern in sensitive_patterns:
            sanitized_msg = re.sub(pattern, '[REDACTED]', sanitized_msg, flags=re.IGNORECASE)
        
        # Log full error if configured (development mode)
        if self.log_sensitive_errors:
            logger.debug(f"Full error details - {context}: {error_msg}")
            logger.debug(f"Error type: {type(error).__name__}")
            if hasattr(error, '__traceback__'):
                import traceback
                logger.debug(f"Traceback: {''.join(traceback.format_tb(error.__traceback__))}")
        
        # Return generic error for security
        if context:
            return f"An error occurred in {context}: {sanitized_msg}"
        else:
            return f"An error occurred: {sanitized_msg}"
    
    def handle_database_error(self, error: Exception) -> str:
        """Handle database-specific errors securely."""
        # Log full error for debugging
        if self.log_sensitive_errors:
            logger.error(f"Database error: {error}")
        
        # Return generic message
        if "connection" in str(error).lower():
            return "Database connection failed. Please check your configuration."
        elif "authentication" in str(error).lower():
            return "Database authentication failed. Please check your credentials."
        elif "permission" in str(error).lower():
            return "Database permission denied. Please check your access rights."
        else:
            return "A database error occurred. Please contact support."
    
    def handle_file_error(self, error: Exception, file_path: str = "") -> str:
        """Handle file operation errors securely."""
        # Log full error for debugging
        if self.log_sensitive_errors:
            logger.error(f"File error for {file_path}: {error}")
        
        # Return generic message without exposing file paths
        if "permission" in str(error).lower():
            return "File permission denied."
        elif "not found" in str(error).lower():
            return "File not found."
        elif "size" in str(error).lower():
            return "File size limit exceeded."
        else:
            return "A file operation error occurred."
    
    def handle_validation_error(self, error: ValidationError) -> str:
        """Handle validation errors (these are usually safe to show)."""
        return str(error)


# Global instances
default_validator = SecureValidator()
default_error_handler = SecureErrorHandler()


# Convenience functions
def validate_string(value: Any, name: str, **kwargs) -> str:
    """Validate string using default validator."""
    return default_validator.validate_string(value, name, **kwargs)


def validate_integer(value: Any, name: str, **kwargs) -> int:
    """Validate integer using default validator."""
    return default_validator.validate_integer(value, name, **kwargs)


def validate_path(value: Any, name: str, **kwargs) -> Path:
    """Validate path using default validator."""
    return default_validator.validate_path(value, name, **kwargs)


def validate_url(value: Any, name: str, **kwargs) -> str:
    """Validate URL using default validator."""
    return default_validator.validate_url(value, name, **kwargs)


def sanitize_error(error: Exception, context: str = "") -> str:
    """Sanitize error message using default error handler."""
    return default_error_handler.sanitize_error_message(error, context)