"""Secure serialization manager replacing unsafe pickle operations."""

import json
import gzip
import lz4.frame
import bz2
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import hashlib
import time
import logging
import os
import tempfile
import shutil
from datetime import datetime
import warnings
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    warnings.warn("zstandard not available. Install with 'pip install zstandard' for better compression.")

logger = logging.getLogger(__name__)

# Maximum file size allowed (100MB by default)
MAX_FILE_SIZE = 100 * 1024 * 1024
# Maximum memory usage allowed (500MB by default)
MAX_MEMORY_USAGE = 500 * 1024 * 1024


class SecureSerializationError(Exception):
    """Custom exception for serialization security errors."""
    pass


class SecureSerializer:
    """Secure serialization manager using JSON and encryption instead of pickle."""
    
    COMPRESSION_METHODS = {
        'gzip': {
            'module': gzip,
            'extension': '.gz',
            'open_kwargs': lambda level: {'compresslevel': min(max(level, 1), 9)}
        },
        'lz4': {
            'module': lz4.frame,
            'extension': '.lz4',
            'open_kwargs': lambda level: {'compression_level': min(max(level, 0), 16)}
        },
        'bz2': {
            'module': bz2,
            'extension': '.bz2',
            'open_kwargs': lambda level: {'compresslevel': min(max(level, 1), 9)}
        },
        'none': {
            'module': None,
            'extension': '',
            'open_kwargs': lambda level: {}
        }
    }
    
    if ZSTD_AVAILABLE:
        COMPRESSION_METHODS['zstd'] = {
            'module': zstd,
            'extension': '.zst',
            'open_kwargs': lambda level: {'cctx': zstd.ZstdCompressor(level=min(max(level, 1), 22))}
        }
    
    def __init__(
        self,
        compression: str = 'lz4',
        compression_level: int = 0,
        encryption_key: Optional[bytes] = None,
        max_file_size: int = MAX_FILE_SIZE,
        max_memory_usage: int = MAX_MEMORY_USAGE
    ):
        """
        Initialize secure serializer.
        
        Args:
            compression: Compression method ('gzip', 'lz4', 'bz2', 'zstd', 'none')
            compression_level: Compression level
            encryption_key: Optional encryption key for sensitive data
            max_file_size: Maximum allowed file size in bytes
            max_memory_usage: Maximum allowed memory usage in bytes
        """
        if compression not in self.COMPRESSION_METHODS:
            raise ValueError(f"Unsupported compression: {compression}")
            
        self.compression = compression
        self.compression_level = compression_level
        self.max_file_size = max_file_size
        self.max_memory_usage = max_memory_usage
        self.encryption_key = encryption_key
        
        # Initialize encryption if key provided
        self.cipher_suite = None
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key)
        
        # Statistics tracking
        self.stats = {
            'bytes_written': 0,
            'bytes_read': 0,
            'compression_ratio': 0,
            'serialization_time': 0,
            'deserialization_time': 0
        }
    
    @staticmethod
    def generate_encryption_key() -> bytes:
        """Generate a secure encryption key."""
        return Fernet.generate_key()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _validate_file_size(self, file_path: Path) -> None:
        """Validate file size against limits."""
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise SecureSerializationError(
                    f"File size {file_size} exceeds maximum allowed size {self.max_file_size}"
                )
    
    def _validate_memory_usage(self, data_size: int) -> None:
        """Validate memory usage against limits."""
        if data_size > self.max_memory_usage:
            raise SecureSerializationError(
                f"Data size {data_size} exceeds maximum allowed memory usage {self.max_memory_usage}"
            )
    
    def _secure_temp_file(self, suffix: str = '.tmp') -> str:
        """Create secure temporary file with proper permissions."""
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix='secure_serialize_')
        # Set restrictive permissions (600 - owner read/write only)
        os.chmod(temp_path, 0o600)
        os.close(fd)
        return temp_path
    
    def _dataframe_to_secure_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert DataFrame to secure dictionary format."""
        # Validate memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        self._validate_memory_usage(memory_usage)
        
        # Convert DataFrame to dict, handling timestamps
        df_copy = df.copy()
        
        # Convert datetime columns to ISO strings for JSON serialization
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif df_copy[col].dtype == 'object':
                # Check if any values are Timestamp objects
                def convert_timestamp(x):
                    if pd.isna(x):
                        return x
                    elif hasattr(x, 'strftime'):  # Timestamp-like object
                        return x.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        return x
                df_copy[col] = df_copy[col].apply(convert_timestamp)
        
        return {
            'type': 'DataFrame',
            'data': df_copy.to_dict('records'),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'index': df.index.tolist() if hasattr(df.index, 'tolist') else list(df.index),
            'shape': df.shape,
            'memory_usage_mb': memory_usage / (1024 * 1024)
        }
    
    def _secure_dict_to_dataframe(self, data_dict: Dict[str, Any]) -> pd.DataFrame:
        """Convert secure dictionary back to DataFrame."""
        if data_dict.get('type') != 'DataFrame':
            raise SecureSerializationError("Invalid DataFrame format in serialized data")
        
        # Validate expected fields
        required_fields = ['data', 'columns', 'dtypes']
        for field in required_fields:
            if field not in data_dict:
                raise SecureSerializationError(f"Missing required field: {field}")
        
        # Reconstruct DataFrame
        df = pd.DataFrame(data_dict['data'], columns=data_dict['columns'])
        
        # Restore data types safely
        for col, dtype_str in data_dict['dtypes'].items():
            if col in df.columns:
                try:
                    # Only allow safe data types
                    safe_dtypes = {
                        'int8', 'int16', 'int32', 'int64',
                        'float16', 'float32', 'float64',
                        'bool', 'object', 'category',
                        'datetime64[ns]', 'timedelta64[ns]'
                    }
                    if dtype_str in safe_dtypes:
                        df[col] = df[col].astype(dtype_str)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not restore dtype for column {col}: {e}")
        
        return df
    
    def _prepare_payload_for_serialization(self, data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert payload with DataFrames to serializable format."""
        serializable_payload = {}
        
        for key, value in data_payload.items():
            if isinstance(value, pd.DataFrame):
                serializable_payload[key] = self._dataframe_to_secure_dict(value)
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                nested_dict = {}
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, pd.DataFrame):
                        nested_dict[nested_key] = self._dataframe_to_secure_dict(nested_value)
                    elif isinstance(nested_value, (str, int, float, bool, list, type(None))):
                        nested_dict[nested_key] = nested_value
                    else:
                        # Convert complex objects to string representation
                        nested_dict[nested_key] = str(nested_value)
                serializable_payload[key] = nested_dict
            elif isinstance(value, (str, int, float, bool, list, type(None))):
                serializable_payload[key] = value
            else:
                # Convert other objects to string representation
                serializable_payload[key] = str(value)
        
        return serializable_payload
    
    def _restore_payload_from_serialization(self, serializable_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Restore payload with DataFrames from serializable format."""
        restored_payload = {}
        
        for key, value in serializable_payload.items():
            if isinstance(value, dict):
                if value.get('type') == 'DataFrame':
                    restored_payload[key] = self._secure_dict_to_dataframe(value)
                else:
                    # Recursively process nested dictionaries
                    nested_dict = {}
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, dict) and nested_value.get('type') == 'DataFrame':
                            nested_dict[nested_key] = self._secure_dict_to_dataframe(nested_value)
                        else:
                            nested_dict[nested_key] = nested_value
                    restored_payload[key] = nested_dict
            else:
                restored_payload[key] = value
        
        return restored_payload
    
    def serialize_datasets(
        self,
        data_payload: Dict[str, Any],
        output_path: Union[str, Path],
        add_metadata: bool = True,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Securely serialize dataset payload.
        
        Args:
            data_payload: Dictionary containing data to serialize
            output_path: Output file path
            add_metadata: Add serialization metadata
            validate: Validate data integrity
            
        Returns:
            Serialization statistics and metadata
        """
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare serializable payload
            serializable_payload = self._prepare_payload_for_serialization(data_payload)
            
            # Add metadata if requested
            if add_metadata:
                serializable_payload['_serialization_metadata'] = {
                    'timestamp': datetime.now().isoformat(),
                    'format_version': '2.0_secure',
                    'compression': self.compression,
                    'encryption_enabled': self.cipher_suite is not None,
                    'serializer': 'SecureSerializer'
                }
            
            # Convert to JSON
            json_data = json.dumps(serializable_payload, indent=None, separators=(',', ':'))
            json_bytes = json_data.encode('utf-8')
            
            # Validate size
            self._validate_memory_usage(len(json_bytes))
            
            # Encrypt if encryption is enabled
            if self.cipher_suite:
                json_bytes = self.cipher_suite.encrypt(json_bytes)
            
            # Get final path with compression extension
            final_path = self._get_final_path(output_path)
            
            # Write with compression
            uncompressed_size = len(json_bytes)
            
            if self.compression == 'none':
                with open(final_path, 'wb') as f:
                    f.write(json_bytes)
            else:
                compression_module = self.COMPRESSION_METHODS[self.compression]['module']
                open_kwargs = self.COMPRESSION_METHODS[self.compression]['open_kwargs'](self.compression_level)
                
                if self.compression == 'zstd' and ZSTD_AVAILABLE:
                    cctx = open_kwargs['cctx']
                    with open(final_path, 'wb') as f:
                        with cctx.stream_writer(f) as compressor:
                            compressor.write(json_bytes)
                else:
                    with compression_module.open(final_path, 'wb', **open_kwargs) as f:
                        f.write(json_bytes)
            
            # Set secure file permissions
            os.chmod(final_path, 0o600)
            
            # Calculate statistics
            compressed_size = final_path.stat().st_size
            compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1
            
            self.stats['bytes_written'] = compressed_size
            self.stats['compression_ratio'] = compression_ratio
            self.stats['serialization_time'] = time.time() - start_time
            
            result = {
                'file_path': str(final_path),
                'file_size_mb': compressed_size / (1024 * 1024),
                'uncompressed_size_mb': uncompressed_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'compression_method': self.compression,
                'encryption_enabled': self.cipher_suite is not None
            }
            
            # Validate if requested
            if validate:
                is_valid = self.validate_integrity(final_path)
                result['validation'] = {
                    'passed': is_valid,
                    'checksum': self._calculate_checksum(final_path)
                }
            
            logger.info(f"Secure serialization complete in {self.stats['serialization_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise SecureSerializationError(f"Serialization failed: {e}")
    
    def deserialize_datasets(
        self,
        input_path: Union[str, Path],
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Securely deserialize dataset from file.
        
        Args:
            input_path: Path to serialized file
            validate: Validate data integrity
            
        Returns:
            Deserialized data payload
        """
        start_time = time.time()
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Validate file size
        self._validate_file_size(input_path)
        
        try:
            # Detect compression
            compression = self._detect_compression(input_path)
            
            # Read and decompress
            if compression == 'none':
                with open(input_path, 'rb') as f:
                    compressed_data = f.read()
            else:
                compression_module = self.COMPRESSION_METHODS[compression]['module']
                
                if compression == 'zstd' and ZSTD_AVAILABLE:
                    dctx = zstd.ZstdDecompressor()
                    with open(input_path, 'rb') as f:
                        with dctx.stream_reader(f) as reader:
                            compressed_data = reader.read()
                else:
                    with compression_module.open(input_path, 'rb') as f:
                        compressed_data = f.read()
            
            # Validate decompressed size
            self._validate_memory_usage(len(compressed_data))
            
            # Decrypt if needed
            if self.cipher_suite:
                try:
                    json_bytes = self.cipher_suite.decrypt(compressed_data)
                except Exception as e:
                    raise SecureSerializationError(f"Decryption failed: {e}")
            else:
                json_bytes = compressed_data
            
            # Parse JSON
            try:
                json_data = json_bytes.decode('utf-8')
                serializable_payload = json.loads(json_data)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise SecureSerializationError(f"Invalid JSON data: {e}")
            
            # Validate format
            if not isinstance(serializable_payload, dict):
                raise SecureSerializationError("Invalid payload format - must be dictionary")
            
            # Restore DataFrames
            data_payload = self._restore_payload_from_serialization(serializable_payload)
            
            self.stats['deserialization_time'] = time.time() - start_time
            self.stats['bytes_read'] = input_path.stat().st_size
            
            logger.info(f"Secure deserialization complete in {self.stats['deserialization_time']:.2f}s")
            return data_payload
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise SecureSerializationError(f"Deserialization failed: {e}")
    
    def validate_integrity(self, file_path: Union[str, Path]) -> bool:
        """
        Validate integrity of serialized file.
        
        Args:
            file_path: Path to serialized file
            
        Returns:
            True if file is valid
        """
        file_path = Path(file_path)
        
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Validate file size
            try:
                self._validate_file_size(file_path)
            except SecureSerializationError:
                logger.error("File size validation failed")
                return False
            
            # Try to deserialize (this validates format and structure)
            try:
                self.deserialize_datasets(file_path, validate=False)
                logger.info(f"File validation passed: {file_path}")
                return True
            except SecureSerializationError:
                logger.error("File format validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _detect_compression(self, file_path: Path) -> str:
        """Detect compression method from file extension."""
        for method, config in self.COMPRESSION_METHODS.items():
            if method != 'none' and file_path.suffix == config['extension']:
                return method
        return 'none'
    
    def _get_final_path(self, file_path: Path) -> Path:
        """Get final file path with compression extension."""
        # Add .json if not present
        stem = file_path.stem
        if not stem.endswith('.json'):
            stem += '.json'
        
        # Add compression extension
        extension = self.COMPRESSION_METHODS[self.compression]['extension']
        
        return file_path.parent / f"{stem}{extension}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()


# Compatibility functions for existing code
def save_data_connector(
    data: Union[Dict[str, pd.DataFrame], Dict[str, Any]],
    file_path: Union[str, Path],
    compression: str = 'lz4',
    encryption_key: Optional[bytes] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Secure save function replacing unsafe pickle operations.
    
    Args:
        data: Data to save
        file_path: Output file path
        compression: Compression method
        encryption_key: Optional encryption key
        **kwargs: Additional arguments
        
    Returns:
        Serialization statistics
    """
    serializer = SecureSerializer(
        compression=compression,
        encryption_key=encryption_key
    )
    
    # Prepare payload structure
    if all(key in data for key in ['raw_data', 'ml_ready', 'metadata', 'ml_artifacts']):
        payload = data
    else:
        payload = {
            'raw_data': data if isinstance(data, dict) else {'data': data},
            'ml_ready': {},
            'metadata': {},
            'ml_artifacts': {}
        }
    
    return serializer.serialize_datasets(payload, file_path, **kwargs)


def load_data_connector(
    file_path: Union[str, Path],
    encryption_key: Optional[bytes] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Secure load function replacing unsafe pickle operations.
    
    Args:
        file_path: Path to serialized file
        encryption_key: Optional encryption key
        **kwargs: Additional arguments
        
    Returns:
        Loaded data
    """
    serializer = SecureSerializer(encryption_key=encryption_key)
    return serializer.deserialize_datasets(file_path, **kwargs)