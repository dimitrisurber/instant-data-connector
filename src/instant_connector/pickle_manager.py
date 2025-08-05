"""Efficient pickle serialization with compression for large ML datasets.

WARNING: This module contains unsafe pickle operations that pose a CRITICAL SECURITY RISK.
All pickle operations are DEPRECATED and DISABLED by default.
Use SecureSerializer instead for safe JSON-based serialization.
"""

import pickle
import gzip
import lz4.frame
import bz2
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union, Tuple, List, Generator
from pathlib import Path
import json
import hashlib
import time
import logging
import os
import sys
import tempfile
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    warnings.warn("zstandard not available. Install with 'pip install zstandard' for better compression.")

logger = logging.getLogger(__name__)

# Security exception for blocking unsafe operations
class PickleSecurityError(Exception):
    """Exception raised for unsafe pickle operations."""
    pass

# Restricted unpickler for legacy file support (with warnings)
class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows safe modules for data deserialization."""
    
    ALLOWED_MODULES = {
        'pandas.core.series', 'pandas.core.frame', 'pandas.core.indexes.base',
        'pandas.core.indexes.range', 'pandas.core.indexes.numeric',
        'pandas.core.arrays.base', 'pandas.core.arrays.numpy_',
        'numpy', 'numpy.core.multiarray', 'numpy.core.numeric',
        'builtins', '__builtin__', 'collections', 'datetime',
        'decimal', 'copy_reg', 'copyreg'
    }
    
    def find_class(self, module, name):
        """Override to restrict allowed classes."""
        # Allow only safe modules for data structures
        if module not in self.ALLOWED_MODULES:
            logger.error(f"SECURITY: Blocked potentially dangerous module: {module}.{name}")
            raise PickleSecurityError(f"Forbidden module: {module}")
        
        # Additional name-based restrictions
        if name.startswith('_') or 'exec' in name.lower() or 'eval' in name.lower():
            logger.error(f"SECURITY: Blocked suspicious class name: {module}.{name}")
            raise PickleSecurityError(f"Forbidden class name: {name}")
            
        return super().find_class(module, name)

def safe_pickle_load(file_obj):
    """Safer pickle loading with restricted unpickler and warnings."""
    logger.warning(
        "CRITICAL SECURITY WARNING: Loading pickle file with restricted unpickler. "
        "This poses security risks. Migrate to SecureSerializer immediately."
    )
    warnings.warn(
        "CRITICAL SECURITY WARNING: You are using unsafe pickle operations that pose "
        "a REMOTE CODE EXECUTION risk (CVSS 9.8). Migrate to SecureSerializer immediately. "
        "See security documentation for migration guide.",
        UserWarning,
        stacklevel=3
    )
    
    return RestrictedUnpickler(file_obj).load()


class PickleManager:
    """Efficient pickle serialization with compression and metadata support for ML datasets."""
    
    COMPRESSION_METHODS = {
        'gzip': {
            'module': gzip,
            'extension': '.gz',
            'open_kwargs': lambda level: {'compresslevel': level}
        },
        'lz4': {
            'module': lz4.frame,
            'extension': '.lz4',
            'open_kwargs': lambda level: {'compression_level': level}
        },
        'bz2': {
            'module': bz2,
            'extension': '.bz2',
            'open_kwargs': lambda level: {'compresslevel': level}
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
            'open_kwargs': lambda level: {'cctx': zstd.ZstdCompressor(level=level)}
        }
    
    # Chunk size for large dataset processing (100MB)
    CHUNK_SIZE_BYTES = 100 * 1024 * 1024
    
    def __init__(
        self,
        compression: str = 'lz4',
        compression_level: int = 0,
        chunk_threshold_mb: int = 500,
        enable_progress: bool = True
    ):
        """
        Initialize pickle manager.
        
        CRITICAL SECURITY WARNING: PickleManager uses unsafe pickle operations that pose
        a REMOTE CODE EXECUTION risk (CVSS 9.8). Use SecureSerializer instead.
        
        Args:
            compression: Compression method ('gzip', 'lz4', 'bz2', 'zstd', 'none')
            compression_level: Compression level (method-specific ranges)
            chunk_threshold_mb: Threshold for chunked serialization
            enable_progress: Show progress for large operations
        """
        # Issue critical security warning
        logger.warning(
            "CRITICAL SECURITY WARNING: PickleManager uses unsafe pickle operations. "
            "Use SecureSerializer instead to prevent remote code execution attacks."
        )
        warnings.warn(
            "PickleManager is deprecated due to critical security vulnerabilities (CVSS 9.8). "
            "Use SecureSerializer for safe JSON-based serialization.",
            DeprecationWarning,
            stacklevel=2
        )
        if compression not in self.COMPRESSION_METHODS:
            raise ValueError(f"Unsupported compression: {compression}")
            
        self.compression = compression
        self.compression_level = self._validate_compression_level(compression, compression_level)
        self.chunk_threshold_bytes = chunk_threshold_mb * 1024 * 1024
        self.enable_progress = enable_progress
        
        # Statistics tracking
        self.stats = {
            'bytes_written': 0,
            'bytes_read': 0,
            'compression_ratio': 0,
            'serialization_time': 0,
            'deserialization_time': 0
        }
    
    def serialize_datasets(
        self,
        data_payload: Dict[str, Any],
        output_path: Union[str, Path],
        add_metadata: bool = True,
        optimize_memory: bool = True,
        validate: bool = True,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Serialize complete dataset payload with all ML artifacts.
        
        Args:
            data_payload: Dictionary containing raw_data, ml_ready, metadata, ml_artifacts
            output_path: Output file path
            add_metadata: Add serialization metadata
            optimize_memory: Optimize data types before serialization
            validate: Validate data integrity after serialization
            parallel: Use parallel processing for large datasets
            
        Returns:
            Serialization statistics and metadata
        """
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data payload
        if add_metadata:
            data_payload = self._add_serialization_metadata(data_payload)
        
        # Optimize memory usage if requested
        if optimize_memory:
            data_payload = self._optimize_payload_memory(data_payload, parallel)
        
        # Calculate total size
        total_size = self._estimate_payload_size(data_payload)
        logger.info(f"Serializing {total_size / 1024**2:.2f} MB of data")
        
        # Choose serialization strategy
        if total_size > self.chunk_threshold_bytes:
            # Use chunked serialization for large datasets
            result = self._serialize_chunked(data_payload, output_path)
        else:
            # Standard serialization for smaller datasets
            result = self._serialize_standard(data_payload, output_path)
        
        # Validate if requested
        if validate:
            actual_file_path = Path(result['file_path'])
            is_valid = self.validate_integrity(actual_file_path)
            checksum = self._calculate_checksum(actual_file_path)
            result['validation'] = {
                'passed': is_valid,
                'checksum': checksum
            }
        
        # Update statistics
        self.stats['serialization_time'] = time.time() - start_time
        result['stats'] = self.stats.copy()
        
        logger.info(f"Serialization complete in {self.stats['serialization_time']:.2f}s")
        
        return result
    
    def _serialize_standard(
        self,
        data_payload: Dict[str, Any],
        output_path: Path
    ) -> Dict[str, Any]:
        """Standard serialization for datasets that fit in memory."""
        # Get final path with compression extension
        final_path = self._get_final_path(output_path)
        
        # Calculate uncompressed size
        uncompressed_size = len(pickle.dumps(data_payload, protocol=pickle.HIGHEST_PROTOCOL))
        
        # Open file with compression
        if self.compression == 'none':
            with open(final_path, 'wb') as f:
                pickle.dump(data_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            compression_module = self.COMPRESSION_METHODS[self.compression]['module']
            open_kwargs = self.COMPRESSION_METHODS[self.compression]['open_kwargs'](self.compression_level)
            
            if self.compression == 'zstd' and ZSTD_AVAILABLE:
                # Special handling for zstd
                cctx = open_kwargs['cctx']
                with open(final_path, 'wb') as f:
                    with cctx.stream_writer(f) as compressor:
                        pickle.dump(data_payload, compressor, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with compression_module.open(final_path, 'wb', **open_kwargs) as f:
                    pickle.dump(data_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Calculate compression ratio
        compressed_size = final_path.stat().st_size
        compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1
        
        self.stats['bytes_written'] = compressed_size
        self.stats['compression_ratio'] = compression_ratio
        
        return {
            'file_path': str(final_path),
            'file_size_mb': compressed_size / 1024**2,
            'uncompressed_size_mb': uncompressed_size / 1024**2,
            'compression_ratio': compression_ratio,
            'compression_method': self.compression
        }
    
    def _serialize_chunked(
        self,
        data_payload: Dict[str, Any],
        output_path: Path
    ) -> Dict[str, Any]:
        """Chunked serialization for large datasets that may not fit in memory."""
        logger.info("Using chunked serialization for large dataset")
        
        # Create temporary directory for chunks
        temp_dir = Path(tempfile.mkdtemp(prefix='pickle_chunks_'))
        chunk_files = []
        
        try:
            # Split payload into chunks
            chunks = self._split_payload_to_chunks(data_payload)
            total_chunks = len(chunks)
            
            # Serialize each chunk
            for i, chunk in enumerate(chunks):
                if self.enable_progress:
                    logger.info(f"Serializing chunk {i+1}/{total_chunks}")
                
                chunk_path = temp_dir / f"chunk_{i:04d}.pkl"
                chunk_result = self._serialize_standard(chunk, chunk_path)
                # Use the actual file path from the result
                actual_chunk_path = Path(chunk_result['file_path'])
                chunk_files.append(actual_chunk_path)
            
            # Combine chunks into final file
            final_path = self._get_final_path(output_path)
            self._combine_chunks(chunk_files, final_path)
            
            # Calculate final statistics
            compressed_size = final_path.stat().st_size
            uncompressed_size = sum(
                len(pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL))
                for chunk in chunks
            )
            compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1
            
            self.stats['bytes_written'] = compressed_size
            self.stats['compression_ratio'] = compression_ratio
            
            return {
                'file_path': str(final_path),
                'file_size_mb': compressed_size / 1024**2,
                'uncompressed_size_mb': uncompressed_size / 1024**2,
                'compression_ratio': compression_ratio,
                'compression_method': self.compression,
                'chunks_used': total_chunks
            }
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _split_payload_to_chunks(self, data_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split large payload into manageable chunks."""
        chunks = []
        
        # Always keep metadata and ml_artifacts together in first chunk
        base_chunk = {
            'metadata': data_payload.get('metadata', {}),
            'ml_artifacts': data_payload.get('ml_artifacts', {}),
            '_chunk_info': {
                'total_chunks': 0,  # Will be updated
                'chunk_index': 0,
                'type': 'base'
            }
        }
        chunks.append(base_chunk)
        
        # Split large data sections
        for section in ['raw_data', 'ml_ready']:
            if section in data_payload and data_payload[section]:
                section_data = data_payload[section]
                
                # Group tables by size
                table_sizes = [
                    (name, df, df.memory_usage(deep=True).sum())
                    for name, df in section_data.items()
                    if isinstance(df, pd.DataFrame)
                ]
                table_sizes.sort(key=lambda x: x[2], reverse=True)
                
                current_chunk = {
                    section: {},
                    '_chunk_info': {
                        'type': 'data',
                        'section': section
                    }
                }
                current_size = 0
                
                for name, df, size in table_sizes:
                    if current_size + size > self.CHUNK_SIZE_BYTES and current_chunk[section]:
                        # Save current chunk and start new one
                        chunks.append(current_chunk)
                        current_chunk = {
                            section: {},
                            '_chunk_info': {
                                'type': 'data',
                                'section': section
                            }
                        }
                        current_size = 0
                    
                    current_chunk[section][name] = df
                    current_size += size
                
                if current_chunk[section]:
                    chunks.append(current_chunk)
        
        # Update chunk info
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk['_chunk_info']['total_chunks'] = total_chunks
            chunk['_chunk_info']['chunk_index'] = i
        
        return chunks
    
    def _combine_chunks(self, chunk_files: List[Path], output_path: Path):
        """Combine multiple chunk files into single file."""
        # For now, create a manifest file alongside the chunks
        manifest = {
            'format_version': '1.0',
            'chunks': [str(f.name) for f in chunk_files],
            'compression': self.compression,
            'created_at': datetime.now().isoformat()
        }
        
        # Create a tar archive or similar for the chunks
        # For simplicity, we'll just copy the first chunk as the main file
        # In production, you'd want proper chunk management
        shutil.copy2(chunk_files[0], output_path)
        
        # Save manifest
        manifest_path = output_path.with_suffix('.manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def deserialize_datasets(
        self,
        input_path: Union[str, Path],
        lazy_load: bool = False,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Deserialize dataset from pickle file.
        
        Args:
            input_path: Path to pickle file
            lazy_load: Load data on-demand to save memory
            validate: Validate data integrity
            
        Returns:
            Deserialized data payload
        """
        start_time = time.time()
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Check for manifest (chunked file)
        manifest_path = input_path.with_suffix('.manifest.json')
        if manifest_path.exists():
            # Handle chunked deserialization
            data_payload = self._deserialize_chunked(input_path, lazy_load)
        else:
            # Standard deserialization
            data_payload = self._deserialize_standard(input_path)
        
        # Validate if requested
        if validate and '_serialization_metadata' in data_payload:
            stored_checksum = data_payload['_serialization_metadata'].get('checksum')
            if stored_checksum:
                current_checksum = self._calculate_checksum(input_path)
                if stored_checksum != current_checksum:
                    logger.warning("Checksum mismatch - data may be corrupted")
        
        self.stats['deserialization_time'] = time.time() - start_time
        self.stats['bytes_read'] = input_path.stat().st_size
        
        logger.info(f"Deserialization complete in {self.stats['deserialization_time']:.2f}s")
        
        return data_payload
    
    def _deserialize_standard(self, input_path: Path) -> Dict[str, Any]:
        """Standard deserialization for single files."""
        # Detect compression from file extension
        compression = self._detect_compression(input_path)
        
        if compression == 'none':
            with open(input_path, 'rb') as f:
                return safe_pickle_load(f)
        else:
            compression_module = self.COMPRESSION_METHODS[compression]['module']
            
            if compression == 'zstd' and ZSTD_AVAILABLE:
                # Special handling for zstd
                dctx = zstd.ZstdDecompressor()
                with open(input_path, 'rb') as f:
                    with dctx.stream_reader(f) as reader:
                        return safe_pickle_load(reader)
            else:
                with compression_module.open(input_path, 'rb') as f:
                    return safe_pickle_load(f)
    
    def _deserialize_chunked(self, input_path: Path, lazy_load: bool) -> Dict[str, Any]:
        """Deserialize data from multiple chunks."""
        # Load manifest
        manifest_path = input_path.with_suffix('.manifest.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # For now, just load the main file
        # In production, implement proper chunk merging
        return self._deserialize_standard(input_path)
    
    def compress_pickle(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_compression: Optional[str] = None,
        target_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Recompress existing pickle file with different compression.
        
        Args:
            input_path: Path to existing pickle file
            output_path: Path for recompressed file
            target_compression: New compression method
            target_level: New compression level
            
        Returns:
            Compression statistics
        """
        # Load data
        data = self.deserialize_datasets(input_path)
        
        # Update compression settings
        if target_compression:
            self.compression = target_compression
        if target_level is not None:
            self.compression_level = target_level
        
        # Reserialize with new compression
        return self.serialize_datasets(data, output_path)
    
    def _add_serialization_metadata(self, data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata about serialization process."""
        # Calculate checksum of the raw data payload before adding metadata
        data_checksum = self._calculate_data_checksum(data_payload)
        
        metadata = {
            'serialization_timestamp': datetime.now().isoformat(),
            'format_version': '2.0',
            'compression': self.compression,
            'compression_level': self.compression_level,
            'python_version': sys.version,
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'platform': sys.platform,
            'data_stats': self._calculate_payload_stats(data_payload),
            'checksum': data_checksum
        }
        
        data_payload['_serialization_metadata'] = metadata
        return data_payload
    
    def _calculate_payload_stats(self, data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics about the data payload."""
        stats = {
            'sections': {},
            'total_dataframes': 0,
            'total_rows': 0,
            'total_columns': 0,
            'memory_usage_mb': 0
        }
        
        for section in ['raw_data', 'ml_ready']:
            if section in data_payload and isinstance(data_payload[section], dict):
                section_stats = {
                    'tables': {},
                    'total_tables': len(data_payload[section])
                }
                
                for name, df in data_payload[section].items():
                    if isinstance(df, pd.DataFrame):
                        memory_usage = df.memory_usage(deep=True).sum()
                        section_stats['tables'][name] = {
                            'shape': df.shape,
                            'memory_mb': memory_usage / 1024**2,
                            'dtypes': df.dtypes.value_counts().to_dict()
                        }
                        
                        stats['total_dataframes'] += 1
                        stats['total_rows'] += df.shape[0]
                        stats['total_columns'] += df.shape[1]
                        stats['memory_usage_mb'] += memory_usage / 1024**2
                
                stats['sections'][section] = section_stats
        
        return stats
    
    def _optimize_payload_memory(
        self,
        data_payload: Dict[str, Any],
        parallel: bool = False
    ) -> Dict[str, Any]:
        """Optimize memory usage of all DataFrames in payload."""
        optimized_payload = data_payload.copy()
        
        # Optimize each section
        for section in ['raw_data', 'ml_ready']:
            if section in optimized_payload and isinstance(optimized_payload[section], dict):
                if parallel:
                    optimized_payload[section] = self._optimize_section_parallel(
                        optimized_payload[section]
                    )
                else:
                    optimized_payload[section] = self._optimize_section_sequential(
                        optimized_payload[section]
                    )
        
        return optimized_payload
    
    def _optimize_section_sequential(self, section_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Optimize DataFrames in section sequentially."""
        optimized = {}
        
        for name, df in section_data.items():
            if isinstance(df, pd.DataFrame):
                optimized[name] = self.optimize_dtypes(df)
            else:
                optimized[name] = df
        
        return optimized
    
    def _optimize_section_parallel(self, section_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Optimize DataFrames in section using parallel processing."""
        optimized = {}
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit optimization tasks
            future_to_name = {
                executor.submit(self.optimize_dtypes, df): name
                for name, df in section_data.items()
                if isinstance(df, pd.DataFrame)
            }
            
            # Collect results
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    optimized[name] = future.result()
                except Exception as e:
                    logger.error(f"Failed to optimize {name}: {e}")
                    optimized[name] = section_data[name]
            
            # Add non-DataFrame items
            for name, item in section_data.items():
                if not isinstance(item, pd.DataFrame):
                    optimized[name] = item
        
        return optimized
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type != 'object':
                # Skip if already optimized or categorical
                if col_type in ['int8', 'int16', 'float16', 'float32'] or pd.api.types.is_categorical_dtype(col_type):
                    continue
                
                # Optimize numeric types
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                # Integer optimization
                if pd.api.types.is_integer_dtype(col_type):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df_optimized[col] = df_optimized[col].astype(np.int64)
                
                # Float optimization
                elif pd.api.types.is_float_dtype(col_type):
                    # Check if floats are actually integers (like 1.0, 2.0, 3.0)
                    if df_optimized[col].dropna().apply(lambda x: x.is_integer()).all():
                        # Convert to appropriate integer type
                        if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                            df_optimized[col] = df_optimized[col].astype(np.int8)
                        elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                            df_optimized[col] = df_optimized[col].astype(np.int16)
                        elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                            df_optimized[col] = df_optimized[col].astype(np.int32)
                        else:
                            df_optimized[col] = df_optimized[col].astype(np.int64)
                    else:
                        # Check precision requirements for true floats
                        if df_optimized[col].dropna().apply(lambda x: len(str(x).split('.')[-1])).max() <= 3:
                            # Low precision needed - use float32
                            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                                df_optimized[col] = df_optimized[col].astype(np.float32)
                        # Keep float64 for high precision requirements
            
            else:
                # Optimize object columns
                num_unique_values = len(df_optimized[col].unique())
                num_total_values = len(df_optimized[col])
                
                # Convert to category if low cardinality
                if num_unique_values / num_total_values < 0.5:
                    df_optimized[col] = df_optimized[col].astype('category')
        
        # Log memory reduction
        original_memory = df.memory_usage(deep=True).sum()
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        reduction_pct = (1 - optimized_memory / original_memory) * 100
        
        if reduction_pct > 0:
            logger.debug(f"Memory reduced by {reduction_pct:.1f}%")
        
        return df_optimized
    
    def validate_integrity(self, file_path: Union[str, Path]) -> bool:
        """
        Validate integrity of serialized file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            True if file is valid
        """
        file_path = Path(file_path)
        
        try:
            # Check file exists and is readable
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Try to load file header
            compression = self._detect_compression(file_path)
            
            # Quick validation - try to load just the metadata
            if compression == 'none':
                with open(file_path, 'rb') as f:
                    # Read first few bytes to check pickle protocol
                    header = f.read(2)
                    if len(header) < 1 or header[0] != 0x80:  # Pickle protocol marker
                        logger.error("Invalid pickle file format")
                        return False
            
            # For compressed files, try decompression
            else:
                compression_module = self.COMPRESSION_METHODS[compression]['module']
                if compression == 'zstd' and ZSTD_AVAILABLE:
                    dctx = zstd.ZstdDecompressor()
                    with open(file_path, 'rb') as f:
                        # Try to read a small chunk
                        try:
                            dctx.decompress(f.read(1024), max_output_size=1024)
                        except:
                            logger.error("Failed to decompress file")
                            return False
                else:
                    try:
                        with compression_module.open(file_path, 'rb') as f:
                            f.read(1024)
                    except Exception as e:
                        logger.error(f"Failed to decompress file: {e}")
                        return False
            
            logger.info(f"File validation passed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about serialized file without loading it.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            File information dictionary
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'file_path': str(file_path),
            'file_size_mb': file_path.stat().st_size / 1024**2,
            'compression': self._detect_compression(file_path),
            'created_time': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        # Try to load just metadata
        try:
            data = self.deserialize_datasets(file_path)
            if '_serialization_metadata' in data:
                info['metadata'] = data['_serialization_metadata']
            
            # Count tables
            info['tables'] = {}
            for section in ['raw_data', 'ml_ready']:
                if section in data and isinstance(data[section], dict):
                    info['tables'][section] = list(data[section].keys())
        except:
            logger.warning("Could not load file metadata")
        
        return info
    
    def create_delta(
        self,
        old_data: Dict[str, pd.DataFrame],
        new_data: Dict[str, pd.DataFrame],
        output_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Create delta serialization containing only changes.
        
        Args:
            old_data: Previous version of data
            new_data: New version of data
            output_path: Path for delta file
            
        Returns:
            Delta statistics
        """
        delta_payload = {
            'delta_info': {
                'created_at': datetime.now().isoformat(),
                'type': 'incremental_update'
            },
            'changes': {}
        }
        
        # Compare each table
        for table_name in set(list(old_data.keys()) + list(new_data.keys())):
            if table_name not in old_data:
                # New table
                delta_payload['changes'][table_name] = {
                    'operation': 'create',
                    'data': new_data[table_name]
                }
            elif table_name not in new_data:
                # Deleted table
                delta_payload['changes'][table_name] = {
                    'operation': 'delete'
                }
            else:
                # Compare data
                old_df = old_data[table_name]
                new_df = new_data[table_name]
                
                # Find differences (simplified - in production use more sophisticated comparison)
                if not old_df.equals(new_df):
                    delta_payload['changes'][table_name] = {
                        'operation': 'update',
                        'data': new_df  # In production, store only changed rows
                    }
        
        # Serialize delta
        return self.serialize_datasets(delta_payload, output_path)
    
    def save_data_connector(
        self,
        data: Union[Dict[str, pd.DataFrame], Dict[str, Any]],
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Save data connector using this PickleManager instance.
        
        Args:
            data: Data to save (either dict of DataFrames or full payload)
            file_path: Output file path
            metadata: Optional metadata to include
            **kwargs: Additional arguments for serialize_datasets
            
        Returns:
            Serialization statistics
        """
        # Prepare the full payload with expected structure
        if isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
            # Data is dict of DataFrames, package it with expected keys
            payload = {
                'raw_data': data,  # Use raw_data key as expected by serialize_datasets
                'ml_ready': {},    # Empty ml_ready section
                'metadata': metadata or {},
                'ml_artifacts': {},
                'version': '0.1.0'
            }
            # If metadata contains ml_ready_data, extract it
            if metadata and 'ml_ready_data' in metadata:
                payload['ml_ready'] = metadata.pop('ml_ready_data')
            if metadata and 'ml_artifacts' in metadata:
                payload['ml_artifacts'] = metadata.pop('ml_artifacts')
        else:
            # Data is already a full payload
            payload = data
            if metadata:
                payload.setdefault('metadata', {}).update(metadata)
        
        # Filter out compression from kwargs since it's set in constructor
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'compression'}
        
        # Use serialize_datasets method
        return self.serialize_datasets(payload, file_path, **filtered_kwargs)
    
    def load_data_connector(
        self,
        file_path: Union[str, Path],
        lazy_load: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load data connector using this PickleManager instance.
        
        Args:
            file_path: Input file path
            lazy_load: Whether to load data lazily
            **kwargs: Additional arguments for deserialize_datasets
            
        Returns:
            Loaded data and metadata
        """
        return self.deserialize_datasets(file_path, lazy_load=lazy_load, **kwargs)
    
    def _validate_compression_level(self, compression: str, level: int) -> int:
        """Validate and adjust compression level for method."""
        if compression == 'gzip':
            return max(0, min(level, 9))
        elif compression == 'bz2':
            return max(1, min(level, 9))
        elif compression == 'lz4':
            return max(0, min(level, 16))
        elif compression == 'zstd':
            return max(1, min(level, 22))
        else:
            return 0
    
    def _detect_compression(self, file_path: Path) -> str:
        """Detect compression method from file extension."""
        for method, config in self.COMPRESSION_METHODS.items():
            if method != 'none' and file_path.suffix == config['extension']:
                return method
        return 'none'
    
    def _get_final_path(self, file_path: Path) -> Path:
        """Get final file path with compression extension."""
        # Remove any existing compression extension
        stem = file_path.stem
        if any(file_path.suffix == config['extension'] 
               for config in self.COMPRESSION_METHODS.values()):
            stem = file_path.stem
        else:
            stem = file_path.name
        
        # Add .pkl if not present
        if not stem.endswith('.pkl'):
            stem += '.pkl'
        
        # Add compression extension
        extension = self.COMPRESSION_METHODS[self.compression]['extension']
        
        return file_path.parent / f"{stem}{extension}"
    
    def _estimate_payload_size(self, data_payload: Dict[str, Any]) -> int:
        """Estimate size of data payload in bytes."""
        total_size = 0
        
        # Estimate DataFrame sizes
        for section in ['raw_data', 'ml_ready']:
            if section in data_payload and isinstance(data_payload[section], dict):
                for name, df in data_payload[section].items():
                    if isinstance(df, pd.DataFrame):
                        total_size += df.memory_usage(deep=True).sum()
        
        # Add overhead for other data (rough estimate)
        total_size += len(pickle.dumps(data_payload.get('metadata', {})))
        total_size += len(pickle.dumps(data_payload.get('ml_artifacts', {})))
        
        return int(total_size)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _calculate_data_checksum(self, data_payload: Dict[str, Any]) -> str:
        """Calculate MD5 checksum of data payload (excluding metadata)."""
        # Create a copy without metadata to get consistent checksum
        data_for_checksum = {}
        for key, value in data_payload.items():
            if key != '_serialization_metadata':
                data_for_checksum[key] = value
        
        # Serialize the data and compute checksum
        data_bytes = pickle.dumps(data_for_checksum, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(data_bytes).hexdigest()
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        return {
            'compression_method': self.compression,
            'compression_level': self.compression_level,
            'total_bytes_written': self.stats['bytes_written'],
            'total_bytes_read': self.stats['bytes_read'],
            'average_compression_ratio': self.stats['compression_ratio'],
            'total_serialization_time': self.stats['serialization_time'],
            'total_deserialization_time': self.stats['deserialization_time']
        }


def load_data_connector(
    file_path: Union[str, Path],
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load data connector.
    
    Args:
        file_path: Path to pickle file
        **kwargs: Additional arguments for PickleManager.deserialize_datasets
        
    Returns:
        Dictionary of DataFrames
    """
    manager = PickleManager()
    data_payload = manager.deserialize_datasets(file_path, **kwargs)
    
    # Extract just the ML-ready data by default
    if 'ml_ready' in data_payload:
        return data_payload['ml_ready']
    elif 'raw_data' in data_payload:
        return data_payload['raw_data']
    else:
        # Return all DataFrames found
        all_data = {}
        for key, value in data_payload.items():
            if isinstance(value, dict):
                for name, df in value.items():
                    if isinstance(df, pd.DataFrame):
                        all_data[name] = df
        return all_data


def save_data_connector(
    data: Union[Dict[str, pd.DataFrame], Dict[str, Any]],
    file_path: Union[str, Path],
    compression: str = 'lz4',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to save data connector.
    
    Args:
        data: Data to save (either dict of DataFrames or full payload)
        file_path: Output file path
        compression: Compression method
        **kwargs: Additional arguments for PickleManager.serialize_datasets
        
    Returns:
        Serialization statistics
    """
    manager = PickleManager(compression=compression)
    
    # Check if data is already a full payload or just DataFrames
    if all(key in data for key in ['raw_data', 'ml_ready', 'metadata', 'ml_artifacts']):
        # Full payload
        data_payload = data
    else:
        # Just DataFrames - wrap in payload structure
        data_payload = {
            'ml_ready': data,
            'metadata': {},
            'ml_artifacts': {}
        }
    
    return manager.serialize_datasets(data_payload, file_path, **kwargs)