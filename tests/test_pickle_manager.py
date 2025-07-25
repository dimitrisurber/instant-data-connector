"""Tests for pickle manager module."""

import pytest
import pandas as pd
import numpy as np
import pickle
import gzip
import tempfile
from pathlib import Path
import json
import hashlib
from unittest.mock import Mock, patch, MagicMock
import time
import shutil

from instant_connector.pickle_manager import (
    PickleManager, load_data_connector, save_data_connector
)


class TestPickleManager:
    """Test suite for PickleManager class."""
    
    @pytest.fixture
    def sample_data_payload(self):
        """Create sample data payload."""
        return {
            'raw_data': {
                'table1': pd.DataFrame({
                    'id': range(100),
                    'value': np.random.randn(100)
                }),
                'table2': pd.DataFrame({
                    'id': range(50),
                    'category': np.random.choice(['A', 'B', 'C'], 50)
                })
            },
            'ml_ready': {
                'features': pd.DataFrame({
                    'feature1': np.random.randn(100),
                    'feature2': np.random.randn(100),
                    'feature3': np.random.choice([0, 1], 100)
                })
            },
            'metadata': {
                'source': 'test',
                'version': '1.0',
                'created_at': '2023-01-01'
            },
            'ml_artifacts': {
                'scaler': {'type': 'StandardScaler', 'params': {}},
                'encoder': {'type': 'LabelEncoder', 'mappings': {'A': 0, 'B': 1}}
            }
        }
    
    @pytest.fixture
    def pickle_manager(self):
        """Create PickleManager instance."""
        return PickleManager(
            compression='gzip',
            compression_level=6,
            chunk_threshold_mb=10
        )
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary file path."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
        # Clean up any related files
        for ext in ['.gz', '.lz4', '.bz2', '.zst', '.manifest.json']:
            related = temp_path.with_suffix(temp_path.suffix + ext)
            if related.exists():
                related.unlink()
    
    def test_init_valid_compression(self):
        """Test initialization with valid compression methods."""
        for method in ['gzip', 'lz4', 'bz2', 'none']:
            manager = PickleManager(compression=method)
            assert manager.compression == method
    
    def test_init_invalid_compression(self):
        """Test initialization with invalid compression method."""
        with pytest.raises(ValueError, match="Unsupported compression"):
            PickleManager(compression='invalid')
    
    def test_compression_level_validation(self):
        """Test compression level validation."""
        # Gzip: 0-9
        manager = PickleManager(compression='gzip', compression_level=10)
        assert manager.compression_level == 9  # Should cap at 9
        
        # LZ4: 0-16
        manager = PickleManager(compression='lz4', compression_level=20)
        assert manager.compression_level == 16  # Should cap at 16
    
    def test_serialize_datasets_basic(self, pickle_manager, sample_data_payload, temp_file):
        """Test basic serialization."""
        result = pickle_manager.serialize_datasets(
            sample_data_payload,
            temp_file,
            add_metadata=True,
            optimize_memory=False,
            validate=True
        )
        
        assert 'file_path' in result
        assert 'file_size_mb' in result
        assert 'compression_ratio' in result
        assert 'validation' in result
        assert result['validation']['passed'] is True
        
        # Check file exists
        output_path = Path(result['file_path'])
        assert output_path.exists()
        assert output_path.suffix == '.gz'  # Should have compression extension
    
    def test_serialize_datasets_no_compression(self, sample_data_payload, temp_file):
        """Test serialization without compression."""
        manager = PickleManager(compression='none')
        result = manager.serialize_datasets(
            sample_data_payload,
            temp_file,
            add_metadata=False
        )
        
        output_path = Path(result['file_path'])
        assert output_path.exists()
        assert output_path.suffix == '.pkl'  # No compression extension
        
        # Compression ratio should be ~1
        assert result['compression_ratio'] == pytest.approx(1.0, rel=0.1)
    
    def test_serialize_with_memory_optimization(self, pickle_manager, temp_file):
        """Test serialization with memory optimization."""
        # Create data with optimizable types
        data_payload = {
            'raw_data': {
                'table': pd.DataFrame({
                    'int64_col': np.array([1, 2, 3, 4, 5], dtype='int64'),
                    'float64_col': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float64'),
                    'object_col': ['A', 'A', 'B', 'A', 'B']  # 2 unique out of 5 = 0.4 < 0.5
                })
            },
            'metadata': {},
            'ml_artifacts': {}
        }
        
        result = pickle_manager.serialize_datasets(
            data_payload,
            temp_file,
            optimize_memory=True
        )
        
        # Load and check optimization
        loaded = pickle_manager.deserialize_datasets(result['file_path'])
        df = loaded['raw_data']['table']
        
        # Check dtype optimization
        assert df['int64_col'].dtype.name in ['int8', 'int16', 'int32']
        assert pd.api.types.is_categorical_dtype(df['object_col'])
    
    def test_serialize_large_dataset_chunking(self, temp_file):
        """Test chunked serialization for large datasets."""
        # Create large dataset
        large_df = pd.DataFrame({
            f'col_{i}': np.random.randn(10000) for i in range(20)
        })
        
        data_payload = {
            'raw_data': {'large_table': large_df},
            'ml_ready': {'processed': large_df},
            'metadata': {'size': 'large'},
            'ml_artifacts': {}
        }
        
        manager = PickleManager(chunk_threshold_mb=1)  # Low threshold to trigger chunking
        result = manager.serialize_datasets(data_payload, temp_file)
        
        assert 'chunks_used' in result
        assert result['chunks_used'] > 1
    
    def test_deserialize_datasets_basic(self, pickle_manager, sample_data_payload, temp_file):
        """Test basic deserialization."""
        # First serialize (disable memory optimization to preserve exact dtypes)
        pickle_manager.serialize_datasets(sample_data_payload, temp_file, optimize_memory=False)
        
        # Then deserialize
        loaded = pickle_manager.deserialize_datasets(temp_file.with_suffix('.pkl.gz'))
        
        assert 'raw_data' in loaded
        assert 'ml_ready' in loaded
        assert 'metadata' in loaded
        assert 'ml_artifacts' in loaded
        
        # Check data integrity
        pd.testing.assert_frame_equal(
            loaded['raw_data']['table1'],
            sample_data_payload['raw_data']['table1']
        )
    
    def test_deserialize_with_validation(self, pickle_manager, sample_data_payload, temp_file):
        """Test deserialization with validation."""
        # Serialize with metadata
        result = pickle_manager.serialize_datasets(
            sample_data_payload,
            temp_file,
            add_metadata=True,
            validate=True
        )
        
        # Deserialize with validation
        loaded = pickle_manager.deserialize_datasets(
            result['file_path'],
            validate=True
        )
        
        assert '_serialization_metadata' in loaded
        assert 'checksum' in loaded['_serialization_metadata']
    
    def test_compress_pickle(self, sample_data_payload, temp_file):
        """Test recompressing existing pickle file."""
        # Create uncompressed pickle
        manager1 = PickleManager(compression='none')
        result1 = manager1.serialize_datasets(sample_data_payload, temp_file)
        
        # Recompress with gzip
        temp_file2 = temp_file.parent / 'compressed.pkl'
        result2 = manager1.compress_pickle(
            result1['file_path'],
            temp_file2,
            target_compression='gzip',
            target_level=9
        )
        
        # Check compression improved
        assert result2['compression_ratio'] > result1['compression_ratio']
        assert result2['file_size_mb'] < result1['file_size_mb']
    
    def test_optimize_dtypes(self, pickle_manager):
        """Test DataFrame dtype optimization."""
        df = pd.DataFrame({
            'int64': np.array([1, 2, 3, 4, 5], dtype='int64'),
            'int32': np.array([1, 2, 3, 4, 5], dtype='int32'),
            'float64': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float64'),
            'float_as_int': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float64'),
            'object_low_card': ['A', 'B', 'A', 'B', 'A'],
            'object_high_card': [f'unique_{i}' for i in range(5)]
        })
        
        optimized = pickle_manager.optimize_dtypes(df)
        
        # Check optimizations
        assert optimized['int64'].dtype.name in ['int8', 'int16']
        assert optimized['int32'].dtype.name in ['int8', 'int16', 'int32']
        assert optimized['float64'].dtype == 'float32'
        assert pd.api.types.is_integer_dtype(optimized['float_as_int'])
        assert pd.api.types.is_categorical_dtype(optimized['object_low_card'])
        assert optimized['object_high_card'].dtype == 'object'  # Should remain object
    
    def test_validate_integrity_valid_file(self, pickle_manager, sample_data_payload, temp_file):
        """Test integrity validation on valid file."""
        result = pickle_manager.serialize_datasets(sample_data_payload, temp_file)
        
        is_valid = pickle_manager.validate_integrity(result['file_path'])
        assert is_valid is True
    
    def test_validate_integrity_missing_file(self, pickle_manager):
        """Test integrity validation on missing file."""
        is_valid = pickle_manager.validate_integrity('/non/existent/file.pkl')
        assert is_valid is False
    
    @patch('builtins.open', side_effect=IOError("Corrupted file"))
    def test_validate_integrity_corrupted_file(self, mock_open, pickle_manager, temp_file):
        """Test integrity validation on corrupted file."""
        # Create a dummy file
        temp_file.write_bytes(b'corrupted data')
        
        is_valid = pickle_manager.validate_integrity(temp_file)
        assert is_valid is False
    
    def test_get_file_info(self, pickle_manager, sample_data_payload, temp_file):
        """Test getting file information."""
        result = pickle_manager.serialize_datasets(sample_data_payload, temp_file)
        
        info = pickle_manager.get_file_info(result['file_path'])
        
        assert 'file_path' in info
        assert 'file_size_mb' in info
        assert 'compression' in info
        assert 'created_time' in info
        assert 'modified_time' in info
        assert 'tables' in info
        
        # Check table information
        assert 'raw_data' in info['tables']
        assert 'table1' in info['tables']['raw_data']
    
    def test_create_delta(self, pickle_manager, temp_file):
        """Test delta serialization."""
        old_data = {
            'table1': pd.DataFrame({'id': [1, 2], 'value': [10, 20]}),
            'table2': pd.DataFrame({'id': [1], 'name': ['Alice']})
        }
        
        new_data = {
            'table1': pd.DataFrame({'id': [1, 2, 3], 'value': [10, 25, 30]}),  # Modified
            'table3': pd.DataFrame({'id': [1], 'category': ['A']})  # New table
        }
        
        result = pickle_manager.create_delta(old_data, new_data, temp_file)
        
        # Load delta
        delta = pickle_manager.deserialize_datasets(result['file_path'])
        
        assert 'delta_info' in delta
        assert 'changes' in delta
        assert delta['changes']['table1']['operation'] == 'update'
        assert delta['changes']['table2']['operation'] == 'delete'
        assert delta['changes']['table3']['operation'] == 'create'
    
    def test_get_compression_stats(self, pickle_manager, sample_data_payload, temp_file):
        """Test compression statistics."""
        pickle_manager.serialize_datasets(sample_data_payload, temp_file)
        
        stats = pickle_manager.get_compression_stats()
        
        assert 'compression_method' in stats
        assert 'compression_level' in stats
        assert 'total_bytes_written' in stats
        assert 'average_compression_ratio' in stats
        assert stats['total_bytes_written'] > 0
    
    def test_detect_compression(self, pickle_manager):
        """Test compression detection from file extension."""
        assert pickle_manager._detect_compression(Path('file.pkl.gz')) == 'gzip'
        assert pickle_manager._detect_compression(Path('file.pkl.lz4')) == 'lz4'
        assert pickle_manager._detect_compression(Path('file.pkl.bz2')) == 'bz2'
        assert pickle_manager._detect_compression(Path('file.pkl')) == 'none'
    
    def test_calculate_checksum(self, pickle_manager, temp_file):
        """Test checksum calculation."""
        temp_file.write_bytes(b'test data')
        
        checksum = pickle_manager._calculate_checksum(temp_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 hash length
        
        # Verify checksum is consistent
        checksum2 = pickle_manager._calculate_checksum(temp_file)
        assert checksum == checksum2
    
    def test_split_payload_to_chunks(self, pickle_manager):
        """Test payload splitting for chunked serialization."""
        # Create large payload
        large_df = pd.DataFrame(np.random.randn(1000, 100))
        
        data_payload = {
            'raw_data': {'large_table': large_df},
            'ml_ready': {'processed': large_df},
            'metadata': {'test': True},
            'ml_artifacts': {'scaler': 'test'}
        }
        
        chunks = pickle_manager._split_payload_to_chunks(data_payload)
        
        assert len(chunks) > 1
        assert chunks[0]['_chunk_info']['type'] == 'base'
        assert 'metadata' in chunks[0]
        assert 'ml_artifacts' in chunks[0]
        
        # Data chunks should have data sections
        for chunk in chunks[1:]:
            assert chunk['_chunk_info']['type'] == 'data'
            assert 'raw_data' in chunk or 'ml_ready' in chunk
    
    def test_parallel_optimization(self, pickle_manager):
        """Test parallel memory optimization."""
        # Create data with multiple tables
        data_payload = {
            'raw_data': {
                f'table_{i}': pd.DataFrame({
                    'col1': np.random.randint(0, 100, 1000),
                    'col2': np.random.choice(['A', 'B', 'C'], 1000)
                }) for i in range(5)
            },
            'metadata': {},
            'ml_artifacts': {}
        }
        
        optimized = pickle_manager._optimize_payload_memory(data_payload, parallel=True)
        
        # Check optimization was applied
        for table_name, df in optimized['raw_data'].items():
            assert pd.api.types.is_integer_dtype(df['col1'])
            assert pd.api.types.is_categorical_dtype(df['col2'])
    
    def test_lz4_compression(self, sample_data_payload, temp_file):
        """Test LZ4 compression."""
        manager = PickleManager(compression='lz4')
        result = manager.serialize_datasets(sample_data_payload, temp_file, optimize_memory=False)
        
        assert Path(result['file_path']).suffix == '.lz4'
        assert result['compression_method'] == 'lz4'
        
        # Test deserialization
        loaded = manager.deserialize_datasets(result['file_path'])
        pd.testing.assert_frame_equal(
            loaded['raw_data']['table1'],
            sample_data_payload['raw_data']['table1']
        )
    
    def test_bz2_compression(self, sample_data_payload, temp_file):
        """Test BZ2 compression."""
        manager = PickleManager(compression='bz2')
        result = manager.serialize_datasets(sample_data_payload, temp_file)
        
        assert Path(result['file_path']).suffix == '.bz2'
        assert result['compression_method'] == 'bz2'
    
    def test_convenience_functions(self, sample_data_payload, temp_file):
        """Test convenience functions."""
        # Test save_data_connector
        save_result = save_data_connector(
            sample_data_payload['ml_ready'],
            temp_file,
            compression='gzip',
            optimize_memory=False
        )
        
        assert 'file_path' in save_result
        
        # Test load_data_connector
        loaded = load_data_connector(save_result['file_path'])
        
        assert isinstance(loaded, dict)
        assert 'features' in loaded
        pd.testing.assert_frame_equal(
            loaded['features'],
            sample_data_payload['ml_ready']['features']
        )
    
    def test_error_handling_empty_payload(self, pickle_manager, temp_file):
        """Test handling of empty payload."""
        empty_payload = {
            'raw_data': {},
            'ml_ready': {},
            'metadata': {},
            'ml_artifacts': {}
        }
        
        result = pickle_manager.serialize_datasets(empty_payload, temp_file)
        assert result['file_size_mb'] > 0  # Should still create a file
        
        loaded = pickle_manager.deserialize_datasets(result['file_path'])
        assert loaded == empty_payload or '_serialization_metadata' in loaded
    
    def test_lazy_load_placeholder(self, pickle_manager, sample_data_payload, temp_file):
        """Test lazy loading placeholder (not fully implemented)."""
        result = pickle_manager.serialize_datasets(sample_data_payload, temp_file)
        
        # Lazy load should still return full data for now
        loaded = pickle_manager.deserialize_datasets(
            result['file_path'],
            lazy_load=True
        )
        
        assert 'raw_data' in loaded
    
    def test_metadata_preservation(self, pickle_manager, temp_file):
        """Test metadata is preserved through serialization."""
        data_payload = {
            'raw_data': {'table': pd.DataFrame({'a': [1, 2, 3]})},
            'ml_ready': {},
            'metadata': {
                'source': 'test_source',
                'version': '1.2.3',
                'custom_field': {'nested': 'value'}
            },
            'ml_artifacts': {
                'preprocessing': {'steps': ['scale', 'encode']}
            }
        }
        
        result = pickle_manager.serialize_datasets(data_payload, temp_file)
        loaded = pickle_manager.deserialize_datasets(result['file_path'])
        
        assert loaded['metadata'] == data_payload['metadata']
        assert loaded['ml_artifacts'] == data_payload['ml_artifacts']
    
    def test_zstd_not_available(self):
        """Test behavior when zstandard is not available."""
        # Temporarily remove zstd from compression methods
        original_methods = PickleManager.COMPRESSION_METHODS.copy()
        if 'zstd' in PickleManager.COMPRESSION_METHODS:
            del PickleManager.COMPRESSION_METHODS['zstd']
        
        try:
            # Should not be in compression methods
            assert 'zstd' not in PickleManager.COMPRESSION_METHODS
            
            # Should raise error if trying to use
            with pytest.raises(ValueError, match="Unsupported compression"):
                PickleManager(compression='zstd')
        finally:
            # Restore original methods
            PickleManager.COMPRESSION_METHODS = original_methods