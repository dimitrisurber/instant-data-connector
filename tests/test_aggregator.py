"""Tests for aggregator module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock
import json

from instant_connector.aggregator import InstantDataConnector


class TestInstantDataConnector:
    """Test suite for InstantDataConnector class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            'sources': [
                {
                    'type': 'database',
                    'name': 'test_db',
                    'connection': {
                        'db_type': 'sqlite',
                        'database': ':memory:'
                    },
                    'tables': ['users', 'orders']
                },
                {
                    'type': 'file',
                    'name': 'test_csv',
                    'path': 'test.csv'
                },
                {
                    'type': 'api',
                    'name': 'test_api',
                    'base_url': 'https://api.example.com',
                    'endpoints': ['/items']
                }
            ],
            'ml_optimization': {
                'enabled': True,
                'handle_missing': 'mean',
                'encode_categorical': 'auto',
                'scale_numeric': 'standard'
            },
            'output': {
                'path': 'output.pkl',
                'compression': 'lz4',
                'optimize_memory': True
            }
        }
    
    @pytest.fixture
    def aggregator(self):
        """Create InstantDataConnector instance."""
        return InstantDataConnector()
    
    @pytest.fixture
    def sample_dataframes(self):
        """Create sample DataFrames."""
        return {
            'users': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [25, 30, 35]
            }),
            'orders': pd.DataFrame({
                'id': [1, 2, 3],
                'user_id': [1, 1, 2],
                'amount': [100.0, 200.0, 150.0]
            }),
            'items': pd.DataFrame({
                'id': [1, 2],
                'name': ['Item1', 'Item2'],
                'price': [10.0, 20.0]
            })
        }
    
    def test_init(self, aggregator):
        """Test initialization."""
        assert aggregator.raw_data == {}
        assert aggregator.metadata == {}
    
    @patch('instant_connector.aggregator.DatabaseSource')
    def test_add_database_source(self, mock_db_class, aggregator):
        """Test adding database source."""
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        
        # Provide complete connection parameters for security validation
        connection_params = {
            'db_type': 'postgresql',
            'host': 'localhost',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        }
        
        aggregator.add_database_source(
            name='test_db',
            connection_params=connection_params,
            tables=['users'],
            optimize_dtypes=True
        )
        
        assert 'test_db' in aggregator.sources
        mock_db_class.assert_called_once_with(connection_params)
    
    @patch('instant_connector.aggregator.FileSource')
    def test_add_file_source(self, mock_file_class, aggregator, temp_csv_file):
        """Test adding file source."""
        mock_file = Mock()
        mock_file_class.return_value = mock_file
        
        aggregator.add_file_source(
            name='test_csv',
            file_path=str(temp_csv_file),
            optimize_dtypes=True
        )
        
        assert 'test_csv' in aggregator.sources
        mock_file_class.assert_called_once()
    
    @patch('instant_connector.aggregator.APISource')
    def test_add_api_source(self, mock_api_class, aggregator):
        """Test adding API source."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        aggregator.add_api_source(
            name='test_api',
            base_url='https://api.example.com',
            endpoints=['/items'],
            headers={'X-API-Key': 'test'}
        )
        
        assert 'test_api' in aggregator.sources
        mock_api_class.assert_called_once_with(
            base_url='https://api.example.com',
            headers={'X-API-Key': 'test'}
        )
    
    @patch('instant_connector.aggregator.DatabaseSource')
    def test_extract_database_data(self, mock_db_class, aggregator, sample_dataframes):
        """Test database data extraction."""
        mock_db = Mock()
        mock_db.extract_table.side_effect = lambda table, **kwargs: sample_dataframes[table]
        mock_db.get_table_info.return_value = pd.DataFrame({
            'table_name': ['users', 'orders'],
            'row_count': [3, 3]
        })
        # Make mock_db work as a context manager
        mock_db.__enter__ = Mock(return_value=mock_db)
        mock_db.__exit__ = Mock(return_value=None)
        mock_db_class.return_value = mock_db
        
        aggregator.add_database_source('test_db', {
            'db_type': 'postgresql',
            'host': 'localhost', 
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password'
        }, tables=['users', 'orders'])
        
        data = aggregator._extract_database_data_for_aggregate('test_db', mock_db, ['users', 'orders'], {})
        
        assert 'users' in data
        assert 'orders' in data
        pd.testing.assert_frame_equal(data['users'], sample_dataframes['users'])
        pd.testing.assert_frame_equal(data['orders'], sample_dataframes['orders'])
    
    @patch('instant_connector.aggregator.FileSource')
    def test_extract_file_data(self, mock_file_class, aggregator, sample_dataframes):
        """Test file data extraction."""
        mock_file = Mock()
        mock_file.extract_data.return_value = sample_dataframes['items']
        mock_file_class.return_value = mock_file
        
        aggregator.add_file_source('test_csv', 'test.csv')
        
        data = aggregator._extract_file_data_for_aggregate('test_csv', mock_file, {})
        
        assert 'test_csv' in data
        pd.testing.assert_frame_equal(data['test_csv'], sample_dataframes['items'])
    
    @patch('instant_connector.aggregator.APISource')
    def test_extract_api_data(self, mock_api_class, aggregator, sample_dataframes):
        """Test API data extraction."""
        mock_api = Mock()
        mock_api.extract_data.return_value = sample_dataframes['items']
        mock_api_class.return_value = mock_api
        
        aggregator.add_api_source('test_api', 'https://api.example.com', ['/items'])
        
        data = aggregator._extract_api_data_for_aggregate('test_api', mock_api, ['/items'])
        
        assert 'items' in data
        pd.testing.assert_frame_equal(data['items'], sample_dataframes['items'])
    
    def test_aggregate_all_empty(self, aggregator):
        """Test aggregating with no sources."""
        aggregator.aggregate_all()
        
        assert aggregator.raw_data == {}
        assert 'extraction_timestamp' in aggregator.metadata
        assert aggregator.metadata['total_sources'] == 0
    
    @patch('instant_connector.aggregator.DatabaseSource')
    @patch('instant_connector.aggregator.FileSource')
    def test_aggregate_all_multiple_sources(self, mock_file_class, mock_db_class, 
                                           aggregator, sample_dataframes):
        """Test aggregating from multiple sources."""
        # Setup database mock
        mock_db = Mock()
        mock_db.extract_table.side_effect = lambda table, **kwargs: sample_dataframes[table]
        mock_db.get_table_info.return_value = pd.DataFrame({'table_name': ['users']})
        # Make mock_db work as a context manager
        mock_db.__enter__ = Mock(return_value=mock_db)
        mock_db.__exit__ = Mock(return_value=None)
        mock_db_class.return_value = mock_db
        
        # Setup file mock
        mock_file = Mock()
        mock_file.extract_data.return_value = sample_dataframes['items']
        mock_file_class.return_value = mock_file
        
        # Add sources
        aggregator.add_database_source('db', {
            'db_type': 'postgresql',
            'host': 'localhost',
            'database': 'test_db', 
            'username': 'test_user',
            'password': 'test_password'
        }, tables=['users'])
        aggregator.add_file_source('file', 'test.csv')
        
        # Aggregate
        aggregator.aggregate_all()
        
        assert 'users' in aggregator.raw_data
        assert 'file' in aggregator.raw_data
        assert aggregator.metadata['total_sources'] == 2
        assert aggregator.metadata['total_tables'] == 2
    
    
    
    
    def test_save_pickle(self, aggregator, sample_dataframes):
        """Test saving to pickle."""
        aggregator.raw_data = sample_dataframes
        aggregator.metadata = {'test': True}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_output.pkl'
            
            result = aggregator.save_pickle(output_path)
            
            assert 'file_path' in result
            assert Path(result['file_path']).exists()
            assert 'compression_method' in result
    
    def test_save_pickle_auto_path(self, aggregator):
        """Test saving with auto-generated path."""
        aggregator.raw_data = {'test': pd.DataFrame({'a': [1, 2, 3]})}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('instant_connector.aggregator.Path.cwd', return_value=Path(temp_dir)):
                result = aggregator.save_pickle()
                
                assert 'file_path' in result
                assert 'instant_data_connector_' in Path(result['file_path']).name
    
    def test_load_from_config(self, aggregator, sample_config):
        """Test loading configuration from dict."""
        # Mock source classes
        with patch('instant_connector.aggregator.DatabaseSource') as mock_db, \
             patch('instant_connector.aggregator.FileSource') as mock_file, \
             patch('instant_connector.aggregator.APISource') as mock_api:
            
            aggregator.load_from_config(sample_config)
            
            # Check sources were added
            assert len(aggregator.sources) == 3
            assert 'test_db' in aggregator.sources
            assert 'test_csv' in aggregator.sources
            assert 'test_api' in aggregator.sources
            
    
    def test_load_from_config_file(self, aggregator, sample_config):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config, f)
            config_path = f.name
        
        try:
            with patch('instant_connector.aggregator.DatabaseSource'), \
                 patch('instant_connector.aggregator.FileSource'), \
                 patch('instant_connector.aggregator.APISource'):
                
                aggregator.load_from_config(config_path)
                
                assert len(aggregator.sources) == 3
        finally:
            Path(config_path).unlink()
    
    def test_get_summary(self, aggregator, sample_dataframes):
        """Test getting summary."""
        aggregator.raw_data = sample_dataframes
        aggregator.metadata = {'total_sources': 2}
        
        summary = aggregator.get_summary()
        
        assert summary['total_sources'] == 2
        assert summary['total_tables'] == 3
        assert summary['total_rows'] > 0
        assert summary['memory_usage_mb'] > 0
    
    def test_error_handling_invalid_source(self, aggregator):
        """Test error handling for invalid source configuration."""
        config = {
            'sources': [{
                'type': 'invalid_type',
                'name': 'test'
            }]
        }
        
        with pytest.raises(ValueError, match="Unknown source type"):
            aggregator.load_from_config(config)
    
    def test_error_handling_extraction_failure(self, aggregator):
        """Test error handling during extraction."""
        mock_source = Mock()
        mock_source.extract_data.side_effect = Exception("Extraction failed")
        
        aggregator.sources['failing_source'] = {
            'type': 'file',
            'source': mock_source,
            'file_path': 'test.csv'
        }
        
        # Should not raise, just log error
        aggregator.aggregate_all()
        
        assert 'failing_source' not in aggregator.raw_data
    
    def test_aggregate_with_extraction_params(self, aggregator):
        """Test aggregation with custom extraction parameters."""
        mock_db = Mock()
        mock_db.extract_table.return_value = pd.DataFrame({'a': [1, 2, 3]})
        mock_db.get_table_info.return_value = pd.DataFrame({'table_name': ['test']})
        # Make mock_db work as a context manager
        mock_db.__enter__ = Mock(return_value=mock_db)
        mock_db.__exit__ = Mock(return_value=None)
        
        aggregator.sources['db'] = {
            'type': 'database',
            'source': mock_db,
            'tables': ['test'],
            'sample_size': 10,
            'where_clause': 'id > 0'
        }
        
        aggregator.aggregate_all()
        
        mock_db.extract_table.assert_called_with(
            'test',
            include_metadata=False,
            optimize_dtypes=True,
            sample_size=10,
            where_clause='id > 0'
        )
    
    def test_complete_workflow(self, aggregator):
        """Test complete workflow from config to pickle."""
        # Create sample data
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30],
            'category': ['A', 'B', 'A']
        })
        
        # Mock sources
        with patch('instant_connector.aggregator.FileSource') as mock_file_class:
            mock_file = Mock()
            mock_file.extract_data.return_value = df1
            mock_file_class.return_value = mock_file
            
            # Configure
            config = {
                'sources': [{
                    'type': 'file',
                    'name': 'test_data',
                    'path': 'test.csv'
                }],
            }
            
            # Run workflow
            aggregator.load_from_config(config)
            aggregator.aggregate_all()
            
            # Check results
            assert 'test_data' in aggregator.raw_data
            
            # Save
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / 'output.pkl'
                result = aggregator.save_pickle(output_path)
                
                assert Path(result['file_path']).exists()
    
    @pytest.fixture
    def temp_csv_file(self):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,value\n")
            f.write("1,test1,10\n")
            f.write("2,test2,20\n")
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink()