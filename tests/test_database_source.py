"""Tests for database source connector."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sqlalchemy as sa
from datetime import datetime

from instant_connector.sources.database_source import DatabaseSource


class TestDatabaseSource:
    """Test suite for DatabaseSource class."""
    
    @pytest.fixture
    def postgres_params(self):
        """PostgreSQL connection parameters."""
        return {
            'db_type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_pass',
            'schema': 'public'
        }
    
    @pytest.fixture
    def mysql_params(self):
        """MySQL connection parameters."""
        return {
            'db_type': 'mysql',
            'host': 'localhost',
            'port': 3306,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_pass'
        }
    
    @pytest.fixture
    def sqlite_params(self):
        """SQLite connection parameters."""
        return {
            'db_type': 'sqlite',
            'database': '/tmp/test.db'
        }
    
    def test_init_with_valid_params(self, postgres_params):
        """Test initialization with valid parameters."""
        source = DatabaseSource(postgres_params)
        assert source.db_type == 'postgresql'
        assert source.schema == 'public'
        assert source.engine is None
    
    def test_init_with_invalid_db_type(self):
        """Test initialization with invalid database type."""
        params = {'db_type': 'invalid_db'}
        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseSource(params)
    
    def test_init_missing_required_params(self):
        """Test initialization with missing required parameters."""
        params = {
            'db_type': 'postgresql',
            'host': 'localhost'
            # Missing database, username, password
        }
        with pytest.raises(ValueError, match="Missing required parameters"):
            DatabaseSource(params)
    
    def test_connection_string_postgres(self, postgres_params):
        """Test PostgreSQL connection string generation."""
        source = DatabaseSource(postgres_params)
        conn_str = source._get_connection_string()
        assert conn_str == "postgresql://test_user:test_pass@localhost:5432/test_db"
    
    def test_connection_string_mysql(self, mysql_params):
        """Test MySQL connection string generation."""
        source = DatabaseSource(mysql_params)
        conn_str = source._get_connection_string()
        assert conn_str == "mysql+pymysql://test_user:test_pass@localhost:3306/test_db"
    
    def test_connection_string_sqlite(self, sqlite_params):
        """Test SQLite connection string generation."""
        source = DatabaseSource(sqlite_params)
        conn_str = source._get_connection_string()
        assert conn_str == "sqlite:////tmp/test.db"
    
    def test_connection_string_special_chars_password(self, postgres_params):
        """Test connection string with special characters in password."""
        postgres_params['password'] = 'pass@word#123'
        source = DatabaseSource(postgres_params)
        conn_str = source._get_connection_string()
        assert "pass%40word%23123" in conn_str
    
    @patch('sqlalchemy.create_engine')
    def test_connect_success(self, mock_create_engine, postgres_params):
        """Test successful database connection."""
        # Mock engine and connection
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        source = DatabaseSource(postgres_params)
        source.connect()
        
        assert source.engine is not None
        mock_create_engine.assert_called_once()
        mock_conn.execute.assert_called_once()
    
    @patch('sqlalchemy.create_engine')
    def test_connect_failure(self, mock_create_engine, postgres_params):
        """Test failed database connection."""
        mock_create_engine.side_effect = Exception("Connection failed")
        
        source = DatabaseSource(postgres_params)
        with pytest.raises(Exception, match="Connection failed"):
            source.connect()
    
    def test_disconnect(self, postgres_params):
        """Test database disconnection."""
        source = DatabaseSource(postgres_params)
        source.engine = Mock()
        source.inspector = Mock()
        
        source.disconnect()
        
        source.engine.dispose.assert_called_once()
        assert source.engine is None
        assert source.inspector is None
    
    @patch('pandas.read_sql')
    def test_extract_table_basic(self, mock_read_sql, postgres_params):
        """Test basic table extraction."""
        # Mock data
        mock_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })
        mock_read_sql.return_value = mock_df
        
        source = DatabaseSource(postgres_params)
        source.engine = Mock()
        source.inspector = Mock()
        
        result = source.extract_table('test_table')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_read_sql.assert_called_once()
    
    @patch('pandas.read_sql')
    def test_extract_table_with_where_clause(self, mock_read_sql, postgres_params):
        """Test table extraction with WHERE clause."""
        mock_df = pd.DataFrame({'id': [1], 'name': ['Alice']})
        mock_read_sql.return_value = mock_df
        
        source = DatabaseSource(postgres_params)
        source.engine = Mock()
        
        result = source.extract_table('test_table', where_clause="id = 1")
        
        args = mock_read_sql.call_args[0]
        assert "WHERE id = 1" in args[0]
    
    @patch('pandas.read_sql')
    def test_extract_table_with_sample_size(self, mock_read_sql, postgres_params):
        """Test table extraction with sample size."""
        mock_df = pd.DataFrame({'id': range(10)})
        mock_read_sql.return_value = mock_df
        
        source = DatabaseSource(postgres_params)
        source.engine = Mock()
        
        result = source.extract_table('test_table', sample_size=5)
        
        args = mock_read_sql.call_args[0]
        assert "LIMIT 5" in args[0]
    
    @patch('pandas.read_sql')
    def test_extract_table_chunked(self, mock_read_sql, postgres_params):
        """Test chunked table extraction."""
        # Mock chunked response
        chunk1 = pd.DataFrame({'id': [1, 2]})
        chunk2 = pd.DataFrame({'id': [3, 4]})
        mock_read_sql.return_value = iter([chunk1, chunk2])
        
        source = DatabaseSource(postgres_params)
        source.engine = Mock()
        
        chunks = list(source.extract_table('test_table', chunksize=2))
        
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
    
    def test_optimize_dtypes(self, postgres_params):
        """Test data type optimization."""
        source = DatabaseSource(postgres_params)
        
        # Create test DataFrame with various types
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c'],
            'date_str': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'cat_col': ['cat1', 'cat1', 'cat2'],
            '_metadata': ['skip', 'skip', 'skip']
        })
        
        optimized = source._optimize_dtypes(df)
        
        # Check optimizations
        assert pd.api.types.is_integer_dtype(optimized['int_col'])
        assert pd.api.types.is_integer_dtype(optimized['float_col'])  # Should convert to int
        assert pd.api.types.is_categorical_dtype(optimized['cat_col'])
        assert optimized['_metadata'].dtype == 'object'  # Should skip metadata columns
    
    def test_is_date_column(self, postgres_params):
        """Test date column detection."""
        source = DatabaseSource(postgres_params)
        
        # Test various date formats
        date_series = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        assert source._is_date_column(date_series) is True
        
        date_series2 = pd.Series(['01/15/2023', '02/20/2023', '03/25/2023'])
        assert source._is_date_column(date_series2) is True
        
        non_date_series = pd.Series(['abc', 'def', 'ghi'])
        assert source._is_date_column(non_date_series) is False
        
        numeric_series = pd.Series([1, 2, 3])
        assert source._is_date_column(numeric_series) is False
    
    def test_add_metadata_columns(self, postgres_params):
        """Test adding metadata columns."""
        source = DatabaseSource(postgres_params)
        source.connection_params['database'] = 'test_db'
        
        df = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        result = source._add_metadata_columns(df, 'test_table')
        
        assert '_source_database' in result.columns
        assert '_source_table' in result.columns
        assert '_extraction_timestamp' in result.columns
        assert '_row_hash' in result.columns
        
        assert result['_source_database'].iloc[0] == 'test_db'
        assert result['_source_table'].iloc[0] == 'test_table'
        assert isinstance(result['_extraction_timestamp'].iloc[0], datetime)
    
    @patch('sqlalchemy.inspect')
    def test_get_table_info(self, mock_inspect, postgres_params):
        """Test getting table information."""
        # Mock inspector
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['table1', 'table2']
        mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': 'INTEGER'},
            {'name': 'name', 'type': 'VARCHAR'}
        ]
        mock_inspector.get_indexes.return_value = []
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': ['id']}
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspect.return_value = mock_inspector
        
        source = DatabaseSource(postgres_params)
        source.engine = Mock()
        source.inspector = mock_inspector
        
        info = source.get_table_info(include_stats=False)
        
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 2
        assert 'table_name' in info.columns
        assert 'column_count' in info.columns
        assert 'primary_key' in info.columns
    
    @patch('sqlalchemy.inspect')
    def test_detect_relationships(self, mock_inspect, postgres_params):
        """Test foreign key relationship detection."""
        # Mock inspector
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['orders', 'customers']
        mock_inspector.get_foreign_keys.side_effect = [
            [{  # orders table has FK to customers
                'name': 'fk_customer',
                'constrained_columns': ['customer_id'],
                'referred_table': 'customers',
                'referred_columns': ['id']
            }],
            []  # customers table has no FKs
        ]
        mock_inspect.return_value = mock_inspector
        
        source = DatabaseSource(postgres_params)
        source.engine = Mock()
        source.inspector = mock_inspector
        
        relationships = source.detect_relationships()
        
        assert 'orders' in relationships
        assert len(relationships['orders']) == 1
        assert relationships['orders'][0]['referred_table'] == 'customers'
    
    def test_validate_data_quality(self, postgres_params):
        """Test data quality validation."""
        source = DatabaseSource(postgres_params)
        
        # Create test DataFrame with quality issues
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [10, 20, None, None, None],  # High null percentage
            'constant': [1, 1, 1, 1, 1],  # Single value
            'unique_id': [1, 2, 3, 4, 5]  # Potential ID column
        })
        
        report = source.validate_data_quality(df, 'test_table')
        
        assert report['table_name'] == 'test_table'
        assert report['row_count'] == 5
        assert report['duplicate_rows'] == 0
        
        # Check column issues
        value_issues = report['columns']['value']['potential_issues']
        assert 'high_null_percentage' in value_issues
        
        constant_issues = report['columns']['constant']['potential_issues']
        assert 'single_value' in constant_issues
        
        id_issues = report['columns']['unique_id']['potential_issues']
        assert 'possible_id_column' in id_issues
    
    def test_context_manager(self, postgres_params):
        """Test context manager functionality."""
        source = DatabaseSource(postgres_params)
        source.connect = Mock()
        source.disconnect = Mock()
        
        with source as db:
            assert db is source
            source.connect.assert_called_once()
        
        source.disconnect.assert_called_once()
    
    @patch('pandas.read_sql')
    @patch('sqlalchemy.inspect')
    def test_extract_related_tables(self, mock_inspect, mock_read_sql, postgres_params):
        """Test extracting related tables."""
        # Mock data
        orders_df = pd.DataFrame({'id': [1, 2], 'customer_id': [1, 2]})
        customers_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        mock_read_sql.side_effect = [orders_df, customers_df]
        
        # Mock relationships
        mock_inspector = Mock()
        mock_inspect.return_value = mock_inspector
        
        source = DatabaseSource(postgres_params)
        source.engine = Mock()
        source.inspector = mock_inspector
        source._relationship_cache = {
            'orders': [{
                'name': 'fk_customer',
                'constrained_columns': ['customer_id'],
                'referred_table': 'customers',
                'referred_columns': ['id']
            }]
        }
        
        related = source.extract_related_tables('orders', max_depth=1)
        
        assert 'orders' in related
        assert 'customers' in related
        assert len(related['orders']) == 2
        assert len(related['customers']) == 2
    
    def test_is_junction_table(self, postgres_params):
        """Test junction table detection."""
        source = DatabaseSource(postgres_params)
        source.inspector = Mock()
        
        # Mock a junction table (2 FKs, 2-3 columns)
        source.inspector.get_columns.return_value = [
            {'name': 'user_id'},
            {'name': 'role_id'}
        ]
        source.inspector.get_foreign_keys.return_value = [
            {'constrained_columns': ['user_id']},
            {'constrained_columns': ['role_id']}
        ]
        
        assert source._is_junction_table('user_roles') is True
        
        # Mock a regular table
        source.inspector.get_columns.return_value = [
            {'name': 'id'},
            {'name': 'name'},
            {'name': 'email'},
            {'name': 'created_at'}
        ]
        source.inspector.get_foreign_keys.return_value = []
        
        assert source._is_junction_table('users') is False