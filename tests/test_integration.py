"""Integration tests for Instant Data Connector."""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import tempfile
from pathlib import Path
import json
import time

from instant_connector import InstantDataConnector


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def sqlite_db(self):
        """Create a temporary SQLite database with test data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        conn = sqlite3.connect(db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                age INTEGER,
                created_at TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                product TEXT,
                amount REAL,
                status TEXT,
                order_date TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Insert test data
        users_data = [
            (1, 'Alice', 'alice@example.com', 25, '2023-01-01'),
            (2, 'Bob', 'bob@example.com', 30, '2023-01-02'),
            (3, 'Charlie', 'charlie@example.com', 35, '2023-01-03'),
            (4, 'David', 'david@example.com', 28, '2023-01-04'),
            (5, 'Eve', 'eve@example.com', 32, '2023-01-05')
        ]
        
        orders_data = [
            (1, 1, 'Laptop', 1200.00, 'completed', '2023-02-01'),
            (2, 1, 'Mouse', 25.00, 'completed', '2023-02-02'),
            (3, 2, 'Keyboard', 75.00, 'pending', '2023-02-03'),
            (4, 3, 'Monitor', 300.00, 'completed', '2023-02-04'),
            (5, 2, 'Headphones', 100.00, 'completed', '2023-02-05'),
            (6, 4, 'Webcam', 80.00, 'shipped', '2023-02-06'),
            (7, 5, 'Desk', 250.00, 'completed', '2023-02-07'),
            (8, 1, 'Chair', 350.00, 'pending', '2023-02-08')
        ]
        
        conn.executemany('INSERT INTO users VALUES (?, ?, ?, ?, ?)', users_data)
        conn.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)', orders_data)
        conn.commit()
        conn.close()
        
        yield Path(db_path)
        Path(db_path).unlink()
    
    @pytest.fixture
    def csv_files(self):
        """Create temporary CSV files with test data."""
        files = []
        
        # Products CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("product_id,product_name,category,price,stock\n")
            f.write("1,Laptop,Electronics,1200.00,50\n")
            f.write("2,Mouse,Electronics,25.00,200\n")
            f.write("3,Keyboard,Electronics,75.00,150\n")
            f.write("4,Monitor,Electronics,300.00,75\n")
            f.write("5,Headphones,Electronics,100.00,100\n")
            f.write("6,Webcam,Electronics,80.00,80\n")
            f.write("7,Desk,Furniture,250.00,30\n")
            f.write("8,Chair,Furniture,350.00,40\n")
            files.append(Path(f.name))
        
        # Sales CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,product_id,quantity,revenue\n")
            f.write("2023-01-01,1,2,2400.00\n")
            f.write("2023-01-02,2,10,250.00\n")
            f.write("2023-01-03,3,5,375.00\n")
            f.write("2023-01-04,4,3,900.00\n")
            f.write("2023-01-05,5,8,800.00\n")
            files.append(Path(f.name))
        
        yield files
        
        for file in files:
            file.unlink()
    
    @pytest.fixture
    def json_file(self):
        """Create temporary JSON file with nested data."""
        data = [
            {
                'id': 1,
                'customer': {
                    'name': 'Alice',
                    'location': {'city': 'New York', 'country': 'USA'}
                },
                'purchases': [
                    {'item': 'Laptop', 'price': 1200},
                    {'item': 'Mouse', 'price': 25}
                ]
            },
            {
                'id': 2,
                'customer': {
                    'name': 'Bob',
                    'location': {'city': 'London', 'country': 'UK'}
                },
                'purchases': [
                    {'item': 'Keyboard', 'price': 75}
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)
        
        yield temp_path
        temp_path.unlink()
    
    def test_database_to_ml_pipeline(self, sqlite_db):
        """Test complete pipeline from database to ML-ready dataset."""
        connector = InstantDataConnector()
        
        # Add database source
        connector.add_database_source(
            name='sales_db',
            connection_params={
                'db_type': 'sqlite',
                'database': str(sqlite_db)
            },
            tables=['users', 'orders']
        )
        
        # Extract data
        connector.aggregate_all()
        
        # Verify extraction
        assert 'users' in connector.raw_data
        assert 'orders' in connector.raw_data
        assert len(connector.raw_data['users']) == 5
        assert len(connector.raw_data['orders']) == 8
        
        # Apply ML optimization
        connector.configure_ml_optimization(
            handle_missing='mean',
            encode_categorical='auto',
            scale_numeric='standard'
        )
        connector.apply_ml_optimization()
        
        # Verify ML processing
        assert len(connector.ml_ready_data) > 0
        assert 'preprocessing_metadata' in connector.ml_artifacts
        
        # Save to pickle
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'sales_data.pkl'
            result = connector.save_pickle(output_path, compression='lz4')
            
            assert Path(result['file_path']).exists()
            assert result['compression_method'] == 'lz4'
    
    def test_multi_source_aggregation(self, sqlite_db, csv_files, json_file):
        """Test aggregating data from multiple source types."""
        connector = InstantDataConnector()
        
        # Add database source
        connector.add_database_source(
            name='db',
            connection_params={
                'db_type': 'sqlite',
                'database': str(sqlite_db)
            },
            tables=['users']
        )
        
        # Add CSV files
        connector.add_file_source(
            name='products',
            file_path=str(csv_files[0])
        )
        connector.add_file_source(
            name='sales',
            file_path=str(csv_files[1])
        )
        
        # Add JSON file
        connector.add_file_source(
            name='customers',
            file_path=str(json_file)
        )
        
        # Aggregate all sources
        connector.aggregate_all()
        
        # Verify all data was extracted
        assert 'users' in connector.raw_data
        assert 'products' in connector.raw_data
        assert 'sales' in connector.raw_data
        assert 'customers' in connector.raw_data
        
        # Check data integrity
        assert len(connector.raw_data['users']) == 5
        assert len(connector.raw_data['products']) == 8
        assert len(connector.raw_data['sales']) == 5
        assert len(connector.raw_data['customers']) == 2
        
        # Verify JSON was properly normalized
        customers_df = connector.raw_data['customers']
        assert 'customer.name' in customers_df.columns or 'customer_name' in customers_df.columns
    
    def test_config_based_workflow(self, sqlite_db, csv_files):
        """Test configuration-based workflow."""
        config = {
            'sources': [
                {
                    'type': 'database',
                    'name': 'main_db',
                    'connection': {
                        'db_type': 'sqlite',
                        'database': str(sqlite_db)
                    },
                    'tables': ['users', 'orders'],
                    'sample_size': 1000
                },
                {
                    'type': 'file',
                    'name': 'products',
                    'path': str(csv_files[0]),
                    'optimize_dtypes': True
                }
            ],
            'ml_optimization': {
                'enabled': True,
                'handle_missing': 'auto',
                'encode_categorical': 'auto',
                'scale_numeric': 'auto',
                'feature_engineering': True,
                'reduce_memory': True
            },
            'output': {
                'compression': 'gzip',
                'compression_level': 6,
                'optimize_memory': True
            }
        }
        
        # Save config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = Path(f.name)
        
        try:
            # Create connector from config
            connector = InstantDataConnector()
            connector.load_from_config(config_path)
            
            # Run aggregation
            connector.aggregate_all()
            
            # Apply ML optimization
            connector.apply_ml_optimization()
            
            # Verify results
            assert len(connector.raw_data) == 3  # users, orders, products
            assert connector.ml_ready_data is not None
            
            # Save with config settings
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / 'output.pkl'
                result = connector.save_pickle(output_path)
                
                assert result['compression_method'] == 'gzip'
                
        finally:
            config_path.unlink()
    
    def test_ml_workflow_with_target(self, csv_files):
        """Test ML workflow with target variable."""
        # Create synthetic ML dataset
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples) * 2 + 1,
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature4': np.random.choice(['X', 'Y'], n_samples),
            'feature5': np.random.rand(n_samples) * 100,
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Add some correlation with target
        df.loc[df['feature3'] == 'A', 'target'] = np.random.choice([0, 1], sum(df['feature3'] == 'A'), p=[0.3, 0.7])
        
        # Save to CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            data_path = Path(f.name)
        
        try:
            connector = InstantDataConnector()
            connector.add_file_source('ml_data', str(data_path))
            connector.aggregate_all()
            
            # Apply ML optimization with target
            connector.apply_ml_optimization(
                target_column='target',
                test_size=0.2,
                stratify=True
            )
            
            # Verify train/test split
            assert 'X_train' in connector.ml_ready_data
            assert 'X_test' in connector.ml_ready_data
            assert 'y_train' in connector.ml_ready_data
            assert 'y_test' in connector.ml_ready_data
            
            # Check sizes
            assert len(connector.ml_ready_data['X_train']) == 800
            assert len(connector.ml_ready_data['X_test']) == 200
            
            # Check preprocessing was applied
            X_train = connector.ml_ready_data['X_train']
            assert 'feature3' not in X_train.columns  # Should be encoded
            assert any('feature3' in col for col in X_train.columns)  # Should have encoded columns
            
        finally:
            data_path.unlink()
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets with chunking."""
        # Create large dataset
        n_rows = 100000
        n_cols = 50
        
        large_df = pd.DataFrame(
            np.random.randn(n_rows, n_cols),
            columns=[f'col_{i}' for i in range(n_cols)]
        )
        
        # Add some categorical columns
        for i in range(5):
            large_df[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_rows)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            large_df.to_parquet(f.name)
            data_path = Path(f.name)
        
        try:
            connector = InstantDataConnector()
            connector.add_file_source('large_data', str(data_path))
            
            # Time the extraction
            start_time = time.time()
            connector.aggregate_all()
            extraction_time = time.time() - start_time
            
            # Apply ML optimization with memory reduction
            connector.configure_ml_optimization(reduce_memory=True)
            connector.apply_ml_optimization()
            
            # Save with chunking
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / 'large_data.pkl'
                result = connector.save_pickle(
                    output_path,
                    compression='lz4',
                    optimize_memory=True
                )
                
                # Verify file was created
                assert Path(result['file_path']).exists()
                
                # Check compression effectiveness
                original_size = large_df.memory_usage(deep=True).sum() / 1024**2
                compressed_size = result['file_size_mb']
                assert compressed_size < original_size
                
                print(f"Extraction time: {extraction_time:.2f}s")
                print(f"Original size: {original_size:.2f} MB")
                print(f"Compressed size: {compressed_size:.2f} MB")
                print(f"Compression ratio: {result['compression_ratio']:.2f}")
                
        finally:
            data_path.unlink()
    
    def test_error_recovery(self, sqlite_db):
        """Test error handling and recovery."""
        connector = InstantDataConnector()
        
        # Add valid source
        connector.add_database_source(
            name='valid_db',
            connection_params={
                'db_type': 'sqlite',
                'database': str(sqlite_db)
            },
            tables=['users']
        )
        
        # Add invalid source (non-existent file)
        connector.add_file_source(
            name='invalid_file',
            file_path='/non/existent/file.csv'
        )
        
        # Aggregate should partially succeed
        connector.aggregate_all()
        
        # Valid source should have data
        assert 'users' in connector.raw_data
        assert len(connector.raw_data['users']) == 5
        
        # Invalid source should not crash the process
        assert 'invalid_file' not in connector.raw_data
        
        # Metadata should indicate partial success
        assert connector.metadata['total_sources'] == 2
        assert connector.metadata['total_tables'] == 1  # Only successful extraction
    
    def test_data_lineage_tracking(self, sqlite_db, csv_files):
        """Test data lineage and metadata tracking."""
        connector = InstantDataConnector()
        
        # Add sources with metadata
        connector.add_database_source(
            name='db',
            connection_params={
                'db_type': 'sqlite',
                'database': str(sqlite_db)
            },
            tables=['users'],
            include_metadata=True
        )
        
        connector.add_file_source(
            name='products',
            file_path=str(csv_files[0]),
            include_metadata=True
        )
        
        # Extract with metadata
        connector.aggregate_all()
        
        # Check metadata columns in extracted data
        users_df = connector.raw_data['users']
        assert '_source_database' in users_df.columns
        assert '_source_table' in users_df.columns
        assert '_extraction_timestamp' in users_df.columns
        
        products_df = connector.raw_data['products']
        assert '_source_file' in products_df.columns
        assert '_file_type' in products_df.columns
        assert '_extraction_timestamp' in products_df.columns
        
        # Apply ML optimization
        connector.apply_ml_optimization()
        
        # Save and verify lineage is preserved
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'lineage_test.pkl'
            result = connector.save_pickle(output_path)
            
            # Load and verify
            from instant_connector.pickle_manager import load_data_connector
            loaded = load_data_connector(result['file_path'])