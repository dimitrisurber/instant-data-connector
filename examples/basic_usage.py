#!/usr/bin/env python3
"""Basic usage example for Instant Data Connector."""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import InstantDataConnector, MLOptimizer, PickleManager
from instant_connector.pickle_manager import load_data_connector


def example_1_simple_file_aggregation():
    """Example 1: Aggregate CSV files and save as pickle."""
    print("\n=== Example 1: Simple File Aggregation ===")
    
    # Create aggregator
    aggregator = InstantDataConnector()
    
    # Add CSV files
    aggregator.add_file_source(
        name='sales_data',
        file_path='data/sales_q1.csv',
        read_options={'parse_dates': ['date']}
    )
    aggregator.add_file_source(
        name='sales_data_q2',
        file_path='data/sales_q2.csv',
        read_options={'parse_dates': ['date']}
    )
    
    # Extract and aggregate data
    data = aggregator.extract_data()
    print(f"Extracted {len(data)} datasets")
    
    # Save as optimized pickle
    stats = aggregator.save_connector('output/sales_data.pkl.lz4')
    print(f"Saved to: {stats['file_path']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")


def example_2_database_with_optimization():
    """Example 2: Extract from database with ML optimization."""
    print("\n=== Example 2: Database with ML Optimization ===")
    
    # Create aggregator
    aggregator = InstantDataConnector()
    
    # Add database source
    aggregator.add_database_source(
        name='analytics_db',
        connection_params={
            'db_type': 'postgresql',
            'host': 'localhost',
            'database': 'analytics',
            'username': 'user',
            'password': 'password'
        },
        queries={
            'user_behavior': """
                SELECT user_id, session_duration, page_views, 
                       bounce_rate, conversion, device_type
                FROM user_analytics
                WHERE date >= '2024-01-01'
            """,
            'product_metrics': """
                SELECT product_id, views, clicks, purchases,
                       revenue, category
                FROM product_performance
            """
        }
    )
    
    # Extract data
    data = aggregator.extract_data()
    
    # Apply ML optimizations
    optimizer = MLOptimizer(
        handle_missing='auto',
        encode_categorical='auto',
        scale_numeric='standard',
        remove_low_variance=True
    )
    
    for name, df in data.items():
        data[name] = optimizer.fit_transform(df)
    
    aggregator.datasets = data
    
    # Save with compression
    stats = aggregator.save_connector(
        'output/analytics_ml_ready.pkl.lz4'
    )
    
    print(f"Original size: {stats['uncompressed_size_mb']:.2f} MB")
    print(f"Compressed size: {stats['file_size_mb']:.2f} MB")


def example_3_multi_source_aggregation():
    """Example 3: Combine data from multiple sources."""
    print("\n=== Example 3: Multi-Source Aggregation ===")
    
    # Create aggregator with config file
    aggregator = InstantDataConnector(config_path='config/sources.yaml')
    
    # Add additional API source programmatically
    aggregator.add_api_source(
        name='customer_api',
        base_url='https://api.example.com/v1',
        headers={'Authorization': 'Bearer YOUR_TOKEN'},
        endpoints={
            'customers': {
                'endpoint': 'customers',
                'method': 'GET',
                'paginate': True,
                'pagination_type': 'offset',
                'params': {'limit': 100}
            }
        }
    )
    
    # Extract and save data
    data = aggregator.extract_data()
    save_stats = aggregator.save_connector(
        'output/multi_source_data.pkl.lz4'
    )
    
    print(f"Aggregated data saved: {save_stats['file_path']}")


def example_4_loading_and_using_data():
    """Example 4: Load and use saved connector data."""
    print("\n=== Example 4: Loading and Using Data ===")
    
    # Simple loading
    data = load_data_connector('output/sales_data.pkl.lz4')
    
    for name, df in data.items():
        print(f"\nDataset: {name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Load with timing
    import time
    start_time = time.time()
    data = load_data_connector('output/multi_source_data.pkl.lz4')
    load_time = time.time() - start_time
    
    print(f"\nTotal datasets: {len(data)}")
    print(f"Load time: {load_time:.2f} seconds")
    
    # Use for ML
    df = data['analytics_db_user_behavior']
    
    # Ready for sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Assuming 'conversion' is the target
    X = df.drop('conversion', axis=1)
    y = df['conversion']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model - data is already preprocessed!
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    print(f"\nModel accuracy: {model.score(X_test, y_test):.3f}")


def example_5_custom_ml_preprocessing():
    """Example 5: Custom ML preprocessing pipeline."""
    print("\n=== Example 5: Custom ML Preprocessing ===")
    
    # Create sample data
    df = pd.DataFrame({
        'age': [25, 30, None, 45, 50, 35],
        'income': [50000, 60000, 75000, None, 90000, 65000],
        'category': ['A', 'B', 'A', 'C', 'B', 'C'],
        'score': [0.8, 0.6, 0.9, 0.7, None, 0.85],
        'target': [1, 0, 1, 1, 0, 1]
    })
    
    # Create ML optimizer
    optimizer = MLOptimizer()
    
    # Configure and apply optimizations
    optimizer = MLOptimizer(
        handle_missing='median',
        encode_categorical='onehot',
        scale_numeric='standard',
        remove_low_variance=False
    )
    
    df_optimized = optimizer.fit_transform(df.drop('target', axis=1))
    df_optimized['target'] = df['target']
    
    print("\nOriginal DataFrame:")
    print(df)
    print("\nOptimized DataFrame:")
    print(df_optimized)
    
    # Get optimization report
    report = optimizer.get_optimization_report()
    print("\nOptimizations applied:")
    for opt in report['optimizations_applied']:
        print(f"  - {opt}")


if __name__ == '__main__':
    print("Instant Data Connector - Examples")
    print("=" * 50)
    
    # Note: These examples assume you have the necessary data files and databases
    # Comment out examples that require resources you don't have
    
    try:
        # example_1_simple_file_aggregation()
        # example_2_database_with_optimization()
        # example_3_multi_source_aggregation()
        # example_4_loading_and_using_data()
        example_5_custom_ml_preprocessing()
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure you have the required data files and database connections")