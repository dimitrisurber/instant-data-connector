#!/usr/bin/env python3
"""ML workflow example using Instant Data Connector."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import DataAggregator, MLOptimizer, PickleManager
from instant_connector.pickle_manager import load_data_connector


def create_sample_ml_data():
    """Create sample datasets for ML demonstrations."""
    np.random.seed(42)
    n_samples = 10000
    
    # Classification dataset
    classification_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 2, n_samples),
        'feature_3': np.random.exponential(1, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),
        'numeric_feature': np.random.uniform(0, 100, n_samples),
        'count_feature': np.random.poisson(5, n_samples),
    })
    
    # Add some missing values
    missing_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    classification_data.loc[missing_idx, 'feature_2'] = np.nan
    
    # Create target based on features
    classification_data['target'] = (
        (classification_data['feature_1'] > 0) & 
        (classification_data['feature_3'] < 1.5) |
        (classification_data['categorical_1'] == 'A')
    ).astype(int)
    
    # Regression dataset
    regression_data = pd.DataFrame({
        'size': np.random.uniform(500, 5000, n_samples),
        'age': np.random.uniform(0, 50, n_samples),
        'location_score': np.random.normal(50, 15, n_samples),
        'amenities': np.random.poisson(3, n_samples),
        'condition': np.random.choice(['poor', 'fair', 'good', 'excellent'], n_samples),
        'neighborhood': np.random.choice(['downtown', 'suburb', 'rural'], n_samples),
    })
    
    # Create price target with some noise
    regression_data['price'] = (
        regression_data['size'] * 100 +
        regression_data['location_score'] * 1000 -
        regression_data['age'] * 500 +
        regression_data['amenities'] * 5000 +
        np.random.normal(0, 10000, n_samples)
    )
    
    return classification_data, regression_data


def example_classification_workflow():
    """Complete classification workflow with data connector."""
    print("\n=== Classification Workflow Example ===")
    
    # Step 1: Create and prepare data
    print("\n1. Creating sample classification data...")
    clf_data, _ = create_sample_ml_data()
    
    # Step 2: Set up aggregator and optimizer
    aggregator = DataAggregator()
    aggregator.datasets = {'classification': clf_data}
    
    # Step 3: Apply ML optimizations
    print("\n2. Applying ML optimizations...")
    optimizer = MLOptimizer()
    optimized_data = aggregator.optimize_datasets(
        optimizer=optimizer,
        target_column='target',
        handle_missing='auto',
        encode_categorical='auto',
        scale_numeric='standard',
        remove_low_variance=True,
        remove_high_correlation=True
    )
    
    # Step 4: Save optimized data
    print("\n3. Saving ML-ready data...")
    pickle_manager = PickleManager(compression='lz4')
    save_stats = aggregator.save_connector(
        'output/classification_data.pkl.lz4',
        pickle_manager=pickle_manager,
        optimize_for_size=True
    )
    
    print(f"   Saved {save_stats['file_size_mb']:.2f} MB")
    print(f"   Compression ratio: {save_stats['compression_ratio']:.2f}x")
    
    # Step 5: Load and train model
    print("\n4. Loading data and training model...")
    start_time = time.time()
    data = load_data_connector('output/classification_data.pkl.lz4')
    load_time = time.time() - start_time
    print(f"   Data loaded in {load_time:.3f} seconds")
    
    df = data['classification']
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\n5. Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"   Training time: {train_time:.2f} seconds")
    print(f"   Train accuracy: {train_score:.3f}")
    print(f"   Test accuracy: {test_score:.3f}")
    
    # Cross-validation
    print("\n6. Cross-validation scores:")
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"   CV scores: {cv_scores}")
    print(f"   Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance
    print("\n7. Top 10 feature importances:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # Get optimization report
    print("\n8. Data optimization report:")
    report = optimizer.get_optimization_report()
    print(f"   Original columns: {len(report['original_columns'])}")
    print(f"   Final columns: {len(X.columns) + 1}")  # +1 for target
    print(f"   Optimizations applied: {', '.join(report['optimizations_applied'])}")


def example_regression_workflow():
    """Complete regression workflow with data connector."""
    print("\n=== Regression Workflow Example ===")
    
    # Step 1: Create data
    print("\n1. Creating sample regression data...")
    _, reg_data = create_sample_ml_data()
    
    # Step 2: Save directly with aggregator
    aggregator = DataAggregator()
    aggregator.datasets = {'house_prices': reg_data}
    
    # Step 3: One-line preprocessing and save
    print("\n2. Preprocessing and saving in one step...")
    save_stats = aggregator.aggregate_and_save(
        'output/regression_data.pkl.lz4',
        optimize=True,
        compression='lz4',
        optimizer_kwargs={
            'target_column': 'price',
            'handle_missing': 'auto',
            'encode_categorical': 'onehot',
            'scale_numeric': 'robust',  # Robust scaler for regression
        }
    )
    
    # Step 4: Instant loading for ML
    print("\n3. Loading for machine learning...")
    start_time = time.time()
    data = load_data_connector('output/regression_data.pkl.lz4')
    print(f"   Loaded in {time.time() - start_time:.3f} seconds")
    
    df = data['house_prices']
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Train model
    print("\n4. Training Gradient Boosting regressor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"\n5. Model performance:")
    print(f"   Train RMSE: ${train_rmse:,.2f}")
    print(f"   Test RMSE: ${test_rmse:,.2f}")
    print(f"   R² score: {model.score(X_test, y_test):.3f}")


def example_large_dataset_workflow():
    """Example handling large datasets efficiently."""
    print("\n=== Large Dataset Workflow Example ===")
    
    # Create a larger dataset
    print("\n1. Creating large dataset (1M rows)...")
    n_samples = 1_000_000
    
    large_data = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(50)
    })
    
    # Add categorical features
    for i in range(10):
        large_data[f'cat_{i}'] = np.random.choice(
            [f'val_{j}' for j in range(20)], 
            n_samples
        )
    
    # Add target
    large_data['target'] = (large_data['feature_0'] + large_data['feature_1'] > 0).astype(int)
    
    print(f"   Dataset shape: {large_data.shape}")
    print(f"   Memory usage: {large_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save with optimization
    print("\n2. Optimizing and saving...")
    aggregator = DataAggregator()
    aggregator.datasets = {'large_dataset': large_data}
    
    start_time = time.time()
    save_stats = aggregator.aggregate_and_save(
        'output/large_dataset.pkl.lz4',
        optimize=True,
        compression='lz4',
        optimizer_kwargs={
            'target_column': 'target',
            'handle_missing': 'drop',
            'encode_categorical': 'label',  # Label encoding for high cardinality
            'scale_numeric': None,  # Skip scaling for speed
            'remove_low_variance': True,
            'remove_high_correlation': True
        }
    )
    
    print(f"   Processing time: {time.time() - start_time:.2f} seconds")
    print(f"   Original size: {save_stats['original_size_mb']:.2f} MB")
    print(f"   Compressed size: {save_stats['file_size_mb']:.2f} MB")
    print(f"   Compression ratio: {save_stats['compression_ratio']:.2f}x")
    
    # Test loading speed
    print("\n3. Testing load performance...")
    start_time = time.time()
    loaded_data = load_data_connector('output/large_dataset.pkl.lz4')
    load_time = time.time() - start_time
    
    df = loaded_data['large_dataset']
    print(f"   Load time: {load_time:.2f} seconds")
    print(f"   Loaded shape: {df.shape}")
    print(f"   MB/second: {save_stats['file_size_mb'] / load_time:.2f}")


def example_iterative_ml_development():
    """Example of iterative ML development with saved connectors."""
    print("\n=== Iterative ML Development Example ===")
    
    # Simulate having a saved connector from previous work
    print("\n1. Loading previously saved ML-ready data...")
    
    # For demo, create and save some data first
    clf_data, _ = create_sample_ml_data()
    aggregator = DataAggregator()
    aggregator.datasets = {'experiment_v1': clf_data}
    aggregator.aggregate_and_save(
        'output/experiment_v1.pkl.lz4',
        optimize=True,
        optimizer_kwargs={'target_column': 'target'}
    )
    
    # Now the actual workflow - instant loading
    data = load_data_connector('output/experiment_v1.pkl.lz4')
    df = data['experiment_v1']
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Quick model experiments
    print("\n2. Running quick model experiments...")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
    }
    
    for name, model in models.items():
        start = time.time()
        scores = cross_val_score(model, X, y, cv=3)
        elapsed = time.time() - start
        
        print(f"\n   {name}:")
        print(f"     CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        print(f"     Time: {elapsed:.2f}s")
    
    print("\n✨ With instant data loading, you can iterate on models quickly!")


if __name__ == '__main__':
    print("Instant Data Connector - ML Workflow Examples")
    print("=" * 60)
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    try:
        example_classification_workflow()
        example_regression_workflow()
        example_large_dataset_workflow()
        example_iterative_ml_development()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n💡 Key benefits demonstrated:")
        print("   - Instant data loading (no preprocessing wait)")
        print("   - Automatic ML-ready transformations")
        print("   - Efficient compression (2-10x ratios)")
        print("   - Consistent preprocessing across experiments")
        print("   - Fast iteration on model development")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()