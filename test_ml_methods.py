#!/usr/bin/env python3
"""Test ML optimizer methods to identify missing signatures."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from instant_connector.ml_optimizer import MLOptimizer

def test_required_methods():
    """Test for methods expected by the test suite."""
    
    # Create test data
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [2, 4, 6, 8, 10],
        'cat1': ['A', 'B', 'A', 'B', 'A']
    })
    
    y = pd.Series([0, 1, 0, 1, 0])
    
    # Test ML optimizer creation
    print("Testing MLOptimizer initialization...")
    try:
        optimizer = MLOptimizer(
            handle_missing='auto',
            encode_categorical='auto', 
            scale_numeric='auto',
            feature_engineering=False
        )
        print("✓ MLOptimizer initialized successfully")
    except Exception as e:
        print(f"✗ MLOptimizer init failed: {e}")
        return
    
    # Test required methods from test_ml_optimizer.py
    methods_to_test = [
        '_identify_column_types',
        '_remove_constant_columns', 
        '_handle_missing_values',
        '_engineer_features',
        '_encode_categoricals',
        '_scale_numeric',
        '_calculate_feature_importance',
        'fit_transform',
        'transform'
    ]
    
    missing_methods = []
    
    for method_name in methods_to_test:
        if hasattr(optimizer, method_name):
            print(f"✓ {method_name} exists")
            
            # Test specific method signatures
            if method_name == '_identify_column_types':
                try:
                    result = optimizer._identify_column_types(df)
                    if isinstance(result, tuple) and len(result) == 4:
                        print(f"  ✓ {method_name} returns 4-tuple as expected")
                    else:
                        print(f"  ✗ {method_name} should return 4-tuple (numeric, categorical, datetime, other)")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
            
            elif method_name == '_remove_constant_columns':
                try:
                    test_df = df.copy()
                    test_df['constant'] = 1
                    cleaned, removed = optimizer._remove_constant_columns(test_df)
                    print(f"  ✓ {method_name} works, returned {type(cleaned)} and {type(removed)}")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
            
            elif method_name == '_handle_missing_values':
                try:
                    result, metadata = optimizer._handle_missing_values(df, None)
                    print(f"  ✓ {method_name} works")
                except Exception as e:
                    print(f"  ✗ {method_name} signature issue: {e}")
            
            elif method_name == '_engineer_features':
                try:
                    result, metadata = optimizer._engineer_features(df)
                    print(f"  ✓ {method_name} works")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
            
            elif method_name == '_encode_categoricals':
                try:
                    result, metadata = optimizer._encode_categoricals(df)
                    print(f"  ✓ {method_name} works")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
            
            elif method_name == '_scale_numeric':
                try:
                    result, metadata = optimizer._scale_numeric(df)
                    print(f"  ✓ {method_name} works")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
            
            elif method_name == '_calculate_feature_importance':
                try:
                    result = optimizer._calculate_feature_importance(df, y)
                    print(f"  ✓ {method_name} works")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
            
            elif method_name == 'fit_transform':
                try:
                    # Test basic fit_transform
                    result = optimizer.fit_transform(
                        df.drop('cat1', axis=1),
                        target_column=None,
                        preserve_artifacts=False
                    )
                    print(f"  ✓ {method_name} basic works")
                    
                    # Test with target
                    result = optimizer.fit_transform(
                        df.copy(),
                        target_column='cat1',
                        test_size=0.2,
                        stratify=True
                    )
                    print(f"  ✓ {method_name} with target works")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
                    
        else:
            missing_methods.append(method_name)
            print(f"✗ {method_name} missing")
    
    if missing_methods:
        print(f"\nMissing methods: {missing_methods}")
    else:
        print("\n✓ All required methods found")

if __name__ == "__main__":
    test_required_methods()