#!/usr/bin/env python3
"""Simple test of ML optimizer methods."""

import sys
sys.path.insert(0, 'src')

def test_ml_basic():
    """Test basic functionality without pandas import."""
    
    print("Testing basic ML optimizer functionality...")
    
    try:
        from instant_connector.ml_optimizer import MLOptimizer
        print("✓ MLOptimizer imported successfully")
        
        # Test basic initialization
        optimizer = MLOptimizer()
        print("✓ MLOptimizer created successfully")
        
        # Test basic properties
        assert hasattr(optimizer, '_identify_column_types')
        assert hasattr(optimizer, '_remove_constant_columns')
        assert hasattr(optimizer, '_handle_missing_values')
        assert hasattr(optimizer, '_engineer_features')
        assert hasattr(optimizer, '_encode_categoricals')
        assert hasattr(optimizer, '_scale_numeric')
        assert hasattr(optimizer, '_calculate_feature_importance')
        assert hasattr(optimizer, 'fit_transform')
        assert hasattr(optimizer, 'transform')
        
        print("✓ All required methods exist")
        
        print("✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ml_basic()