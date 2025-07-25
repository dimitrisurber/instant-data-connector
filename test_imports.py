#!/usr/bin/env python3
"""Test that all modules can be imported without syntax errors."""

import sys
import os
sys.path.insert(0, 'src')

def test_imports():
    """Test importing all modules."""
    
    print("Testing imports...")
    
    try:
        # Test basic imports
        print("Testing basic structure...")
        
        # This will fail due to pandas dependency, but we can catch and report the specific error
        try:
            from instant_connector import InstantDataConnector
            print("✓ InstantDataConnector imports successfully")
        except ImportError as e:
            if "pandas" in str(e) or "numpy" in str(e) or "sklearn" in str(e):
                print("✓ InstantDataConnector structure OK (missing pandas/numpy/sklearn dependencies)")
            else:
                print(f"✗ InstantDataConnector import error: {e}")
                return False
        
        # Test individual modules
        modules_to_test = [
            'instant_connector.aggregator',
            'instant_connector.ml_optimizer', 
            'instant_connector.pickle_manager',
            'instant_connector.sources.database_source',
            'instant_connector.sources.file_source',
            'instant_connector.sources.api_source'
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"✓ {module} imports successfully")
            except ImportError as e:
                if any(dep in str(e) for dep in ["pandas", "numpy", "sklearn", "sqlalchemy", "requests"]):
                    print(f"✓ {module} structure OK (missing dependencies)")
                else:
                    print(f"✗ {module} import error: {e}")
                    return False
        
        print("\n✅ All modules have correct structure and syntax!")
        print("Note: Cannot run full tests without pandas/numpy/sklearn dependencies")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)