#!/usr/bin/env python3
"""Validate Python syntax of all modules."""

import ast
import sys
from pathlib import Path

def test_syntax_validation():
    """Test that all Python files have valid syntax."""
    
    print("Testing Python syntax validation...")
    
    src_dir = Path("src")
    test_dir = Path("tests")
    
    python_files = []
    python_files.extend(src_dir.rglob("*.py"))
    python_files.extend(test_dir.rglob("*.py"))
    
    errors = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to check syntax
            ast.parse(content, filename=str(py_file))
            print(f"✓ {py_file}")
            
        except SyntaxError as e:
            error_msg = f"✗ {py_file}: {e}"
            print(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"✗ {py_file}: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    if errors:
        print(f"\n❌ Found {len(errors)} syntax errors:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print(f"\n✅ All {len(python_files)} Python files have valid syntax!")
        return True

def check_method_signatures():
    """Check that key method signatures look correct."""
    
    print("\nChecking key method signatures...")
    
    # Read the ML optimizer file and check key signatures
    ml_optimizer_file = Path("src/instant_connector/ml_optimizer.py")
    
    try:
        with open(ml_optimizer_file, 'r') as f:
            content = f.read()
        
        # Check for key method signatures
        required_signatures = [
            "def _identify_column_types(self, df:",
            "def _remove_constant_columns(self, df:",
            "def _handle_missing_values(",
            "def _engineer_features(",
            "def _encode_categoricals(",
            "def _scale_numeric(",
            "def _calculate_feature_importance(",
            "def fit_transform(",
            "def transform("
        ]
        
        for signature in required_signatures:
            if signature in content:
                print(f"✓ Found: {signature}")
            else:
                print(f"✗ Missing: {signature}")
                return False
        
        print("✅ All required method signatures found!")
        return True
        
    except Exception as e:
        print(f"✗ Error checking method signatures: {e}")
        return False

if __name__ == "__main__":
    syntax_ok = test_syntax_validation()
    signatures_ok = check_method_signatures()
    
    if syntax_ok and signatures_ok:
        print("\n🎉 All validation tests passed!")
        print("The code is syntactically correct and has the required structure.")
        print("Run 'pip install -r requirements.txt' to install dependencies and run full tests.")
    else:
        print("\n❌ Some validation tests failed.")
    
    sys.exit(0 if (syntax_ok and signatures_ok) else 1)