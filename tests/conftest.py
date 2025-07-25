"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import shutil


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup temporary test files after each test."""
    yield
    # Cleanup any test files created during tests
    test_patterns = [
        'test_*.pkl*',
        'temp_*.csv',
        'temp_*.json',
        'temp_*.xlsx',
        '*.pkl.gz',
        '*.pkl.lz4',
        '*.pkl.bz2'
    ]
    
    for pattern in test_patterns:
        for file in Path.cwd().glob(pattern):
            try:
                file.unlink()
            except:
                pass


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100),
        'numeric': np.random.randn(100),
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.rand(100) * 100
    })


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_api_responses():
    """Common API response patterns for testing."""
    return {
        'simple': [
            {'id': 1, 'name': 'Item 1', 'value': 10},
            {'id': 2, 'name': 'Item 2', 'value': 20}
        ],
        'nested': {
            'data': {
                'items': [
                    {'id': 1, 'meta': {'category': 'A'}},
                    {'id': 2, 'meta': {'category': 'B'}}
                ]
            }
        },
        'paginated': {
            'results': [{'id': 1}, {'id': 2}],
            'next': 'https://api.example.com/items?page=2'
        }
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )