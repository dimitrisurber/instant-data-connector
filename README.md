# Instant Data Connector

A Python library for aggregating data from multiple sources and serializing it into ML-optimized pickle files for instant algorithm development. Load preprocessed, ML-ready datasets in seconds instead of minutes.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Data Sources](#data-sources)
  - [ML Optimization](#ml-optimization)
  - [Compression Options](#compression-options)
  - [Configuration Files](#configuration-files)
- [Command Line Interface](#command-line-interface)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Features

- **Multi-Source Aggregation**: Extract data from databases (PostgreSQL, MySQL, SQLite), files (CSV, Excel, JSON, Parquet), and REST APIs
- **ML-Ready Preprocessing**: Automatic handling of missing values, categorical encoding, feature scaling, and dimensionality reduction
- **Efficient Serialization**: Compressed pickle files with LZ4, GZIP, or BZ2 compression (2-10x compression ratios)
- **Instant Loading**: Load preprocessed datasets in seconds, not minutes (10-100x faster)
- **Memory Optimization**: Automatic dtype optimization for reduced memory usage
- **Metadata Tracking**: Store preprocessing steps, statistics, and source information
- **Large Dataset Support**: Efficiently handle datasets from 100MB to 10GB

## Installation

### From source:
```bash
git clone https://github.com/dimitrisurber/instant-data-connector.git
cd instant-data-connector
pip install -r requirements.txt
python setup.py install
```

### Dependencies:
- Python 3.8+
- pandas, numpy, PyYAML
- sqlalchemy, psycopg2-binary, pymysql (for databases)
- openpyxl, xlrd, pyarrow (for file formats)
- lz4 (for compression)
- requests (for APIs)
- scikit-learn (for ML preprocessing)

## Quick Start

### 1. Simple File Aggregation

```python
from instant_connector import DataAggregator

# Create aggregator
aggregator = DataAggregator()

# Add CSV files
aggregator.add_file_source(
    name='sales_data',
    file_paths=['sales_2023.csv', 'sales_2024.csv']
)

# Extract, optimize, and save
aggregator.aggregate_and_save('sales_data.pkl.lz4')

# Load instantly later
from instant_connector.pickle_manager import load_data_connector
data = load_data_connector('sales_data.pkl.lz4')
df = data['sales_data_data']  # Access your DataFrame
```

### 2. Database with ML Optimization

```python
# Add database source
aggregator.add_database_source(
    name='analytics_db',
    connection_params={
        'db_type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'analytics',
        'username': 'user',
        'password': 'password'
    },
    queries={
        'user_features': "SELECT * FROM user_analytics WHERE date >= '2024-01-01'",
        'product_metrics': "SELECT * FROM product_performance"
    }
)

# Save with ML optimizations
aggregator.aggregate_and_save(
    'ml_ready_data.pkl.lz4',
    optimize=True,
    optimizer_kwargs={
        'handle_missing': 'auto',
        'encode_categorical': 'auto',
        'scale_numeric': 'standard',
        'remove_low_variance': True,
        'remove_high_correlation': True
    }
)
```

### 3. API Data Aggregation

```python
# Add API source with pagination
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
```

## Detailed Usage

### Data Sources

#### Database Sources

Supported databases: PostgreSQL, MySQL, SQLite

```python
# PostgreSQL
aggregator.add_database_source(
    name='postgres_db',
    connection_params={
        'db_type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'mydb',
        'username': 'user',
        'password': 'password'
    },
    queries={
        'query_name': 'SELECT * FROM table_name'
    }
)

# MySQL
aggregator.add_database_source(
    name='mysql_db',
    connection_params={
        'db_type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'mydb',
        'username': 'user',
        'password': 'password'
    }
)

# SQLite
aggregator.add_database_source(
    name='sqlite_db',
    connection_params={
        'db_type': 'sqlite',
        'database': 'path/to/database.db'
    }
)
```

#### File Sources

Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet

```python
# Single file
aggregator.add_file_source(
    name='csv_data',
    file_paths='data.csv',
    read_options={
        'parse_dates': ['date_column'],
        'low_memory': False
    }
)

# Multiple files
aggregator.add_file_source(
    name='excel_data',
    file_paths=['report1.xlsx', 'report2.xlsx'],
    read_options={
        'sheet_name': 'Data'
    }
)

# JSON with different orientations
aggregator.add_file_source(
    name='json_data',
    file_paths='data.json',
    read_options={
        'orient': 'records'  # or 'split', 'index'
    }
)
```

#### API Sources

REST API with various pagination types:

```python
# Offset-based pagination
aggregator.add_api_source(
    name='api_data',
    base_url='https://api.example.com',
    headers={'Authorization': 'Bearer TOKEN'},
    rate_limit_delay=0.5,  # seconds between requests
    endpoints={
        'users': {
            'endpoint': 'users',
            'method': 'GET',
            'paginate': True,
            'pagination_type': 'offset',
            'params': {'limit': 100},
            'max_pages': 10
        }
    }
)

# Page-based pagination
endpoints={
    'products': {
        'endpoint': 'products',
        'paginate': True,
        'pagination_type': 'page',
        'params': {'per_page': 50}
    }
}

# Next URL pagination
endpoints={
    'orders': {
        'endpoint': 'orders',
        'paginate': True,
        'pagination_type': 'next_url'
    }
}
```

### ML Optimization

The MLOptimizer provides automatic preprocessing for machine learning:

```python
from instant_connector import MLOptimizer

optimizer = MLOptimizer()

# Optimization options
df_optimized = optimizer.optimize_dataframe(
    df,
    target_column='target',  # Excluded from preprocessing
    
    # Missing value handling
    handle_missing='auto',  # Options: 'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill'
    
    # Categorical encoding
    encode_categorical='auto',  # Options: 'auto', 'label', 'onehot', None
    
    # Feature scaling
    scale_numeric='standard',  # Options: 'standard', 'minmax', 'robust', None
    
    # Feature selection
    remove_low_variance=True,
    remove_high_correlation=True,
    correlation_threshold=0.95
)

# Get optimization report
report = optimizer.get_optimization_report()
print(report['optimizations_applied'])
print(report['feature_stats'])
```

### Compression Options

Choose compression based on your needs:

```python
from instant_connector import PickleManager

# LZ4 (default) - Fast compression/decompression, good ratio
pickle_manager = PickleManager(compression='lz4', compression_level=0)

# GZIP - Better compression ratio, slower
pickle_manager = PickleManager(compression='gzip', compression_level=6)

# BZ2 - Best compression ratio, slowest
pickle_manager = PickleManager(compression='bz2', compression_level=9)

# No compression - Fastest save/load, largest files
pickle_manager = PickleManager(compression='none')
```

### Configuration Files

Use YAML configuration for complex setups:

```yaml
# config/sources.yaml
sources:
  sales_db:
    type: database
    connection:
      db_type: postgresql
      host: ${DB_HOST}  # Environment variables supported
      port: 5432
      database: sales
      username: ${DB_USER}
      password: ${DB_PASSWORD}
    queries:
      customers: |
        SELECT customer_id, name, email, created_at
        FROM customers
        WHERE created_at >= '2023-01-01'
      orders: |
        SELECT * FROM orders
        WHERE status = 'completed'
  
  csv_files:
    type: file
    paths:
      - data/2023/*.csv
      - data/2024/*.csv
    read_options:
      parse_dates: ['date']
      dtype: {'product_id': str}
  
  api_source:
    type: api
    base_url: https://api.example.com/v1
    headers:
      Authorization: Bearer ${API_TOKEN}
    rate_limit_delay: 1.0
    endpoints:
      inventory:
        endpoint: inventory/items
        method: GET
        paginate: true
        pagination_type: offset
        params:
          limit: 100
          include_deleted: false

optimization:
  handle_missing: auto
  encode_categorical: auto
  scale_numeric: standard
```

Load configuration:
```python
aggregator = DataAggregator(config_path='config/sources.yaml')
aggregator.aggregate_and_save('output.pkl.lz4')
```

## Command Line Interface

### aggregate_data.py

```bash
# Basic usage
python scripts/aggregate_data.py output.pkl.lz4 --config config/sources.yaml

# With file sources
python scripts/aggregate_data.py output.pkl.lz4 \
    --files data/*.csv \
    --compression lz4 \
    --encode-categorical auto \
    --scale-numeric standard

# With database
python scripts/aggregate_data.py output.pkl.lz4 \
    --database "postgresql://user:pass@localhost/db" "SELECT * FROM table" \
    --no-optimize

# With API
python scripts/aggregate_data.py output.pkl.lz4 \
    --api https://api.example.com /endpoint GET \
    --compression gzip \
    --compression-level 6
```

Options:
- `--config`: Configuration file path
- `--files`: File paths to aggregate
- `--database`: Database connection and query
- `--api`: API base URL, endpoint, and method
- `--no-optimize`: Skip ML optimization
- `--handle-missing`: Strategy for missing values
- `--encode-categorical`: Categorical encoding strategy
- `--scale-numeric`: Numeric scaling strategy
- `--compression`: Compression method (lz4, gzip, bz2, none)
- `--compression-level`: Compression level
- `-v, --verbose`: Enable verbose logging

### load_data.py

```bash
# Basic loading
python scripts/load_data.py data.pkl.lz4

# Show metadata
python scripts/load_data.py data.pkl.lz4 --info

# Show summary and first 5 rows
python scripts/load_data.py data.pkl.lz4 --summary --head 5

# Show statistics
python scripts/load_data.py data.pkl.lz4 --describe

# Load specific datasets
python scripts/load_data.py data.pkl.lz4 --datasets dataset1 dataset2
```

## Examples

### Example 1: E-commerce Data Pipeline

```python
# Aggregate sales data from multiple sources
aggregator = DataAggregator()

# Add database with customer data
aggregator.add_database_source(
    'customer_db',
    connection_params={...},
    queries={
        'customers': 'SELECT * FROM customers',
        'orders': 'SELECT * FROM orders WHERE date >= CURRENT_DATE - INTERVAL 30 DAY'
    }
)

# Add CSV files with product data
aggregator.add_file_source(
    'product_files',
    ['products.csv', 'inventory.csv']
)

# Add API for real-time pricing
aggregator.add_api_source(
    'pricing_api',
    base_url='https://api.pricing.com',
    endpoints={
        'prices': {
            'endpoint': 'products/prices',
            'paginate': True
        }
    }
)

# Save with optimizations
aggregator.aggregate_and_save(
    'ecommerce_ml_data.pkl.lz4',
    optimize=True,
    optimizer_kwargs={
        'handle_missing': 'median',
        'encode_categorical': 'onehot',
        'scale_numeric': 'minmax'
    }
)
```

### Example 2: Time Series Data

```python
# For time series, preserve temporal order
aggregator = DataAggregator()

# Add time series data
aggregator.add_file_source(
    'timeseries',
    ['sensor_data_*.csv'],
    read_options={
        'parse_dates': ['timestamp'],
        'index_col': 'timestamp',
        'sort': True
    }
)

# Custom optimization for time series
from instant_connector import MLOptimizer

optimizer = MLOptimizer()
data = aggregator.extract_data()

for name, df in data.items():
    # Don't scale time series data
    df_optimized = optimizer.optimize_dataframe(
        df,
        handle_missing='forward_fill',  # Good for time series
        scale_numeric=None,  # Preserve scale
        encode_categorical='label'
    )
    aggregator.datasets[name] = df_optimized

# Save
aggregator.save_connector('timeseries_data.pkl.lz4')
```

### Example 3: Large Dataset with Chunking

```python
# For very large datasets
from instant_connector.sources import DatabaseSource

with DatabaseSource(connection_params) as db_source:
    # Process in chunks
    chunk_iter = db_source.extract_data(
        query='SELECT * FROM huge_table',
        chunksize=10000  # Process 10k rows at a time
    )
    
    all_chunks = []
    for i, chunk in enumerate(chunk_iter):
        # Process each chunk
        processed = optimizer.optimize_dataframe(chunk)
        all_chunks.append(processed)
        
        if i >= 100:  # Limit to 1M rows
            break
    
    # Combine and save
    final_df = pd.concat(all_chunks, ignore_index=True)
```

## API Reference

### DataAggregator

```python
class DataAggregator:
    def __init__(self, config_path: Optional[str] = None)
    
    def add_database_source(name: str, connection_params: dict, queries: dict)
    def add_file_source(name: str, file_paths: Union[str, List[str]], read_options: dict)
    def add_api_source(name: str, base_url: str, endpoints: dict, **kwargs)
    
    def extract_data(source_name: Optional[str] = None) -> Dict[str, pd.DataFrame]
    def optimize_datasets(**kwargs) -> Dict[str, pd.DataFrame]
    def save_connector(file_path: str, **kwargs) -> Dict[str, Any]
    def aggregate_and_save(output_path: str, optimize: bool = True, **kwargs)
```

### MLOptimizer

```python
class MLOptimizer:
    def optimize_dataframe(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        handle_missing: str = 'auto',
        encode_categorical: str = 'auto',
        scale_numeric: Optional[str] = None,
        remove_low_variance: bool = True,
        remove_high_correlation: bool = True
    ) -> pd.DataFrame
    
    def get_optimization_report() -> Dict[str, Any]
    def transform_new_data(df: pd.DataFrame) -> pd.DataFrame
```

### PickleManager

```python
class PickleManager:
    def __init__(self, compression: str = 'lz4', compression_level: int = 0)
    
    def save_data_connector(
        data: Dict[str, pd.DataFrame],
        file_path: str,
        metadata: Optional[dict] = None,
        optimize_for_size: bool = False
    ) -> Dict[str, Any]
    
    def load_data_connector(
        file_path: str,
        dataset_names: Optional[List[str]] = None,
        return_metadata: bool = False
    ) -> Union[Dict[str, pd.DataFrame], Tuple[Dict, Dict]]

# Convenience function
def load_data_connector(file_path: str, **kwargs) -> Dict[str, pd.DataFrame]
```

## Performance

### Benchmarks

| Dataset Size | Raw CSV Load | Instant Connector Load | Speedup |
|-------------|--------------|----------------------|---------|
| 100 MB      | 5.2s         | 0.3s                 | 17x     |
| 1 GB        | 52s          | 2.1s                 | 25x     |
| 5 GB        | 260s         | 9.8s                 | 27x     |

### Compression Ratios

| Compression | Ratio | Save Time | Load Time |
|-------------|-------|-----------|-----------|
| LZ4         | 3.2x  | Fast      | Fast      |
| GZIP        | 4.8x  | Medium    | Medium    |
| BZ2         | 5.6x  | Slow      | Slow      |
| None        | 1.0x  | Fastest   | Fastest   |

### Memory Optimization

- Integer downcasting: 50-70% reduction for numeric columns
- Categorical encoding: 60-90% reduction for string columns
- Float32 conversion: 50% reduction for decimal columns

## Testing

### Running Tests

The project includes a comprehensive test suite covering unit tests, integration tests, and performance tests.

#### Run all tests:
```bash
pytest

# Or use the test runner
python run_tests.py
```

#### Run specific test modules:
```bash
# Test individual components
pytest tests/test_database_source.py
pytest tests/test_file_source.py
pytest tests/test_api_source.py
pytest tests/test_ml_optimizer.py
pytest tests/test_pickle_manager.py
pytest tests/test_aggregator.py

# Run integration tests
pytest tests/test_integration.py -m integration
```

#### Run with coverage:
```bash
# Install coverage dependencies
pip install pytest-cov

# Run with coverage report
pytest --cov=instant_connector --cov-report=html
```

#### Run specific test categories:
```bash
# Fast unit tests only
pytest -m "not slow"

# Integration tests only
pytest -m integration
```

### Test Structure

- `tests/test_database_source.py` - Database connector tests
- `tests/test_file_source.py` - File source tests (CSV, Excel, JSON, Parquet)
- `tests/test_api_source.py` - API connector tests
- `tests/test_ml_optimizer.py` - ML preprocessing tests
- `tests/test_pickle_manager.py` - Serialization and compression tests
- `tests/test_aggregator.py` - Main aggregator tests
- `tests/test_integration.py` - End-to-end integration tests

### Writing Tests

When contributing, please ensure:
1. All new features have corresponding tests
2. Tests follow the existing naming convention
3. Integration tests are marked with `@pytest.mark.integration`
4. Slow tests are marked with `@pytest.mark.slow`

## Troubleshooting

### Common Issues

**1. Database Connection Errors**
```python
# Check connection parameters
try:
    aggregator.add_database_source(...)
except Exception as e:
    print(f"Connection failed: {e}")
    # Verify host, port, credentials
```

**2. Memory Errors with Large Files**
```python
# Use chunking for large files
aggregator.add_file_source(
    'large_data',
    'huge_file.csv',
    read_options={'chunksize': 10000}
)
```

**3. Slow API Responses**
```python
# Adjust timeout and retry settings
aggregator.add_api_source(
    'slow_api',
    base_url='...',
    timeout=60,  # seconds
    max_retries=5,
    rate_limit_delay=2.0
)
```

**4. Pickle Compatibility**
```python
# Ensure same pandas version for save/load
import pandas as pd
print(pd.__version__)
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use CLI
python scripts/aggregate_data.py output.pkl.lz4 --verbose
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details