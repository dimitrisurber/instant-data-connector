# Instant Data Connector

A Python library for aggregating data from multiple sources and serializing it into ML-optimized formats for instant algorithm development. Load preprocessed, ML-ready datasets in seconds instead of minutes.

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
- **Secure Serialization**: Compressed formats with LZ4, GZIP, BZ2, or ZSTD compression (2-10x compression ratios)
- **Instant Loading**: Load preprocessed datasets in seconds, not minutes (10-100x faster)
- **Memory Optimization**: Automatic dtype optimization for reduced memory usage
- **Metadata Tracking**: Store preprocessing steps, statistics, and source information
- **Large Dataset Support**: Efficiently handle datasets from 100MB to 10GB
- **Security Features**: Secure credential management and encrypted serialization
- **Resource Monitoring**: Built-in memory and processing limits

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
- pandas>=2.1.0, numpy>=1.24.0, PyYAML>=6.0.2
- sqlalchemy>=2.0.0, psycopg2-binary>=2.9.10, pymysql>=1.1.0 (for databases)
- openpyxl>=3.1.5, pyarrow>=15.0.0 (for file formats)
- lz4>=4.4.0, zstandard>=0.22.0 (for compression)
- requests>=2.32.0 (for APIs)
- scikit-learn>=1.5.0 (for ML preprocessing)
- cryptography>=42.0.0, keyring>=25.0.0 (for security)

## Quick Start

### 1. Simple File Aggregation

```python
from instant_connector import InstantDataConnector
from instant_connector.pickle_manager import load_data_connector

# Create connector
connector = InstantDataConnector()

# Add file source (single file)
connector.add_file_source(
    name='sales_data',
    file_path='sales_2023.csv'
)

# Extract and save
raw_data = connector.extract_data()
save_stats = connector.save_connector('sales_data.pkl.lz4')

# Load instantly later
data = load_data_connector('sales_data.pkl.lz4')
df = data['sales_data_data']  # Access your DataFrame
```

### 2. Database with ML Optimization

```python
from instant_connector import InstantDataConnector, MLOptimizer

# Create connector
connector = InstantDataConnector()

# Add database source
connector.add_database_source(
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

# Extract data
raw_data = connector.extract_data()

# Apply ML optimizations
optimizer = MLOptimizer(
    handle_missing='auto',
    encode_categorical='auto',
    scale_numeric='standard'
)

for dataset_name, df in raw_data.items():
    result = optimizer.fit_transform(df)
    optimized_df = result.get('X_processed', df)
    connector.raw_data[dataset_name] = optimized_df

# Save with compression
save_stats = connector.save_connector('ml_ready_data.pkl.lz4')
```

### 3. API Data Aggregation

```python
# Add API source with pagination
connector.add_api_source(
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
connector.add_database_source(
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
    },
    optimize_dtypes=True,
    include_metadata=False
)

# MySQL
connector.add_database_source(
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
connector.add_database_source(
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
connector.add_file_source(
    name='csv_data',
    file_path='data.csv',
    read_options={
        'parse_dates': ['date_column'],
        'low_memory': False
    }
)

# Multiple files (using file_paths parameter)
connector.add_file_source(
    name='excel_data',
    file_path=None,
    file_paths=['report1.xlsx', 'report2.xlsx'],
    read_options={
        'sheet_name': 'Data'
    }
)

# JSON with different orientations
connector.add_file_source(
    name='json_data',
    file_path='data.json',
    read_options={
        'orient': 'records'  # or 'split', 'index'
    }
)
```

#### API Sources

REST API with various pagination types:

```python
# Offset-based pagination
connector.add_api_source(
    name='api_data',
    base_url='https://api.example.com',
    headers={'Authorization': 'Bearer TOKEN'},
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
endpoints = {
    'products': {
        'endpoint': 'products',
        'paginate': True,
        'pagination_type': 'page',
        'params': {'per_page': 50}
    }
}

# Next URL pagination
endpoints = {
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

# Initialize with configuration
optimizer = MLOptimizer(
    random_state=42,
    handle_missing='auto',  # 'auto', 'drop', 'mean', 'median', 'mode', 'knn'
    encode_categorical='auto',  # 'auto', 'label', 'onehot', 'target', 'ordinal'
    scale_numeric='auto',  # 'auto', 'standard', 'minmax', 'robust', None
    feature_engineering=False,
    reduce_memory=False
)

# Complete ML preprocessing pipeline
result = optimizer.fit_transform(
    df,
    target_column='target',  # Optional target column
    test_size=0.2,
    stratify=False,
    preserve_artifacts=True,
    reduce_memory=False
)

# Access processed data
if 'X_train' in result:
    # Training/test splits created
    X_train, X_test = result['X_train'], result['X_test']
    y_train, y_test = result['y_train'], result['y_test']
else:
    # Single processed dataset
    df_optimized = result['X_processed']

# Get preprocessing information
preprocessing_info = optimizer.get_preprocessing_info()
print(f"Applied {len(preprocessing_info['preprocessing_history'])} optimization steps")

# Transform new data using saved artifacts
new_data_transformed = optimizer.transform(new_df, result.get('ml_artifacts'))
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

# ZSTD - High performance compression
pickle_manager = PickleManager(compression='zstd', compression_level=3)

# No compression - Fastest save/load, largest files
pickle_manager = PickleManager(compression='none')

# Use with connector
save_stats = connector.save_connector('output.pkl.lz4', pickle_manager=pickle_manager)
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
    path: data/sales_2023.csv  # Single file
    read_options:
      parse_dates: ['date']
      dtype: {'product_id': str}
  
  api_source:
    type: api
    base_url: https://api.example.com/v1
    headers:
      Authorization: Bearer ${API_TOKEN}
    endpoints:
      inventory:
        endpoint: inventory/items
        method: GET
        paginate: true
        pagination_type: offset
        params:
          limit: 100

# ML optimization settings (for constructor)
optimization:
  handle_missing: auto
  encode_categorical: auto
  scale_numeric: standard
  feature_engineering: false
  reduce_memory: false

# Output settings
output:
  compression: lz4
  compression_level: 0
```

Load configuration:
```python
connector = InstantDataConnector(config_path='config/sources.yaml')
save_stats = connector.save_connector('output.pkl.lz4')
```

## Command Line Interface

### aggregate_data.py

```bash
# Basic usage with config
python scripts/aggregate_data.py output.pkl.lz4 --config config/sources.yaml

# With file sources
python scripts/aggregate_data.py output.pkl.lz4 \
    --files data/*.csv \
    --compression lz4

# With database
python scripts/aggregate_data.py output.pkl.lz4 \
    --database "postgresql://user:pass@localhost/db" "SELECT * FROM table"

# With API
python scripts/aggregate_data.py output.pkl.lz4 \
    --api https://api.example.com /endpoint GET
```

### load_data.py

```bash
# Basic loading
python scripts/load_data.py data.pkl.lz4

# Show metadata
python scripts/load_data.py data.pkl.lz4 --info

# Show summary and first 5 rows
python scripts/load_data.py data.pkl.lz4 --summary --head 5
```

## Examples

### Example 1: E-commerce Data Pipeline

```python
from instant_connector import InstantDataConnector, MLOptimizer

# Create connector
connector = InstantDataConnector(
    max_memory_mb=4096,  # 4GB limit
    max_rows=5_000_000,  # 5M rows limit
    use_secure_serialization=True
)

# Add database with customer data
connector.add_database_source(
    'customer_db',
    connection_params={
        'db_type': 'postgresql',
        'host': 'localhost',
        'database': 'ecommerce'
    },
    queries={
        'customers': 'SELECT * FROM customers',
        'orders': 'SELECT * FROM orders WHERE date >= CURRENT_DATE - INTERVAL 30 DAY'
    }
)

# Add CSV files with product data
connector.add_file_source(
    'product_files',
    file_path=None,
    file_paths=['products.csv', 'inventory.csv']
)

# Extract data
raw_data = connector.extract_data()

# Apply ML optimizations
optimizer = MLOptimizer(
    handle_missing='auto',
    encode_categorical='onehot',
    scale_numeric='minmax',
    feature_engineering=True
)

for name, df in raw_data.items():
    result = optimizer.fit_transform(df)
    optimized_df = result.get('X_processed', df)
    connector.raw_data[name] = optimized_df

# Save
save_stats = connector.save_connector('ecommerce_ml_data.pkl.lz4')
```

### Example 2: Time Series Data

```python
# For time series, preserve temporal order
connector = InstantDataConnector()

# Add time series data
connector.add_file_source(
    'timeseries',
    file_path='sensor_data.csv',
    read_options={
        'parse_dates': ['timestamp'],
        'index_col': 'timestamp'
    }
)

# Custom optimization for time series
optimizer = MLOptimizer(
    handle_missing='forward_fill',  # Good for time series
    scale_numeric=None,  # Preserve scale
    encode_categorical='label'
)

raw_data = connector.extract_data()
for name, df in raw_data.items():
    result = optimizer.fit_transform(df)
    connector.raw_data[name] = result.get('X_processed', df)

# Save
connector.save_connector('timeseries_data.pkl.lz4')
```

## API Reference

### InstantDataConnector

```python
class InstantDataConnector:
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        max_memory_mb: int = 2048,
        max_rows: int = 10_000_000,
        allowed_directories: Optional[List[Union[str, Path]]] = None,
        use_secure_serialization: bool = True,
        credential_manager: Optional[SecureCredentialManager] = None
    )
    
    def add_database_source(
        self,
        name: str,
        connection_params: Dict[str, Any],
        tables: Optional[List[str]] = None,
        queries: Optional[Dict[str, str]] = None,
        optimize_dtypes: bool = True,
        include_metadata: bool = False
    )
    
    def add_file_source(
        self,
        name: str,
        file_path: Union[str, Path],
        file_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        read_options: Optional[Dict[str, Any]] = None,
        optimize_dtypes: bool = True,
        include_metadata: bool = False
    )
    
    def add_api_source(
        self,
        name: str,
        base_url: str,
        endpoints: Union[List[str], Dict[str, Dict[str, Any]]],
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    )
    
    def extract_data(
        self,
        source_name: Optional[str] = None,
        dataset_name: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]
    
    def optimize_datasets(
        self,
        optimizer: Optional[MLOptimizer] = None,
        **optimizer_kwargs
    ) -> Dict[str, pd.DataFrame]
    
    def save_connector(
        self,
        file_path: Union[str, Path],
        pickle_manager: Optional[PickleManager] = None,
        include_metadata: bool = True,
        **save_kwargs
    ) -> Dict[str, Any]
```

### MLOptimizer

```python
class MLOptimizer:
    def __init__(
        self, 
        random_state: int = 42,
        handle_missing: str = 'auto',
        encode_categorical: str = 'auto', 
        scale_numeric: str = 'auto',
        feature_engineering: bool = False,
        reduce_memory: bool = False
    )
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        stratify: bool = False,
        preserve_artifacts: bool = False,
        reduce_memory: bool = False
    ) -> Dict[str, Any]
    
    def transform(
        self, 
        df: pd.DataFrame, 
        ml_artifacts: Optional[Dict] = None
    ) -> pd.DataFrame
    
    def optimize_dataframe(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> pd.DataFrame
    
    def optimize_for_ml(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        test_size: float = 0.2, 
        stratify: bool = False, 
        **kwargs
    ) -> Dict[str, pd.DataFrame]
    
    def get_preprocessing_info(self) -> Dict[str, Any]
    
    def get_feature_names(self) -> List[str]
```

### PickleManager

```python
class PickleManager:
    def __init__(
        self,
        compression: str = 'lz4',
        compression_level: int = 0,
        chunk_threshold_mb: int = 500,
        enable_progress: bool = True
    )
    
    def serialize_datasets(
        self,
        data_payload: Dict[str, Any],
        output_path: Union[str, Path],
        add_metadata: bool = True,
        optimize_memory: bool = True,
        validate: bool = True,
        parallel: bool = False
    ) -> Dict[str, Any]
    
    def save_data_connector(
        self,
        data: Union[Dict[str, pd.DataFrame], Dict[str, Any]],
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]
    
    def load_data_connector(
        self,
        file_path: Union[str, Path],
        dataset_names: Optional[List[str]] = None,
        return_metadata: bool = False
    ) -> Union[Dict[str, pd.DataFrame], Tuple[Dict, Dict]]

# Convenience functions
def load_data_connector(
    file_path: Union[str, Path], 
    **kwargs
) -> Dict[str, pd.DataFrame]

def save_data_connector(
    data: Union[Dict[str, pd.DataFrame], Dict[str, Any]],
    file_path: Union[str, Path],
    compression: str = 'lz4',
    **kwargs
) -> Dict[str, Any]
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
| ZSTD        | 4.1x  | Fast      | Fast      |
| None        | 1.0x  | Fastest   | Fastest   |

### Memory Optimization

- Integer downcasting: 50-70% reduction for numeric columns
- Categorical encoding: 60-90% reduction for string columns
- Float32 conversion: 50% reduction for decimal columns

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Or use the test runner
python run_tests.py

# Run specific test modules
pytest tests/test_aggregator.py
pytest tests/test_ml_optimizer.py
pytest tests/test_pickle_manager.py

# Run with coverage
pytest --cov=instant_connector --cov-report=html

# Fast unit tests only
pytest -m "not slow"

# Integration tests only
pytest -m integration
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Wrong
from instant_connector import DataAggregator

# Correct
from instant_connector import InstantDataConnector
```

**2. File Path Errors**
```python
# Wrong
connector.add_file_source('data', file_paths=['file.csv'])

# Correct
connector.add_file_source('data', file_path='file.csv')
# OR
connector.add_file_source('data', file_path=None, file_paths=['file.csv'])
```

**3. Memory Errors with Large Files**
```python
# Use resource limits
connector = InstantDataConnector(
    max_memory_mb=8192,  # 8GB limit
    max_rows=20_000_000  # 20M rows limit
)
```

**4. ML Optimization Errors**
```python
# Wrong
optimizer.optimize_dataframe(df, handle_missing='auto')

# Correct - configure in constructor
optimizer = MLOptimizer(handle_missing='auto')
result = optimizer.fit_transform(df)
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Features

- **Secure Credential Management**: Store database credentials securely using keyring
- **Encrypted Serialization**: Optional encryption for sensitive data
- **Resource Limits**: Built-in memory and processing limits
- **Path Validation**: Restrict file operations to allowed directories
- **Input Validation**: Comprehensive data validation and sanitization

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details