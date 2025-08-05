# Instant Data Connector

A high-performance PostgreSQL Foreign Data Wrapper (FDW) based data connector that provides unified access to multiple data sources through PostgreSQL's powerful query engine. Built for modern data architectures with lazy loading, push-down optimization, and enterprise-grade security.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 12+](https://img.shields.io/badge/postgresql-12+-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://www.docker.com/)
[![Security](https://img.shields.io/badge/security-hardened-green.svg)](#security-features)

## 🏗️ Architecture Overview

Instead of extracting and moving data, this connector uses **PostgreSQL Foreign Data Wrappers** to create a unified query interface across all your data sources. PostgreSQL acts as an intelligent query router, pushing filters, aggregations, and limits directly to source systems.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ML ML    │    │  Instant Data    │    │   PostgreSQL    │
│   Platform      │◄───┤   Connector      │◄───┤   FDW Hub       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                              ┌─────────┼─────────┐
                                              │         │         │
                                      ┌───────▼───┐ ┌───▼───┐ ┌───▼────┐
                                      │MySQL/NoSQL│ │ APIs  │ │Files   │
                                      │ Databases │ │ & Web │ │& S3    │
                                      └───────────┘ └───────┘ └────────┘
```

## 🚀 Key Features

- **🔗 Universal Connectivity**: Connect PostgreSQL, MySQL, MongoDB, REST APIs, file systems, and more through FDW
- **⚡ Push-down Optimization**: Filters, aggregations, and joins executed at source for maximum performance
- **🦥 Lazy Loading**: Query only what you need, when you need it - no unnecessary data movement
- **🔒 Enterprise Security**: SQL injection prevention, encrypted credentials, secure serialization
- **📈 Query Optimization**: Cost-based optimization with execution plan analysis
- **🐳 Production Ready**: Docker/Kubernetes deployment with monitoring and health checks
- **🔄 Async/Sync Support**: Full async architecture with sync wrappers for compatibility
- **📊 ML Integration**: Perfect integration with ML ML platform via pandas DataFrames

## 📦 Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 12+ with FDW extensions
- Docker (recommended for development)

### Quick Install
```bash
git clone https://github.com/your-org/instant-data-connector.git
cd instant-data-connector
pip install -r requirements.txt
python -m pip install -e .
```

### Docker Installation (Recommended)
```bash
# Development environment with all services
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

## 🏃 Quick Start

### 1. Basic FDW Setup

```python
import asyncio
from instant_connector import InstantDataConnector

async def main():
    # Initialize connector with PostgreSQL hub
    connector = InstantDataConnector(
        postgres_config={
            'host': 'localhost',
            'port': 5432,
            'database': 'data_hub',
            'username': 'connector_user',
            'password': 'secure_password'
        }
    )
    
    # Setup FDW infrastructure
    await connector.setup_fdw_infrastructure()
    
    # List available tables
    tables = await connector.list_available_tables()
    print(f"Available tables: {list(tables.keys())}")
    
    # Query with lazy loading and push-down optimization
    df = await connector.lazy_load_table(
        'remote_customers',
        filters={'status': 'active', 'region': 'US'},
        columns=['id', 'name', 'email', 'created_at'],
        limit=1000
    )
    
    print(f"Loaded {len(df)} customers efficiently!")
    await connector.close()

# Run async code
asyncio.run(main())
```

### 2. Configuration-Driven Setup

Create `config/sources.yaml`:
```yaml
# PostgreSQL FDW Configuration
extension: postgres_fdw

server:
  name: production_db
  options:
    host: prod-db.example.com
    port: 5432
    dbname: analytics
    
user_mapping:
  options:
    user: ${PROD_DB_USER}      # Environment variables supported
    password: ${PROD_DB_PASS}  # Secure credential management

tables:
  - name: customers
    columns:
      - {name: customer_id, type: integer}
      - {name: name, type: text}
      - {name: email, type: text}
      - {name: status, type: text}
      - {name: created_at, type: timestamp}
    options:
      table_name: customers
      schema_name: public
      
  - name: orders
    columns:
      - {name: order_id, type: integer}
      - {name: customer_id, type: integer}
      - {name: amount, type: numeric}
      - {name: order_date, type: date}
    options:
      table_name: orders
      schema_name: sales
```

Load and use configuration:
```python
# Load from configuration file
connector = InstantDataConnector(config_path='config/sources.yaml')
await connector.setup_fdw_infrastructure()

# Query across multiple tables with optimization
recent_orders = await connector.lazy_load_table(
    'orders',
    filters={
        'order_date': {'ge': '2024-01-01'},  # Greater than or equal
        'amount': {'gt': 100}                # Greater than
    },
    order_by=['-order_date'],  # Descending order
    limit=5000
)
```

### 3. ML Integration

Perfect integration with ML ML platform:

```python
# In your ML pipeline
import pandas as pd
from instant_connector import InstantDataConnector

async def get_training_data():
    connector = InstantDataConnector(config_path='ml_sources.yaml')
    await connector.setup_fdw_infrastructure()
    
    # Get features with automatic push-down optimization
    features_df = await connector.lazy_load_table(
        'customer_features',
        filters={'feature_date': {'ge': '2024-01-01'}},
        columns=['customer_id', 'age', 'income', 'purchase_history']
    )
    
    # Get labels
    labels_df = await connector.lazy_load_table(
        'customer_labels',
        filters={'label_date': {'ge': '2024-01-01'}},
        columns=['customer_id', 'churn_risk']
    )
    
    # ML can now use these DataFrames directly
    return features_df, labels_df

# Use in ML
features, labels = await get_training_data()
# Train your models with ML...
```

## 🔧 Advanced Configuration

### Multi-Source FDW Setup

```yaml
# Support multiple database types
sources:
  # PostgreSQL source
  postgres_prod:
    extension: postgres_fdw
    server:
      name: postgres_server
      options:
        host: postgres.example.com
        port: 5432
        dbname: production
    user_mapping:
      options:
        user: ${POSTGRES_USER}
        password: ${POSTGRES_PASS}
    tables:
      - name: users
        columns: [{name: id, type: integer}, {name: email, type: text}]
        options: {table_name: users}

  # MySQL source via mysql_fdw
  mysql_analytics:
    extension: mysql_fdw
    server:
      name: mysql_server
      options:
        host: mysql.example.com
        port: 3306
        database: analytics
    user_mapping:
      options:
        username: ${MYSQL_USER}
        password: ${MYSQL_PASS}
    tables:
      - name: events
        columns: [{name: event_id, type: integer}, {name: user_id, type: integer}]
        options: {table_name: user_events}

  # REST API source via multicorn
  api_service:
    extension: multicorn
    server:
      name: api_server
      options:
        wrapper: multicorn.restfdw.RestFdw
        base_url: https://api.example.com/v1
        auth_token: ${API_TOKEN}
    tables:
      - name: external_data
        columns: [{name: id, type: text}, {name: data, type: jsonb}]
        options: {endpoint: /data}
```

### Query Optimization Features

```python
# Cost-based optimization
query_info = connector.query_builder.build_select_query(
    'large_table',
    filters={'date_range': {'between': ['2024-01-01', '2024-12-31']}},
    columns=['id', 'revenue', 'region']
)

# Analyze query performance
optimized_query = await connector.query_builder.optimize_query(
    query_info, 
    connector.fdw_connector
)

print(f"Estimated cost: {optimized_query['cost_estimate']}")
print(f"Push-down eligible: {optimized_query['push_down_eligible']}")

# Execute with caching
df = await connector.execute_query(
    optimized_query['sql'],
    optimized_query['params'],
    cache_key='revenue_analysis_2024',
    cache_ttl=3600  # 1 hour cache
)
```

### Security Configuration

```python
from instant_connector import SecureCredentialManager

# Secure credential management
cred_manager = SecureCredentialManager()

# Store credentials securely (not in code!)
cred_manager.store_credential(
    'production_db',
    'db_user',
    'secure_password_from_vault'
)

# Use with connector
connector = InstantDataConnector(
    postgres_config={
        'host': 'localhost',
        'database': 'hub'
        # password retrieved automatically from credential manager
    },
    credential_manager=cred_manager
)
```

## 🐳 Docker Deployment

### Development Environment

```bash
# Start all services (PostgreSQL FDW + Redis + Monitoring)
docker-compose up -d

# Check service health
docker-compose ps

# View application logs
docker-compose logs -f connector-app

# Run tests
docker-compose exec connector-app python -m pytest tests/
```

### Production Deployment

```bash
# Production deployment with monitoring
docker-compose -f docker-compose.prod.yml up -d

# Health check
curl http://localhost:8000/health

# Metrics endpoint
curl http://localhost:8000/metrics
```

Services included:
- **Application**: FastAPI app with FDW connector
- **PostgreSQL**: FDW-enabled database with extensions
- **Redis**: Caching and session storage  
- **NGINX**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **AlertManager**: Alert routing

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=instant-data-connector

# Port forward for local access
kubectl port-forward service/connector-service 8000:8000
```

## 📊 Performance & Optimization

### Push-down Optimization Benefits

| Query Type | Without FDW | With FDW Push-down | Improvement |
|------------|-------------|-------------------|-------------|
| Filtered SELECT | Transfer 1M rows | Transfer 10K rows | 100x less data |
| Aggregation | Process locally | Process at source | 50x faster |
| JOIN operations | Multiple round trips | Single optimized query | 20x faster |
| Large datasets | Memory issues | Streaming results | No memory limit |

### Query Performance Monitoring

```python
# Enable performance tracking
async with connector.performance_monitor() as monitor:
    df = await connector.lazy_load_table(
        'large_dataset',
        filters={'status': 'active'}
    )
    
    # Get performance metrics
    stats = monitor.get_stats()
    print(f"Query time: {stats['execution_time']:.2f}s")
    print(f"Rows returned: {stats['rows_returned']:,}")
    print(f"Push-down used: {stats['push_down_optimized']}")
    print(f"Cache hit: {stats['cache_hit']}")
```

## 🔒 Security Features

### Comprehensive Security Implementation

✅ **SQL Injection Prevention**: Parameterized queries with identifier validation  
✅ **Secure Serialization**: JSON-based serialization replacing unsafe pickle  
✅ **Credential Management**: Encrypted credential storage with environment variable support  
✅ **Connection Security**: SSL/TLS enforcement with timeout controls  
✅ **Input Validation**: Multi-layered validation for all user inputs  
✅ **Container Security**: Non-root containers with security hardening  

### Security Testing

The codebase includes comprehensive security tests:

```bash
# Run security test suite (22 tests covering all vulnerabilities)
pytest tests/test_security.py -v

# Security-specific tests
pytest -k "security" -v

# All tests pass with 100% success rate
```

### Credential Security Best Practices

```python
# ✅ SECURE - Use environment variables
connection_params = {
    'host': 'prod-db.example.com',
    'username': 'app_user',
    'password': '${DB_PASSWORD}'  # Retrieved from environment
}

# ✅ SECURE - Use credential manager
cred_manager = SecureCredentialManager()
password = cred_manager.get_credential('prod_db', 'app_user')

# ❌ INSECURE - Never hardcode credentials
connection_params = {
    'password': 'hardcoded_password'  # Will trigger security warnings
}
```

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest

# Security tests specifically
pytest tests/test_security.py

# Integration tests with real databases
pytest tests/integration/ --db-url="postgresql://test:test@localhost/test"

# Performance benchmarks
pytest tests/test_performance.py --benchmark

# With coverage report
pytest --cov=instant_connector --cov-report=html
```

### Test Coverage

- ✅ **Security Tests**: 22 tests, 100% pass rate
- ✅ **FDW Integration**: Real PostgreSQL testing with testcontainers
- ✅ **Query Building**: Comprehensive SQL generation testing
- ✅ **Async Operations**: Full async/await pattern testing
- ✅ **Error Handling**: Edge cases and failure scenarios

## 📚 API Reference

### InstantDataConnector

Main connector class for FDW-based data access:

```python
class InstantDataConnector:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        postgres_config: Optional[Dict[str, Any]] = None,
        schema_path: Optional[Path] = None,
        enable_caching: bool = True,
        credential_manager: Optional[SecureCredentialManager] = None
    )
    
    async def setup_fdw_infrastructure(
        self,
        force_refresh: bool = False,
        validate_connections: bool = True
    ) -> bool
    
    async def lazy_load_table(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        optimize_query: bool = True
    ) -> pd.DataFrame
    
    async def execute_query(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        return_dataframe: bool = True,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]
    
    async def health_check(self) -> Dict[str, Any]
    
    async def list_available_tables(
        self, 
        refresh: bool = False
    ) -> Dict[str, Dict[str, Any]]
```

### Query Filters

Supported filter operators:

```python
# Comparison operators
filters = {
    'age': {'gt': 18},           # Greater than
    'income': {'ge': 50000},     # Greater than or equal
    'status': {'eq': 'active'},  # Equal (default)
    'region': {'ne': 'inactive'} # Not equal
}

# Range operations
filters = {
    'date': {'between': ['2024-01-01', '2024-12-31']},
    'amount': {'in': [100, 200, 300]},
    'category': {'not_in': ['spam', 'test']}
}

# Null checks
filters = {
    'optional_field': {'is_null': True},
    'required_field': {'is_not_null': True}
}

# Text operations
filters = {
    'name': {'like': 'John%'},
    'email': {'ilike': '%@GMAIL.COM'}  # Case insensitive
}
```

### PostgreSQL FDW Manager

Direct FDW management for advanced use cases:

```python
from instant_connector import PostgreSQLFDWConnector

async def advanced_fdw():
    fdw = PostgreSQLFDWConnector(
        host='localhost',
        database='hub',
        pool_size=10
    )
    
    await fdw.initialize()
    
    # Install FDW extension
    await fdw.install_extension('postgres_fdw')
    
    # Create foreign server
    await fdw.create_foreign_server(
        'remote_server',
        'postgres_fdw',
        {'host': 'remote.example.com', 'port': '5432', 'dbname': 'prod'}
    )
    
    # Create user mapping
    await fdw.create_user_mapping(
        'remote_server',
        options={'user': 'remote_user', 'password': 'secure_pass'}
    )
    
    # Create foreign table
    await fdw.create_foreign_table(
        'remote_users',
        'remote_server',
        [
            {'name': 'id', 'type': 'integer'},
            {'name': 'email', 'type': 'text'}
        ],
        {'table_name': 'users'}
    )
```

## 🚀 Migration from Legacy Version

### Migration Guide

If you're upgrading from the legacy aggregation-based version:

```python
# OLD (Legacy) - Don't use this anymore
from instant_connector import InstantDataConnector as LegacyConnector
connector = LegacyConnector()
connector.add_database_source('db', params, queries)
raw_data = connector.extract_data()
connector.save_connector('data.pkl')

# NEW (FDW-based) - Use this instead
from instant_connector import InstantDataConnector
connector = InstantDataConnector(config_path='fdw_config.yaml')
await connector.setup_fdw_infrastructure()
df = await connector.lazy_load_table('table_name', filters={'status': 'active'})
```

### Automatic Migration Helper

```python
# Migration utility for existing users
async def migrate_from_legacy(legacy_config_path: str):
    from instant_connector.migration import LegacyMigrator
    
    migrator = LegacyMigrator()
    
    # Convert legacy config to FDW config
    fdw_config = migrator.convert_legacy_config(legacy_config_path)
    
    # Setup new FDW-based connector
    connector = InstantDataConnector(config=fdw_config)
    await connector.setup_fdw_infrastructure()
    
    print("Migration completed! Your data sources are now available via FDW.")
    return connector
```

## 🔍 Troubleshooting

### Common Issues

**1. FDW Extension Not Found**
```bash
# Install FDW extensions in PostgreSQL
# Connect to your PostgreSQL database and run:
CREATE EXTENSION IF NOT EXISTS postgres_fdw;
CREATE EXTENSION IF NOT EXISTS mysql_fdw;
CREATE EXTENSION IF NOT EXISTS multicorn;
```

**2. Connection Issues**
```python
# Test connectivity
async def test_connection():
    connector = InstantDataConnector(postgres_config=your_config)
    health = await connector.health_check()
    print(f"Overall healthy: {health['overall_healthy']}")
    for component, status in health['components'].items():
        print(f"{component}: {status['healthy']} - {status['details']}")
```

**3. Permission Denied for FDW Operations**
```sql
-- Grant necessary permissions to your PostgreSQL user
GRANT USAGE ON FOREIGN DATA WRAPPER postgres_fdw TO connector_user;
GRANT CREATE ON DATABASE your_database TO connector_user;
```

**4. Query Performance Issues**
```python
# Analyze query performance
query_info = connector.query_builder.build_select_query(
    'slow_table',
    filters={'date': {'ge': '2024-01-01'}}
)

# Check if push-down optimization is working
optimized = await connector.query_builder.optimize_query(query_info, connector)
print(f"Push-down optimized: {optimized.get('push_down_eligible', False)}")
print(f"Estimated cost: {optimized.get('cost_estimate', 'Unknown')}")
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('instant_connector')
logger.setLevel(logging.DEBUG)

# Now you'll see detailed logs about FDW operations
```

### Health Monitoring

```bash
# Application health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics

# Database health via Docker
docker-compose exec postgres-fdw pg_isready -U connector_user
```

## 🤝 Integration Examples

### ML Platform Integration

```python
# Complete ;L integration example
import asyncio
from instant_connector import InstantDataConnector

class MLDataLoader:
    def __init__(self, config_path: str):
        self.connector = InstantDataConnector(config_path=config_path)
    
    async def get_ml_features(self, feature_set: str, date_range: tuple):
        """Get ML features for ML training"""
        await self.connector.setup_fdw_infrastructure()
        
        features_df = await self.connector.lazy_load_table(
            f'{feature_set}_features',
            filters={
                'feature_date': {'between': date_range},
                'quality_score': {'ge': 0.8}
            },
            columns=['customer_id', 'feature_vector', 'timestamp']
        )
        
        return features_df
    
    async def get_training_labels(self, label_set: str, date_range: tuple):
        """Get training labels for supervised learning"""
        labels_df = await self.connector.lazy_load_table(
            f'{label_set}_labels',
            filters={'label_date': {'between': date_range}},
            columns=['customer_id', 'target_value', 'confidence']
        )
        
        return labels_df

# Use in ML pipeline
async def ML_training_pipeline():
    loader = MLDataLoader('ml_config.yaml')
    
    # Get training data efficiently
    features = await loader.get_ml_features(
        'customer_behavior', 
        ('2024-01-01', '2024-12-31')
    )
    
    labels = await loader.get_training_labels(
        'churn_prediction',
        ('2024-01-01', '2024-12-31')
    )
    
    # ML can now train models on this data
    print(f"Features: {features.shape}, Labels: {labels.shape}")
    return features, labels
```

### Jupyter Notebook Integration

```python
# Perfect for data science workflows
%load_ext async
import pandas as pd
from instant_connector import InstantDataConnector

# Setup connector
connector = InstantDataConnector(config_path='notebook_config.yaml')
await connector.setup_fdw_infrastructure()

# Interactive data exploration
customers = await connector.lazy_load_table(
    'customers',
    filters={'signup_date': {'ge': '2024-01-01'}},
    limit=10000
)

# Standard pandas operations work perfectly
customers.describe()
customers.groupby('region')['revenue'].sum().plot(kind='bar')
```

## 📈 Monitoring & Observability

### Built-in Metrics

The connector exposes Prometheus metrics:

- `fdw_query_duration_seconds`: Query execution time
- `fdw_rows_returned_total`: Number of rows returned
- `fdw_cache_hits_total`: Cache hit rate
- `fdw_connection_pool_size`: Active connections
- `fdw_push_down_optimized_total`: Push-down optimization usage

### Grafana Dashboards

Pre-built dashboards available in `monitoring/grafana/dashboards/`:

- **FDW Overview**: System health and performance
- **Query Performance**: Execution times and optimization metrics
- **Security Monitoring**: Failed queries and security events
- **ML Integration**: ML pipeline specific metrics

## 🎯 Best Practices

### Query Optimization

```python
# ✅ GOOD: Use specific columns and filters
df = await connector.lazy_load_table(
    'large_table',
    columns=['id', 'name', 'status'],  # Specific columns
    filters={'active': True},          # Push-down filters
    limit=1000                         # Reasonable limit
)

# ❌ AVOID: Select all without filters
df = await connector.lazy_load_table('large_table')  # May return millions of rows
```

### Configuration Management

```python
# ✅ GOOD: Use environment variables for secrets
config = {
    'host': 'db.example.com',
    'password': '${DB_PASSWORD}'  # From environment
}

# ❌ AVOID: Hardcoded credentials
config = {
    'password': 'hardcoded_secret'  # Security risk
}
```

### Error Handling

```python
# ✅ GOOD: Proper error handling
try:
    df = await connector.lazy_load_table('table_name')
except Exception as e:
    logger.error(f"Query failed: {e}")
    # Handle gracefully
finally:
    await connector.close()

# Or use context manager
async with InstantDataConnector(config_path='config.yaml') as connector:
    await connector.setup_fdw_infrastructure()
    df = await connector.lazy_load_table('table_name')
    # Automatic cleanup
```

## 🛣️ Roadmap

### Upcoming Features

- **Multi-region FDW Support**: Cross-region data federation
- **GraphQL API Layer**: Flexible query interface for web applications  
- **Real-time Streaming**: Live data feeds via PostgreSQL logical replication
- **Advanced ML Integration**: Built-in feature stores and model serving
- **Enhanced Security**: Row-level security and data masking
- **Performance Enhancements**: Query result caching and connection multiplexing

### ML Platform Enhancements

- **Feature Store Integration**: Automated feature engineering pipelines
- **Model Versioning**: Track model performance across data versions
- **A/B Testing Support**: Easy experiment data segmentation
- **AutoML Integration**: Automated model selection and tuning

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/instant-data-connector.git
cd instant-data-connector

# Development environment
docker-compose -f docker-compose.dev.yml up -d

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Code quality checks
black src/ tests/
flake8 src/ tests/
mypy src/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.instant-data-connector.com](https://docs.instant-data-connector.com)
- **Issues**: [GitHub Issues](https://github.com/your-org/instant-data-connector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/instant-data-connector/discussions)
- **Security**: [security@instant-data-connector.com](mailto:security@instant-data-connector.com)

---

**Ready to revolutionize your data architecture?** 🚀

Start with our [Quick Start Guide](#-quick-start) or explore the [Docker deployment](#-docker-deployment) for a complete development environment.

The future of data connectivity is here - no more ETL, no more data movement, just intelligent queries across all your sources! 🎉