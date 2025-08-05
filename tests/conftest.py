"""
Comprehensive test configuration and fixtures for FDW-based data connector testing.

This module provides:
- PostgreSQL testcontainer with FDW extensions
- MySQL testcontainer for multi-database testing
- Mock REST API server setup
- Test data fixtures and factories
- Database initialization scripts
- Common test utilities
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid

import pandas as pd
import pytest
import pytest_asyncio
from faker import Faker
from fastapi import FastAPI
from testcontainers.postgres import PostgresContainer
from testcontainers.mysql import MySqlContainer
import uvicorn
import httpx
import factory
from factory import LazyFunction
import asyncpg
import pymysql

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import (
    InstantDataConnector,
    PostgreSQLFDWConnector,
    FDWManager,
    VirtualTableManager,
    ConfigParser
)

# Initialize faker for test data generation
fake = Faker()

# Test configuration
TEST_DB_NAME = "test_fdw_db"
TEST_MYSQL_DB = "test_mysql_db"
TEST_POSTGRES_USER = "test_user"
TEST_POSTGRES_PASSWORD = "test_password"
TEST_MYSQL_USER = "test_user"
TEST_MYSQL_PASSWORD = "test_password"

# Mock API server configuration
MOCK_API_PORT = 8999
MOCK_API_HOST = "localhost"


class TestDataFactory(factory.Factory):
    """Factory for generating test data."""
    
    class Meta:
        model = dict
    
    id = factory.Sequence(lambda n: n + 1)
    name = factory.LazyAttribute(lambda obj: fake.name())
    email = factory.LazyAttribute(lambda obj: fake.email())
    age = factory.LazyFunction(lambda: fake.random_int(min=18, max=80))
    created_at = factory.LazyFunction(fake.date_time_this_year)
    is_active = factory.LazyFunction(lambda: fake.boolean(chance_of_getting_true=80))


class UserFactory(factory.Factory):
    """Factory for generating user test data."""
    
    class Meta:
        model = dict
    
    user_id = factory.Sequence(lambda n: n + 1)
    username = factory.Sequence(lambda n: f"testuser_{n}_{fake.user_name()}")
    first_name = factory.LazyAttribute(lambda obj: fake.first_name())
    last_name = factory.LazyAttribute(lambda obj: fake.last_name())
    email = factory.Sequence(lambda n: f"testuser_{n}_{fake.random_int(1000, 9999)}@example.com")
    registration_date = factory.LazyFunction(fake.date_this_year)
    last_login = factory.LazyFunction(fake.date_time_this_month)
    is_active = factory.LazyFunction(lambda: fake.boolean(chance_of_getting_true=90))


class OrderFactory(factory.Factory):
    """Factory for generating order test data."""
    
    class Meta:
        model = dict
    
    order_id = factory.Sequence(lambda n: n + 1000)
    customer_id = factory.LazyFunction(lambda: fake.random_int(min=1, max=100))
    order_number = factory.Sequence(lambda n: f"ORD-TEST-{n:08d}")
    status = factory.LazyFunction(
        lambda: fake.random_element(elements=['pending', 'processing', 'shipped', 'delivered', 'cancelled'])
    )
    total_amount = factory.LazyFunction(lambda: round(fake.random.uniform(10.0, 1000.0), 2))
    order_date = factory.LazyFunction(fake.date_time_this_year)
    items = factory.LazyFunction(lambda: fake.random_int(min=1, max=5))


class ProductFactory(factory.Factory):
    """Factory for generating product test data."""
    
    class Meta:
        model = dict
    
    product_id = factory.Sequence(lambda n: n + 1)
    sku = factory.Sequence(lambda n: f"SKU-TEST-{n:06d}")
    name = factory.LazyAttribute(lambda obj: fake.catch_phrase())
    description = factory.LazyAttribute(lambda obj: fake.text(max_nb_chars=200))
    category = factory.LazyFunction(
        lambda: fake.random_element(elements=['Electronics', 'Clothing', 'Books', 'Home', 'Sports'])
    )
    price = factory.LazyFunction(lambda: round(fake.random.uniform(5.0, 500.0), 2))
    cost = factory.LazyFunction(lambda: round(fake.random.uniform(2.0, 250.0), 2))
    weight = factory.LazyFunction(lambda: round(fake.random.uniform(0.1, 10.0), 3))
    stock_quantity = factory.LazyFunction(lambda: fake.random_int(min=0, max=1000))
    is_active = factory.LazyFunction(lambda: fake.boolean(chance_of_getting_true=85))


# Mock API Server
class MockAPIServer:
    """Mock REST API server for testing API integrations."""
    
    def __init__(self, host: str = MOCK_API_HOST, port: int = MOCK_API_PORT):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.server = None
        self.thread = None
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up mock API routes."""
        
        @self.app.get("/users")
        async def get_users(limit: int = 100, offset: int = 0):
            """Mock users endpoint."""
            users = [UserFactory() for _ in range(min(limit, 50))]
            return {
                "data": users,
                "total": 1000,
                "limit": limit,
                "offset": offset
            }
        
        @self.app.get("/users/{user_id}")
        async def get_user(user_id: int):
            """Mock single user endpoint."""
            user = UserFactory(user_id=user_id)
            return {"data": user}
        
        @self.app.get("/orders")
        async def get_orders(customer_id: Optional[int] = None, status: Optional[str] = None):
            """Mock orders endpoint."""
            orders = []
            for _ in range(20):
                order = OrderFactory()
                if customer_id:
                    order['customer_id'] = customer_id
                if status:
                    order['status'] = status
                orders.append(order)
            
            return {"data": orders}
        
        @self.app.get("/products")
        async def get_products(category: Optional[str] = None):
            """Mock products endpoint."""
            products = []
            for _ in range(30):
                product = ProductFactory()
                if category:
                    product['category'] = category
                products.append(product)
            
            return {"data": products}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": fake.iso8601()}
    
    def start(self):
        """Start the mock API server in a separate thread."""
        def run_server():
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="error"  # Suppress uvicorn logs in tests
            )
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Verify server is running
        try:
            import requests
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=5)
            if response.status_code != 200:
                raise Exception("Mock API server failed to start")
        except Exception as e:
            raise Exception(f"Mock API server not accessible: {e}")
    
    def stop(self):
        """Stop the mock API server."""
        # Note: uvicorn doesn't provide a clean shutdown method in this setup
        # In a production test environment, you'd use a more sophisticated approach
        pass


# Pytest fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def mock_api_server():
    """Start mock API server for testing."""
    server = MockAPIServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQL testcontainer with FDW extensions."""
    with PostgresContainer(
        image="postgres:15-alpine",
        username=TEST_POSTGRES_USER,
        password=TEST_POSTGRES_PASSWORD,
        dbname=TEST_DB_NAME,
        port=5432
    ) as postgres:
        # Install FDW extensions (simulate with SQL commands)
        connection_url = postgres.get_connection_url()
        
        # Wait for container to be ready
        time.sleep(5)
        
        yield {
            "host": postgres.get_container_host_ip(),
            "port": postgres.get_exposed_port(5432),
            "database": TEST_DB_NAME,
            "username": TEST_POSTGRES_USER,
            "password": TEST_POSTGRES_PASSWORD,
            "connection_url": connection_url
        }


@pytest.fixture(scope="session")
def mysql_container():
    """MySQL testcontainer for FDW testing."""
    with MySqlContainer(
        image="mysql:8.0",
        username=TEST_MYSQL_USER,
        password=TEST_MYSQL_PASSWORD,
        dbname=TEST_MYSQL_DB
    ) as mysql:
        # Wait for container to be ready
        time.sleep(10)
        
        yield {
            "host": mysql.get_container_host_ip(),
            "port": mysql.get_exposed_port(3306),
            "database": TEST_MYSQL_DB,
            "username": TEST_MYSQL_USER,
            "password": TEST_MYSQL_PASSWORD
        }


@pytest_asyncio.fixture
async def postgres_connection(postgres_container):
    """Async PostgreSQL connection fixture."""
    config = postgres_container
    conn = await asyncpg.connect(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        user=config["username"],
        password=config["password"]
    )
    
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def mysql_connection(mysql_container):
    """MySQL connection fixture."""
    config = mysql_container
    conn = pymysql.connect(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        user=config["username"],
        password=config["password"],
        autocommit=True
    )
    
    try:
        yield conn
    finally:
        conn.close()


@pytest_asyncio.fixture
async def initialized_postgres(postgres_connection):
    """PostgreSQL database initialized with test data and FDW setup."""
    conn = postgres_connection
    
    # Create extensions (simulate FDW extensions)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS postgres_fdw")
    
    # Create test tables
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            email VARCHAR(255) UNIQUE NOT NULL,
            registration_date DATE,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id SERIAL PRIMARY KEY,
            customer_id INTEGER,
            order_number VARCHAR(50) UNIQUE NOT NULL,
            status VARCHAR(20) NOT NULL,
            total_amount DECIMAL(12,2) NOT NULL,
            order_date TIMESTAMP NOT NULL,
            items INTEGER DEFAULT 1
        )
    """)
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id SERIAL PRIMARY KEY,
            sku VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            category VARCHAR(100),
            price DECIMAL(10,2) NOT NULL,
            cost DECIMAL(10,2),
            weight DECIMAL(8,3),
            stock_quantity INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)
    
    # Insert test data with conflict handling
    users_data = [UserFactory() for _ in range(100)]
    for user in users_data:
        await conn.execute("""
            INSERT INTO users (username, first_name, last_name, email, registration_date, last_login, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (username) DO NOTHING
        """, user['username'], user['first_name'], user['last_name'], user['email'],
            user['registration_date'], user['last_login'], user['is_active'])
    
    orders_data = [OrderFactory() for _ in range(200)]
    for order in orders_data:
        await conn.execute("""
            INSERT INTO orders (customer_id, order_number, status, total_amount, order_date, items)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (order_number) DO NOTHING
        """, order['customer_id'], order['order_number'], order['status'],
            order['total_amount'], order['order_date'], order['items'])
    
    products_data = [ProductFactory() for _ in range(50)]
    for product in products_data:
        await conn.execute("""
            INSERT INTO products (sku, name, description, category, price, cost, weight, stock_quantity, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (sku) DO NOTHING
        """, product['sku'], product['name'], product['description'], product['category'],
            product['price'], product['cost'], product['weight'], product['stock_quantity'], product['is_active'])
    
    yield conn


@pytest.fixture
def initialized_mysql(mysql_connection):
    """MySQL database initialized with test data."""
    conn = mysql_connection
    cursor = conn.cursor()
    
    # Create test tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS legacy_customers (
            customer_id INT AUTO_INCREMENT PRIMARY KEY,
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            email VARCHAR(100),
            phone VARCHAR(20),
            address_line1 VARCHAR(100),
            city VARCHAR(50),
            state VARCHAR(30),
            zip_code VARCHAR(10),
            date_created DATE,
            last_purchase_date DATE,
            total_purchases DECIMAL(12,2),
            is_active BOOLEAN DEFAULT TRUE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            product_id INT,
            warehouse_id INT,
            quantity_on_hand INT NOT NULL,
            quantity_reserved INT DEFAULT 0,
            reorder_point INT DEFAULT 0,
            max_stock_level INT,
            last_count_date DATE,
            PRIMARY KEY (product_id, warehouse_id)
        )
    """)
    
    # Insert test data
    customers_data = [UserFactory() for _ in range(50)]
    for customer in customers_data:
        cursor.execute("""
            INSERT INTO legacy_customers 
            (first_name, last_name, email, phone, address_line1, city, state, zip_code, date_created, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            customer['first_name'], customer['last_name'], customer['email'],
            fake.phone_number(), fake.street_address(), fake.city(), fake.state(),
            fake.zipcode(), customer['registration_date'], customer['is_active']
        ))
    
    # Insert inventory data
    for product_id in range(1, 51):
        for warehouse_id in range(1, 4):
            cursor.execute("""
                INSERT INTO inventory (product_id, warehouse_id, quantity_on_hand, quantity_reserved, reorder_point)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                product_id, warehouse_id,
                fake.random_int(min=0, max=1000),
                fake.random_int(min=0, max=50),
                fake.random_int(min=10, max=100)
            ))
    
    yield conn


@pytest.fixture
def test_config_dir():
    """Create temporary directory with test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        
        # Create test configuration
        test_config = {
            "version": "1.0",
            "metadata": {
                "name": "Test Configuration",
                "description": "Test configuration for FDW connector",
                "author": "Test Suite"
            },
            "global_settings": {
                "connection_timeout": 30,
                "query_timeout": 300,
                "enable_push_down": True
            },
            "sources": {
                "test_postgres": {
                    "type": "postgres_fdw",
                    "description": "Test PostgreSQL source",
                    "enabled": True,
                    "server_options": {
                        "host": "${TEST_POSTGRES_HOST}",
                        "port": "${TEST_POSTGRES_PORT}",
                        "dbname": "test_fdw_db"
                    },
                    "user_mapping": {
                        "options": {
                            "user": "${TEST_POSTGRES_USER}",
                            "password": "${TEST_POSTGRES_PASSWORD}"
                        }
                    },
                    "tables": [
                        {
                            "name": "users",
                            "description": "User data",
                            "options": {
                                "table_name": "users",
                                "schema_name": "public"
                            }
                        }
                    ]
                }
            }
        }
        
        config_file = config_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(test_config, f)
        
        yield config_dir


@pytest.fixture
def test_data_files():
    """Create test data files (CSV, JSON, Excel)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        # Generate test data
        test_data = [TestDataFactory() for _ in range(100)]
        df = pd.DataFrame(test_data)
        
        # Save as CSV
        csv_file = data_dir / "test_data.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as JSON
        json_file = data_dir / "test_data.json"
        df.to_json(json_file, orient='records', date_format='iso')
        
        # Save as Excel
        excel_file = data_dir / "test_data.xlsx"
        df.to_excel(excel_file, index=False)
        
        yield {
            "csv": csv_file,
            "json": json_file,
            "excel": excel_file,
            "dataframe": df
        }


@pytest_asyncio.fixture
async def fdw_manager(postgres_container):
    """FDW Manager fixture."""
    config = postgres_container
    
    # Set environment variables for testing
    os.environ.update({
        "TEST_POSTGRES_HOST": config["host"],
        "TEST_POSTGRES_PORT": str(config["port"]),
        "TEST_POSTGRES_USER": config["username"],
        "TEST_POSTGRES_PASSWORD": config["password"]
    })
    
    postgres_config = {
        "host": config["host"],
        "port": config["port"],
        "database": config["database"],
        "username": config["username"],
        "password": config["password"]
    }
    
    fdw_connector = PostgreSQLFDWConnector(**postgres_config)
    await fdw_connector.initialize()
    
    manager = FDWManager(fdw_connector)
    
    try:
        yield manager
    finally:
        await fdw_connector.close()


@pytest_asyncio.fixture
async def instant_connector(postgres_container, test_config_dir):
    """InstantDataConnector fixture with test configuration."""
    config = postgres_container
    
    # Set environment variables
    os.environ.update({
        "TEST_POSTGRES_HOST": config["host"],
        "TEST_POSTGRES_PORT": str(config["port"]),
        "TEST_POSTGRES_USER": config["username"],
        "TEST_POSTGRES_PASSWORD": config["password"]
    })
    
    postgres_config = {
        "host": config["host"],
        "port": config["port"],
        "database": config["database"],
        "username": config["username"],
        "password": config["password"]
    }
    
    config_file = test_config_dir / "test_config.yaml"
    
    connector = InstantDataConnector(
        config_path=config_file,
        postgres_config=postgres_config,
        enable_caching=True
    )
    
    try:
        yield connector
    finally:
        await connector.close()


@pytest.fixture
def benchmark_data():
    """Generate large dataset for performance benchmarking."""
    return pd.DataFrame([TestDataFactory() for _ in range(10000)])


@pytest.fixture
def mock_credentials():
    """Mock credential manager for testing."""
    mock = MagicMock()
    mock.get_credential.return_value = "mock_password"
    mock.store_credential.return_value = True
    return mock


# Legacy fixtures (keeping for backward compatibility)
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    import numpy as np
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
    import shutil
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


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def create_test_table_config(table_name: str, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a test table configuration."""
        return {
            "name": table_name,
            "description": f"Test table: {table_name}",
            "options": {
                "table_name": table_name,
                "schema_name": "public"
            },
            "columns": columns
        }
    
    @staticmethod
    def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = False):
        """Assert two DataFrames are equal with better error messages."""
        try:
            pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
        except AssertionError as e:
            print(f"DataFrame assertion failed:\nDF1:\n{df1}\nDF2:\n{df2}")
            raise e
    
    @staticmethod
    async def wait_for_async_condition(condition_func, timeout: float = 30.0, interval: float = 0.1):
        """Wait for an async condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        return False
    
    @staticmethod
    def generate_sql_test_cases() -> List[Dict[str, Any]]:
        """Generate SQL test cases for query testing."""
        return [
            {
                "name": "simple_select",
                "sql": "SELECT * FROM users LIMIT 10",
                "expected_columns": ["user_id", "username", "email"]
            },
            {
                "name": "filtered_select",
                "sql": "SELECT * FROM users WHERE is_active = true",
                "expected_min_rows": 1
            },
            {
                "name": "aggregation",
                "sql": "SELECT COUNT(*) as user_count FROM users",
                "expected_columns": ["user_count"]
            },
            {
                "name": "join_query",
                "sql": """
                    SELECT u.username, COUNT(o.order_id) as order_count
                    FROM users u
                    LEFT JOIN orders o ON u.user_id = o.customer_id
                    GROUP BY u.user_id, u.username
                    LIMIT 10
                """,
                "expected_columns": ["username", "order_count"]
            }
        ]


@pytest.fixture
def test_utils():
    """Test utilities fixture."""
    return TestUtils


# Pytest configuration
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
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "benchmark" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)


# Cleanup function
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Cleanup environment variables after each test."""
    original_env = os.environ.copy()
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


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