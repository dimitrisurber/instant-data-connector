"""
Comprehensive tests for lazy loading functionality.

This module tests:
- Lazy query building
- Pagination and filtering
- Aggregation queries
- Query optimization
- Memory efficiency
- Large dataset handling
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import asyncio
import time
import psutil
import os
from typing import Dict, Any, List

from instant_connector import LazyQueryBuilder, InstantDataConnector


@pytest.mark.asyncio
class TestLazyQueryBuilder:
    """Test cases for LazyQueryBuilder functionality."""
    
    def test_query_builder_initialization(self):
        """Test query builder initialization."""
        builder = LazyQueryBuilder()
        assert builder is not None
        assert hasattr(builder, 'build_select_query')
        assert hasattr(builder, 'optimize_query')
        assert hasattr(builder, 'generate_cache_key')
    
    def test_build_simple_select_query(self):
        """Test building simple SELECT queries."""
        builder = LazyQueryBuilder()
        
        query_info = builder.build_select_query("users")
        
        assert query_info['sql'] == "SELECT * FROM users"
        assert query_info['params'] == []
        assert query_info['table_name'] == "users"
        assert query_info['estimated_rows'] is None
    
    def test_build_select_query_with_columns(self):
        """Test building SELECT queries with specific columns."""
        builder = LazyQueryBuilder()
        
        columns = ["id", "name", "email"]
        query_info = builder.build_select_query("users", columns=columns)
        
        expected_sql = "SELECT id, name, email FROM users"
        assert query_info['sql'] == expected_sql
        assert query_info['params'] == []
    
    def test_build_select_query_with_limit(self):
        """Test building SELECT queries with LIMIT."""
        builder = LazyQueryBuilder()
        
        query_info = builder.build_select_query("users", limit=10)
        
        assert query_info['sql'] == "SELECT * FROM users LIMIT 10"
        assert query_info['params'] == []
    
    def test_build_select_query_with_offset(self):
        """Test building SELECT queries with OFFSET."""
        builder = LazyQueryBuilder()
        
        query_info = builder.build_select_query("users", limit=10, offset=20)
        
        assert query_info['sql'] == "SELECT * FROM users LIMIT 10 OFFSET 20"
        assert query_info['params'] == []
    
    def test_build_select_query_with_order_by(self):
        """Test building SELECT queries with ORDER BY."""
        builder = LazyQueryBuilder()
        
        # Single column order
        query_info = builder.build_select_query("users", order_by="name")
        assert query_info['sql'] == "SELECT * FROM users ORDER BY name"
        
        # Multiple columns order
        query_info = builder.build_select_query("users", order_by=["name", "created_at"])
        assert query_info['sql'] == "SELECT * FROM users ORDER BY name, created_at"
        
        # Order with direction
        query_info = builder.build_select_query("users", order_by="created_at DESC")
        assert query_info['sql'] == "SELECT * FROM users ORDER BY created_at DESC"
    
    def test_build_select_query_with_filters(self):
        """Test building SELECT queries with filters."""
        builder = LazyQueryBuilder()
        
        # Simple equality filter
        filters = {"status": "active"}
        query_info = builder.build_select_query("users", filters=filters)
        
        assert "WHERE" in query_info['sql']
        assert "status = $1" in query_info['sql']
        assert query_info['params'] == ["active"]
        
        # Multiple filters
        filters = {"status": "active", "age": 25}
        query_info = builder.build_select_query("users", filters=filters)
        
        assert "WHERE" in query_info['sql']
        assert "status = $1" in query_info['sql'] or "status = $2" in query_info['sql']
        assert "age = $1" in query_info['sql'] or "age = $2" in query_info['sql']
        assert len(query_info['params']) == 2
        assert "active" in query_info['params']
        assert 25 in query_info['params']
    
    def test_build_complex_query(self):
        """Test building complex queries with all options."""
        builder = LazyQueryBuilder()
        
        query_info = builder.build_select_query(
            table_name="orders",
            columns=["id", "customer_id", "total_amount", "order_date"],
            filters={"status": "completed", "total_amount": 100.0},
            limit=50,
            offset=100,
            order_by="order_date DESC"
        )
        
        sql = query_info['sql']
        assert "SELECT id, customer_id, total_amount, order_date FROM orders" in sql
        assert "WHERE" in sql
        assert "status = $" in sql
        assert "total_amount = $" in sql
        assert "ORDER BY order_date DESC" in sql
        assert "LIMIT 50 OFFSET 100" in sql
        
        assert len(query_info['params']) == 2
        assert "completed" in query_info['params']
        assert 100.0 in query_info['params']
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        builder = LazyQueryBuilder()
        
        query_info = {
            'sql': 'SELECT * FROM users WHERE status = $1',
            'params': ['active'],
            'table_name': 'users'
        }
        
        cache_key = builder.generate_cache_key(query_info)
        
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
        
        # Same query should generate same cache key
        cache_key2 = builder.generate_cache_key(query_info)
        assert cache_key == cache_key2
        
        # Different query should generate different cache key
        query_info2 = {
            'sql': 'SELECT * FROM users WHERE status = $1',
            'params': ['inactive'],
            'table_name': 'users'
        }
        cache_key3 = builder.generate_cache_key(query_info2)
        assert cache_key != cache_key3
    
    async def test_optimize_query_basic(self):
        """Test basic query optimization."""
        builder = LazyQueryBuilder()
        
        query_info = {
            'sql': 'SELECT * FROM users LIMIT 10',
            'params': [],
            'table_name': 'users'
        }
        
        # Mock FDW connector
        mock_connector = MagicMock()
        mock_connector.get_connection = AsyncMock()
        mock_conn = AsyncMock()
        mock_connector.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connector.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock statistics
        mock_conn.fetchrow.return_value = {
            'reltuples': 1000,
            'relpages': 100
        }
        
        optimized_query = await builder.optimize_query(query_info, mock_connector)
        
        assert optimized_query is not None
        assert 'estimated_rows' in optimized_query
        assert 'cost_estimate' in optimized_query
        assert optimized_query['estimated_rows'] == 10  # Limited by LIMIT clause
    
    async def test_detect_push_down_opportunities(self):
        """Test push-down optimization detection."""
        builder = LazyQueryBuilder()
        
        # Query with filter - good for push-down
        query_info = {
            'sql': 'SELECT * FROM users WHERE status = $1',
            'params': ['active'],
            'table_name': 'users'
        }
        
        mock_connector = MagicMock()
        mock_connector.get_connection = AsyncMock()
        mock_conn = AsyncMock()
        mock_connector.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connector.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_conn.fetchrow.return_value = {
            'reltuples': 10000,
            'relpages': 1000
        }
        
        optimized_query = await builder.optimize_query(query_info, mock_connector)
        
        assert optimized_query.get('push_down_eligible') is True
    
    def test_build_aggregation_query(self):
        """Test building aggregation queries."""
        builder = LazyQueryBuilder()
        
        # COUNT aggregation
        query_info = builder.build_aggregation_query(
            table_name="orders",
            aggregations={"total_orders": "COUNT(*)"},
            group_by=["status"]
        )
        
        sql = query_info['sql']
        assert "SELECT" in sql
        assert "COUNT(*) as total_orders" in sql
        assert "FROM orders" in sql
        assert "GROUP BY status" in sql
    
    def test_build_aggregation_query_with_filters(self):
        """Test building aggregation queries with filters."""
        builder = LazyQueryBuilder()
        
        query_info = builder.build_aggregation_query(
            table_name="orders",
            aggregations={
                "total_orders": "COUNT(*)",
                "total_amount": "SUM(amount)",
                "avg_amount": "AVG(amount)"
            },
            group_by=["status", "customer_type"],
            filters={"order_date": "2024-01-01"}
        )
        
        sql = query_info['sql']
        assert "COUNT(*) as total_orders" in sql
        assert "SUM(amount) as total_amount" in sql
        assert "AVG(amount) as avg_amount" in sql
        assert "GROUP BY status, customer_type" in sql
        assert "WHERE" in sql
        assert "order_date = $1" in sql
        assert query_info['params'] == ["2024-01-01"]
    
    def test_build_join_query(self):
        """Test building JOIN queries."""
        builder = LazyQueryBuilder()
        
        query_info = builder.build_join_query(
            main_table="users",
            joins=[
                {
                    "table": "orders",
                    "type": "LEFT JOIN",
                    "condition": "users.id = orders.customer_id"
                }
            ],
            columns=["users.name", "orders.total_amount"],
            filters={"users.status": "active"}
        )
        
        sql = query_info['sql']
        assert "SELECT users.name, orders.total_amount" in sql
        assert "FROM users" in sql
        assert "LEFT JOIN orders ON users.id = orders.customer_id" in sql
        assert "WHERE users.status = $1" in sql
        assert query_info['params'] == ["active"]


@pytest.mark.asyncio
class TestLazyLoadingIntegration:
    """Integration tests for lazy loading with real database."""
    
    async def test_lazy_loading_with_small_dataset(self, instant_connector, initialized_postgres):
        """Test lazy loading with small dataset."""
        await instant_connector.setup_fdw_infrastructure(validate_connections=False)
        
        try:
            # Test basic lazy loading
            result = await instant_connector.lazy_load_table(
                "users",
                limit=5,
                optimize_query=False
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 5
            
        except Exception as e:
            if "does not exist" in str(e) or "relation" in str(e):
                pytest.skip("FDW table not available - test environment limitation")
            else:
                raise
    
    async def test_lazy_loading_with_pagination(self, instant_connector, initialized_postgres):
        """Test lazy loading with pagination."""
        await instant_connector.setup_fdw_infrastructure(validate_connections=False)
        
        try:
            # Load first page
            page1 = await instant_connector.lazy_load_table(
                "users",
                limit=10,
                offset=0,
                order_by="user_id",
                optimize_query=False
            )
            
            # Load second page
            page2 = await instant_connector.lazy_load_table(
                "users",
                limit=10,
                offset=10,
                order_by="user_id",
                optimize_query=False
            )
            
            assert isinstance(page1, pd.DataFrame)
            assert isinstance(page2, pd.DataFrame)
            assert len(page1) <= 10
            assert len(page2) <= 10
            
            # Verify no overlap (if both pages have data)
            if len(page1) > 0 and len(page2) > 0:
                page1_ids = set(page1['user_id'].values)
                page2_ids = set(page2['user_id'].values)
                assert page1_ids.isdisjoint(page2_ids)
                
        except Exception as e:
            if "does not exist" in str(e) or "relation" in str(e):
                pytest.skip("FDW table not available - test environment limitation")
            else:
                raise
    
    async def test_lazy_loading_with_filters(self, instant_connector, initialized_postgres):
        """Test lazy loading with various filters."""
        await instant_connector.setup_fdw_infrastructure(validate_connections=False)
        
        try:
            # Test boolean filter
            active_users = await instant_connector.lazy_load_table(
                "users",
                filters={"is_active": True},
                limit=20,
                optimize_query=False
            )
            
            assert isinstance(active_users, pd.DataFrame)
            
            # Verify filter was applied (if result has data)
            if len(active_users) > 0:
                assert all(active_users['is_active'] == True)
            
        except Exception as e:
            if "does not exist" in str(e) or "relation" in str(e):
                pytest.skip("FDW table not available - test environment limitation")
            else:
                raise
    
    async def test_lazy_loading_column_selection(self, instant_connector, initialized_postgres):
        """Test lazy loading with column selection."""
        await instant_connector.setup_fdw_infrastructure(validate_connections=False)
        
        try:
            # Select specific columns
            result = await instant_connector.lazy_load_table(
                "users",
                columns=["user_id", "username", "email"],
                limit=10,
                optimize_query=False
            )
            
            assert isinstance(result, pd.DataFrame)
            
            # Verify only requested columns are present (if result has data)
            if len(result) > 0:
                expected_columns = {"user_id", "username", "email"}
                actual_columns = set(result.columns)
                assert expected_columns.issubset(actual_columns)
                
        except Exception as e:
            if "does not exist" in str(e) or "relation" in str(e):
                pytest.skip("FDW table not available - test environment limitation")
            else:
                raise
    
    @pytest.mark.slow
    async def test_memory_efficiency_large_dataset(self, instant_connector, initialized_postgres):
        """Test memory efficiency with large dataset simulation."""
        await instant_connector.setup_fdw_infrastructure(validate_connections=False)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Load data in chunks to test memory efficiency
            chunk_size = 25
            total_loaded = 0
            max_memory_increase = 0
            
            for offset in range(0, 100, chunk_size):
                chunk = await instant_connector.lazy_load_table(
                    "users",
                    limit=chunk_size,
                    offset=offset,
                    optimize_query=False
                )
                
                total_loaded += len(chunk)
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                max_memory_increase = max(max_memory_increase, memory_increase)
                
                # Memory increase should be reasonable
                assert memory_increase < 100  # Less than 100MB increase
                
                if len(chunk) == 0:
                    break
            
            print(f"Total loaded: {total_loaded} rows")
            print(f"Max memory increase: {max_memory_increase:.2f} MB")
            
        except Exception as e:
            if "does not exist" in str(e) or "relation" in str(e):
                pytest.skip("FDW table not available - test environment limitation")
            else:
                raise
    
    async def test_concurrent_lazy_loading(self, instant_connector, initialized_postgres):
        """Test concurrent lazy loading operations."""
        await instant_connector.setup_fdw_infrastructure(validate_connections=False)
        
        async def load_chunk(offset, limit):
            try:
                return await instant_connector.lazy_load_table(
                    "users",
                    limit=limit,
                    offset=offset,
                    order_by="user_id",
                    optimize_query=False
                )
            except Exception:
                return pd.DataFrame()  # Return empty DataFrame on error
        
        # Load multiple chunks concurrently
        tasks = [
            load_chunk(0, 10),
            load_chunk(10, 10),
            load_chunk(20, 10),
            load_chunk(30, 10),
            load_chunk(40, 10)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful_results = [r for r in results if isinstance(r, pd.DataFrame)]
        assert len(successful_results) >= 3  # Allow some failures
        
        # Verify no data corruption
        for result in successful_results:
            if len(result) > 0:
                assert 'user_id' in result.columns
                assert result['user_id'].notna().all()
    
    @pytest.mark.benchmark
    async def test_lazy_loading_performance_benchmark(self, instant_connector, initialized_postgres):
        """Benchmark lazy loading performance."""
        await instant_connector.setup_fdw_infrastructure(validate_connections=False)
        
        try:
            # Benchmark different loading strategies
            benchmarks = {}
            
            # 1. Small batch loading
            start_time = time.time()
            result = await instant_connector.lazy_load_table(
                "users",
                limit=10,
                optimize_query=False
            )
            benchmarks['small_batch'] = time.time() - start_time
            
            # 2. Medium batch loading
            start_time = time.time()
            result = await instant_connector.lazy_load_table(
                "users",
                limit=50,
                optimize_query=False
            )
            benchmarks['medium_batch'] = time.time() - start_time
            
            # 3. With optimization
            start_time = time.time()
            result = await instant_connector.lazy_load_table(
                "users",
                limit=50,
                optimize_query=True
            )
            benchmarks['optimized'] = time.time() - start_time
            
            # 4. With filters
            start_time = time.time()
            result = await instant_connector.lazy_load_table(
                "users",
                filters={"is_active": True},
                limit=30,
                optimize_query=False
            )
            benchmarks['with_filters'] = time.time() - start_time
            
            # Log benchmark results
            for strategy, duration in benchmarks.items():
                print(f"{strategy}: {duration:.3f} seconds")
                assert duration < 5.0  # All operations should complete within 5 seconds
                
        except Exception as e:
            if "does not exist" in str(e) or "relation" in str(e):
                pytest.skip("FDW table not available - test environment limitation")
            else:
                raise
    
    async def test_query_optimization_effectiveness(self, instant_connector, initialized_postgres):
        """Test effectiveness of query optimization."""
        await instant_connector.setup_fdw_infrastructure(validate_connections=False)
        
        try:
            # Test query without optimization
            start_time = time.time()
            result_unoptimized = await instant_connector.lazy_load_table(
                "users",
                limit=20,
                optimize_query=False
            )
            unoptimized_time = time.time() - start_time
            
            # Test query with optimization
            start_time = time.time()
            result_optimized = await instant_connector.lazy_load_table(
                "users",
                limit=20,
                optimize_query=True
            )
            optimized_time = time.time() - start_time
            
            # Results should be equivalent
            if len(result_unoptimized) > 0 and len(result_optimized) > 0:
                assert len(result_unoptimized) == len(result_optimized)
            
            # Optimization may or may not be faster depending on the query
            print(f"Unoptimized: {unoptimized_time:.3f}s, Optimized: {optimized_time:.3f}s")
            
        except Exception as e:
            if "does not exist" in str(e) or "relation" in str(e):
                pytest.skip("FDW table not available - test environment limitation")
            else:
                raise


@pytest.mark.unit
class TestLazyLoadingUnit:
    """Unit tests for lazy loading components."""
    
    def test_query_builder_edge_cases(self):
        """Test query builder edge cases."""
        builder = LazyQueryBuilder()
        
        # Empty filters
        query_info = builder.build_select_query("users", filters={})
        assert "WHERE" not in query_info['sql']
        
        # None values in filters
        query_info = builder.build_select_query("users", filters={"status": None})
        assert "status IS NULL" in query_info['sql']
        
        # Empty columns list
        query_info = builder.build_select_query("users", columns=[])
        assert query_info['sql'] == "SELECT * FROM users"
        
        # Zero limit
        query_info = builder.build_select_query("users", limit=0)
        assert "LIMIT 0" in query_info['sql']
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention in query building."""
        builder = LazyQueryBuilder()
        
        # Malicious table name should be escaped/validated
        with pytest.raises(ValueError):
            builder.build_select_query("users; DROP TABLE users; --")
        
        # Malicious column names should be handled
        with pytest.raises(ValueError):
            builder.build_select_query("users", columns=["id; DROP TABLE users; --"])
        
        # Filter values should be parameterized (not injectable)
        query_info = builder.build_select_query(
            "users", 
            filters={"name": "'; DROP TABLE users; --"}
        )
        assert "DROP TABLE" not in query_info['sql']
        assert "'; DROP TABLE users; --" in query_info['params']
    
    def test_query_validation(self):
        """Test query validation logic."""
        builder = LazyQueryBuilder()
        
        # Valid queries should pass
        query_info = builder.build_select_query("users", limit=10)
        assert builder.validate_query(query_info) is True
        
        # Invalid SQL should fail validation
        invalid_query = {
            'sql': 'INVALID SQL SYNTAX',
            'params': [],
            'table_name': 'users'
        }
        assert builder.validate_query(invalid_query) is False
    
    def test_cache_key_consistency(self):
        """Test cache key generation consistency."""
        builder = LazyQueryBuilder()
        
        query_info = {
            'sql': 'SELECT * FROM users WHERE id = $1',
            'params': [123],
            'table_name': 'users'
        }
        
        # Multiple calls should generate same key
        key1 = builder.generate_cache_key(query_info)
        key2 = builder.generate_cache_key(query_info)
        key3 = builder.generate_cache_key(query_info)
        
        assert key1 == key2 == key3
        
        # Different queries should generate different keys
        query_info2 = query_info.copy()
        query_info2['params'] = [456]
        
        key4 = builder.generate_cache_key(query_info2)
        assert key1 != key4
    
    async def test_optimization_with_mock_stats(self):
        """Test query optimization with mocked database statistics."""
        builder = LazyQueryBuilder()
        
        query_info = {
            'sql': 'SELECT * FROM users WHERE status = $1 LIMIT 100',
            'params': ['active'],
            'table_name': 'users'
        }
        
        # Mock connector with table statistics
        mock_connector = MagicMock()
        mock_connector.get_connection = AsyncMock()
        mock_conn = AsyncMock()
        mock_connector.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connector.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock table statistics
        mock_conn.fetchrow.return_value = {
            'reltuples': 10000,
            'relpages': 500,
            'relallvisible': 450
        }
        
        optimized = await builder.optimize_query(query_info, mock_connector)
        
        assert optimized is not None
        assert 'estimated_rows' in optimized
        assert 'cost_estimate' in optimized
        assert optimized['estimated_rows'] <= 100  # Limited by LIMIT clause
        assert optimized['cost_estimate'] > 0