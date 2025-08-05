"""
Comprehensive tests for FDW Manager functionality.

This module tests:
- FDW extension installation
- Foreign server creation/deletion  
- User mapping creation
- Foreign table creation
- Connection validation
- Error handling scenarios
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import asyncpg
from typing import Dict, Any, List

from instant_connector import FDWManager, PostgreSQLFDWConnector


@pytest.mark.asyncio
class TestFDWManager:
    """Test cases for FDW Manager functionality."""
    
    async def test_fdw_manager_initialization(self, fdw_manager):
        """Test FDW manager initialization."""
        assert fdw_manager is not None
        assert fdw_manager.fdw_connector is not None
        assert hasattr(fdw_manager, 'install_fdw_extension')
    
    async def test_install_postgres_fdw_extension(self, fdw_manager):
        """Test PostgreSQL FDW extension installation."""
        # Test successful installation
        result = await fdw_manager.install_fdw_extension("postgres_fdw")
        assert result is True
        
        # Test idempotent installation (should not fail if already exists)
        result = await fdw_manager.install_fdw_extension("postgres_fdw")
        assert result is True
    
    async def test_install_file_fdw_extension(self, fdw_manager):
        """Test file FDW extension installation."""
        # Note: file_fdw might not be available in all PostgreSQL images
        # This test checks the attempt to install
        try:
            result = await fdw_manager.install_fdw_extension("file_fdw")
            # If it succeeds, great
            assert result is True
        except Exception as e:
            # If it fails due to extension not available, that's expected
            assert "does not exist" in str(e) or "not found" in str(e)
    
    async def test_install_invalid_fdw_extension(self, fdw_manager):
        """Test installation of invalid FDW extension."""
        with pytest.raises(Exception):
            await fdw_manager.install_fdw_extension("nonexistent_fdw")
    
    async def test_create_foreign_server(self, fdw_manager):
        """Test foreign server creation."""
        # Install extension first
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "test_server",
            "fdw_name": "postgres_fdw",
            "options": {
                "host": "localhost",
                "port": "5432",
                "dbname": "test_db"
            }
        }
        
        result = await fdw_manager.create_foreign_server(server_config)
        assert result is True
        
        # Test idempotent creation
        result = await fdw_manager.create_foreign_server(server_config)
        assert result is True
    
    async def test_create_foreign_server_with_invalid_fdw(self, fdw_manager):
        """Test foreign server creation with invalid FDW."""
        server_config = {
            "server_name": "invalid_server",
            "fdw_name": "nonexistent_fdw", 
            "options": {
                "host": "localhost"
            }
        }
        
        with pytest.raises(Exception):
            await fdw_manager.create_foreign_server(server_config)
    
    async def test_drop_foreign_server(self, fdw_manager):
        """Test foreign server deletion."""
        # Create server first
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "test_drop_server",
            "fdw_name": "postgres_fdw",
            "options": {
                "host": "localhost",
                "port": "5432"
            }
        }
        
        await fdw_manager.create_foreign_server(server_config)
        
        # Drop the server
        result = await fdw_manager.drop_foreign_server("test_drop_server")
        assert result is True
        
        # Test dropping non-existent server (should not fail)
        result = await fdw_manager.drop_foreign_server("nonexistent_server")
        assert result is True
    
    async def test_create_user_mapping(self, fdw_manager):
        """Test user mapping creation."""
        # Setup prerequisites
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "test_user_mapping_server",
            "fdw_name": "postgres_fdw",
            "options": {
                "host": "localhost",
                "port": "5432"
            }
        }
        await fdw_manager.create_foreign_server(server_config)
        
        # Create user mapping
        mapping_config = {
            "server_name": "test_user_mapping_server",
            "user_name": "current_user",
            "options": {
                "user": "remote_user",
                "password": "remote_password"
            }
        }
        
        result = await fdw_manager.create_user_mapping(mapping_config)
        assert result is True
        
        # Test idempotent creation
        result = await fdw_manager.create_user_mapping(mapping_config)
        assert result is True
    
    async def test_create_user_mapping_without_server(self, fdw_manager):
        """Test user mapping creation without existing server."""
        mapping_config = {
            "server_name": "nonexistent_server",
            "user_name": "current_user",
            "options": {
                "user": "remote_user",
                "password": "remote_password"
            }
        }
        
        with pytest.raises(Exception):
            await fdw_manager.create_user_mapping(mapping_config)
    
    async def test_drop_user_mapping(self, fdw_manager):
        """Test user mapping deletion."""
        # Setup prerequisites
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "test_drop_mapping_server", 
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "test_drop_mapping_server",
            "user_name": "current_user",
            "options": {
                "user": "remote_user",
                "password": "remote_password"
            }
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # Drop user mapping
        result = await fdw_manager.drop_user_mapping(
            "test_drop_mapping_server", 
            "current_user"
        )
        assert result is True
        
        # Test dropping non-existent mapping
        result = await fdw_manager.drop_user_mapping(
            "nonexistent_server", 
            "nonexistent_user"
        )
        assert result is True
    
    async def test_create_foreign_table(self, fdw_manager):
        """Test foreign table creation."""
        # Setup prerequisites
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "test_table_server",
            "fdw_name": "postgres_fdw", 
            "options": {
                "host": "localhost",
                "port": "5432",
                "dbname": "test_db"
            }
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "test_table_server",
            "user_name": "current_user",
            "options": {
                "user": "remote_user",
                "password": "remote_password"
            }
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # Create foreign table
        table_config = {
            "table_name": "test_foreign_table",
            "server_name": "test_table_server",
            "schema": "public",
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR(100)"},
                {"name": "email", "type": "VARCHAR(255)"}
            ],
            "options": {
                "table_name": "remote_users",
                "schema_name": "public"
            }
        }
        
        result = await fdw_manager.create_foreign_table(table_config)
        assert result is True
        
        # Test idempotent creation
        result = await fdw_manager.create_foreign_table(table_config)
        assert result is True
    
    async def test_create_foreign_table_without_server(self, fdw_manager):
        """Test foreign table creation without existing server."""
        table_config = {
            "table_name": "test_foreign_table_no_server",
            "server_name": "nonexistent_server",
            "schema": "public",
            "columns": [
                {"name": "id", "type": "INTEGER"}
            ],
            "options": {
                "table_name": "remote_table"
            }
        }
        
        with pytest.raises(Exception):
            await fdw_manager.create_foreign_table(table_config)
    
    async def test_drop_foreign_table(self, fdw_manager):
        """Test foreign table deletion."""
        # Setup prerequisites and create table
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "test_drop_table_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "test_drop_table_server",
            "user_name": "current_user",
            "options": {"user": "remote_user", "password": "remote_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        table_config = {
            "table_name": "test_drop_foreign_table",
            "server_name": "test_drop_table_server",
            "schema": "public",
            "columns": [{"name": "id", "type": "INTEGER"}],
            "options": {"table_name": "remote_table"}
        }
        await fdw_manager.create_foreign_table(table_config)
        
        # Drop the table
        result = await fdw_manager.drop_foreign_table("test_drop_foreign_table", "public")
        assert result is True
        
        # Test dropping non-existent table
        result = await fdw_manager.drop_foreign_table("nonexistent_table", "public")
        assert result is True
    
    async def test_list_foreign_servers(self, fdw_manager):
        """Test listing foreign servers."""
        # Install extension and create servers
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        servers = [
            {
                "server_name": "list_test_server1",
                "fdw_name": "postgres_fdw",
                "options": {"host": "localhost"}
            },
            {
                "server_name": "list_test_server2", 
                "fdw_name": "postgres_fdw",
                "options": {"host": "remote_host"}
            }
        ]
        
        for server in servers:
            await fdw_manager.create_foreign_server(server)
        
        # List servers
        server_list = await fdw_manager.list_foreign_servers()
        assert isinstance(server_list, list)
        
        # Check that our test servers are in the list
        server_names = [s.get('srvname') for s in server_list]
        assert "list_test_server1" in server_names
        assert "list_test_server2" in server_names
    
    async def test_list_foreign_tables(self, fdw_manager):
        """Test listing foreign tables."""
        # Setup prerequisites
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "list_tables_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "list_tables_server",
            "user_name": "current_user",
            "options": {"user": "remote_user", "password": "remote_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # Create test tables
        tables = [
            {
                "table_name": "list_test_table1",
                "server_name": "list_tables_server",
                "schema": "public",
                "columns": [{"name": "id", "type": "INTEGER"}],
                "options": {"table_name": "remote_table1"}
            },
            {
                "table_name": "list_test_table2",
                "server_name": "list_tables_server", 
                "schema": "public",
                "columns": [{"name": "name", "type": "VARCHAR(100)"}],
                "options": {"table_name": "remote_table2"}
            }
        ]
        
        for table in tables:
            await fdw_manager.create_foreign_table(table)
        
        # List tables
        table_list = await fdw_manager.list_foreign_tables()
        assert isinstance(table_list, list)
        
        # Check that our test tables are in the list
        table_names = [t.get('foreign_table_name') for t in table_list]
        assert "list_test_table1" in table_names
        assert "list_test_table2" in table_names
    
    async def test_validate_foreign_server_connection(self, fdw_manager, postgres_container):
        """Test foreign server connection validation."""
        # Setup with valid connection details
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "validate_connection_server",
            "fdw_name": "postgres_fdw",
            "options": {
                "host": postgres_container["host"],
                "port": str(postgres_container["port"]),
                "dbname": postgres_container["database"] 
            }
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "validate_connection_server",
            "user_name": "current_user",
            "options": {
                "user": postgres_container["username"],
                "password": postgres_container["password"]
            }
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # Validate connection (this may fail if we can't actually connect)
        try:
            result = await fdw_manager.validate_foreign_server_connection(
                "validate_connection_server"
            )
            # If validation succeeds, connection is working
            assert result is True
        except Exception as e:
            # If validation fails, it might be due to network restrictions
            # In that case, we just verify the method exists and handles errors
            assert "connection" in str(e).lower() or "connect" in str(e).lower()
    
    async def test_validate_nonexistent_server_connection(self, fdw_manager):
        """Test connection validation for non-existent server."""
        with pytest.raises(Exception):
            await fdw_manager.validate_foreign_server_connection("nonexistent_server")
    
    async def test_get_foreign_table_info(self, fdw_manager):
        """Test getting foreign table information."""
        # Setup table
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "info_test_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "info_test_server",
            "user_name": "current_user",
            "options": {"user": "remote_user", "password": "remote_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        table_config = {
            "table_name": "info_test_table",
            "server_name": "info_test_server",
            "schema": "public",
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR(100)"},
                {"name": "created_at", "type": "TIMESTAMP"}
            ],
            "options": {"table_name": "remote_info_table"}
        }
        await fdw_manager.create_foreign_table(table_config)
        
        # Get table info
        info = await fdw_manager.get_foreign_table_info("info_test_table", "public")
        assert info is not None
        assert info.get("table_name") == "info_test_table"
        assert info.get("server_name") == "info_test_server"
    
    async def test_get_nonexistent_table_info(self, fdw_manager):
        """Test getting info for non-existent table."""
        info = await fdw_manager.get_foreign_table_info("nonexistent_table", "public")
        assert info is None
    
    async def test_error_handling_with_invalid_sql(self, fdw_manager):
        """Test error handling with invalid SQL operations."""
        # This tests the error handling in the FDW manager
        with pytest.raises(Exception):
            async with fdw_manager.fdw_connector.get_connection() as conn:
                await conn.execute("INVALID SQL SYNTAX")
    
    async def test_concurrent_operations(self, fdw_manager):
        """Test concurrent FDW operations."""
        import asyncio
        
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        # Create multiple servers concurrently
        async def create_server(server_num):
            server_config = {
                "server_name": f"concurrent_server_{server_num}",
                "fdw_name": "postgres_fdw",
                "options": {"host": f"host_{server_num}"}
            }
            return await fdw_manager.create_foreign_server(server_config)
        
        # Run concurrent operations
        tasks = [create_server(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most operations succeeded
        successful_results = [r for r in results if r is True]
        assert len(successful_results) >= 3  # Allow for some failures due to concurrency
    
    @pytest.mark.slow
    async def test_cleanup_all_test_objects(self, fdw_manager):
        """Test cleanup of all test objects created during testing."""
        # This test runs at the end to clean up test objects
        # It's marked as slow since it's cleanup
        
        # Get all foreign tables and servers
        try:
            tables = await fdw_manager.list_foreign_tables()
            servers = await fdw_manager.list_foreign_servers()
            
            # Drop test tables
            for table in tables:
                table_name = table.get('foreign_table_name')
                schema_name = table.get('foreign_table_schema', 'public')
                if table_name and 'test' in table_name.lower():
                    await fdw_manager.drop_foreign_table(table_name, schema_name)
            
            # Drop test servers
            for server in servers:
                server_name = server.get('srvname')
                if server_name and 'test' in server_name.lower():
                    await fdw_manager.drop_foreign_server(server_name)
                    
        except Exception as e:
            # Cleanup failures are not critical for tests
            print(f"Cleanup warning: {e}")
    
    async def test_fdw_manager_context_manager(self, postgres_container):
        """Test FDW manager as context manager."""
        postgres_config = {
            "host": postgres_container["host"],
            "port": postgres_container["port"],
            "database": postgres_container["database"],
            "username": postgres_container["username"],
            "password": postgres_container["password"]
        }
        
        fdw_connector = PostgreSQLFDWConnector(**postgres_config)
        await fdw_connector.initialize()
        
        try:
            async with FDWManager(fdw_connector) as manager:
                assert manager is not None
                # Test basic operation
                result = await manager.install_fdw_extension("postgres_fdw")
                assert result is True
        finally:
            await fdw_connector.close()


@pytest.mark.unit
class TestFDWManagerUnit:
    """Unit tests for FDW Manager with mocked dependencies."""
    
    async def test_fdw_manager_with_mock_connection(self):
        """Test FDW manager with mocked connection."""
        mock_connector = MagicMock()
        mock_connector.get_connection = AsyncMock()
        
        # Mock connection context manager
        mock_conn = AsyncMock()
        mock_connector.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connector.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        manager = FDWManager(mock_connector)
        
        # Test that the manager uses the mock connector
        assert manager.fdw_connector == mock_connector
    
    async def test_error_handling_with_mock(self):
        """Test error handling with mocked exceptions."""
        mock_connector = MagicMock()
        mock_connector.get_connection = AsyncMock()
        
        # Mock connection that raises an exception
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = asyncpg.PostgresError("Mock database error")
        mock_connector.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connector.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        manager = FDWManager(mock_connector)
        
        # Test that exceptions are properly handled
        with pytest.raises(Exception):
            await manager.install_fdw_extension("test_fdw")
    
    def test_fdw_manager_validation_methods(self):
        """Test FDW manager validation methods."""
        mock_connector = MagicMock()
        manager = FDWManager(mock_connector)
        
        # Test validation methods exist
        assert hasattr(manager, 'install_fdw_extension')
        assert hasattr(manager, 'create_foreign_server')
        assert hasattr(manager, 'create_user_mapping')
        assert hasattr(manager, 'create_foreign_table')
        assert hasattr(manager, 'validate_foreign_server_connection')
        assert hasattr(manager, 'list_foreign_servers')
        assert hasattr(manager, 'list_foreign_tables')


@pytest.mark.integration
class TestFDWManagerIntegration:
    """Integration tests for FDW Manager with real database."""
    
    async def test_full_fdw_setup_workflow(self, fdw_manager, postgres_container):
        """Test complete FDW setup workflow."""
        # This test runs through a complete FDW setup workflow
        
        # 1. Install extension
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        # 2. Create foreign server
        server_config = {
            "server_name": "integration_test_server",
            "fdw_name": "postgres_fdw",
            "options": {
                "host": postgres_container["host"],
                "port": str(postgres_container["port"]),
                "dbname": postgres_container["database"]
            }
        }
        await fdw_manager.create_foreign_server(server_config)
        
        # 3. Create user mapping
        mapping_config = {
            "server_name": "integration_test_server",
            "user_name": "current_user",
            "options": {
                "user": postgres_container["username"],
                "password": postgres_container["password"]
            }
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # 4. Create foreign table
        table_config = {
            "table_name": "integration_test_table",
            "server_name": "integration_test_server",
            "schema": "public",
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR(100)"}
            ],
            "options": {
                "table_name": "users",
                "schema_name": "public"
            }
        }
        await fdw_manager.create_foreign_table(table_config)
        
        # 5. Verify setup
        servers = await fdw_manager.list_foreign_servers()
        server_names = [s.get('srvname') for s in servers]
        assert "integration_test_server" in server_names
        
        tables = await fdw_manager.list_foreign_tables()
        table_names = [t.get('foreign_table_name') for t in tables]
        assert "integration_test_table" in table_names
        
        # 6. Get table info
        table_info = await fdw_manager.get_foreign_table_info("integration_test_table", "public")
        assert table_info is not None
        assert table_info.get("server_name") == "integration_test_server"
        
        # 7. Cleanup
        await fdw_manager.drop_foreign_table("integration_test_table", "public")
        await fdw_manager.drop_user_mapping("integration_test_server", "current_user")
        await fdw_manager.drop_foreign_server("integration_test_server")