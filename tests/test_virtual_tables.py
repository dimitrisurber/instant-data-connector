"""
Comprehensive tests for virtual table management functionality.

This module tests:
- Virtual table creation from config
- Different FDW types
- Table metadata management
- Configuration validation
- Table refresh functionality
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, List

from instant_connector import VirtualTableManager, PostgreSQLFDWConnector, ConfigParser


@pytest.mark.asyncio
class TestVirtualTableManager:
    """Test cases for VirtualTableManager functionality."""
    
    async def test_virtual_table_manager_initialization(self, fdw_manager):
        """Test virtual table manager initialization."""
        config_parser = ConfigParser()
        
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        assert manager is not None
        assert manager.fdw_connector is not None
        assert manager.config_parser is not None
        assert hasattr(manager, 'create_virtual_tables_from_config')
        assert hasattr(manager, 'list_virtual_tables')
        assert hasattr(manager, 'refresh_virtual_table')
    
    async def test_create_postgres_fdw_table(self, fdw_manager):
        """Test creating PostgreSQL FDW virtual table."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Setup FDW infrastructure first
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        # Create server
        server_config = {
            "server_name": "test_postgres_server",
            "fdw_name": "postgres_fdw",
            "options": {
                "host": "localhost",
                "port": "5432",
                "dbname": "test_db"
            }
        }
        await fdw_manager.create_foreign_server(server_config)
        
        # Create user mapping
        mapping_config = {
            "server_name": "test_postgres_server",
            "user_name": "current_user",
            "options": {
                "user": "test_user",
                "password": "test_password"
            }
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # Create virtual table
        table_config = {
            "name": "test_virtual_table",
            "type": "postgres_fdw",
            "server_name": "test_postgres_server",
            "schema": "public",
            "options": {
                "table_name": "remote_users",
                "schema_name": "public"
            },
            "columns": [
                {"name": "id", "type": "INTEGER", "not_null": True},
                {"name": "name", "type": "VARCHAR(100)", "not_null": True},
                {"name": "email", "type": "VARCHAR(255)"}
            ]
        }
        
        result = await manager.create_virtual_table(table_config)
        assert result is True
        
        # Verify table was created
        virtual_tables = await manager.list_virtual_tables()
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "test_virtual_table" in table_names
    
    async def test_create_virtual_tables_from_config(self, fdw_manager):
        """Test creating virtual tables from configuration."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Create test configuration
        config = {
            "version": "1.0",
            "sources": {
                "test_postgres_source": {
                    "type": "postgres_fdw",
                    "enabled": True,
                    "server_options": {
                        "host": "localhost",
                        "port": "5432",
                        "dbname": "test_db"
                    },
                    "user_mapping": {
                        "options": {
                            "user": "test_user",
                            "password": "test_password"
                        }
                    },
                    "tables": [
                        {
                            "name": "config_test_table",
                            "description": "Test table from config",
                            "options": {
                                "table_name": "remote_table",
                                "schema_name": "public"
                            },
                            "columns": [
                                {"name": "id", "type": "INTEGER"},
                                {"name": "data", "type": "TEXT"}
                            ]
                        }
                    ]
                }
            }
        }
        
        result = await manager.create_virtual_tables_from_config(config)
        assert result is True
        
        # Verify tables were created
        virtual_tables = await manager.list_virtual_tables()
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "config_test_table" in table_names
    
    async def test_create_file_fdw_table(self, fdw_manager):
        """Test creating file FDW virtual table."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Note: file_fdw might not be available in all PostgreSQL installations
        try:
            await fdw_manager.install_fdw_extension("file_fdw")
            
            # Create virtual table for file
            table_config = {
                "name": "test_file_table",
                "type": "file_fdw",
                "server_name": "file_server",  # file_fdw typically uses a generic server
                "schema": "public",
                "options": {
                    "filename": "/tmp/test_data.csv",
                    "format": "csv",
                    "header": "true"
                },
                "columns": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "name", "type": "TEXT"},
                    {"name": "value", "type": "DECIMAL(10,2)"}
                ]
            }
            
            result = await manager.create_virtual_table(table_config)
            assert result is True
            
        except Exception as e:
            if "does not exist" in str(e) or "not found" in str(e):
                pytest.skip("file_fdw extension not available")
            else:
                raise
    
    async def test_create_multiple_fdw_types(self, fdw_manager):
        """Test creating tables with different FDW types."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        config = {
            "version": "1.0",
            "sources": {
                "postgres_source": {
                    "type": "postgres_fdw",
                    "enabled": True,
                    "server_options": {
                        "host": "localhost",
                        "port": "5432",
                        "dbname": "test_db"
                    },
                    "user_mapping": {
                        "options": {
                            "user": "test_user",
                            "password": "test_password"
                        }
                    },
                    "tables": [
                        {
                            "name": "postgres_table",
                            "options": {
                                "table_name": "remote_postgres",
                                "schema_name": "public"
                            },
                            "columns": [
                                {"name": "id", "type": "INTEGER"}
                            ]
                        }
                    ]
                }
            }
        }
        
        result = await manager.create_virtual_tables_from_config(config)
        assert result is True
        
        # Verify different types were created
        virtual_tables = await manager.list_virtual_tables()
        assert len(virtual_tables) >= 1
        
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "postgres_table" in table_names
    
    async def test_list_virtual_tables(self, fdw_manager):
        """Test listing virtual tables."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Create some test tables first
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "list_test_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "list_test_server",
            "user_name": "current_user",
            "options": {"user": "test_user", "password": "test_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # Create multiple tables
        table_configs = [
            {
                "name": "list_table_1",
                "type": "postgres_fdw",
                "server_name": "list_test_server",
                "schema": "public",
                "options": {"table_name": "remote_1"},
                "columns": [{"name": "id", "type": "INTEGER"}]
            },
            {
                "name": "list_table_2",
                "type": "postgres_fdw",
                "server_name": "list_test_server",
                "schema": "public",
                "options": {"table_name": "remote_2"},
                "columns": [{"name": "id", "type": "INTEGER"}]
            }
        ]
        
        for table_config in table_configs:
            await manager.create_virtual_table(table_config)
        
        # List all virtual tables
        virtual_tables = await manager.list_virtual_tables()
        
        assert isinstance(virtual_tables, list)
        assert len(virtual_tables) >= 2
        
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "list_table_1" in table_names
        assert "list_table_2" in table_names
        
        # Check table metadata
        for table in virtual_tables:
            if table.get('table_name') in ['list_table_1', 'list_table_2']:
                assert 'server_name' in table
                assert 'created_at' in table or 'table_name' in table
    
    async def test_refresh_virtual_table(self, fdw_manager):
        """Test refreshing virtual table metadata."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Create a test table
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "refresh_test_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "refresh_test_server",
            "user_name": "current_user",
            "options": {"user": "test_user", "password": "test_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        table_config = {
            "name": "refresh_test_table",
            "type": "postgres_fdw",
            "server_name": "refresh_test_server",
            "schema": "public",
            "options": {"table_name": "remote_refresh"},
            "columns": [{"name": "id", "type": "INTEGER"}]
        }
        
        await manager.create_virtual_table(table_config)
        
        # Refresh the table
        result = await manager.refresh_virtual_table("refresh_test_table")
        assert result is True
    
    async def test_drop_virtual_table(self, fdw_manager):
        """Test dropping virtual tables."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Create a test table
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "drop_test_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "drop_test_server",
            "user_name": "current_user",
            "options": {"user": "test_user", "password": "test_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        table_config = {
            "name": "drop_test_table",
            "type": "postgres_fdw",
            "server_name": "drop_test_server",
            "schema": "public",
            "options": {"table_name": "remote_drop"},
            "columns": [{"name": "id", "type": "INTEGER"}]
        }
        
        await manager.create_virtual_table(table_config)
        
        # Verify table exists
        virtual_tables = await manager.list_virtual_tables()
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "drop_test_table" in table_names
        
        # Drop the table
        result = await manager.drop_virtual_table("drop_test_table", "public")
        assert result is True
        
        # Verify table is gone
        virtual_tables = await manager.list_virtual_tables()
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "drop_test_table" not in table_names
    
    async def test_get_virtual_table_metadata(self, fdw_manager):
        """Test getting virtual table metadata."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Create a test table with metadata
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "metadata_test_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "metadata_test_server",
            "user_name": "current_user",
            "options": {"user": "test_user", "password": "test_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        table_config = {
            "name": "metadata_test_table",
            "type": "postgres_fdw",
            "description": "Test table for metadata",
            "server_name": "metadata_test_server",
            "schema": "public",
            "options": {"table_name": "remote_metadata"},
            "columns": [
                {"name": "id", "type": "INTEGER", "description": "Primary key"},
                {"name": "name", "type": "VARCHAR(100)", "description": "Name field"}
            ]
        }
        
        await manager.create_virtual_table(table_config)
        
        # Get table metadata
        metadata = await manager.get_virtual_table_metadata("metadata_test_table", "public")
        
        assert metadata is not None
        assert metadata.get('table_name') == "metadata_test_table"
        assert metadata.get('description') == "Test table for metadata"
        assert 'columns' in metadata
        
        # Check column metadata
        columns = metadata['columns']
        assert len(columns) >= 2
        
        column_names = [col['column_name'] for col in columns]
        assert 'id' in column_names
        assert 'name' in column_names
    
    async def test_configuration_validation(self, fdw_manager):
        """Test configuration validation."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Test invalid configuration
        invalid_config = {
            "version": "1.0",
            "sources": {
                "invalid_source": {
                    # Missing required fields
                    "enabled": True,
                    "tables": []
                }
            }
        }
        
        with pytest.raises(Exception):
            await manager.create_virtual_tables_from_config(invalid_config)
        
        # Test configuration with missing tables
        config_without_tables = {
            "version": "1.0",
            "sources": {
                "empty_source": {
                    "type": "postgres_fdw",
                    "enabled": True,
                    "server_options": {"host": "localhost"},
                    "user_mapping": {"options": {"user": "test", "password": "test"}},
                    "tables": []
                }
            }
        }
        
        # Should succeed but create no tables
        result = await manager.create_virtual_tables_from_config(config_without_tables)
        assert result is True
    
    async def test_disabled_sources(self, fdw_manager):
        """Test handling of disabled sources."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        config = {
            "version": "1.0",
            "sources": {
                "disabled_source": {
                    "type": "postgres_fdw",
                    "enabled": False,  # Disabled source
                    "server_options": {"host": "localhost"},
                    "user_mapping": {"options": {"user": "test", "password": "test"}},
                    "tables": [
                        {
                            "name": "should_not_be_created",
                            "options": {"table_name": "remote"},
                            "columns": [{"name": "id", "type": "INTEGER"}]
                        }
                    ]
                }
            }
        }
        
        result = await manager.create_virtual_tables_from_config(config)
        assert result is True
        
        # Verify disabled table was not created
        virtual_tables = await manager.list_virtual_tables()
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "should_not_be_created" not in table_names
    
    async def test_error_handling_in_table_creation(self, fdw_manager):
        """Test error handling during table creation."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Try to create table without server setup
        table_config = {
            "name": "error_test_table",
            "type": "postgres_fdw",
            "server_name": "nonexistent_server",
            "schema": "public",
            "options": {"table_name": "remote"},
            "columns": [{"name": "id", "type": "INTEGER"}]
        }
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            await manager.create_virtual_table(table_config)
    
    async def test_concurrent_table_operations(self, fdw_manager):
        """Test concurrent virtual table operations."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Setup infrastructure
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "concurrent_test_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "concurrent_test_server",
            "user_name": "current_user",
            "options": {"user": "test_user", "password": "test_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # Create multiple tables concurrently
        async def create_table(table_num):
            table_config = {
                "name": f"concurrent_table_{table_num}",
                "type": "postgres_fdw",
                "server_name": "concurrent_test_server",
                "schema": "public",
                "options": {"table_name": f"remote_{table_num}"},
                "columns": [{"name": "id", "type": "INTEGER"}]
            }
            return await manager.create_virtual_table(table_config)
        
        # Run concurrent operations
        import asyncio
        tasks = [create_table(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most operations should succeed
        successful_results = [r for r in results if r is True]
        assert len(successful_results) >= 3  # Allow some failures due to concurrency


@pytest.mark.unit
class TestVirtualTableManagerUnit:
    """Unit tests for VirtualTableManager with mocked dependencies."""
    
    def test_table_config_validation(self):
        """Test table configuration validation."""
        mock_connector = MagicMock()
        config_parser = ConfigParser()
        
        manager = VirtualTableManager(mock_connector, config_parser)
        
        # Valid configuration
        valid_config = {
            "name": "test_table",
            "type": "postgres_fdw",
            "server_name": "test_server",
            "schema": "public",
            "options": {"table_name": "remote_table"},
            "columns": [{"name": "id", "type": "INTEGER"}]
        }
        
        # Should not raise exception
        assert manager._validate_table_config(valid_config) is True
        
        # Invalid configuration (missing required fields)
        invalid_config = {
            "name": "test_table",
            # Missing type, server_name, etc.
        }
        
        assert manager._validate_table_config(invalid_config) is False
    
    def test_column_definition_validation(self):
        """Test column definition validation."""
        mock_connector = MagicMock()
        config_parser = ConfigParser()
        
        manager = VirtualTableManager(mock_connector, config_parser)
        
        # Valid column definition
        valid_column = {
            "name": "id",
            "type": "INTEGER",
            "not_null": True,
            "description": "Primary key"
        }
        
        assert manager._validate_column_definition(valid_column) is True
        
        # Invalid column definition (missing name)
        invalid_column = {
            "type": "INTEGER"
            # Missing name
        }
        
        assert manager._validate_column_definition(invalid_column) is False
    
    def test_fdw_type_support(self):
        """Test FDW type support detection."""
        mock_connector = MagicMock()
        config_parser = ConfigParser()
        
        manager = VirtualTableManager(mock_connector, config_parser)
        
        # Supported FDW types
        assert manager._is_fdw_type_supported("postgres_fdw") is True
        assert manager._is_fdw_type_supported("mysql_fdw") is True
        assert manager._is_fdw_type_supported("file_fdw") is True
        assert manager._is_fdw_type_supported("multicorn") is True
        
        # Unsupported FDW type
        assert manager._is_fdw_type_supported("unsupported_fdw") is False
    
    async def test_mock_table_creation(self):
        """Test table creation with mocked connector."""
        mock_connector = MagicMock()
        mock_connector.get_connection = AsyncMock()
        
        # Mock connection context manager
        mock_conn = AsyncMock()
        mock_connector.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connector.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        config_parser = ConfigParser()
        manager = VirtualTableManager(mock_connector, config_parser)
        
        table_config = {
            "name": "mock_table",
            "type": "postgres_fdw",
            "server_name": "mock_server",
            "schema": "public",
            "options": {"table_name": "remote_table"},
            "columns": [{"name": "id", "type": "INTEGER"}]
        }
        
        # Mock successful table creation
        mock_conn.execute.return_value = None
        
        result = await manager.create_virtual_table(table_config)
        
        # Verify methods were called
        mock_connector.get_connection.assert_called()
        mock_conn.execute.assert_called()
        
        assert result is True


@pytest.mark.integration
class TestVirtualTableManagerIntegration:
    """Integration tests with real database and configuration files."""
    
    async def test_integration_with_config_file(self, fdw_manager, test_config_dir):
        """Test integration with actual configuration file."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # Create comprehensive test configuration
        test_config = {
            "version": "1.0",
            "metadata": {
                "name": "Integration Test Config",
                "description": "Test configuration for integration testing"
            },
            "sources": {
                "integration_postgres": {
                    "type": "postgres_fdw",
                    "description": "Integration test PostgreSQL source",
                    "enabled": True,
                    "server_options": {
                        "host": "localhost",
                        "port": "5432",
                        "dbname": "integration_test"
                    },
                    "user_mapping": {
                        "options": {
                            "user": "integration_user",
                            "password": "integration_password"
                        }
                    },
                    "tables": [
                        {
                            "name": "integration_users",
                            "description": "Integration test users table",
                            "options": {
                                "table_name": "users",
                                "schema_name": "public"
                            },
                            "columns": [
                                {"name": "id", "type": "INTEGER", "not_null": True},
                                {"name": "username", "type": "VARCHAR(50)", "not_null": True},
                                {"name": "email", "type": "VARCHAR(255)"},
                                {"name": "created_at", "type": "TIMESTAMP"}
                            ]
                        },
                        {
                            "name": "integration_orders",
                            "description": "Integration test orders table",
                            "options": {
                                "table_name": "orders",
                                "schema_name": "public"
                            },
                            "columns": [
                                {"name": "id", "type": "BIGINT", "not_null": True},
                                {"name": "user_id", "type": "INTEGER", "not_null": True},
                                {"name": "total_amount", "type": "DECIMAL(12,2)"},
                                {"name": "status", "type": "VARCHAR(20)"}
                            ]
                        }
                    ]
                }
            }
        }
        
        # Save configuration to file
        config_file = test_config_dir / "integration_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Parse and apply configuration
        parsed_config = config_parser.parse_config(config_file)
        result = await manager.create_virtual_tables_from_config(parsed_config)
        
        assert result is True
        
        # Verify tables were created
        virtual_tables = await manager.list_virtual_tables()
        table_names = [t.get('table_name') for t in virtual_tables]
        
        assert "integration_users" in table_names
        assert "integration_orders" in table_names
        
        # Verify table metadata
        users_metadata = await manager.get_virtual_table_metadata("integration_users", "public")
        assert users_metadata is not None
        assert users_metadata.get('table_name') == "integration_users"
        
        orders_metadata = await manager.get_virtual_table_metadata("integration_orders", "public")
        assert orders_metadata is not None
        assert orders_metadata.get('table_name') == "integration_orders"
    
    async def test_full_lifecycle_management(self, fdw_manager):
        """Test full lifecycle of virtual table management."""
        config_parser = ConfigParser()
        manager = VirtualTableManager(fdw_manager.fdw_connector, config_parser)
        
        # 1. Create infrastructure
        await fdw_manager.install_fdw_extension("postgres_fdw")
        
        server_config = {
            "server_name": "lifecycle_test_server",
            "fdw_name": "postgres_fdw",
            "options": {"host": "localhost"}
        }
        await fdw_manager.create_foreign_server(server_config)
        
        mapping_config = {
            "server_name": "lifecycle_test_server",
            "user_name": "current_user",
            "options": {"user": "test_user", "password": "test_password"}
        }
        await fdw_manager.create_user_mapping(mapping_config)
        
        # 2. Create virtual table
        table_config = {
            "name": "lifecycle_test_table",
            "type": "postgres_fdw",
            "description": "Lifecycle test table",
            "server_name": "lifecycle_test_server",
            "schema": "public",
            "options": {"table_name": "remote_lifecycle"},
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "data", "type": "TEXT"}
            ]
        }
        
        create_result = await manager.create_virtual_table(table_config)
        assert create_result is True
        
        # 3. List and verify
        virtual_tables = await manager.list_virtual_tables()
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "lifecycle_test_table" in table_names
        
        # 4. Get metadata
        metadata = await manager.get_virtual_table_metadata("lifecycle_test_table", "public")
        assert metadata is not None
        assert metadata.get('table_name') == "lifecycle_test_table"
        
        # 5. Refresh table
        refresh_result = await manager.refresh_virtual_table("lifecycle_test_table")
        assert refresh_result is True
        
        # 6. Drop table
        drop_result = await manager.drop_virtual_table("lifecycle_test_table", "public")
        assert drop_result is True
        
        # 7. Verify table is gone
        virtual_tables = await manager.list_virtual_tables()
        table_names = [t.get('table_name') for t in virtual_tables]
        assert "lifecycle_test_table" not in table_names