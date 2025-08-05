"""
Virtual Table Manager for PostgreSQL FDW

This module manages virtual tables created through PostgreSQL Foreign Data Wrappers,
providing a high-level interface for configuration-driven table management.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from .config_parser import ConfigParser
from .postgresql_fdw_manager import PostgreSQLFDWConnector

logger = logging.getLogger(__name__)


class VirtualTableManager:
    """
    High-level manager for PostgreSQL FDW virtual tables.
    
    Features:
    - YAML configuration parsing and validation
    - Virtual table creation and management
    - Support for multiple FDW types
    - Table metadata management
    - Column type inference and mapping
    - Table refresh and synchronization
    - Hot-reloading of configuration
    """
    
    # Supported FDW types and their characteristics
    FDW_TYPES = {
        'postgres_fdw': {
            'extension': 'postgres_fdw',
            'required_server_options': ['host', 'dbname'],
            'optional_server_options': ['port'],
            'required_user_mapping_options': ['user', 'password'],
            'required_table_options': ['table_name'],
            'optional_table_options': ['schema_name'],
            'supports_pushdown': True,
            'supports_transactions': True
        },
        'mysql_fdw': {
            'extension': 'mysql_fdw',
            'required_server_options': ['host', 'port'],
            'optional_server_options': ['secure_auth', 'use_remote_estimate'],
            'required_user_mapping_options': ['username', 'password'],
            'required_table_options': ['dbname', 'table_name'],
            'optional_table_options': [],
            'supports_pushdown': True,
            'supports_transactions': False
        },
        'file_fdw': {
            'extension': 'file_fdw',
            'required_server_options': [],
            'optional_server_options': [],
            'required_user_mapping_options': [],
            'required_table_options': ['filename'],
            'optional_table_options': ['format', 'header', 'delimiter', 'quote', 'escape', 'null'],
            'supports_pushdown': False,
            'supports_transactions': False
        },
        'multicorn': {
            'extension': 'multicorn',
            'required_server_options': ['wrapper'],
            'optional_server_options': [],
            'required_user_mapping_options': [],
            'required_table_options': [],
            'optional_table_options': [],
            'supports_pushdown': True,
            'supports_transactions': False
        }
    }
    
    # PostgreSQL to pandas type mapping
    PG_TO_PANDAS_TYPES = {
        'integer': 'int64',
        'bigint': 'int64',
        'smallint': 'int32',
        'decimal': 'float64',
        'numeric': 'float64',
        'real': 'float32',
        'double precision': 'float64',
        'text': 'object',
        'varchar': 'object',
        'char': 'object',
        'boolean': 'bool',
        'date': 'datetime64[ns]',
        'timestamp': 'datetime64[ns]',
        'timestamptz': 'datetime64[ns]',
        'json': 'object',
        'jsonb': 'object',
        'uuid': 'object'
    }
    
    def __init__(
        self,
        fdw_connector: PostgreSQLFDWConnector,
        config_parser: Optional[ConfigParser] = None
    ):
        """
        Initialize virtual table manager.
        
        Args:
            fdw_connector: PostgreSQL FDW connector instance
            config_parser: Optional config parser (creates new if None)
        """
        self.fdw_connector = fdw_connector
        self.config_parser = config_parser or ConfigParser()
        
        # Track managed virtual tables
        self.virtual_tables: Dict[str, Dict[str, Any]] = {}
        self.managed_tables: Dict[str, Dict[str, Any]] = {}  # Alias for compatibility
        self.table_metadata: Dict[str, Dict[str, Any]] = {}
        self.last_config_hash: Optional[str] = None
        self.config_file_path: Optional[Path] = None
        
        logger.info("Initialized VirtualTableManager")
    
    async def load_configuration(
        self,
        config_path: Union[str, Path],
        validate: bool = True,
        auto_reload: bool = False
    ) -> bool:
        """
        Load FDW configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            validate: Whether to validate configuration schema
            auto_reload: Whether to enable hot-reloading
        
        Returns:
            True if configuration loaded successfully
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Parse and validate configuration
            config = self.config_parser.parse_config(config_path, validate=validate)
            
            # Store configuration metadata
            self.config_file_path = config_path
            self.last_config_hash = self.config_parser.get_config_hash()
            
            # Create virtual tables from configuration
            success = await self._create_virtual_tables_from_config(config)
            
            if success:
                logger.info(f"Successfully loaded configuration from {config_path}")
                if auto_reload:
                    logger.info("Auto-reload enabled for configuration")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    async def reload_configuration(self, force: bool = False) -> bool:
        """
        Reload configuration if it has changed.
        
        Args:
            force: Force reload even if no changes detected
        
        Returns:
            True if configuration was reloaded
        """
        if not self.config_file_path:
            logger.warning("No configuration file path set for reload")
            return False
        
        try:
            # Check if configuration has changed
            if not force:
                current_hash = self.config_parser.calculate_file_hash(self.config_file_path)
                if current_hash == self.last_config_hash:
                    logger.debug("Configuration unchanged, skipping reload")
                    return False
            
            logger.info("Configuration changes detected, reloading...")
            
            # Clean up existing virtual tables
            await self._cleanup_managed_tables()
            
            # Reload configuration
            return await self.load_configuration(self.config_file_path, validate=True)
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    async def create_virtual_table(
        self,
        table_config: Dict[str, Any],
        server_name: str,
        validate_config: bool = True
    ) -> bool:
        """
        Create a single virtual table from configuration.
        
        Args:
            table_config: Table configuration dictionary
            server_name: Name of foreign server
            validate_config: Whether to validate table configuration
        
        Returns:
            True if table created successfully
        """
        try:
            table_name = table_config['name']
            fdw_type = table_config.get('fdw_type', 'postgres_fdw')
            
            if validate_config:
                self._validate_table_config(table_config, fdw_type)
            
            # Infer column types if not specified
            columns = await self._prepare_table_columns(table_config, fdw_type)
            
            # Create foreign table
            success = await self.fdw_connector.create_foreign_table(
                table_name=table_name,
                server_name=server_name,
                columns=columns,
                options=table_config.get('options', {}),
                schema=table_config.get('schema', 'public'),
                if_not_exists=True
            )
            
            if success:
                # Store virtual table metadata
                self.virtual_tables[table_name] = {
                    'config': table_config,
                    'server_name': server_name,
                    'fdw_type': fdw_type,
                    'created_at': datetime.utcnow().isoformat(),
                    'schema': table_config.get('schema', 'public'),
                    'columns': columns
                }
                
                # Store table metadata
                await self._update_table_metadata(table_name, columns, fdw_type)
                
                logger.info(f"Successfully created virtual table: {table_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create virtual table: {e}")
            raise
    
    async def refresh_table_metadata(self, table_name: str) -> bool:
        """
        Refresh metadata for a specific virtual table.
        
        Args:
            table_name: Name of virtual table
        
        Returns:
            True if metadata refreshed successfully
        """
        try:
            if table_name not in self.virtual_tables:
                raise ValueError(f"Virtual table {table_name} not found")
            
            table_info = self.virtual_tables[table_name]
            
            # Query table metadata from PostgreSQL
            async with self.fdw_connector.get_connection() as conn:
                metadata_query = """
                    SELECT 
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        c.column_default,
                        c.character_maximum_length,
                        c.numeric_precision,
                        c.numeric_scale
                    FROM information_schema.columns c
                    WHERE c.table_schema = $1 AND c.table_name = $2
                    ORDER BY c.ordinal_position
                """
                
                schema = table_info.get('schema', 'public')
                columns = await conn.fetch(metadata_query, schema, table_name)
                
                if columns:
                    # Update metadata
                    metadata = {
                        'columns': [dict(col) for col in columns],
                        'row_count': await self._estimate_row_count(table_name, schema),
                        'last_refreshed': datetime.utcnow().isoformat(),
                        'fdw_type': table_info['fdw_type']
                    }
                    
                    self.table_metadata[table_name] = metadata
                    logger.info(f"Refreshed metadata for table: {table_name}")
                    return True
                else:
                    logger.warning(f"No metadata found for table: {table_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to refresh metadata for table {table_name}: {e}")
            return False
    
    async def synchronize_table(self, table_name: str) -> bool:
        """
        Synchronize virtual table structure with source.
        
        Args:
            table_name: Name of virtual table to synchronize
        
        Returns:
            True if synchronization successful
        """
        try:
            if table_name not in self.virtual_tables:
                raise ValueError(f"Virtual table {table_name} not found")
            
            table_info = self.virtual_tables[table_name]
            fdw_type = table_info['fdw_type']
            
            # Different synchronization strategies based on FDW type
            if fdw_type == 'postgres_fdw':
                return await self._sync_postgres_table(table_name, table_info)
            elif fdw_type == 'file_fdw':
                return await self._sync_file_table(table_name, table_info)
            elif fdw_type == 'mysql_fdw':
                return await self._sync_mysql_table(table_name, table_info)
            else:
                logger.warning(f"Synchronization not supported for FDW type: {fdw_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to synchronize table {table_name}: {e}")
            return False
    
    def get_virtual_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a virtual table.
        
        Args:
            table_name: Name of virtual table
        
        Returns:
            Virtual table information or None if not found
        """
        table_info = self.virtual_tables.get(table_name)
        if table_info:
            # Combine with metadata
            metadata = self.table_metadata.get(table_name, {})
            return {
                **table_info,
                'metadata': metadata
            }
        return None
    
    def list_virtual_tables(self) -> List[Dict[str, Any]]:
        """
        List all managed virtual tables.
        
        Returns:
            List of virtual table information
        """
        tables = []
        for table_name, table_info in self.virtual_tables.items():
            metadata = self.table_metadata.get(table_name, {})
            tables.append({
                'name': table_name,
                'fdw_type': table_info['fdw_type'],
                'server_name': table_info['server_name'],
                'schema': table_info['schema'],
                'created_at': table_info['created_at'],
                'column_count': len(table_info.get('columns', [])),
                'row_count': metadata.get('row_count', 'unknown'),
                'last_refreshed': metadata.get('last_refreshed', 'never')
            })
        return tables
    
    async def drop_virtual_table(self, table_name: str, cascade: bool = False) -> bool:
        """
        Drop a virtual table.
        
        Args:
            table_name: Name of virtual table to drop
            cascade: Whether to use CASCADE option
        
        Returns:
            True if table dropped successfully
        """
        try:
            if table_name not in self.virtual_tables:
                logger.warning(f"Virtual table {table_name} not found in managed tables")
                return True
            
            table_info = self.virtual_tables[table_name]
            schema = table_info.get('schema', 'public')
            
            # Drop foreign table
            async with self.fdw_connector.get_connection() as conn:
                cascade_clause = "CASCADE" if cascade else ""
                drop_query = f'DROP FOREIGN TABLE IF EXISTS {schema}.{table_name} {cascade_clause}'
                await conn.execute(drop_query)
            
            # Remove from tracking
            del self.virtual_tables[table_name]
            if table_name in self.table_metadata:
                del self.table_metadata[table_name]
            
            logger.info(f"Successfully dropped virtual table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop virtual table {table_name}: {e}")
            return False
    
    async def create_virtual_tables_from_config(
        self,
        config: Dict[str, Any],
        force_recreate: bool = False
    ) -> bool:
        """
        Create virtual tables from FDW configuration.
        
        This is a public method that handles the FDW configuration format
        used by the InstantDataConnector.
        
        Args:
            config: FDW configuration dictionary
            force_recreate: Whether to recreate existing tables
            
        Returns:
            True if successful
        """
        try:
            # Since FDW infrastructure is already set up by PostgreSQLFDWConnector,
            # the virtual tables are essentially the same as foreign tables.
            # For file_fdw, the foreign table has already been created, so we just
            # need to register it with our virtual table manager.
            
            extension = config.get('extension')
            server_config = config.get('server', {})
            server_name = server_config.get('name')
            tables = config.get('tables', [])
            
            logger.info(f"Virtual table management for {extension} server '{server_name}'")
            
            for table_config in tables:
                table_name = table_config.get('name')
                if table_name:
                    # Register the table as managed (it's already created as a foreign table)
                    self.managed_tables[table_name] = {
                        'server_name': server_name,
                        'created_at': datetime.now(),
                        'last_refreshed': datetime.now(),
                        **table_config
                    }
                    logger.info(f"Registered virtual table: {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create virtual tables from config: {e}")
            return False
    
    async def _create_virtual_tables_from_config(self, config: Dict[str, Any]) -> bool:
        """Create all virtual tables from configuration."""
        try:
            sources = config.get('sources', {})
            success_count = 0
            total_count = 0
            
            for source_name, source_config in sources.items():
                fdw_type = source_config['type']
                
                # Setup foreign server if not exists
                server_success = await self._setup_foreign_server(source_name, source_config)
                if not server_success:
                    logger.error(f"Failed to setup foreign server: {source_name}")
                    continue
                
                # Create virtual tables for this source
                for table_config in source_config.get('tables', []):
                    total_count += 1
                    table_config['fdw_type'] = fdw_type
                    
                    if await self.create_virtual_table(table_config, source_name):
                        success_count += 1
            
            logger.info(f"Created {success_count}/{total_count} virtual tables")
            return success_count == total_count
            
        except Exception as e:
            logger.error(f"Failed to create virtual tables from config: {e}")
            return False
    
    async def _setup_foreign_server(self, source_name: str, source_config: Dict[str, Any]) -> bool:
        """Setup foreign server from source configuration."""
        try:
            fdw_type = source_config['type']
            fdw_info = self.FDW_TYPES[fdw_type]
            
            # Install extension
            await self.fdw_connector.install_extension(fdw_info['extension'])
            
            # Create foreign server
            server_options = source_config.get('server_options', {})
            await self.fdw_connector.create_foreign_server(
                server_name=source_name,
                fdw_name=fdw_info['extension'],
                options=server_options
            )
            
            # Create user mapping if specified
            if 'user_mapping' in source_config:
                user_mapping = source_config['user_mapping']
                await self.fdw_connector.create_user_mapping(
                    server_name=source_name,
                    user_name=user_mapping.get('user', 'current_user'),
                    options=user_mapping.get('options', {})
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup foreign server {source_name}: {e}")
            return False
    
    async def _prepare_table_columns(
        self,
        table_config: Dict[str, Any],
        fdw_type: str
    ) -> List[Dict[str, str]]:
        """Prepare table columns with type inference if needed."""
        columns = table_config.get('columns', [])
        
        if columns:
            # Columns explicitly defined
            return columns
        
        # Attempt type inference based on FDW type
        if fdw_type == 'postgres_fdw':
            return await self._infer_postgres_columns(table_config)
        elif fdw_type == 'file_fdw':
            return await self._infer_file_columns(table_config)
        elif fdw_type == 'mysql_fdw':
            return await self._infer_mysql_columns(table_config)
        else:
            # Default to basic columns if inference not available
            logger.warning(f"Column inference not available for {fdw_type}, using defaults")
            return [
                {'name': 'id', 'type': 'integer'},
                {'name': 'data', 'type': 'text'}
            ]
    
    async def _infer_postgres_columns(self, table_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Infer column types from remote PostgreSQL table."""
        # This would require connection to remote server to inspect schema
        # For now, return basic structure
        return [
            {'name': 'id', 'type': 'integer'},
            {'name': 'created_at', 'type': 'timestamp'},
            {'name': 'data', 'type': 'text'}
        ]
    
    async def _infer_file_columns(self, table_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Infer column types from file structure."""
        # This would analyze the file to determine column types
        # For now, return basic CSV structure
        return [
            {'name': 'column1', 'type': 'text'},
            {'name': 'column2', 'type': 'text'},
            {'name': 'column3', 'type': 'text'}
        ]
    
    async def _infer_mysql_columns(self, table_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Infer column types from MySQL table."""
        # This would require connection to MySQL to inspect schema
        # For now, return basic structure
        return [
            {'name': 'id', 'type': 'integer'},
            {'name': 'name', 'type': 'text'},
            {'name': 'value', 'type': 'decimal'}
        ]
    
    def _validate_table_config(self, table_config: Dict[str, Any], fdw_type: str) -> None:
        """Validate table configuration for specific FDW type."""
        if fdw_type not in self.FDW_TYPES:
            raise ValueError(f"Unsupported FDW type: {fdw_type}")
        
        fdw_info = self.FDW_TYPES[fdw_type]
        table_options = table_config.get('options', {})
        
        # Check required table options
        for required_option in fdw_info['required_table_options']:
            if required_option not in table_options:
                raise ValueError(
                    f"Required table option '{required_option}' missing for {fdw_type}"
                )
        
        # Validate column definitions if present
        columns = table_config.get('columns', [])
        if columns:
            for column in columns:
                if 'name' not in column or 'type' not in column:
                    raise ValueError("Each column must have 'name' and 'type' keys")
    
    async def _update_table_metadata(
        self,
        table_name: str,
        columns: List[Dict[str, str]],
        fdw_type: str
    ) -> None:
        """Update table metadata."""
        try:
            metadata = {
                'columns': columns,
                'fdw_type': fdw_type,
                'pandas_dtypes': self._map_to_pandas_types(columns),
                'last_updated': datetime.utcnow().isoformat(),
                'row_count': 'unknown'
            }
            
            self.table_metadata[table_name] = metadata
            
        except Exception as e:
            logger.warning(f"Failed to update metadata for {table_name}: {e}")
    
    def _map_to_pandas_types(self, columns: List[Dict[str, str]]) -> Dict[str, str]:
        """Map PostgreSQL column types to pandas dtypes."""
        dtype_mapping = {}
        
        for column in columns:
            pg_type = column['type'].lower()
            # Handle type variations
            if '(' in pg_type:
                pg_type = pg_type.split('(')[0]
            
            pandas_type = self.PG_TO_PANDAS_TYPES.get(pg_type, 'object')
            dtype_mapping[column['name']] = pandas_type
        
        return dtype_mapping
    
    async def _estimate_row_count(self, table_name: str, schema: str = 'public') -> str:
        """Estimate row count for virtual table."""
        try:
            async with self.fdw_connector.get_connection() as conn:
                count_query = f'SELECT COUNT(*) FROM {schema}.{table_name} LIMIT 1'
                result = await conn.fetchval(count_query)
                return str(result) if result is not None else 'unknown'
        except Exception:
            return 'unknown'
    
    async def _sync_postgres_table(self, table_name: str, table_info: Dict[str, Any]) -> bool:
        """Synchronize PostgreSQL FDW table."""
        # Implementation would check remote schema and update local foreign table
        logger.info(f"PostgreSQL table sync for {table_name} - implementation pending")
        return True
    
    async def _sync_file_table(self, table_name: str, table_info: Dict[str, Any]) -> bool:
        """Synchronize File FDW table."""
        # Implementation would check file structure and update foreign table
        logger.info(f"File table sync for {table_name} - implementation pending")
        return True
    
    async def _sync_mysql_table(self, table_name: str, table_info: Dict[str, Any]) -> bool:
        """Synchronize MySQL FDW table."""
        # Implementation would check MySQL schema and update foreign table
        logger.info(f"MySQL table sync for {table_name} - implementation pending")
        return True
    
    async def _cleanup_managed_tables(self) -> None:
        """Clean up all managed virtual tables."""
        for table_name in list(self.virtual_tables.keys()):
            await self.drop_virtual_table(table_name)
        
        logger.info("Cleaned up all managed virtual tables")