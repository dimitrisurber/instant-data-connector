"""
PostgreSQL Foreign Data Wrapper (FDW) Management System

This module provides a comprehensive FDW management system for PostgreSQL,
enabling connection to various external data sources through FDW extensions.
"""

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote_plus

import asyncpg
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from .fdw_manager import FDWManager
from .lazy_query_builder import LazyQueryBuilder
from .secure_credentials import SecureCredentialManager, get_global_credential_manager

logger = logging.getLogger(__name__)


class PostgreSQLFDWConnector:
    """
    Main PostgreSQL FDW connector class for managing foreign data wrappers.
    
    This class provides high-level interface for:
    - Connection management with pooling
    - FDW extension installation and management
    - Foreign server and table creation
    - Lazy loading query execution
    - Query optimization for large datasets
    """
    
    SUPPORTED_EXTENSIONS = {
        'postgres_fdw': 'PostgreSQL Foreign Data Wrapper',
        'file_fdw': 'File Foreign Data Wrapper', 
        'mysql_fdw': 'MySQL Foreign Data Wrapper',
        'multicorn': 'Multicorn Python FDW'
    }
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'postgres',
        username: str = 'postgres',
        password: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        credential_manager: Optional[SecureCredentialManager] = None,
        **kwargs
    ):
        """
        Initialize PostgreSQL FDW connector.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            username: Username for connection
            password: Password (if None, will try credential manager)
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
            pool_timeout: Pool timeout in seconds
            credential_manager: Optional credential manager
            **kwargs: Additional connection parameters
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.credential_manager = credential_manager or get_global_credential_manager()
        
        # Get password from credential manager if not provided
        if password is None:
            try:
                password = self.credential_manager.get_credential(
                    f"postgres://{username}@{host}:{port}/{database}",
                    username
                )
            except Exception as e:
                logger.warning(f"Could not retrieve password from credential manager: {e}")
                password = os.getenv('POSTGRES_PASSWORD', '')
        
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.connection_kwargs = kwargs
        
        # Initialize components
        self._engine: Optional[Engine] = None
        self._async_pool: Optional[asyncpg.Pool] = None
        self.fdw_manager = FDWManager(fdw_connector=self)
        self.query_builder = LazyQueryBuilder()
        
        # Track managed foreign objects
        self.foreign_servers: Dict[str, Dict[str, Any]] = {}
        self.foreign_tables: Dict[str, Dict[str, Any]] = {}
        self.user_mappings: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized PostgreSQL FDW connector for {host}:{port}/{database}")
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        password_encoded = quote_plus(self.password) if self.password else ''
        auth_part = f"{self.username}:{password_encoded}@" if self.username else ""
        
        return f"postgresql://{auth_part}{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """Generate async PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    async def initialize(self) -> None:
        """Initialize connection pools and FDW manager."""
        try:
            # Initialize SQLAlchemy engine with connection pooling
            self._engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_pre_ping=True,
                **self.connection_kwargs
            )
            
            # Initialize async connection pool
            self._async_pool = await asyncpg.create_pool(
                self.async_connection_string,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=60
            )
            
            # Initialize FDW manager with our connection
            await self.fdw_manager.initialize(self._async_pool)
            
            logger.info("PostgreSQL FDW connector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL FDW connector: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self) -> None:
        """Clean up connections and resources."""
        try:
            if self._async_pool:
                await self._async_pool.close()
                self._async_pool = None
            
            if self._engine:
                self._engine.dispose()
                self._engine = None
            
            logger.info("PostgreSQL FDW connector cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def close(self) -> None:
        """Close connector and clean up resources (alias for cleanup)."""
        await self.cleanup()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get async database connection from pool."""
        if not self._async_pool:
            raise RuntimeError("Connector not initialized. Call initialize() first.")
        
        async with self._async_pool.acquire() as conn:
            yield conn
    
    def get_sync_connection(self):
        """Get synchronous database connection."""
        if not self._engine:
            raise RuntimeError("Connector not initialized. Call initialize() first.")
        
        return self._engine.connect()
    
    async def install_extension(self, extension_name: str, if_not_exists: bool = True) -> bool:
        """
        Install FDW extension.
        
        Args:
            extension_name: Name of extension to install
            if_not_exists: Whether to use IF NOT EXISTS clause
        
        Returns:
            True if installation successful
        """
        if extension_name not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {extension_name}")
        
        try:
            async with self.get_connection() as conn:
                result = await self.fdw_manager.install_extension(
                    conn, extension_name, if_not_exists
                )
                
                if result:
                    logger.info(f"Successfully installed extension: {extension_name}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to install extension {extension_name}: {e}")
            raise
    
    async def create_foreign_server(
        self,
        server_name: str,
        fdw_name: str,
        options: Dict[str, str],
        if_not_exists: bool = True
    ) -> bool:
        """
        Create foreign server.
        
        Args:
            server_name: Name of foreign server
            fdw_name: FDW extension name
            options: Server options (host, port, dbname, etc.)
            if_not_exists: Whether to skip if server exists
        
        Returns:
            True if creation successful
        """
        try:
            # Substitute environment variables in options
            processed_options = self._substitute_env_vars(options)
            
            async with self.get_connection() as conn:
                result = await self.fdw_manager.create_foreign_server(
                    conn, server_name, fdw_name, processed_options, if_not_exists
                )
                
                if result:
                    self.foreign_servers[server_name] = {
                        'fdw_name': fdw_name,
                        'options': processed_options
                    }
                    logger.info(f"Successfully created foreign server: {server_name}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to create foreign server {server_name}: {e}")
            raise
    
    async def create_user_mapping(
        self,
        server_name: str,
        user_name: str = 'current_user',
        options: Optional[Dict[str, str]] = None,
        if_not_exists: bool = True
    ) -> bool:
        """
        Create user mapping for foreign server.
        
        Args:
            server_name: Name of foreign server
            user_name: PostgreSQL user name (default: current_user)
            options: User mapping options (user, password, etc.)
            if_not_exists: Whether to skip if mapping exists
        
        Returns:
            True if creation successful
        """
        try:
            # Substitute environment variables and handle credentials
            processed_options = {}
            if options:
                processed_options = self._substitute_env_vars(options)
                
                # Handle secure credential retrieval
                if 'password' in processed_options and processed_options['password'].startswith('${'):
                    # Try to get from credential manager
                    credential_key = processed_options.get('user', server_name)
                    try:
                        secure_password = self.credential_manager.get_credential(
                            server_name, credential_key
                        )
                        if secure_password:
                            processed_options['password'] = secure_password
                    except Exception as e:
                        logger.warning(f"Could not retrieve secure password for {server_name}: {e}")
            
            async with self.get_connection() as conn:
                result = await self.fdw_manager.create_user_mapping(
                    conn, server_name, user_name, processed_options, if_not_exists
                )
                
                if result:
                    mapping_key = f"{server_name}:{user_name}"
                    self.user_mappings[mapping_key] = {
                        'server_name': server_name,
                        'user_name': user_name,
                        'options': {k: v for k, v in processed_options.items() if k != 'password'}
                    }
                    logger.info(f"Successfully created user mapping for {server_name}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to create user mapping for {server_name}: {e}")
            raise
    
    async def create_foreign_table(
        self,
        table_name: str,
        server_name: str,
        columns: List[Dict[str, str]],
        options: Dict[str, str],
        schema: str = 'public',
        if_not_exists: bool = True
    ) -> bool:
        """
        Create foreign table.
        
        Args:
            table_name: Name of foreign table
            server_name: Name of foreign server
            columns: List of column definitions [{'name': 'col1', 'type': 'text'}, ...]
            options: Table options (table_name, schema_name, etc.)
            schema: PostgreSQL schema name
            if_not_exists: Whether to skip if table exists
        
        Returns:
            True if creation successful
        """
        try:
            # Substitute environment variables in options
            processed_options = self._substitute_env_vars(options)
            
            async with self.get_connection() as conn:
                # Build config for FDWManager.create_foreign_table
                table_config = {
                    'table_name': table_name,
                    'server_name': server_name,
                    'columns': columns,
                    'options': processed_options,
                    'schema': schema
                }
                result = await self.fdw_manager.create_foreign_table(table_config)
                
                if result:
                    table_key = f"{schema}.{table_name}"
                    self.foreign_tables[table_key] = {
                        'server_name': server_name,
                        'columns': columns,
                        'options': processed_options,
                        'schema': schema
                    }
                    logger.info(f"Successfully created foreign table: {table_key}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to create foreign table {table_name}: {e}")
            raise
    
    async def setup_foreign_data_source(
        self,
        config: Dict[str, Any]
    ) -> bool:
        """
        Setup complete foreign data source from configuration.
        
        Args:
            config: Configuration dictionary with extension, server, user_mapping, and tables
        
        Returns:
            True if setup successful
        
        Example config:
        {
            "extension": "postgres_fdw",
            "server": {
                "name": "remote_pg",
                "options": {"host": "remote.example.com", "port": "5432", "dbname": "remotedb"}
            },
            "user_mapping": {
                "options": {"user": "remote_user", "password": "${REMOTE_PASSWORD}"}
            },
            "tables": [
                {
                    "name": "remote_users",
                    "columns": [{"name": "id", "type": "integer"}, {"name": "name", "type": "text"}],
                    "options": {"table_name": "users", "schema_name": "public"}
                }
            ]
        }
        """
        try:
            # Install extension
            extension_name = config['extension']
            await self.install_extension(extension_name)
            
            # Create foreign server
            server_config = config['server']
            await self.create_foreign_server(
                server_config['name'],
                extension_name,
                server_config['options']
            )
            
            # Create user mapping
            if 'user_mapping' in config:
                user_mapping_config = config['user_mapping']
                await self.create_user_mapping(
                    server_config['name'],
                    user_mapping_config.get('user', 'current_user'),
                    user_mapping_config.get('options', {})
                )
            
            # Create foreign tables
            for table_config in config.get('tables', []):
                await self.create_foreign_table(
                    table_config['name'],
                    server_config['name'],
                    table_config['columns'],
                    table_config['options'],
                    table_config.get('schema', 'public')
                )
            
            logger.info(f"Successfully setup foreign data source: {server_config['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup foreign data source: {e}")
            raise
    
    def execute_lazy_query(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        schema: str = 'public'
    ) -> pd.DataFrame:
        """
        Execute lazy loading query via pandas.read_sql.
        
        Args:
            table_name: Name of foreign table
            columns: List of columns to select
            filters: Filter conditions
            limit: Maximum number of rows
            offset: Number of rows to skip
            order_by: List of columns to order by
            schema: Schema name
        
        Returns:
            pandas DataFrame with results
        """
        try:
            # Build optimized query
            query = self.query_builder.build_select_query(
                table_name=f"{schema}.{table_name}",
                columns=columns,
                filters=filters,
                limit=limit,
                offset=offset,
                order_by=order_by
            )
            
            logger.debug(f"Executing lazy query: {query}")
            
            # Execute query using pandas for efficient data loading
            with self.get_sync_connection() as conn:
                df = pd.read_sql(text(query), conn)
                
                logger.info(f"Loaded {len(df)} rows from {schema}.{table_name}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to execute lazy query on {table_name}: {e}")
            raise
    
    def execute_aggregation_query(
        self,
        table_name: str,
        aggregations: Dict[str, str],
        group_by: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        having: Optional[Dict[str, Any]] = None,
        schema: str = 'public'
    ) -> pd.DataFrame:
        """
        Execute aggregation query with push-down optimization.
        
        Args:
            table_name: Name of foreign table
            aggregations: Aggregation functions {'column': 'function'}
            group_by: Columns to group by
            filters: WHERE clause filters
            having: HAVING clause filters
            schema: Schema name
        
        Returns:
            pandas DataFrame with aggregated results
        """
        try:
            # Build aggregation query
            query = self.query_builder.build_aggregation_query(
                table_name=f"{schema}.{table_name}",
                aggregations=aggregations,
                group_by=group_by,
                filters=filters,
                having=having
            )
            
            logger.debug(f"Executing aggregation query: {query}")
            
            # Execute query
            with self.get_sync_connection() as conn:
                df = pd.read_sql(text(query), conn)
                
                logger.info(f"Executed aggregation query on {schema}.{table_name}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to execute aggregation query on {table_name}: {e}")
            raise
    
    async def analyze_query_plan(self, query: str) -> Dict[str, Any]:
        """
        Analyze query execution plan.
        
        Args:
            query: SQL query to analyze
        
        Returns:
            Query plan analysis
        """
        try:
            plan_info = await self.query_builder.analyze_query_plan(query, self._async_pool)
            logger.debug(f"Analyzed query plan: {plan_info}")
            return plan_info
            
        except Exception as e:
            logger.error(f"Failed to analyze query plan: {e}")
            raise
    
    async def estimate_query_cost(self, query: str) -> float:
        """
        Estimate query execution cost.
        
        Args:
            query: SQL query to estimate
        
        Returns:
            Estimated cost
        """
        try:
            cost = await self.query_builder.estimate_query_cost(query, self._async_pool)
            logger.debug(f"Estimated query cost: {cost}")
            return cost
            
        except Exception as e:
            logger.error(f"Failed to estimate query cost: {e}")
            raise
    
    async def check_server_health(self, server_name: str) -> Dict[str, Any]:
        """
        Check health of foreign server.
        
        Args:
            server_name: Name of foreign server to check
        
        Returns:
            Health check results
        """
        try:
            async with self.get_connection() as conn:
                health_info = await self.fdw_manager.check_server_health(conn, server_name)
                logger.debug(f"Health check for {server_name}: {health_info}")
                return health_info
                
        except Exception as e:
            logger.error(f"Health check failed for {server_name}: {e}")
            raise
    
    async def cleanup_foreign_server(self, server_name: str, cascade: bool = False) -> bool:
        """
        Clean up foreign server and related objects.
        
        Args:
            server_name: Name of server to clean up
            cascade: Whether to cascade drop
        
        Returns:
            True if cleanup successful
        """
        try:
            async with self.get_connection() as conn:
                result = await self.fdw_manager.cleanup_server(conn, server_name, cascade)
                
                if result:
                    # Remove from tracking
                    if server_name in self.foreign_servers:
                        del self.foreign_servers[server_name]
                    
                    # Remove related user mappings
                    mappings_to_remove = [
                        key for key in self.user_mappings.keys()
                        if key.startswith(f"{server_name}:")
                    ]
                    for mapping_key in mappings_to_remove:
                        del self.user_mappings[mapping_key]
                    
                    # Remove related foreign tables
                    tables_to_remove = [
                        key for key, info in self.foreign_tables.items()
                        if info['server_name'] == server_name
                    ]
                    for table_key in tables_to_remove:
                        del self.foreign_tables[table_key]
                    
                    logger.info(f"Successfully cleaned up foreign server: {server_name}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to cleanup foreign server {server_name}: {e}")
            raise
    
    def _substitute_env_vars(self, options: Dict[str, str]) -> Dict[str, str]:
        """
        Substitute environment variables in option values.
        
        Args:
            options: Dictionary with option values that may contain ${VAR} patterns
        
        Returns:
            Dictionary with environment variables substituted
        """
        substituted = {}
        env_var_pattern = re.compile(r'\$\{([^}]+)\}')
        
        for key, value in options.items():
            if isinstance(value, str):
                # Find all environment variable references
                matches = env_var_pattern.findall(value)
                substituted_value = value
                
                for var_name in matches:
                    env_value = os.getenv(var_name, '')
                    substituted_value = substituted_value.replace(f"${{{var_name}}}", env_value)
                
                substituted[key] = substituted_value
            else:
                substituted[key] = value
        
        return substituted
    
    def get_managed_objects(self) -> Dict[str, Any]:
        """
        Get summary of managed foreign objects.
        
        Returns:
            Dictionary with managed servers, tables, and user mappings
        """
        return {
            'foreign_servers': list(self.foreign_servers.keys()),
            'foreign_tables': list(self.foreign_tables.keys()),
            'user_mappings': list(self.user_mappings.keys()),
            'supported_extensions': list(self.SUPPORTED_EXTENSIONS.keys())
        }