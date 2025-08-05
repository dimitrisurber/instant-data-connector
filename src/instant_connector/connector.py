"""
Modern FDW-based InstantDataConnector

This module provides the new InstantDataConnector class that uses PostgreSQL Foreign Data Wrappers
for efficient data access and aggregation. It replaces the legacy direct extraction approach
with a more scalable FDW-based architecture.
"""

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncGenerator
import pandas as pd
import numpy as np
from datetime import datetime

from .postgresql_fdw_manager import PostgreSQLFDWConnector
from .virtual_table_manager import VirtualTableManager
from .config_parser import ConfigParser
from .lazy_query_builder import LazyQueryBuilder
from .pickle_manager import PickleManager
from .secure_credentials import SecureCredentialManager
from .secure_serializer import SecureSerializer

logger = logging.getLogger(__name__)


class InstantDataConnector:
    """
    Modern FDW-based data connector for efficient data aggregation and processing.
    
    This class provides a high-level interface for connecting to multiple data sources
    using PostgreSQL Foreign Data Wrappers, enabling efficient cross-database queries
    and data aggregation without moving data.
    
    Features:
    - PostgreSQL FDW-based connectivity
    - Lazy loading and query optimization
    - Configuration-driven setup
    - Async and sync operation support
    - Backward compatibility with serialization
    - Comprehensive error handling and health checks
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        postgres_config: Optional[Dict[str, Any]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        enable_caching: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        credential_manager: Optional[SecureCredentialManager] = None
    ):
        """
        Initialize the FDW-based data connector.
        
        Args:
            config_path: Path to YAML configuration file
            postgres_config: PostgreSQL connection configuration
            schema_path: Path to JSON schema for configuration validation
            enable_caching: Whether to enable result caching
            cache_dir: Directory for cache storage
            credential_manager: Optional credential manager instance
        """
        self.config_path = Path(config_path) if config_path else None
        self.postgres_config = postgres_config or {}
        self.enable_caching = enable_caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "cache"
        
        # Initialize components
        self.credential_manager = credential_manager or SecureCredentialManager()
        self.config_parser = ConfigParser(schema_path, self.credential_manager)
        self.fdw_connector: Optional[PostgreSQLFDWConnector] = None
        self.virtual_table_manager: Optional[VirtualTableManager] = None
        self.query_builder = LazyQueryBuilder()
        
        # State management
        self.config: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        self._available_tables: Dict[str, Dict[str, Any]] = {}
        self._table_schemas: Dict[str, List[Dict[str, Any]]] = {}
        
        # Secure caching with JSON serialization instead of unsafe pickle
        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.secure_serializer = SecureSerializer(compression='lz4')
            # Keep pickle_manager for backward compatibility but don't use it for new operations
            self.pickle_manager = None
        else:
            self.secure_serializer = None
            self.pickle_manager = None
        
        logger.info("Initialized InstantDataConnector with FDW support")
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given cache key."""
        # Create a safe filename from the cache key
        import hashlib
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json.lz4"
    
    def _save_to_cache(self, data: Any, cache_key: str, ttl: Optional[int] = None) -> None:
        """Save data to cache with optional TTL using secure serialization."""
        if not self.enable_caching or not self.secure_serializer:
            return
        
        try:
            cache_path = self._get_cache_path(cache_key)
            cache_data = {
                'data': data,
                'timestamp': pd.Timestamp.now().isoformat(),  # Convert to ISO string for JSON
                'ttl': ttl
            }
            
            # Use secure serializer instead of unsafe pickle
            self.secure_serializer.serialize_datasets(cache_data, cache_path)
            
            logger.debug(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache, checking TTL using secure deserialization."""
        if not self.enable_caching or not self.secure_serializer:
            return None
        
        try:
            cache_path = self._get_cache_path(cache_key)
            if not cache_path.exists():
                return None
            
            # Use secure deserializer instead of unsafe pickle
            cache_data = self.secure_serializer.deserialize_datasets(cache_path)
            
            # Check TTL if specified
            if cache_data.get('ttl'):
                timestamp = pd.Timestamp.fromisoformat(cache_data['timestamp'])
                age_seconds = (pd.Timestamp.now() - timestamp).total_seconds()
                if age_seconds > cache_data['ttl']:
                    # Cache expired, remove file
                    cache_path.unlink(missing_ok=True)
                    return None
            
            logger.debug(f"Loaded data from cache: {cache_key}")
            return cache_data['data']
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    async def setup_fdw_infrastructure(
        self,
        force_refresh: bool = False,
        validate_connections: bool = True
    ) -> bool:
        """
        Set up the FDW infrastructure including extensions, servers, and virtual tables.
        
        Args:
            force_refresh: Whether to force recreation of existing infrastructure
            validate_connections: Whether to validate all connections after setup
        
        Returns:
            True if setup was successful
        """
        try:
            logger.info("Setting up FDW infrastructure")
            
            # Load configuration if not already loaded
            if not self.config and self.config_path:
                self.config = self.config_parser.parse_config(self.config_path)
                
                # Validate credentials for security
                credential_errors = self.config_parser.validate_required_credentials(self.config)
                if credential_errors:
                    error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in credential_errors)
                    logger.error(error_msg)
                    raise ValueError(f"Configuration validation failed: {len(credential_errors)} errors found. Check logs for details.")
            
            if not self.config:
                raise ValueError("No configuration available. Provide config_path or call load_config first.")
            
            # Initialize FDW connector if not already done
            if not self.fdw_connector:
                self.fdw_connector = PostgreSQLFDWConnector(**self.postgres_config)
                await self.fdw_connector.initialize()
            
            # Set up foreign data sources from configuration
            await self.fdw_connector.setup_foreign_data_source(self.config)
            
            # Initialize virtual table manager
            self.virtual_table_manager = VirtualTableManager(
                self.fdw_connector,
                self.config_parser
            )
            
            # Create virtual tables from configuration
            await self.virtual_table_manager.create_virtual_tables_from_config(
                self.config,
                force_recreate=force_refresh
            )
            
            # Validate connections if requested
            if validate_connections:
                health_status = await self.health_check()
                if not health_status['overall_healthy']:
                    logger.warning("Some connections failed validation during setup")
            
            # Update available tables cache
            await self._refresh_table_cache()
            
            self.is_initialized = True
            logger.info("FDW infrastructure setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup FDW infrastructure: {e}")
            raise
    
    def load_config(
        self,
        config_path: Union[str, Path],
        validate: bool = True,
        migrate_legacy: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            validate: Whether to validate configuration against schema
            migrate_legacy: Whether to migrate legacy configuration formats
        
        Returns:
            Loaded configuration dictionary
        """
        try:
            self.config_path = Path(config_path)
            self.config = self.config_parser.parse_config(
                config_path,
                validate=validate,
                migrate_legacy=migrate_legacy
            )
            
            # Validate credentials for security
            credential_errors = self.config_parser.validate_required_credentials(self.config)
            if credential_errors:
                error_msg = "Configuration credential validation failed:\n" + "\n".join(f"  - {error}" for error in credential_errors)
                logger.error(error_msg)
                if any("SECURITY WARNING" in error or "Hardcoded credential" in error for error in credential_errors):
                    raise ValueError("SECURITY ERROR: Hardcoded credentials detected in configuration. Use environment variables or credential manager.")
                else:
                    logger.warning(f"Configuration has {len(credential_errors)} credential issues. Set missing environment variables.")
            logger.info(f"Loaded configuration from {config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def list_available_tables(self, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        List all available tables across all configured data sources.
        
        Args:
            refresh: Whether to refresh the table cache
        
        Returns:
            Dictionary mapping table names to their metadata
        """
        try:
            if not self.is_initialized:
                await self.setup_fdw_infrastructure()
            
            if refresh or not self._available_tables:
                await self._refresh_table_cache()
            
            return self._available_tables.copy()
            
        except Exception as e:
            logger.error(f"Failed to list available tables: {e}")
            raise
    
    async def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get the schema information for a specific table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            List of column definitions with type information
        """
        try:
            if not self.is_initialized:
                await self.setup_fdw_infrastructure()
            
            if table_name not in self._table_schemas:
                # Query PostgreSQL information_schema for column information
                query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    ordinal_position
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """
                
                async with self.fdw_connector.get_connection() as conn:
                    result = await conn.fetch(query, table_name)
                    
                schema = []
                for row in result:
                    schema.append({
                        'name': row['column_name'],
                        'type': row['data_type'],
                        'nullable': row['is_nullable'] == 'YES',
                        'default': row['column_default'],
                        'max_length': row['character_maximum_length'],
                        'precision': row['numeric_precision'],
                        'scale': row['numeric_scale'],
                        'position': row['ordinal_position']
                    })
                
                self._table_schemas[table_name] = schema
            
            return self._table_schemas[table_name].copy()
            
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            raise
    
    async def execute_query(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        return_dataframe: bool = True,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute a SQL query and return results as DataFrame or list of dictionaries.
        
        Args:
            sql: SQL query to execute
            params: Optional query parameters
            return_dataframe: Whether to return pandas DataFrame (True) or list of dicts (False)
            cache_key: Optional cache key for result caching
            cache_ttl: Cache time-to-live in seconds
        
        Returns:
            Query results as DataFrame or list of dictionaries
        """
        try:
            if not self.is_initialized:
                await self.setup_fdw_infrastructure()
            
            # Check cache first if caching is enabled
            if self.enable_caching and cache_key:
                cached_result = self._load_from_cache(cache_key)
                if cached_result is not None:
                    logger.debug(f"Returning cached result for key: {cache_key}")
                    return cached_result
            
            logger.debug(f"Executing query: {sql[:100]}...")
            
            async with self.fdw_connector.get_connection() as conn:
                if params:
                    result = await conn.fetch(sql, *params)
                else:
                    result = await conn.fetch(sql)
            
            if return_dataframe:
                # Convert to pandas DataFrame
                if result:
                    df = pd.DataFrame([dict(row) for row in result])
                else:
                    df = pd.DataFrame()
                
                # Cache result if caching is enabled
                if self.enable_caching and cache_key:
                    self._save_to_cache(df, cache_key, ttl=cache_ttl)
                
                return df
            else:
                # Return as list of dictionaries
                result_list = [dict(row) for row in result]
                
                # Cache result if caching is enabled
                if self.enable_caching and cache_key:
                    self._save_to_cache(result_list, cache_key, ttl=cache_ttl)
                
                return result_list
            
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise
    
    async def lazy_load_table(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        optimize_query: bool = True
    ) -> pd.DataFrame:
        """
        Lazy load data from a table with optional filtering and optimization.
        
        Args:
            table_name: Name of the table to query
            filters: Optional filters to apply (e.g., {'column': 'value'})
            columns: Optional list of columns to select
            limit: Maximum number of rows to return
            offset: Number of rows to skip
            order_by: Column(s) to order by
            optimize_query: Whether to apply query optimization
        
        Returns:
            DataFrame with the requested data
        """
        try:
            if not self.is_initialized:
                await self.setup_fdw_infrastructure()
            
            # Build optimized query
            query_info = self.query_builder.build_select_query(
                table_name,
                columns=columns,
                filters=filters,
                limit=limit,
                offset=offset,
                order_by=order_by
            )
            
            # Apply optimization if requested
            if optimize_query:
                query_info = await self.query_builder.optimize_query(
                    query_info,
                    self.fdw_connector
                )
            
            # Generate cache key for this query
            cache_key = None
            if self.enable_caching:
                cache_key = self.query_builder.generate_cache_key(query_info)
            
            # Execute query
            result = await self.execute_query(
                query_info['sql'],
                query_info.get('params'),
                return_dataframe=True,
                cache_key=cache_key,
                cache_ttl=3600  # 1 hour default TTL
            )
            
            logger.info(f"Loaded {len(result)} rows from table {table_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to lazy load table {table_name}: {e}")
            raise
    
    async def refresh_virtual_tables(self, table_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Refresh virtual table definitions and metadata.
        
        Args:
            table_names: Optional list of specific tables to refresh (all if None)
        
        Returns:
            Dictionary mapping table names to refresh success status
        """
        try:
            if not self.is_initialized:
                await self.setup_fdw_infrastructure()
            
            if not self.virtual_table_manager:
                raise RuntimeError("Virtual table manager not initialized")
            
            # Refresh specified tables or all tables
            if table_names:
                tables_to_refresh = table_names
            else:
                available_tables = await self.list_available_tables()
                tables_to_refresh = list(available_tables.keys())
            
            refresh_results = {}
            for table_name in tables_to_refresh:
                try:
                    await self.virtual_table_manager.refresh_virtual_table(table_name)
                    refresh_results[table_name] = True
                    logger.debug(f"Successfully refreshed table: {table_name}")
                except Exception as e:
                    logger.error(f"Failed to refresh table {table_name}: {e}")
                    refresh_results[table_name] = False
            
            # Update table cache after refresh
            await self._refresh_table_cache()
            
            logger.info(f"Refreshed {sum(refresh_results.values())} of {len(refresh_results)} tables")
            return refresh_results
            
        except Exception as e:
            logger.error(f"Failed to refresh virtual tables: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all connections and components.
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_healthy': True,
            'components': {}
        }
        
        try:
            # Check PostgreSQL FDW connector
            if self.fdw_connector:
                try:
                    fdw_healthy = await self.fdw_connector.health_check()
                    health_status['components']['fdw_connector'] = {
                        'healthy': fdw_healthy,
                        'details': 'FDW connector operational' if fdw_healthy else 'FDW connector failed'
                    }
                except Exception as e:
                    health_status['components']['fdw_connector'] = {
                        'healthy': False,
                        'details': f'FDW connector error: {e}'
                    }
                    health_status['overall_healthy'] = False
            else:
                health_status['components']['fdw_connector'] = {
                    'healthy': False,
                    'details': 'FDW connector not initialized'
                }
                health_status['overall_healthy'] = False
            
            # Check virtual table manager
            if self.virtual_table_manager:
                try:
                    # Test basic functionality
                    tables = await self.list_available_tables()
                    health_status['components']['virtual_tables'] = {
                        'healthy': True,
                        'details': f'Virtual table manager operational with {len(tables)} tables'
                    }
                except Exception as e:
                    health_status['components']['virtual_tables'] = {
                        'healthy': False,
                        'details': f'Virtual table manager error: {e}'
                    }
                    health_status['overall_healthy'] = False
            else:
                health_status['components']['virtual_tables'] = {
                    'healthy': False,
                    'details': 'Virtual table manager not initialized'
                }
                health_status['overall_healthy'] = False
            
            # Check configuration parser
            if self.config:
                health_status['components']['configuration'] = {
                    'healthy': True,
                    'details': 'Configuration loaded and validated'
                }
            else:
                health_status['components']['configuration'] = {
                    'healthy': False,
                    'details': 'No configuration loaded'
                }
                health_status['overall_healthy'] = False
            
            # Check caching system
            if self.enable_caching and self.secure_serializer:
                try:
                    # Test cache write/read
                    test_key = 'health_check_test'
                    test_data = {'test': True, 'timestamp': datetime.utcnow().isoformat()}
                    self._save_to_cache(test_data, test_key, ttl=60)
                    retrieved_data = self._load_from_cache(test_key)
                    
                    if retrieved_data and retrieved_data.get('test'):
                        health_status['components']['caching'] = {
                            'healthy': True,
                            'details': 'Caching system operational'
                        }
                    else:
                        health_status['components']['caching'] = {
                            'healthy': False,
                            'details': 'Cache read/write test failed'
                        }
                        health_status['overall_healthy'] = False
                except Exception as e:
                    health_status['components']['caching'] = {
                        'healthy': False,
                        'details': f'Caching system error: {e}'
                    }
                    health_status['overall_healthy'] = False
            else:
                health_status['components']['caching'] = {
                    'healthy': True,
                    'details': 'Caching disabled'
                }
            
            logger.info(f"Health check completed. Overall healthy: {health_status['overall_healthy']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['overall_healthy'] = False
            health_status['error'] = str(e)
            return health_status
    
    # Backward compatibility methods with deprecation warnings
    
    def save_connector(self, file_path: str) -> bool:
        """
        Save connector state for backward compatibility.
        
        Args:
            file_path: Path to save the connector state
        
        Returns:
            True if successful
        """
        warnings.warn(
            "save_connector is deprecated. Use the new caching system instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not self.secure_serializer:
            logger.warning("Caching is disabled, cannot save connector state")
            return False
        
        try:
            connector_state = {
                'config': self.config,
                'postgres_config': self.postgres_config,
                'config_path': str(self.config_path) if self.config_path else None,
                'is_initialized': self.is_initialized,
                'available_tables': self._available_tables,
                'version': '0.2.0'
            }
            
            self._save_to_cache(connector_state, f"connector_state_{file_path}")
            logger.info(f"Saved connector state to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save connector state: {e}")
            return False
    
    @classmethod
    def load_connector(cls, file_path: str) -> 'InstantDataConnector':
        """
        Load connector state for backward compatibility.
        
        Args:
            file_path: Path to load the connector state from
        
        Returns:
            Loaded InstantDataConnector instance
        """
        warnings.warn(
            "load_connector is deprecated. Create new instance and call load_config instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Create a temporary instance to load the state
            temp_instance = cls(enable_caching=True)
            connector_state = temp_instance._load_from_cache(f"connector_state_{file_path}")
            
            if not connector_state:
                raise ValueError(f"No connector state found at {file_path}")
            
            # Create new instance with loaded state
            instance = cls(
                config_path=connector_state.get('config_path'),
                postgres_config=connector_state.get('postgres_config', {}),
                enable_caching=True
            )
            
            instance.config = connector_state.get('config')
            instance._available_tables = connector_state.get('available_tables', {})
            
            logger.info(f"Loaded connector state from {file_path}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load connector state: {e}")
            raise
    
    # Migration helpers for existing users
    
    async def migrate_from_legacy(
        self,
        legacy_connector,
        setup_fdw: bool = True,
        preserve_cache: bool = True
    ) -> bool:
        """
        Migrate from legacy InstantDataConnector to FDW-based system.
        
        Args:
            legacy_connector: Legacy InstantDataConnector instance
            setup_fdw: Whether to set up FDW infrastructure
            preserve_cache: Whether to preserve existing cache data
        
        Returns:
            True if migration was successful
        """
        try:
            logger.info("Starting migration from legacy connector")
            
            # Preserve cache data if requested
            if preserve_cache and hasattr(legacy_connector, 'pickle_manager'):
                # Copy cache directory contents if possible
                legacy_cache_dir = getattr(legacy_connector.pickle_manager, 'storage_dir', None)
                if legacy_cache_dir and Path(legacy_cache_dir).exists():
                    import shutil
                    shutil.copytree(legacy_cache_dir, self.cache_dir, dirs_exist_ok=True)
                    logger.info("Preserved legacy cache data")
            
            # Set up FDW infrastructure if requested
            if setup_fdw:
                await self.setup_fdw_infrastructure()
            
            logger.info("Migration from legacy connector completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate from legacy connector: {e}")
            return False
    
    async def _refresh_table_cache(self) -> None:
        """Refresh the internal table cache."""
        try:
            if not self.virtual_table_manager:
                return
            
            # Get list of virtual tables
            virtual_tables = await self.virtual_table_manager.list_virtual_tables()
            
            self._available_tables = {}
            for table_info in virtual_tables:
                table_name = table_info['table_name']
                self._available_tables[table_name] = {
                    'name': table_name,
                    'schema': table_info.get('schema', 'public'),
                    'source_type': table_info.get('source_type', 'unknown'),
                    'description': table_info.get('description', ''),
                    'created_at': table_info.get('created_at'),
                    'updated_at': table_info.get('updated_at')
                }
            
            logger.debug(f"Refreshed table cache with {len(self._available_tables)} tables")
            
        except Exception as e:
            logger.error(f"Failed to refresh table cache: {e}")
    
    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        try:
            if self.fdw_connector:
                await self.fdw_connector.close()
                
            logger.info("InstantDataConnector closed successfully")
            
        except Exception as e:
            logger.error(f"Error during connector cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Synchronous wrapper for backward compatibility
class SyncInstantDataConnector:
    """
    Synchronous wrapper for InstantDataConnector for backward compatibility.
    
    This class provides synchronous methods that wrap the async methods
    of the main InstantDataConnector class.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as async version."""
        self._async_connector = InstantDataConnector(*args, **kwargs)
        self._loop = None
    
    def _get_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop
    
    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        loop = self._get_loop()
        if loop.is_running():
            # If loop is already running, we need to use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    
    def setup_fdw_infrastructure(self, **kwargs) -> bool:
        """Synchronous version of setup_fdw_infrastructure."""
        return self._run_async(self._async_connector.setup_fdw_infrastructure(**kwargs))
    
    def list_available_tables(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Synchronous version of list_available_tables."""
        return self._run_async(self._async_connector.list_available_tables(**kwargs))
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Synchronous version of get_table_schema."""
        return self._run_async(self._async_connector.get_table_schema(table_name))
    
    def execute_query(self, sql: str, **kwargs) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Synchronous version of execute_query."""
        return self._run_async(self._async_connector.execute_query(sql, **kwargs))
    
    def lazy_load_table(self, table_name: str, **kwargs) -> pd.DataFrame:
        """Synchronous version of lazy_load_table."""
        return self._run_async(self._async_connector.lazy_load_table(table_name, **kwargs))
    
    def refresh_virtual_tables(self, **kwargs) -> Dict[str, bool]:
        """Synchronous version of refresh_virtual_tables."""
        return self._run_async(self._async_connector.refresh_virtual_tables(**kwargs))
    
    def health_check(self) -> Dict[str, Any]:
        """Synchronous version of health_check."""
        return self._run_async(self._async_connector.health_check())
    
    def load_config(self, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous version of load_config."""
        return self._async_connector.load_config(*args, **kwargs)
    
    def close(self) -> None:
        """Close connections and cleanup."""
        self._run_async(self._async_connector.close())
        if self._loop and not self._loop.is_closed():
            self._loop.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# For backward compatibility, provide both sync and async versions
__all__ = ['InstantDataConnector', 'SyncInstantDataConnector']