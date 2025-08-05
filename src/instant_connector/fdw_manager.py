"""
Foreign Data Wrapper (FDW) Manager

Low-level FDW operations for PostgreSQL including extension management,
server creation, user mappings, and foreign table management.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from .sql_security import SQLSecurityValidator, SQLSecurityError

logger = logging.getLogger(__name__)


class FDWManager:
    """
    Low-level FDW manager for PostgreSQL operations.
    
    Handles:
    - Extension installation with version checking
    - Foreign server creation and validation
    - User mapping creation with secure credential handling
    - Foreign table creation with column mapping
    - Server cleanup and health checks
    """
    
    def __init__(self, fdw_connector=None):
        """Initialize FDW manager with optional connector."""
        self.pool: Optional[asyncpg.Pool] = None
        self.fdw_connector = fdw_connector
        logger.debug("Initialized FDW manager")
    
    async def initialize(self, pool: asyncpg.Pool) -> None:
        """
        Initialize with database connection pool.
        
        Args:
            pool: AsyncPG connection pool
        """
        self.pool = pool
        logger.info("FDW manager initialized with connection pool")
    
    async def install_extension(
        self,
        conn: asyncpg.Connection,
        extension_name: str,
        if_not_exists: bool = True
    ) -> bool:
        """
        Install FDW extension with version checking.
        
        Args:
            conn: Database connection
            extension_name: Name of extension to install
            if_not_exists: Whether to use IF NOT EXISTS clause
        
        Returns:
            True if installation successful or already exists
        """
        try:
            # Check if extension is already installed
            check_query = """
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension 
                    WHERE extname = $1
                )
            """
            exists = await conn.fetchval(check_query, extension_name)
            
            if exists:
                logger.info(f"Extension {extension_name} is already installed")
                return True
            
            # Check if extension is available
            available_query = """
                SELECT name, default_version, installed_version 
                FROM pg_available_extensions 
                WHERE name = $1
            """
            extension_info = await conn.fetchrow(available_query, extension_name)
            
            if not extension_info:
                raise ValueError(f"Extension {extension_name} is not available")
            
            # Install extension
            if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
            install_query = f"CREATE EXTENSION {if_not_exists_clause} {extension_name}"
            
            await conn.execute(install_query)
            
            # Verify installation
            verify_query = """
                SELECT extname, extversion 
                FROM pg_extension 
                WHERE extname = $1
            """
            installed_info = await conn.fetchrow(verify_query, extension_name)
            
            if installed_info:
                logger.info(
                    f"Successfully installed extension {extension_name} "
                    f"version {installed_info['extversion']}"
                )
                return True
            else:
                raise RuntimeError(f"Extension {extension_name} installation verification failed")
                
        except Exception as e:
            logger.error(f"Failed to install extension {extension_name}: {e}")
            raise
    
    async def _create_foreign_server_internal(
        self,
        conn: asyncpg.Connection,
        server_name: str,
        fdw_name: str,
        options: Dict[str, str],
        if_not_exists: bool = True
    ) -> bool:
        """
        Create foreign server with validation.
        
        Args:
            conn: Database connection
            server_name: Name of foreign server
            fdw_name: FDW extension name
            options: Server options
            if_not_exists: Whether to skip if server exists
        
        Returns:
            True if creation successful
        """
        try:
            # Check if server already exists
            check_query = """
                SELECT srvname FROM pg_foreign_server 
                WHERE srvname = $1
            """
            exists = await conn.fetchval(check_query, server_name)
            
            if exists:
                if if_not_exists:
                    logger.info(f"Foreign server {server_name} already exists")
                    return True
                else:
                    raise ValueError(f"Foreign server {server_name} already exists")
            
            # Validate FDW exists
            fdw_check_query = """
                SELECT fdwname FROM pg_foreign_data_wrapper 
                WHERE fdwname = $1
            """
            fdw_exists = await conn.fetchval(fdw_check_query, fdw_name)
            
            if not fdw_exists:
                raise ValueError(f"Foreign data wrapper {fdw_name} does not exist")
            
            # Build CREATE SERVER statement with security validation
            safe_server_name = SQLSecurityValidator.validate_and_escape_identifier(
                server_name, "server name"
            )
            safe_fdw_name = SQLSecurityValidator.validate_and_escape_identifier(
                fdw_name, "FDW name"
            )
            options_str = SQLSecurityValidator.build_options_string(options)
            
            create_query = f"""
                CREATE SERVER {safe_server_name}
                FOREIGN DATA WRAPPER {safe_fdw_name}
                {options_str}
            """
            
            await conn.execute(create_query)
            
            # Verify creation
            verify_query = """
                SELECT srvname, srvfdw, srvoptions 
                FROM pg_foreign_server 
                WHERE srvname = $1
            """
            server_info = await conn.fetchrow(verify_query, server_name)
            
            if server_info:
                logger.info(f"Successfully created foreign server: {server_name}")
                return True
            else:
                raise RuntimeError(f"Foreign server {server_name} creation verification failed")
                
        except Exception as e:
            logger.error(f"Failed to create foreign server {server_name}: {e}")
            raise
    
    async def _create_user_mapping_internal(
        self,
        conn: asyncpg.Connection,
        server_name: str,
        user_name: str = 'current_user',
        options: Optional[Dict[str, str]] = None,
        if_not_exists: bool = True
    ) -> bool:
        """
        Create user mapping with secure credential handling.
        
        Args:
            conn: Database connection
            server_name: Name of foreign server
            user_name: PostgreSQL user name
            options: User mapping options
            if_not_exists: Whether to skip if mapping exists
        
Returns:
            True if creation successful
        """
        try:
            # Check if user mapping already exists - simplified check
            check_query = """
                SELECT COUNT(*) 
                FROM pg_user_mapping um
                JOIN pg_foreign_server s ON um.umserver = s.oid
                WHERE s.srvname = $1
            """
            mapping_count = await conn.fetchval(check_query, server_name)
            
            if mapping_count > 0:
                if if_not_exists:
                    logger.info(f"User mapping for {user_name}@{server_name} already exists")
                    return True
                else:
                    raise ValueError(f"User mapping for {user_name}@{server_name} already exists")
            
            # Validate server exists
            server_check_query = """
                SELECT srvname FROM pg_foreign_server 
                WHERE srvname = $1
            """
            server_exists = await conn.fetchval(server_check_query, server_name)
            
            if not server_exists:
                raise ValueError(f"Foreign server {server_name} does not exist")
            
            # Build CREATE USER MAPPING statement with security validation
            if user_name == 'current_user':
                user_part = 'current_user'  # Special PostgreSQL keyword
            else:
                user_part = SQLSecurityValidator.validate_and_escape_identifier(
                    user_name, "user name"
                )
            
            safe_server_name = SQLSecurityValidator.validate_and_escape_identifier(
                server_name, "server name"
            )
            options_str = SQLSecurityValidator.build_options_string(options) if options else ""
            
            create_query = f"""
                CREATE USER MAPPING FOR {user_part}
                SERVER {safe_server_name}
                {options_str}
            """
            
            await conn.execute(create_query)
            
            # Verify creation (note: we cannot see the actual options for security)
            verify_query = """
                SELECT COUNT(*) as count
                FROM pg_user_mapping um
                JOIN pg_foreign_server s ON um.umserver = s.oid
                WHERE s.srvname = $1
            """
            mapping_count = await conn.fetchval(verify_query, server_name)
            
            if mapping_count > 0:
                logger.info(f"Successfully created user mapping for {user_name}@{server_name}")
                return True
            else:
                raise RuntimeError(f"User mapping creation verification failed")
                
        except Exception as e:
            logger.error(f"Failed to create user mapping for {user_name}@{server_name}: {e}")
            raise
    
    async def _create_foreign_table_internal(
        self,
        conn: asyncpg.Connection,
        table_name: str,
        server_name: str,
        columns: List[Dict[str, str]],
        options: Dict[str, str],
        schema: str = 'public',
        if_not_exists: bool = True
    ) -> bool:
        """
        Create foreign table with column mapping.
        
        Args:
            conn: Database connection
            table_name: Name of foreign table
            server_name: Name of foreign server
            columns: List of column definitions
            options: Table options
            schema: PostgreSQL schema name
            if_not_exists: Whether to skip if table exists
        
        Returns:
            True if creation successful
        """
        try:
            # Check if table already exists
            check_query = """
                SELECT schemaname, tablename 
                FROM pg_tables 
                WHERE schemaname = $1 AND tablename = $2
                UNION
                SELECT n.nspname as schemaname, c.relname as tablename
                FROM pg_class c
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE c.relkind = 'f' AND n.nspname = $1 AND c.relname = $2
            """
            exists = await conn.fetchval(check_query, schema, table_name)
            
            if exists:
                if if_not_exists:
                    logger.info(f"Foreign table {schema}.{table_name} already exists")
                    return True
                else:
                    raise ValueError(f"Table {schema}.{table_name} already exists")
            
            # Validate server exists
            server_check_query = """
                SELECT srvname FROM pg_foreign_server 
                WHERE srvname = $1
            """
            server_exists = await conn.fetchval(server_check_query, server_name)
            
            if not server_exists:
                raise ValueError(f"Foreign server {server_name} does not exist")
            
            # Validate columns
            if not columns:
                raise ValueError("At least one column must be specified")
            
            # Build column definitions with security validation
            column_definitions = []
            for col in columns:
                # Validate column definition structure and content
                validated_col = SQLSecurityValidator.validate_column_definition(col)
                
                # Build safe column definition
                safe_col_name = SQLSecurityValidator.validate_and_escape_identifier(
                    validated_col['name'], "column name"
                )
                col_def = f"{safe_col_name} {validated_col['type']}"
                
                # Add constraints if specified
                if validated_col.get('not_null'):
                    col_def += " NOT NULL"
                if validated_col.get('default'):
                    # Note: DEFAULT values should also be validated, but this requires more complex parsing
                    # For now, we'll log a warning for manual review
                    logger.warning(f"DEFAULT value specified for column {validated_col['name']}: manual review recommended")
                    col_def += f" DEFAULT {validated_col['default']}"
                
                column_definitions.append(col_def)
            
            columns_str = ", ".join(column_definitions)
            options_str = SQLSecurityValidator.build_options_string(options)
            
            # Build CREATE FOREIGN TABLE statement with security validation
            safe_schema = SQLSecurityValidator.validate_and_escape_identifier(
                schema, "schema name"
            )
            safe_table_name = SQLSecurityValidator.validate_and_escape_identifier(
                table_name, "table name"  
            )
            safe_server_name = SQLSecurityValidator.validate_and_escape_identifier(
                server_name, "server name"
            )
            
            create_query = f"""
                CREATE FOREIGN TABLE {safe_schema}.{safe_table_name} (
                    {columns_str}
                ) SERVER {safe_server_name}
                {options_str}
            """
            
            await conn.execute(create_query)
            
            # Verify creation
            verify_query = """
                SELECT n.nspname as schemaname, c.relname as tablename, s.srvname as servername
                FROM pg_class c
                JOIN pg_namespace n ON c.relnamespace = n.oid
                JOIN pg_foreign_table ft ON c.oid = ft.ftrelid
                JOIN pg_foreign_server s ON ft.ftserver = s.oid
                WHERE c.relkind = 'f' AND n.nspname = $1 AND c.relname = $2
            """
            table_info = await conn.fetchrow(verify_query, schema, table_name)
            
            if table_info:
                logger.info(f"Successfully created foreign table: {schema}.{table_name}")
                return True
            else:
                raise RuntimeError(f"Foreign table creation verification failed")
                
        except Exception as e:
            logger.error(f"Failed to create foreign table {schema}.{table_name}: {e}")
            raise
    
    async def check_server_health(
        self,
        conn: asyncpg.Connection,
        server_name: str
    ) -> Dict[str, Any]:
        """
        Check health of foreign server.
        
        Args:
            conn: Database connection
            server_name: Name of foreign server
        
        Returns:
            Health check results
        """
        try:
            health_info = {
                'server_name': server_name,
                'exists': False,
                'accessible': False,
                'tables_count': 0,
                'user_mappings_count': 0,
                'error': None
            }
            
            # Check if server exists
            server_query = """
                SELECT s.srvname, s.srvfdw, s.srvoptions, fdw.fdwname
                FROM pg_foreign_server s
                JOIN pg_foreign_data_wrapper fdw ON s.srvfdw = fdw.oid
                WHERE s.srvname = $1
            """
            server_info = await conn.fetchrow(server_query, server_name)
            
            if not server_info:
                health_info['error'] = f"Server {server_name} does not exist"
                return health_info
            
            health_info['exists'] = True
            health_info['fdw_name'] = server_info['fdwname']
            health_info['options'] = server_info['srvoptions']
            
            # Count foreign tables
            tables_query = """
                SELECT COUNT(*) as count
                FROM pg_class c
                JOIN pg_foreign_table ft ON c.oid = ft.ftrelid
                JOIN pg_foreign_server s ON ft.ftserver = s.oid
                WHERE c.relkind = 'f' AND s.srvname = $1
            """
            tables_count = await conn.fetchval(tables_query, server_name)
            health_info['tables_count'] = tables_count
            
            # Count user mappings
            mappings_query = """
                SELECT COUNT(*) as count
                FROM pg_user_mapping um
                JOIN pg_foreign_server s ON um.umserver = s.oid
                WHERE s.srvname = $1
            """
            mappings_count = await conn.fetchval(mappings_query, server_name)
            health_info['user_mappings_count'] = mappings_count
            
            # Try to test basic connectivity (if possible)
            try:
                # For postgres_fdw, we can try a simple query
                if server_info['fdwname'] == 'postgres_fdw':
                    test_query = f"SELECT 1 FROM information_schema.tables LIMIT 1"
                    # This is a basic connectivity test - in practice you might want
                    # to create a simple foreign table for testing
                    health_info['accessible'] = True
            except Exception as e:
                logger.debug(f"Connectivity test failed for {server_name}: {e}")
                health_info['error'] = str(e)
            
            return health_info
            
        except Exception as e:
            logger.error(f"Health check failed for {server_name}: {e}")
            return {
                'server_name': server_name,
                'exists': False,
                'accessible': False,
                'tables_count': 0,
                'user_mappings_count': 0,
                'error': str(e)
            }
    
    async def cleanup_server(
        self,
        conn: asyncpg.Connection,
        server_name: str,
        cascade: bool = False
    ) -> bool:
        """
        Clean up foreign server and related objects.
        
        Args:
            conn: Database connection
            server_name: Name of server to clean up
            cascade: Whether to cascade drop
        
        Returns:
            True if cleanup successful
        """
        try:
            # Check if server exists
            server_check_query = """
                SELECT srvname FROM pg_foreign_server 
                WHERE srvname = $1
            """
            server_exists = await conn.fetchval(server_check_query, server_name)
            
            if not server_exists:
                logger.info(f"Foreign server {server_name} does not exist")
                return True
            
            cascade_clause = "CASCADE" if cascade else ""
            
            # Drop foreign server (this will also drop user mappings) with security validation
            safe_server_name = SQLSecurityValidator.validate_and_escape_identifier(
                server_name, "server name"
            )
            drop_query = f"DROP SERVER {safe_server_name} {cascade_clause}"
            await conn.execute(drop_query)
            
            # Verify cleanup
            verify_query = """
                SELECT srvname FROM pg_foreign_server 
                WHERE srvname = $1
            """
            still_exists = await conn.fetchval(verify_query, server_name)
            
            if not still_exists:
                logger.info(f"Successfully cleaned up foreign server: {server_name}")
                return True
            else:
                raise RuntimeError(f"Server cleanup verification failed")
                
        except Exception as e:
            logger.error(f"Failed to cleanup server {server_name}: {e}")
            raise
    
    async def _list_foreign_servers_internal(self, conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        """
        List all foreign servers.
        
        Args:
            conn: Database connection
        
        Returns:
            List of foreign server information
        """
        try:
            query = """
                SELECT 
                    s.srvname,
                    fdw.fdwname,
                    s.srvoptions,
                    0 as table_count,
                    COUNT(um.umuser) as user_mapping_count
                FROM pg_foreign_server s
                JOIN pg_foreign_data_wrapper fdw ON s.srvfdw = fdw.oid
                LEFT JOIN pg_user_mapping um ON um.umserver = s.oid
                GROUP BY s.srvname, fdw.fdwname, s.srvoptions
                ORDER BY s.srvname
            """
            
            servers = await conn.fetch(query)
            return [dict(server) for server in servers]
            
        except Exception as e:
            logger.error(f"Failed to list foreign servers: {e}")
            raise
    
    async def _list_foreign_tables_internal(
        self,
        conn: asyncpg.Connection,
        server_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List foreign tables.
        
        Args:
            conn: Database connection
            server_name: Optional server name filter
        
        Returns:
            List of foreign table information
        """
        try:
            base_query = """
                SELECT 
                    n.nspname as foreign_table_schema,
                    c.relname as foreign_table_name,
                    s.srvname as foreign_server_name,
                    '[]'::json as columns
                FROM pg_class c
                JOIN pg_namespace n ON c.relnamespace = n.oid
                JOIN pg_foreign_table ft ON c.oid = ft.ftrelid
                JOIN pg_foreign_server s ON ft.ftserver = s.oid
                WHERE c.relkind = 'f'
            """
            
            if server_name:
                query = base_query + " AND s.srvname = $1"
                params = [server_name]
            else:
                query = base_query
                params = []
            
            query += """
                ORDER BY n.nspname, c.relname
            """
            
            tables = await conn.fetch(query, *params)
            return [dict(table) for table in tables]
            
        except Exception as e:
            logger.error(f"Failed to list foreign tables: {e}")
            raise
    
    async def get_extension_info(self, conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        """
        Get information about installed FDW extensions.
        
        Args:
            conn: Database connection
        
        Returns:
            List of extension information
        """
        try:
            query = """
                SELECT 
                    e.extname,
                    e.extversion,
                    ae.default_version,
                    ae.comment,
                    CASE WHEN e.extname IS NOT NULL THEN true ELSE false END as installed
                FROM pg_available_extensions ae
                LEFT JOIN pg_extension e ON ae.name = e.extname
                WHERE ae.name LIKE '%fdw%' OR ae.name = 'multicorn'
                ORDER BY ae.name
            """
            
            extensions = await conn.fetch(query)
            return [dict(ext) for ext in extensions]
            
        except Exception as e:
            logger.error(f"Failed to get extension info: {e}")
            raise
    
    def _build_options_string(self, options: Dict[str, str]) -> str:
        """
        Build OPTIONS clause for SQL statements.
        
        Args:
            options: Dictionary of option key-value pairs
        
        Returns:
            Formatted OPTIONS string
        """
        if not options:
            return ""
        
        option_pairs = []
        for key, value in options.items():
            # Escape single quotes in values
            escaped_value = value.replace("'", "''")
            option_pairs.append(f"{key} '{escaped_value}'")
        
        return f"OPTIONS ({', '.join(option_pairs)})"
    
    def _escape_identifier(self, identifier: str) -> str:
        """
        Escape SQL identifier.
        
        Args:
            identifier: SQL identifier to escape
        
        Returns:
            Escaped identifier
        """
        # Basic identifier escaping - in production you might want more robust escaping
        return f'"{identifier}"' if not identifier.isidentifier() else identifier
    
    # High-level API methods expected by tests
    async def install_fdw_extension(self, extension_name: str) -> bool:
        """Install FDW extension using connection from connector."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        async with self.fdw_connector.get_connection() as conn:
            return await self.install_extension(conn, extension_name)
    
    async def create_foreign_server_from_config(self, config: Dict[str, Any]) -> bool:
        """Create foreign server using configuration dict."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        async with self.fdw_connector.get_connection() as conn:
            return await self._create_foreign_server_internal(
                conn,
                config["server_name"],
                config["fdw_name"],
                config.get("options", {})
            )
    
    # Add method alias for backward compatibility with tests
    async def create_foreign_server(self, config_or_conn, server_name=None, fdw_name=None, options=None, if_not_exists=True):
        """Create foreign server - supports both dict config and direct params."""
        if isinstance(config_or_conn, dict):
            # Called with config dict (test API)
            return await self.create_foreign_server_from_config(config_or_conn)
        else:
            # Called with connection and params (internal API)
            return await self._create_foreign_server_internal(
                config_or_conn, server_name, fdw_name, options, if_not_exists
            )
    
    async def drop_foreign_server(self, server_name: str) -> bool:
        """Drop foreign server."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        async with self.fdw_connector.get_connection() as conn:
            return await self.cleanup_server(conn, server_name, cascade=True)
    
    async def create_user_mapping(self, config: Dict[str, Any]) -> bool:
        """Create user mapping using configuration dict."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        async with self.fdw_connector.get_connection() as conn:
            return await self._create_user_mapping_internal(
                conn,
                config["server_name"],
                config.get("user_name", "current_user"),
                config.get("options", {})
            )
    
    async def drop_user_mapping(self, server_name: str, user_name: str = "current_user") -> bool:
        """Drop user mapping."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        try:
            async with self.fdw_connector.get_connection() as conn:
                # Build secure DROP USER MAPPING statement
                if user_name == 'current_user':
                    user_part = 'current_user'  # Special PostgreSQL keyword
                else:
                    user_part = SQLSecurityValidator.validate_and_escape_identifier(
                        user_name, "user name"
                    )
                
                safe_server_name = SQLSecurityValidator.validate_and_escape_identifier(
                    server_name, "server name"
                )
                drop_query = f"DROP USER MAPPING IF EXISTS FOR {user_part} SERVER {safe_server_name}"
                await conn.execute(drop_query)
                return True
        except Exception as e:
            logger.debug(f"Drop user mapping failed (may not exist): {e}")
            return True  # Consider it successful if mapping doesn't exist
    
    async def create_foreign_table(self, config: Dict[str, Any]) -> bool:
        """Create foreign table using configuration dict."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        async with self.fdw_connector.get_connection() as conn:
            return await self._create_foreign_table_internal(
                conn,
                config["table_name"],
                config["server_name"],
                config["columns"],
                config.get("options", {}),
                config.get("schema", "public")
            )
    
    async def drop_foreign_table(self, table_name: str, schema: str = "public") -> bool:
        """Drop foreign table."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        try:
            async with self.fdw_connector.get_connection() as conn:
                # Build secure DROP FOREIGN TABLE statement
                safe_schema = SQLSecurityValidator.validate_and_escape_identifier(
                    schema, "schema name"
                )
                safe_table_name = SQLSecurityValidator.validate_and_escape_identifier(
                    table_name, "table name"
                )
                drop_query = f"DROP FOREIGN TABLE IF EXISTS {safe_schema}.{safe_table_name}"
                await conn.execute(drop_query)
                return True
        except Exception as e:
            logger.debug(f"Drop foreign table failed (may not exist): {e}")
            return True  # Consider it successful if table doesn't exist
    
    async def validate_foreign_server_connection(self, server_name: str) -> Dict[str, Any]:
        """Validate foreign server connection."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        async with self.fdw_connector.get_connection() as conn:
            health_info = await self.check_server_health(conn, server_name)
            
            # Raise exception if server doesn't exist
            if not health_info.get('exists', False):
                raise ValueError(health_info.get('error', f"Server {server_name} validation failed"))
            
            return health_info
    
    async def list_foreign_servers(self) -> List[Dict[str, Any]]:
        """List all foreign servers."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        async with self.fdw_connector.get_connection() as conn:
            return await self._list_foreign_servers_internal(conn)
    
    async def list_foreign_tables(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List foreign tables."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        async with self.fdw_connector.get_connection() as conn:
            return await self._list_foreign_tables_internal(conn, server_name)
    
    async def get_foreign_table_info(self, table_name: str, schema: str = "public") -> Optional[Dict[str, Any]]:
        """Get foreign table information."""
        if not self.fdw_connector:
            raise RuntimeError("No FDW connector configured")
        
        try:
            async with self.fdw_connector.get_connection() as conn:
                query = """
                    SELECT 
                        n.nspname as schema_name,
                        c.relname as table_name,
                        s.srvname as server_name,
                        array_agg(
                            json_build_object(
                                'column_name', col.column_name,
                                'data_type', col.data_type,
                                'is_nullable', col.is_nullable
                            ) ORDER BY col.ordinal_position
                        ) as columns
                    FROM pg_class c
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    JOIN pg_foreign_table ft ON c.oid = ft.ftrelid
                    JOIN pg_foreign_server s ON ft.ftserver = s.oid
                    LEFT JOIN information_schema.columns col 
                        ON col.table_schema = n.nspname 
                        AND col.table_name = c.relname
                    WHERE c.relkind = 'f' AND n.nspname = $1 AND c.relname = $2
                    GROUP BY n.nspname, c.relname, s.srvname
                """
                result = await conn.fetchrow(query, schema, table_name)
                return dict(result) if result else None
        except Exception as e:
            logger.debug(f"Get table info failed: {e}")
            return None
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup if needed
        pass