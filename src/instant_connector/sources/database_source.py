"""Database source connector for PostgreSQL, MySQL, and SQLite with ML-optimized extraction."""

import pandas as pd
from typing import Optional, Dict, Any, Union, List, Tuple, Generator
import logging
from urllib.parse import quote_plus
import sqlalchemy as sa
from sqlalchemy import inspect, MetaData, Table
import numpy as np
from datetime import datetime
import hashlib
import os

from ..secure_credentials import SecureCredentialManager, get_global_credential_manager

logger = logging.getLogger(__name__)


class DatabaseSource:
    """Connector for database sources with ML-optimized data extraction and relationship preservation."""
    
    def __init__(
        self, 
        connection_params: Dict[str, Any],
        credential_manager: Optional[SecureCredentialManager] = None
    ):
        """
        Initialize database connection with secure credential management.
        
        Args:
            connection_params: Dictionary containing:
                - db_type: 'postgresql', 'mysql', or 'sqlite'
                - host, port, database, username for connection
                - password: can be plain text, environment variable, or 'credential:name'
            credential_manager: Optional secure credential manager
                - host: Database host (not for SQLite)
                - port: Database port (not for SQLite)
                - database: Database name
                - username: Username (not for SQLite)
                - password: Password (not for SQLite)
                - schema: Optional schema name (PostgreSQL)
                - Additional driver-specific parameters
        """
        self.connection_params = connection_params
        self.db_type = connection_params.get('db_type', '').lower()
        self.schema = connection_params.get('schema', None)
        self.credential_manager = credential_manager or get_global_credential_manager()
        self.engine = None
        self.inspector = None
        self.metadata = MetaData()
        self._table_cache = {}
        self._relationship_cache = {}
        self._validate_params()
        
    def _validate_params(self):
        """Validate connection parameters."""
        if self.db_type not in ['postgresql', 'mysql', 'sqlite']:
            raise ValueError(f"Unsupported database type: {self.db_type}")
            
        if self.db_type != 'sqlite':
            required = ['host', 'database', 'username', 'password']
            missing = [p for p in required if p not in self.connection_params]
            if missing:
                raise ValueError(f"Missing required parameters: {missing}")
    
    def _get_connection_string(self) -> str:
        """Build SQLAlchemy connection string."""
        if self.db_type == 'sqlite':
            return f"sqlite:///{self.connection_params['database']}"
        
        # Resolve password securely
        password = self._resolve_password()
        # URL encode password to handle special characters  
        password = quote_plus(password) if password else ''
        username = self.connection_params['username']
        host = self.connection_params['host']
        database = self.connection_params['database']
        port = self.connection_params.get('port', '')
        
        if self.db_type == 'postgresql':
            port = port or 5432
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif self.db_type == 'mysql':
            port = port or 3306
            return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    
    def _resolve_password(self) -> str:
        """
        Securely resolve password from various sources.
        
        Supports:
        - credential:name - from encrypted credential manager
        - ${ENV_VAR} - from environment variables 
        - Plain text (discouraged, shows warning)
        
        Returns:
            Resolved password string
        """
        password_config = self.connection_params.get('password', '')
        
        if not password_config:
            return ''
            
        # Handle credential manager reference
        if password_config.startswith('credential:'):
            credential_name = password_config.replace('credential:', '')
            try:
                password = self.credential_manager.get_credential(credential_name)
                if password:
                    logger.info(f"Retrieved password from credential manager: {credential_name}")
                    return password
                else:
                    logger.error(f"Credential '{credential_name}' not found in credential manager")
                    return ''
            except Exception as e:
                logger.error(f"Error retrieving credential '{credential_name}': {e}")
                return ''
        
        # Handle environment variable reference  
        if password_config.startswith('${') and password_config.endswith('}'):
            env_var = password_config[2:-1]  # Remove ${ and }
            password = os.getenv(env_var, '')
            if password:
                logger.info(f"Retrieved password from environment variable: {env_var}")
                return password
            else:
                logger.error(f"Environment variable '{env_var}' not found")
                return ''
        
        # Handle plain text password (security warning)
        if password_config and password_config not in ['', 'password', 'changeme']:
            logger.warning(
                "SECURITY WARNING: Plain text password detected in configuration. "
                "Use environment variables (${PASS}) or credential manager (credential:name) instead."
            )
            return password_config
            
        return ''
    
    def _get_secure_connect_args(self) -> Dict[str, Any]:
        """Get security-enhanced connection arguments for database drivers."""
        connect_args = {}
        
        if self.db_type == 'postgresql':
            connect_args.update({
                'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', '10')),
                'command_timeout': int(os.getenv('DB_COMMAND_TIMEOUT', '60')),
                'server_settings': {
                    'application_name': 'instant_data_connector',
                    'search_path': self.schema if self.schema else 'public'
                }
            })
            
            # SSL configuration for PostgreSQL
            ssl_mode = os.getenv('DB_SSL_MODE', 'prefer')  # prefer, require, verify-ca, verify-full
            if ssl_mode in ['require', 'verify-ca', 'verify-full']:
                connect_args.update({
                    'sslmode': ssl_mode,
                    'sslcert': os.getenv('DB_SSL_CERT'),
                    'sslkey': os.getenv('DB_SSL_KEY'),
                    'sslrootcert': os.getenv('DB_SSL_CA')
                })
                # Remove None values
                connect_args = {k: v for k, v in connect_args.items() if v is not None}
                
        elif self.db_type == 'mysql':
            connect_args.update({
                'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', '10')),
                'read_timeout': int(os.getenv('DB_READ_TIMEOUT', '60')),
                'write_timeout': int(os.getenv('DB_WRITE_TIMEOUT', '60')),
                'charset': 'utf8mb4',
                'use_unicode': True
            })
            
            # SSL configuration for MySQL
            if os.getenv('DB_SSL_MODE', 'PREFERRED') in ['REQUIRED', 'VERIFY_CA', 'VERIFY_IDENTITY']:
                connect_args.update({
                    'ssl_mode': os.getenv('DB_SSL_MODE'),
                    'ssl_cert': os.getenv('DB_SSL_CERT'),
                    'ssl_key': os.getenv('DB_SSL_KEY'),
                    'ssl_ca': os.getenv('DB_SSL_CA')
                })
                # Remove None values
                connect_args = {k: v for k, v in connect_args.items() if v is not None}
        
        return connect_args
    
    def _mask_connection_string(self, connection_string: str) -> str:
        """Mask sensitive information in connection string for logging."""
        import re
        # Replace password with asterisks
        masked = re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', connection_string)
        return masked
    
    def connect(self):
        """Establish database connection and initialize inspector."""
        if not self.engine:
            try:
                connection_string = self._get_connection_string()
                
                # Security-enhanced connection parameters
                connect_args = self._get_secure_connect_args()
                
                self.engine = sa.create_engine(
                    connection_string,
                    pool_pre_ping=True,
                    pool_size=min(10, int(os.getenv('DB_POOL_SIZE', '10'))),
                    max_overflow=min(20, int(os.getenv('DB_MAX_OVERFLOW', '20'))),
                    pool_recycle=min(3600, int(os.getenv('DB_POOL_RECYCLE', '3600'))),
                    pool_timeout=min(30, int(os.getenv('DB_POOL_TIMEOUT', '30'))),
                    connect_args=connect_args
                )
                # Test connection and log securely (mask sensitive info)
                masked_connection_string = self._mask_connection_string(connection_string)
                logger.info(f"Connecting to database: {masked_connection_string}")
                
                with self.engine.connect() as conn:
                    conn.execute(sa.text("SELECT 1"))
                    logger.info("Database connection established successfully")
                
                self.inspector = inspect(self.engine)
                self.metadata.bind = self.engine
                logger.info(f"Connected to {self.db_type} database")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise
    
    def disconnect(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.inspector = None
            logger.info("Disconnected from database")
    
    def _validate_table_name(self, table_name: str) -> str:
        """Validate and sanitize table name to prevent SQL injection."""
        if not table_name or not isinstance(table_name, str):
            raise ValueError("Table name must be a non-empty string")
        
        # Allow alphanumeric, underscore, hyphen, and dot (for schema.table)
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-')
        if not set(table_name) <= allowed_chars:
            raise ValueError(f"Invalid table name: {table_name}. Only alphanumeric characters, underscores, hyphens, and dots allowed.")
        
        return table_name
    
    def _build_safe_where_clause(self, conditions: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build a safe parameterized WHERE clause from conditions dictionary."""
        if not conditions:
            return "", {}
        
        where_parts = []
        params = {}
        
        for i, (column, value) in enumerate(conditions.items()):
            # Validate column name
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
            if not set(column) <= allowed_chars:
                raise ValueError(f"Invalid column name in WHERE clause: {column}")
            
            param_name = f"param_{i}"
            
            if isinstance(value, (list, tuple)):
                # Handle IN clause
                placeholders = []
                for j, v in enumerate(value):
                    list_param_name = f"{param_name}_{j}"
                    placeholders.append(f":{list_param_name}")
                    params[list_param_name] = v
                where_parts.append(f"{column} IN ({', '.join(placeholders)})")
            elif value is None:
                where_parts.append(f"{column} IS NULL")
            else:
                where_parts.append(f"{column} = :{param_name}")
                params[param_name] = value
        
        return " AND ".join(where_parts), params

    def extract_table(
        self,
        table_name: str,
        chunksize: Optional[int] = None,
        sample_size: Optional[int] = None,
        optimize_dtypes: bool = True,
        include_metadata: bool = True,
        where_conditions: Optional[Dict[str, Any]] = None,
        preserve_index: bool = False
    ) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        """
        Extract complete table with proper data type inference and secure parameterized queries.
        
        Args:
            table_name: Name of the table to extract
            chunksize: If specified, returns iterator of DataFrames
            sample_size: If specified, limit results to this many rows
            optimize_dtypes: Automatically optimize data types for ML
            include_metadata: Add metadata columns to DataFrame
            where_conditions: Dictionary of column:value conditions for secure WHERE clause
            preserve_index: Preserve database index as DataFrame index
            
        Returns:
            DataFrame or generator of DataFrames
        """
        if not self.engine:
            self.connect()
        
        # Validate table name
        table_name = self._validate_table_name(table_name)
        
        # Validate sample_size
        if sample_size is not None and (not isinstance(sample_size, int) or sample_size < 0):
            raise ValueError("Sample size must be a positive integer")
        
        # Build WHERE clause with parameters
        where_clause, params = self._build_safe_where_clause(where_conditions)
        
        # Build query safely
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if sample_size:
            query += f" LIMIT {sample_size}"
        
        try:
            logger.info(f"Extracting table: {table_name}")
            
            if chunksize:
                # Return chunked iterator for memory efficiency
                chunks = pd.read_sql(sa.text(query), self.engine, chunksize=chunksize, params=params)
                if optimize_dtypes or include_metadata:
                    return self._process_chunks(
                        chunks, table_name, optimize_dtypes, include_metadata
                    )
                return chunks
            else:
                # Extract full table
                df = pd.read_sql(sa.text(query), self.engine, params=params)
                
                if optimize_dtypes:
                    df = self._optimize_dtypes(df)
                
                if include_metadata:
                    df = self._add_metadata_columns(df, table_name)
                
                if preserve_index:
                    # Try to set primary key as index
                    pk_cols = self._get_primary_key_columns(table_name)
                    if pk_cols and all(col in df.columns for col in pk_cols):
                        df.set_index(pk_cols, inplace=True)
                
                logger.info(f"Extracted {len(df)} rows, {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                return df
                
        except Exception as e:
            logger.error(f"Failed to extract table {table_name}: {e}")
            raise
    
    def _process_chunks(
        self,
        chunks: Generator,
        table_name: str,
        optimize_dtypes: bool,
        include_metadata: bool
    ) -> Generator[pd.DataFrame, None, None]:
        """Process chunks with optimization and metadata."""
        for chunk in chunks:
            if optimize_dtypes:
                chunk = self._optimize_dtypes(chunk)
            if include_metadata:
                chunk = self._add_metadata_columns(chunk, table_name)
            yield chunk
    
    def _add_metadata_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Add metadata columns for data lineage."""
        df['_source_database'] = self.connection_params.get('database', 'unknown')
        df['_source_table'] = table_name
        df['_extraction_timestamp'] = datetime.now()
        df['_row_hash'] = df.apply(
            lambda row: hashlib.md5(
                ''.join(str(row.values)).encode()
            ).hexdigest()[:8], axis=1
        )
        return df
    
    def extract_data(
        self,
        query: Optional[str] = None,
        table: Optional[str] = None,
        chunksize: Optional[int] = None,
        optimize_dtypes: bool = True,
        sample_size: Optional[int] = None,
        query_params: Optional[Dict[str, Any]] = None
    ) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        """
        Extract data from database with secure parameterized queries.
        
        Args:
            query: SQL query to execute (should use :param_name for parameters)
            table: Table name (alternative to query)
            chunksize: If specified, returns iterator of DataFrames
            optimize_dtypes: Automatically optimize data types for ML
            sample_size: If specified, limit results to this many rows
            query_params: Parameters for the SQL query
            
        Returns:
            DataFrame or iterator of DataFrames
        """
        if table:
            return self.extract_table(
                table, chunksize=chunksize, sample_size=sample_size,
                optimize_dtypes=optimize_dtypes
            )
        
        if not query:
            raise ValueError("Either query or table must be specified")
        
        if not self.engine:
            self.connect()
        
        # Validate sample_size
        if sample_size is not None and (not isinstance(sample_size, int) or sample_size < 0):
            raise ValueError("Sample size must be a positive integer")
        
        # Add LIMIT clause safely
        if sample_size:
            query = f"SELECT * FROM ({query}) AS subquery LIMIT {sample_size}"
        
        # Prepare parameters
        params = query_params or {}
        
        try:
            logger.info(f"Executing parameterized query: {query[:100]}...")
            
            if chunksize:
                return pd.read_sql(sa.text(query), self.engine, chunksize=chunksize, params=params)
            else:
                df = pd.read_sql(sa.text(query), self.engine, params=params)
                
                if optimize_dtypes:
                    df = self._optimize_dtypes(df)
                
                logger.info(f"Extracted {len(df)} rows, {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                return df
                
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for ML workflows and memory efficiency."""
        for col in df.columns:
            if col.startswith('_'):  # Skip metadata columns
                continue
                
            col_type = df[col].dtype
            
            # Optimize numeric types
            if col_type != 'object':
                try:
                    # Try to downcast integers
                    if pd.api.types.is_integer_dtype(col_type):
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    # Try to downcast floats
                    elif pd.api.types.is_float_dtype(col_type):
                        # Check if column can be converted to int
                        if df[col].dropna().apply(lambda x: x.is_integer()).all():
                            df[col] = df[col].fillna(-1).astype('int64')
                            df[col] = pd.to_numeric(df[col], downcast='integer')
                        else:
                            df[col] = pd.to_numeric(df[col], downcast='float')
                except Exception:
                    pass
            else:
                # Handle object columns
                n_unique = df[col].nunique()
                n_total = len(df[col])
                
                # Try to parse dates
                if self._is_date_column(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        continue
                    except:
                        pass
                
                # Convert low-cardinality strings to categorical
                if n_unique / n_total < 0.5 and n_unique < 1000:
                    df[col] = df[col].astype('category')
                
                # Try to convert numeric strings
                if df[col].str.match(r'^-?\d+\.?\d*$', na=False).all():
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = pd.to_numeric(df[col], downcast='float')
                    except:
                        pass
        
        return df
    
    def _is_date_column(self, series: pd.Series, sample_size: int = 100) -> bool:
        """Check if a column likely contains dates."""
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(sample_size)
        if len(sample) == 0:
            return False
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            if sample.str.match(pattern, na=False).mean() > 0.8:
                return True
        
        return False
    
    def get_table_info(self, include_stats: bool = True) -> pd.DataFrame:
        """Get comprehensive information about available tables."""
        if not self.inspector:
            self.connect()
        
        tables = []
        table_names = self.inspector.get_table_names(schema=self.schema)
        
        for table_name in table_names:
            try:
                columns = self.inspector.get_columns(table_name, schema=self.schema)
                indexes = self.inspector.get_indexes(table_name, schema=self.schema)
                pk = self.inspector.get_pk_constraint(table_name, schema=self.schema)
                fks = self.inspector.get_foreign_keys(table_name, schema=self.schema)
                
                table_info = {
                    'table_name': table_name,
                    'column_count': len(columns),
                    'index_count': len(indexes),
                    'primary_key': pk['constrained_columns'] if pk else [],
                    'foreign_key_count': len(fks),
                    'column_names': [col['name'] for col in columns],
                    'column_types': {col['name']: str(col['type']) for col in columns}
                }
                
                if include_stats:
                    stats = self._get_table_statistics(table_name)
                    table_info.update(stats)
                
                tables.append(table_info)
                
            except Exception as e:
                logger.warning(f"Failed to get info for table {table_name}: {e}")
                continue
        
        return pd.DataFrame(tables)
    
    def _get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics including row count and size."""
        stats = {'row_count': -1, 'size_mb': -1}
        
        try:
            # Get row count
            with self.engine.connect() as conn:
                result = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}"))
                stats['row_count'] = result.scalar()
                
                # Get table size (database-specific)
                if self.db_type == 'postgresql':
                    size_query = f"""
                        SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))
                    """
                    result = conn.execute(sa.text(size_query))
                    size_str = result.scalar()
                    # Parse size string (e.g., "123 MB")
                    if size_str:
                        parts = size_str.split()
                        if len(parts) == 2:
                            value = float(parts[0])
                            unit = parts[1].upper()
                            if unit == 'KB':
                                stats['size_mb'] = value / 1024
                            elif unit == 'MB':
                                stats['size_mb'] = value
                            elif unit == 'GB':
                                stats['size_mb'] = value * 1024
                
                elif self.db_type == 'mysql':
                    size_query = f"""
                        SELECT 
                            ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb
                        FROM information_schema.TABLES 
                        WHERE table_schema = '{self.connection_params['database']}'
                        AND table_name = '{table_name}'
                    """
                    result = conn.execute(sa.text(size_query))
                    stats['size_mb'] = result.scalar() or -1
                
        except Exception as e:
            logger.warning(f"Failed to get statistics for {table_name}: {e}")
        
        return stats
    
    def detect_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detect and preserve foreign key relationships between tables."""
        if not self.inspector:
            self.connect()
        
        relationships = {}
        table_names = self.inspector.get_table_names(schema=self.schema)
        
        for table_name in table_names:
            try:
                fks = self.inspector.get_foreign_keys(table_name, schema=self.schema)
                if fks:
                    relationships[table_name] = []
                    for fk in fks:
                        rel = {
                            'name': fk.get('name', 'unnamed'),
                            'constrained_columns': fk['constrained_columns'],
                            'referred_table': fk['referred_table'],
                            'referred_columns': fk['referred_columns']
                        }
                        relationships[table_name].append(rel)
                        
            except Exception as e:
                logger.warning(f"Failed to get relationships for {table_name}: {e}")
        
        self._relationship_cache = relationships
        return relationships
    
    def _get_primary_key_columns(self, table_name: str) -> List[str]:
        """Get primary key columns for a table."""
        try:
            pk = self.inspector.get_pk_constraint(table_name, schema=self.schema)
            return pk['constrained_columns'] if pk else []
        except:
            return []
    
    def extract_related_tables(
        self,
        root_table: str,
        max_depth: int = 2,
        include_many_to_many: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Extract a table and its related tables following foreign key relationships."""
        if not self._relationship_cache:
            self.detect_relationships()
        
        extracted_tables = {}
        tables_to_process = [(root_table, 0)]
        processed = set()
        
        while tables_to_process:
            table_name, depth = tables_to_process.pop(0)
            
            if table_name in processed or depth > max_depth:
                continue
            
            # Extract table
            logger.info(f"Extracting related table: {table_name} (depth: {depth})")
            df = self.extract_table(table_name, optimize_dtypes=True)
            extracted_tables[table_name] = df
            processed.add(table_name)
            
            # Find related tables
            if depth < max_depth:
                # Tables this table references
                if table_name in self._relationship_cache:
                    for rel in self._relationship_cache[table_name]:
                        ref_table = rel['referred_table']
                        if ref_table not in processed:
                            tables_to_process.append((ref_table, depth + 1))
                
                # Tables that reference this table
                for other_table, rels in self._relationship_cache.items():
                    for rel in rels:
                        if rel['referred_table'] == table_name and other_table not in processed:
                            # Check if it's a junction table (many-to-many)
                            if include_many_to_many or not self._is_junction_table(other_table):
                                tables_to_process.append((other_table, depth + 1))
        
        return extracted_tables
    
    def _is_junction_table(self, table_name: str) -> bool:
        """Check if a table is likely a junction table for many-to-many relationships."""
        try:
            columns = self.inspector.get_columns(table_name, schema=self.schema)
            fks = self.inspector.get_foreign_keys(table_name, schema=self.schema)
            
            # Junction tables typically have 2-3 columns and 2 foreign keys
            if len(columns) <= 3 and len(fks) == 2:
                # Check if all non-FK columns are keys
                fk_cols = set()
                for fk in fks:
                    fk_cols.update(fk['constrained_columns'])
                
                non_fk_cols = [col['name'] for col in columns if col['name'] not in fk_cols]
                return len(non_fk_cols) <= 1  # Only allow one additional column (like a timestamp)
                
        except:
            pass
        
        return False
    
    def validate_data_quality(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Perform data quality validation and profiling."""
        quality_report = {
            'table_name': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum(),
            'columns': {}
        }
        
        for col in df.columns:
            if col.startswith('_'):  # Skip metadata columns
                continue
                
            col_report = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            # Additional statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_report.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'zeros': (df[col] == 0).sum(),
                    'negative_count': (df[col] < 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0
                })
            
            # Check for potential issues
            col_report['potential_issues'] = []
            
            # High null percentage
            if col_report['null_percentage'] > 50:
                col_report['potential_issues'].append('high_null_percentage')
            
            # Single value column
            if col_report['unique_count'] == 1:
                col_report['potential_issues'].append('single_value')
            
            # Potential ID column
            if col_report['unique_percentage'] > 99 and pd.api.types.is_numeric_dtype(df[col]):
                col_report['potential_issues'].append('possible_id_column')
            
            quality_report['columns'][col] = col_report
        
        return quality_report
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()