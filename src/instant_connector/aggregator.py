"""Main data aggregation engine for combining multiple data sources."""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path
import yaml
import psutil
import os
from .sources import DatabaseSource, FileSource, APISource
from .ml_optimizer import MLOptimizer
from .secure_serializer import SecureSerializer, save_data_connector, load_data_connector
from .secure_credentials import SecureCredentialManager, get_global_credential_manager
from .pickle_manager import PickleManager

logger = logging.getLogger(__name__)

# Security limits
MAX_MEMORY_USAGE_MB = 2048  # 2GB default
MAX_TOTAL_ROWS = 10_000_000  # 10M rows default
MAX_FILE_SIZE_MB = 1024  # 1GB default


class InstantDataConnector:
    """Main engine for aggregating data from multiple sources."""
    
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        max_memory_mb: int = MAX_MEMORY_USAGE_MB,
        max_rows: int = MAX_TOTAL_ROWS,
        allowed_directories: Optional[List[Union[str, Path]]] = None,
        use_secure_serialization: bool = True,
        credential_manager: Optional[SecureCredentialManager] = None
    ):
        """
        Initialize data aggregator with security controls.
        
        Args:
            config_path: Path to YAML configuration file
            max_memory_mb: Maximum memory usage in MB
            max_rows: Maximum total rows across all datasets
            allowed_directories: List of allowed directories for file operations
            use_secure_serialization: Whether to use secure serialization instead of pickle
            credential_manager: Optional credential manager for secure credentials
        """
        self.config = {}
        self.sources = {}
        self.raw_data = {}
        self.ml_ready_data = {}
        self.metadata = {}
        self.ml_artifacts = {}
        self.ml_optimizer = None
        
        # Security settings
        self.max_memory_mb = max_memory_mb
        self.max_rows = max_rows
        self.allowed_directories = allowed_directories
        self.use_secure_serialization = use_secure_serialization
        self.credential_manager = credential_manager or get_global_credential_manager()
        
        # Initialize secure serializer
        if use_secure_serialization:
            self.serializer = SecureSerializer(
                max_memory_usage=max_memory_mb * 1024 * 1024,
                max_file_size=MAX_FILE_SIZE_MB * 1024 * 1024
            )
        
        if config_path:
            self.load_from_config(config_path)
    
    def _validate_memory_usage(self) -> None:
        """Validate current memory usage against limits."""
        total_memory = 0
        for section in ['raw_data', 'ml_ready_data']:
            data_dict = getattr(self, section, {})
            for name, df in data_dict.items():
                if isinstance(df, pd.DataFrame):
                    total_memory += df.memory_usage(deep=True).sum()
        
        total_memory_mb = total_memory / (1024 * 1024)
        if total_memory_mb > self.max_memory_mb:
            raise ValueError(f"Memory usage ({total_memory_mb:.1f} MB) exceeds limit ({self.max_memory_mb} MB)")
    
    def _validate_total_rows(self) -> None:
        """Validate total row count against limits."""
        total_rows = 0
        for section in ['raw_data', 'ml_ready_data']:
            data_dict = getattr(self, section, {})
            for name, df in data_dict.items():
                if isinstance(df, pd.DataFrame):
                    total_rows += len(df)
        
        if total_rows > self.max_rows:
            raise ValueError(f"Total rows ({total_rows:,}) exceeds limit ({self.max_rows:,})")
    
    def _validate_system_resources(self) -> None:
        """Validate system resource availability."""
        # Check available memory
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        if available_memory_mb < self.max_memory_mb * 0.5:  # Need at least 50% buffer
            logger.warning(f"Low system memory: {available_memory_mb:.1f} MB available")
    
    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
    
    def add_database_source(
        self,
        name: str,
        connection_params: Dict[str, Any],
        tables: Optional[List[str]] = None,
        queries: Optional[Dict[str, str]] = None,
        optimize_dtypes: bool = True,
        include_metadata: bool = False
    ):
        """
        Add a database source.
        
        Args:
            name: Source identifier
            connection_params: Database connection parameters
            queries: Dictionary of query names to SQL queries
        """
        source = DatabaseSource(connection_params)
        self.sources[name] = {
            'type': 'database',
            'source': source,
            'tables': tables or [],
            'queries': queries or {},
            'optimize_dtypes': optimize_dtypes,
            'include_metadata': include_metadata
        }
        logger.info(f"Added database source: {name}")
    
    def add_file_source(
        self,
        name: str,
        file_path: Union[str, Path],
        file_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        read_options: Optional[Dict[str, Any]] = None,
        optimize_dtypes: bool = True,
        include_metadata: bool = False
    ):
        """
        Add a file source.
        
        Args:
            name: Source identifier
            file_paths: Path(s) to files
            read_options: Options for reading files
        """
        # Use file_path parameter if provided, otherwise fall back to file_paths
        paths = file_path if file_path is not None else file_paths
        source = FileSource(
            paths,
            allowed_directories=self.allowed_directories,
            max_file_size_mb=MAX_FILE_SIZE_MB
        )
        self.sources[name] = {
            'type': 'file',
            'source': source,
            'read_options': read_options or {},
            'optimize_dtypes': optimize_dtypes,
            'include_metadata': include_metadata
        }
        logger.info(f"Added file source: {name}")
    
    def add_api_source(
        self,
        name: str,
        base_url: str,
        endpoints: Union[List[str], Dict[str, Dict[str, Any]]],
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Add an API source.
        
        Args:
            name: Source identifier
            base_url: Base URL for API
            endpoints: Dictionary of endpoint configurations
            **kwargs: Additional arguments for APISource
        """
        # Add headers to kwargs if provided
        if headers:
            kwargs['headers'] = headers
        source = APISource(base_url=base_url, **kwargs)
        self.sources[name] = {
            'type': 'api',
            'source': source,
            'endpoints': endpoints
        }
        logger.info(f"Added API source: {name}")
    
    def extract_data(
        self,
        source_name: Optional[str] = None,
        dataset_name: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract data from sources.
        
        Args:
            source_name: Specific source to extract from (None for all)
            dataset_name: Specific dataset to extract (None for all)
            
        Returns:
            Dictionary of DataFrames
        """
        extracted_data = {}
        
        # Load from config if available
        if self.config and 'sources' in self.config:
            self._load_sources_from_config()
        
        # Extract from specified sources
        sources_to_extract = [source_name] if source_name else list(self.sources.keys())
        
        for src_name in sources_to_extract:
            if src_name not in self.sources:
                logger.warning(f"Source not found: {src_name}")
                continue
            
            source_info = self.sources[src_name]
            source_type = source_info['type']
            
            try:
                if source_type == 'database':
                    data = self._extract_database_data(src_name, dataset_name)
                elif source_type == 'file':
                    data = self._extract_file_data(src_name)
                elif source_type == 'api':
                    data = self._extract_api_data(src_name, dataset_name)
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    continue
                
                # Add source metadata
                for name, df in data.items():
                    df['_source'] = src_name
                    df['_source_type'] = source_type
                    extracted_data[f"{src_name}_{name}"] = df
                    
            except Exception as e:
                logger.error(f"Failed to extract from {src_name}: {e}")
                continue
        
        self.raw_data.update(extracted_data)
        logger.info(f"Extracted {len(extracted_data)} datasets")
        
        return extracted_data
    
    def _extract_database_data(
        self,
        source_name: str,
        dataset_name: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Extract data from database source."""
        source_info = self.sources[source_name]
        source = source_info['source']
        queries = source_info['queries']
        
        data = {}
        queries_to_run = [dataset_name] if dataset_name and dataset_name in queries else queries.keys()
        
        with source:
            for query_name in queries_to_run:
                query = queries[query_name]
                logger.info(f"Executing query: {query_name}")
                df = source.extract_data(query=query)
                data[query_name] = df
        
        return data
    
    def _extract_file_data(self, source_name: str) -> Dict[str, pd.DataFrame]:
        """Extract data from file source."""
        source_info = self.sources[source_name]
        source = source_info['source']
        read_options = source_info['read_options']
        
        df = source.extract_data(**read_options)
        
        # Handle multiple files
        if isinstance(df, list):
            return {f"file_{i}": d for i, d in enumerate(df)}
        else:
            return {"data": df}
    
    def _extract_api_data(
        self,
        source_name: str,
        dataset_name: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Extract data from API source."""
        source_info = self.sources[source_name]
        source = source_info['source']
        endpoints = source_info['endpoints']
        
        data = {}
        endpoints_to_call = [dataset_name] if dataset_name and dataset_name in endpoints else endpoints.keys()
        
        for endpoint_name in endpoints_to_call:
            endpoint_config = endpoints[endpoint_name].copy()
            endpoint = endpoint_config.pop('endpoint')
            logger.info(f"Calling endpoint: {endpoint_name}")
            df = source.extract_data(endpoint, **endpoint_config)
            data[endpoint_name] = df
        
        return data
    
    def _load_sources_from_config(self):
        """Load sources from configuration."""
        if 'sources' not in self.config:
            return
        
        for source_name, source_config in self.config['sources'].items():
            source_type = source_config.get('type')
            
            if source_type == 'database':
                self.add_database_source(
                    source_name,
                    source_config['connection'],
                    source_config.get('queries', {})
                )
            elif source_type == 'file':
                self.add_file_source(
                    source_name,
                    source_config['paths'],
                    source_config.get('read_options', {})
                )
            elif source_type == 'api':
                api_config = source_config.copy()
                base_url = api_config.pop('base_url')
                endpoints = api_config.pop('endpoints', {})
                self.add_api_source(source_name, base_url, endpoints, **api_config)
    
    def optimize_datasets(
        self,
        optimizer: Optional[MLOptimizer] = None,
        **optimizer_kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply ML optimizations to all datasets.
        
        Args:
            optimizer: MLOptimizer instance (creates new if None)
            **optimizer_kwargs: Arguments for optimize_dataframe
            
        Returns:
            Optimized datasets
        """
        if not self.raw_data:
            raise ValueError("No datasets to optimize. Run extract_data first.")
        
        if optimizer is None:
            optimizer = MLOptimizer()
        
        optimized = {}
        
        for name, df in self.raw_data.items():
            logger.info(f"Optimizing dataset: {name}")
            try:
                df_optimized = optimizer.optimize_dataframe(df, **optimizer_kwargs)
                optimized[name] = df_optimized
            except Exception as e:
                logger.error(f"Failed to optimize {name}: {e}")
                optimized[name] = df
        
        self.ml_ready_data = optimized
        return optimized
    
    def save_connector(
        self,
        file_path: Union[str, Path],
        pickle_manager: Optional[PickleManager] = None,
        include_metadata: bool = True,
        **save_kwargs
    ) -> Dict[str, Any]:
        """
        Save aggregated data as a connector pickle.
        
        Args:
            file_path: Output file path
            pickle_manager: PickleManager instance (creates new if None)
            include_metadata: Include aggregation metadata
            **save_kwargs: Additional arguments for save_data_connector
            
        Returns:
            Save statistics
        """
        if not self.raw_data:
            raise ValueError("No datasets to save. Run extract_data first.")
        
        if pickle_manager is None:
            pickle_manager = PickleManager()
        
        # Prepare metadata
        metadata = {}
        if include_metadata:
            metadata = {
                'sources': {name: info['type'] for name, info in self.sources.items()},
                'dataset_info': {
                    name: {
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                    }
                    for name, df in self.raw_data.items()
                },
                'aggregation_config': self.config
            }
        
        # Save connector
        return pickle_manager.save_data_connector(
            self.raw_data,
            file_path,
            metadata=metadata,
            **save_kwargs
        )
    
    def aggregate_and_save(
        self,
        output_path: Union[str, Path],
        optimize: bool = True,
        compression: str = 'lz4',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete pipeline: extract, optimize, and save data.
        
        Args:
            output_path: Output file path
            optimize: Whether to apply ML optimizations
            compression: Compression method
            **kwargs: Additional arguments
            
        Returns:
            Save statistics
        """
        # Extract data
        logger.info("Starting data aggregation pipeline")
        self.extract_data()
        
        # Optimize if requested
        if optimize:
            optimizer_kwargs = kwargs.pop('optimizer_kwargs', {})
            self.optimize_datasets(**optimizer_kwargs)
        
        # Save connector
        pickle_manager = PickleManager(compression=compression)
        save_stats = self.save_connector(output_path, pickle_manager, **kwargs)
        
        logger.info("Data aggregation pipeline complete")
        return save_stats
    
    def aggregate_all(self):
        """Aggregate data from all configured sources with security validation."""
        logger.info("Starting data aggregation from all sources")
        
        # Validate system resources before starting
        self._validate_system_resources()
        
        # Initialize metadata
        from datetime import datetime
        self.metadata['extraction_timestamp'] = datetime.now().isoformat()
        self.metadata['total_sources'] = len(self.sources)
        self.metadata['total_tables'] = 0
        
        if not self.sources:
            logger.info("No sources configured")
            return
        
        # Extract from all sources
        for source_name, source_info in self.sources.items():
            source_type = source_info['type']
            source_obj = source_info['source']
            
            try:
                if source_type == 'database':
                    tables = source_info.get('tables', [])
                    if not tables:
                        # Get all tables if none specified
                        with source_obj:
                            table_info = source_obj.get_table_info()
                            tables = table_info['table_name'].tolist()
                    
                    data = self._extract_database_data_for_aggregate(source_name, source_obj, tables, source_info)
                elif source_type == 'file':
                    data = self._extract_file_data_for_aggregate(source_name, source_obj, source_info)
                elif source_type == 'api':
                    endpoints = source_info.get('endpoints', [])
                    data = self._extract_api_data_for_aggregate(source_name, source_obj, endpoints)
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    continue
                
                self.raw_data.update(data)
                self.metadata['total_tables'] += len(data)
                
                # Validate after each source to catch issues early
                try:
                    self._validate_memory_usage()
                    self._validate_total_rows()
                except ValueError as ve:
                    logger.error(f"Resource limit exceeded after processing {source_name}: {ve}")
                    # Remove the data that caused the issue
                    for key in data.keys():
                        self.raw_data.pop(key, None)
                    break
                
            except Exception as e:
                logger.error(f"Failed to extract from {source_name}: {e}")
                continue
        
        logger.info(f"Aggregated {len(self.raw_data)} datasets from {len(self.sources)} sources")
    
    def _extract_database_data_for_aggregate(self, source_name: str, source_obj, tables: List[str], source_info: Dict) -> Dict[str, pd.DataFrame]:
        """Extract database data for aggregate_all method."""
        data = {}
        include_metadata = source_info.get('include_metadata', False)
        optimize_dtypes = source_info.get('optimize_dtypes', True)
        
        with source_obj:
            for table in tables:
                try:
                    # Extract additional parameters for table extraction
                    extract_params = {
                        'include_metadata': include_metadata,
                        'optimize_dtypes': optimize_dtypes
                    }
                    
                    # Add any additional extraction parameters
                    if 'sample_size' in source_info:
                        extract_params['sample_size'] = source_info['sample_size']
                    if 'where_clause' in source_info:
                        extract_params['where_clause'] = source_info['where_clause']
                    
                    df = source_obj.extract_table(table, **extract_params)
                    data[table] = df
                except Exception as e:
                    logger.error(f"Failed to extract table {table} from {source_name}: {e}")
        return data
    
    def _extract_file_data_for_aggregate(self, source_name: str, source_obj, source_info: Dict) -> Dict[str, pd.DataFrame]:
        """Extract file data for aggregate_all method."""
        try:
            include_metadata = source_info.get('include_metadata', False)
            optimize_dtypes = source_info.get('optimize_dtypes', True)
            
            df = source_obj.extract_data(
                include_metadata=include_metadata,
                optimize_dtypes=optimize_dtypes
            )
            return {source_name: df}
        except Exception as e:
            logger.error(f"Failed to extract from file source {source_name}: {e}")
            return {}
    
    def _extract_api_data_for_aggregate(self, source_name: str, source_obj, endpoints: List[str]) -> Dict[str, pd.DataFrame]:
        """Extract API data for aggregate_all method."""
        data = {}
        for endpoint in endpoints:
            try:
                df = source_obj.extract_data(endpoint)
                endpoint_name = endpoint.strip('/').replace('/', '_')
                data[endpoint_name] = df
            except Exception as e:
                logger.error(f"Failed to extract from endpoint {endpoint} in {source_name}: {e}")
        return data
    
    def configure_ml_optimization(self, **kwargs):
        """Configure ML optimization settings."""
        if self.ml_optimizer is None:
            from .ml_optimizer import MLOptimizer
            self.ml_optimizer = MLOptimizer()
        
        # Set configuration options
        for key, value in kwargs.items():
            if hasattr(self.ml_optimizer, key):
                setattr(self.ml_optimizer, key, value)
        
        logger.info(f"Configured ML optimizer with options: {kwargs}")
    
    def apply_ml_optimization(self, target_column=None, **kwargs):
        """Apply ML optimization to raw data."""
        if not self.raw_data:
            raise ValueError("No raw data to optimize. Run aggregate_all first.")
        
        if self.ml_optimizer is None:
            from .ml_optimizer import MLOptimizer
            self.ml_optimizer = MLOptimizer()
        
        # Apply optimization to each dataset
        for name, df in self.raw_data.items():
            try:
                if target_column and target_column in df.columns:
                    # Handle target column separately for train/test split
                    optimized_data = self.ml_optimizer.optimize_for_ml(
                        df, target_column=target_column, **kwargs
                    )
                    # Add train/test splits to ml_ready_data
                    for key, value in optimized_data.items():
                        self.ml_ready_data[key] = value
                else:
                    # Standard optimization
                    optimized_df = self.ml_optimizer.optimize_dataframe(df)
                    self.ml_ready_data[name] = optimized_df
                
            except Exception as e:
                logger.error(f"Failed to optimize dataset {name}: {e}")
                # Keep original data if optimization fails
                self.ml_ready_data[name] = df
        
        # Store preprocessing metadata
        self.ml_artifacts['preprocessing_metadata'] = self.ml_optimizer.get_preprocessing_info()
        
        logger.info(f"Applied ML optimization to {len(self.ml_ready_data)} datasets")
    
    def save_pickle(self, output_path=None, **kwargs):
        """Save data using secure serialization (or legacy pickle if configured)."""
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.use_secure_serialization:
                output_path = Path.cwd() / f"instant_data_connector_{timestamp}.json"
            else:
                output_path = Path.cwd() / f"instant_data_connector_{timestamp}.pkl"
        
        if not self.raw_data:
            raise ValueError("No data to save. Run aggregate_all first.")
        
        # Validate data before saving
        self._validate_memory_usage()
        self._validate_total_rows()
        
        # Prepare data for saving
        data_to_save = {
            'raw_data': self.raw_data,
            'ml_ready_data': self.ml_ready_data,
            'metadata': self.metadata,
            'ml_artifacts': self.ml_artifacts
        }
        
        if self.use_secure_serialization:
            # Use secure serializer - filter out unsupported kwargs
            secure_kwargs = {k: v for k, v in kwargs.items() if k in ['add_metadata', 'validate']}
            return self.serializer.serialize_datasets(data_to_save, output_path, **secure_kwargs)
        else:
            # Use legacy pickle manager (deprecated)
            logger.warning("Using legacy pickle serialization - consider enabling secure serialization")
            from .pickle_manager import PickleManager
            pickle_manager = PickleManager()
            
            return pickle_manager.save_data_connector(
                data_to_save['raw_data'],
                output_path,
                metadata={
                    'ml_ready_data': data_to_save['ml_ready_data'],
                    'metadata': data_to_save['metadata'],
                    'ml_artifacts': data_to_save['ml_artifacts']
                },
                **kwargs
            )
    
    def load_from_config(self, config):
        """Load configuration from dict or file path."""
        if isinstance(config, (str, Path)):
            # Load from file
            config_path = Path(config)
            if config_path.suffix.lower() == '.json':
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Process sources
        if 'sources' in config:
            for source_config in config['sources']:
                source_type = source_config.get('type')
                source_name = source_config.get('name')
                
                if source_type == 'database':
                    self.add_database_source(
                        name=source_name,
                        connection_params=source_config['connection'],
                        tables=source_config.get('tables', []),
                        optimize_dtypes=source_config.get('optimize_dtypes', True),
                        include_metadata=source_config.get('include_metadata', False)
                    )
                    
                    # Add any additional parameters directly to the source info
                    if source_name in self.sources:
                        for param in ['sample_size', 'where_clause']:
                            if param in source_config:
                                self.sources[source_name][param] = source_config[param]
                elif source_type == 'file':
                    self.add_file_source(
                        name=source_name,
                        file_path=source_config['path'],
                        optimize_dtypes=source_config.get('optimize_dtypes', True),
                        include_metadata=source_config.get('include_metadata', False)
                    )
                elif source_type == 'api':
                    self.add_api_source(
                        name=source_name,
                        base_url=source_config['base_url'],
                        endpoints=source_config.get('endpoints', [])
                    )
                else:
                    raise ValueError(f"Unknown source type: {source_type}")
        
        # Configure ML optimization
        if 'ml_optimization' in config and config['ml_optimization'].get('enabled', False):
            ml_config = config['ml_optimization'].copy()
            ml_config.pop('enabled', None)
            self.configure_ml_optimization(**ml_config)
        
        logger.info(f"Loaded configuration with {len(self.sources)} sources")
    
    def get_summary(self):
        """Get summary of aggregated data."""
        summary = {
            'total_sources': self.metadata.get('total_sources', len(self.sources)),
            'total_raw_tables': len(self.raw_data),
            'total_ml_ready_tables': len(self.ml_ready_data),
            'total_rows': sum(len(df) for df in self.raw_data.values()),
            'memory_usage_mb': sum(df.memory_usage(deep=True).sum() for df in self.raw_data.values()) / 1024**2
        }
        
        if self.metadata:
            summary.update(self.metadata)
        
        return summary