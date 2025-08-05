"""
Configuration Parser for FDW Data Connector

This module provides comprehensive YAML configuration parsing with schema validation,
environment variable substitution, and credential management integration.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema
import yaml

from .secure_credentials import SecureCredentialManager, get_global_credential_manager

logger = logging.getLogger(__name__)


class ConfigParser:
    """
    YAML configuration parser with schema validation and credential management.
    
    Features:
    - YAML configuration parsing
    - JSON schema validation
    - Environment variable substitution
    - Credential management integration
    - Configuration inheritance and overrides
    - Hot-reload support with change detection
    - Migration from old configuration formats
    """
    
    def __init__(
        self,
        schema_path: Optional[Union[str, Path]] = None,
        credential_manager: Optional[SecureCredentialManager] = None
    ):
        """
        Initialize configuration parser.
        
        Args:
            schema_path: Path to JSON schema file for validation
            credential_manager: Optional credential manager instance
        """
        self.credential_manager = credential_manager or get_global_credential_manager()
        self.schema_path = Path(schema_path) if schema_path else None
        self.schema: Optional[Dict[str, Any]] = None
        self._config_cache: Dict[str, Any] = {}
        self._config_hash: Optional[str] = None
        
        # Load schema if provided
        if self.schema_path and self.schema_path.exists():
            self._load_schema()
        
        logger.debug("Initialized ConfigParser")
    
    def parse_config(
        self,
        config_path: Union[str, Path],
        validate: bool = True,
        substitute_env_vars: bool = True,
        migrate_legacy: bool = True
    ) -> Dict[str, Any]:
        """
        Parse YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            validate: Whether to validate against schema
            substitute_env_vars: Whether to substitute environment variables
            migrate_legacy: Whether to migrate legacy configuration format
        
        Returns:
            Parsed and processed configuration
        """
        try:
            config_path = Path(config_path)
            
            # Load YAML configuration
            config = self._load_yaml_config(config_path)
            
            # Migrate legacy format if needed
            if migrate_legacy:
                config = self._migrate_legacy_config(config)
            
            # Substitute environment variables
            if substitute_env_vars:
                config = self._substitute_environment_variables(config)
            
            # Handle credential resolution
            config = self._resolve_credentials(config)
            
            # Validate configuration
            if validate and self.schema:
                self._validate_configuration(config)
            
            # Cache configuration and calculate hash
            self._config_cache = config
            self._config_hash = self.calculate_file_hash(config_path)
            
            logger.info(f"Successfully parsed configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to parse configuration from {config_path}: {e}")
            raise
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            List of validation errors (empty if valid)
        """
        if not self.schema:
            logger.warning("No schema loaded for validation")
            return ["No schema available for validation"]
        
        try:
            jsonschema.validate(config, self.schema)
            return []
        except jsonschema.ValidationError as e:
            return [self._format_validation_error(e)]
        except jsonschema.SchemaError as e:
            return [f"Schema error: {e.message}"]
    
    def substitute_environment_variables(
        self,
        config: Dict[str, Any],
        prefix: str = "",
        fail_on_missing: bool = False
    ) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration.
        
        Args:
            config: Configuration dictionary
            prefix: Prefix for environment variable names
            fail_on_missing: Whether to fail if environment variable not found
        
        Returns:
            Configuration with environment variables substituted
        """
        return self._substitute_environment_variables(config, prefix, fail_on_missing)
    
    def merge_configurations(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configurations with override precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
        
        Returns:
            Merged configuration
        """
        return self._deep_merge_dicts(base_config, override_config)
    
    def load_schema(self, schema_path: Union[str, Path]) -> bool:
        """
        Load JSON schema for validation.
        
        Args:
            schema_path: Path to JSON schema file
        
        Returns:
            True if schema loaded successfully
        """
        try:
            self.schema_path = Path(schema_path)
            return self._load_schema()
        except Exception as e:
            logger.error(f"Failed to load schema from {schema_path}: {e}")
            return False
    
    def get_config_hash(self) -> Optional[str]:
        """Get hash of currently loaded configuration."""
        return self._config_hash
    
    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Calculate hash of configuration file for change detection.
        
        Args:
            file_path: Path to file
        
        Returns:
            SHA256 hash of file contents
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return ""
            
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a dictionary")
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")
    
    def _load_schema(self) -> bool:
        """Load JSON schema from file."""
        try:
            if not self.schema_path or not self.schema_path.exists():
                logger.warning(f"Schema file not found: {self.schema_path}")
                return False
            
            with open(self.schema_path, 'r', encoding='utf-8') as file:
                self.schema = json.load(file)
            
            # Validate the schema itself
            jsonschema.Draft7Validator.check_schema(self.schema)
            
            logger.info(f"Loaded schema from {self.schema_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            self.schema = None
            return False
    
    def _substitute_environment_variables(
        self,
        obj: Any,
        prefix: str = "",
        fail_on_missing: bool = False
    ) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(obj, dict):
            return {
                key: self._substitute_environment_variables(value, prefix, fail_on_missing)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [
                self._substitute_environment_variables(item, prefix, fail_on_missing)
                for item in obj
            ]
        elif isinstance(obj, str):
            return self._substitute_string_variables(obj, prefix, fail_on_missing)
        else:
            return obj
    
    def _substitute_string_variables(
        self,
        text: str,
        prefix: str = "",
        fail_on_missing: bool = False
    ) -> str:
        """Substitute environment variables in a string."""
        # Pattern matches ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            
            # Add prefix if specified
            full_var_name = f"{prefix}{var_name}" if prefix else var_name
            
            # Get environment variable value
            env_value = os.getenv(full_var_name)
            
            if env_value is not None:
                return env_value
            elif default_value:
                return default_value
            elif fail_on_missing:
                raise ValueError(f"Environment variable {full_var_name} not found")
            else:
                logger.warning(f"Environment variable {full_var_name} not found, using empty string")
                return ""
        
        return re.sub(pattern, replace_var, text)
    
    def validate_required_credentials(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate that all required credentials are available.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of missing credential errors
        """
        errors = []
        missing_vars = []
        
        def find_env_vars(obj, path=""):
            """Recursively find environment variable references."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    find_env_vars(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_env_vars(item, f"{path}[{i}]")
            elif isinstance(obj, str):
                # Find ${VAR_NAME} without default values - these are required
                pattern = r'\$\{([^}:]+)\}'  # No colon means no default
                matches = re.findall(pattern, obj)
                for var_name in matches:
                    if not os.getenv(var_name):
                        missing_vars.append((var_name, path))
                        
                # Also check for credential: references
                if obj.startswith('credential:'):
                    cred_name = obj.replace('credential:', '')
                    try:
                        cred_manager = get_global_credential_manager()
                        if not cred_manager.get_credential(cred_name):
                            errors.append(f"Missing credential '{cred_name}' at {path}")
                    except Exception as e:
                        errors.append(f"Error accessing credential '{cred_name}' at {path}: {e}")
        
        # Scan configuration for missing credentials
        find_env_vars(config)
        
        # Add missing environment variable errors
        for var_name, path in missing_vars:
            errors.append(f"Missing required environment variable '{var_name}' at {path}")
        
        # Check for critical security patterns
        if self._has_hardcoded_credentials(config):
            errors.append("SECURITY WARNING: Hardcoded credentials detected in configuration")
            
        return errors
    
    def _has_hardcoded_credentials(self, obj: Any, path: str = "") -> bool:
        """Check for hardcoded credentials in configuration."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                # Check for suspicious key names with hardcoded values
                if key.lower() in ['password', 'secret', 'token', 'key', 'credential'] and isinstance(value, str):
                    # Skip if it's an environment variable or credential reference  
                    if not (value.startswith('${') or value.startswith('credential:')):
                        # Check if it looks like a real credential (not empty/default)
                        if value and value not in ['', 'password', 'secret', 'changeme', 'default']:
                            logger.error(f"SECURITY: Hardcoded credential detected at {current_path}")
                            return True
                if self._has_hardcoded_credentials(value, current_path):
                    return True
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if self._has_hardcoded_credentials(item, f"{path}[{i}]"):
                    return True
        return False
    
    def _resolve_credentials(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve credentials using credential manager."""
        try:
            return self._resolve_credentials_recursive(config)
        except Exception as e:
            logger.error(f"Failed to resolve credentials: {e}")
            return config
    
    def _resolve_credentials_recursive(self, obj: Any) -> Any:
        """Recursively resolve credentials in configuration."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == 'password' and isinstance(value, str) and value.startswith('credential:'):
                    # Resolve credential reference
                    credential_key = value[11:]  # Remove 'credential:' prefix
                    try:
                        resolved_password = self.credential_manager.get_credential(
                            credential_key, key
                        )
                        result[key] = resolved_password or value
                    except Exception as e:
                        logger.warning(f"Failed to resolve credential {credential_key}: {e}")
                        result[key] = value
                else:
                    result[key] = self._resolve_credentials_recursive(value)
            return result
        elif isinstance(obj, list):
            return [self._resolve_credentials_recursive(item) for item in obj]
        else:
            return obj
    
    def _migrate_legacy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy configuration format to new format."""
        try:
            # Check if this is a legacy format
            if not self._is_legacy_format(config):
                return config
            
            logger.info("Migrating legacy configuration format")
            
            migrated_config = {
                'version': '1.0',
                'sources': {}
            }
            
            # Migrate old 'databases' section to new format
            if 'databases' in config:
                for db_name, db_config in config['databases'].items():
                    migrated_config['sources'][db_name] = {
                        'type': 'postgres_fdw',
                        'server_options': {
                            'host': db_config.get('host', 'localhost'),
                            'port': str(db_config.get('port', 5432)),
                            'dbname': db_config.get('database', db_name)
                        },
                        'user_mapping': {
                            'options': {
                                'user': db_config.get('username', 'postgres'),
                                'password': db_config.get('password', '')
                            }
                        },
                        'tables': []
                    }
                    
                    # Migrate table configurations
                    if 'tables' in db_config:
                        for table_name, table_config in db_config['tables'].items():
                            table_def = {
                                'name': table_name,
                                'options': {
                                    'table_name': table_config.get('table_name', table_name),
                                    'schema_name': table_config.get('schema', 'public')
                                }
                            }
                            
                            if 'columns' in table_config:
                                table_def['columns'] = table_config['columns']
                            
                            migrated_config['sources'][db_name]['tables'].append(table_def)
            
            # Migrate old 'files' section
            if 'files' in config:
                for file_name, file_config in config['files'].items():
                    migrated_config['sources'][file_name] = {
                        'type': 'file_fdw',
                        'server_options': {},
                        'tables': [{
                            'name': file_name,
                            'options': {
                                'filename': file_config.get('path', ''),
                                'format': file_config.get('format', 'csv'),
                                'header': str(file_config.get('header', True)).lower()
                            },
                            'columns': file_config.get('columns', [])
                        }]
                    }
            
            logger.info("Successfully migrated legacy configuration")
            return migrated_config
            
        except Exception as e:
            logger.error(f"Failed to migrate legacy configuration: {e}")
            return config
    
    def _is_legacy_format(self, config: Dict[str, Any]) -> bool:
        """Check if configuration is in legacy format."""
        # Legacy format has 'databases' or 'files' at root level
        # New format has 'sources' at root level
        return (
            'sources' not in config and 
            ('databases' in config or 'files' in config)
        )
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        errors = self.validate_configuration(config)
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(errors)
            raise ValueError(error_message)
    
    def _format_validation_error(self, error: jsonschema.ValidationError) -> str:
        """Format validation error for better readability."""
        path = " -> ".join(str(p) for p in error.absolute_path)
        if path:
            return f"Error at '{path}': {error.message}"
        else:
            return f"Error: {error.message}"
    
    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result