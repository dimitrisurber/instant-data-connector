"""SQL security utilities for preventing injection attacks and validating identifiers."""

import re
import logging
from typing import List, Optional, Union, Dict, Any

logger = logging.getLogger(__name__)


class SQLSecurityError(Exception):
    """Exception raised for SQL security violations."""
    pass


class SQLSecurityValidator:
    """Comprehensive SQL security validator for identifiers and values."""
    
    # PostgreSQL identifier rules
    IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    SCHEMA_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    MAX_IDENTIFIER_LENGTH = 63  # PostgreSQL limit
    
    # Dangerous SQL patterns that indicate injection attempts (not just keywords)
    DANGEROUS_PATTERNS = {
        r';\s*drop\s+table', r';\s*delete\s+from', r';\s*truncate\s+table',
        r';\s*alter\s+table', r';\s*grant\s+', r';\s*revoke\s+',
        r'union\s+select', r'exec\s*\(', r'execute\s*\(', 
        r'sp_\w+', r'xp_\w+', r'eval\s*\(', r'script\s*:', 
        r'javascript\s*:', r'vbscript\s*:', r'--\s*', r'/\*.*\*/',
        r'waitfor\s+delay', r'benchmark\s*\('
    }
    
    # Allowed PostgreSQL data types for column definitions
    ALLOWED_DATA_TYPES = {
        'text', 'varchar', 'char', 'bpchar', 'name',
        'integer', 'int', 'int4', 'bigint', 'int8', 'smallint', 'int2',
        'real', 'float4', 'double precision', 'float8', 'numeric', 'decimal',
        'boolean', 'bool',
        'date', 'time', 'timestamp', 'timestamptz', 'interval',
        'uuid', 'json', 'jsonb', 'xml',
        'bytea', 'bit', 'varbit',
        'inet', 'cidr', 'macaddr', 'macaddr8',
        'point', 'line', 'lseg', 'box', 'path', 'polygon', 'circle',
        'tsvector', 'tsquery'
    }
    
    @classmethod
    def validate_identifier(
        cls, 
        identifier: str, 
        identifier_type: str = "identifier",
        allow_qualified: bool = False
    ) -> str:
        """
        Validate and sanitize a SQL identifier (table, column, schema name).
        
        Args:
            identifier: The identifier to validate
            identifier_type: Type description for error messages
            allow_qualified: Whether to allow schema.table format
            
        Returns:
            Validated identifier
            
        Raises:
            SQLSecurityError: If identifier is invalid or dangerous
        """
        if not identifier:
            raise SQLSecurityError(f"Empty {identifier_type} not allowed")
            
        if not isinstance(identifier, str):
            raise SQLSecurityError(f"{identifier_type} must be a string")
            
        # Check length
        if len(identifier) > cls.MAX_IDENTIFIER_LENGTH:
            raise SQLSecurityError(
                f"{identifier_type} '{identifier}' exceeds maximum length of {cls.MAX_IDENTIFIER_LENGTH}"
            )
        
        # Handle qualified identifiers (schema.table)
        if allow_qualified and '.' in identifier:
            parts = identifier.split('.')
            if len(parts) != 2:
                raise SQLSecurityError(f"Invalid qualified {identifier_type}: '{identifier}'")
            
            schema_part, name_part = parts
            cls.validate_identifier(schema_part, "schema", allow_qualified=False)
            cls.validate_identifier(name_part, identifier_type, allow_qualified=False)
            return identifier
        
        # Validate format
        if not cls.IDENTIFIER_PATTERN.match(identifier):
            raise SQLSecurityError(
                f"Invalid {identifier_type} '{identifier}'. Must start with letter/underscore "
                f"and contain only letters, numbers, and underscores."
            )
        
        # Check for dangerous patterns
        identifier_lower = identifier.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, identifier_lower, re.IGNORECASE):
                raise SQLSecurityError(
                    f"Potentially dangerous {identifier_type} '{identifier}' matches forbidden pattern"
                )
        
        logger.debug(f"Validated {identifier_type}: {identifier}")
        return identifier
    
    @classmethod
    def escape_identifier(cls, identifier: str) -> str:
        """
        Escape a SQL identifier with double quotes for safe interpolation.
        
        Args:
            identifier: Pre-validated identifier
            
        Returns:
            Escaped identifier ready for SQL interpolation
        """
        # PostgreSQL identifiers with double quotes
        return f'"{identifier}"'
    
    @classmethod
    def validate_and_escape_identifier(
        cls,
        identifier: str, 
        identifier_type: str = "identifier",
        allow_qualified: bool = False
    ) -> str:
        """
        Validate and escape identifier in one step.
        
        Args:
            identifier: The identifier to validate and escape
            identifier_type: Type description for error messages
            allow_qualified: Whether to allow schema.table format
            
        Returns:
            Validated and escaped identifier
        """
        validated = cls.validate_identifier(identifier, identifier_type, allow_qualified)
        
        if allow_qualified and '.' in validated:
            # Handle qualified identifiers
            parts = validated.split('.')
            return f'"{parts[0]}"."{parts[1]}"'
        else:
            return cls.escape_identifier(validated)
    
    @classmethod
    def validate_data_type(cls, data_type: str) -> str:
        """
        Validate PostgreSQL data type.
        
        Args:
            data_type: Data type to validate
            
        Returns:
            Validated data type
            
        Raises:
            SQLSecurityError: If data type is invalid
        """
        if not data_type:
            raise SQLSecurityError("Empty data type not allowed")
            
        data_type_clean = data_type.lower().strip()
        
        # Extract base type (handle things like varchar(255))
        base_type = re.sub(r'\([^)]*\)', '', data_type_clean).strip()
        
        if base_type not in cls.ALLOWED_DATA_TYPES:
            raise SQLSecurityError(f"Invalid or unsafe data type: '{data_type}'")
        
        # Additional validation for parameterized types
        if '(' in data_type:
            # Validate parentheses content is numeric
            param_match = re.search(r'\(([^)]+)\)', data_type)
            if param_match:
                params = param_match.group(1)
                # Allow numeric parameters and precision/scale (e.g., "10,2")
                if not re.match(r'^[\d,\s]+$', params):
                    raise SQLSecurityError(f"Invalid data type parameters: '{data_type}'")
        
        return data_type
    
    @classmethod
    def validate_options_dict(cls, options: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate and sanitize FDW options dictionary.
        
        Args:
            options: Dictionary of option key-value pairs
            
        Returns:
            Validated options as string dictionary
            
        Raises:
            SQLSecurityError: If options contain dangerous values
        """
        if not isinstance(options, dict):
            raise SQLSecurityError("Options must be a dictionary")
        
        validated_options = {}
        
        for key, value in options.items():
            # Validate option keys
            if not isinstance(key, str) or not key:
                raise SQLSecurityError(f"Invalid option key: {key}")
            
            # Check for dangerous patterns in keys
            key_lower = key.lower()
            for pattern in cls.DANGEROUS_PATTERNS:
                if re.search(pattern, key_lower, re.IGNORECASE):
                    raise SQLSecurityError(f"Dangerous option key: '{key}'")
            
            # Validate and sanitize values
            if value is None:
                continue
                
            value_str = str(value)
            
            # Check for SQL injection patterns in values
            dangerous_patterns = [
                r"['\";\-\-]",  # Quotes, semicolons, SQL comments
                r"(?i)(union|select|insert|update|delete|drop|exec|execute)",
                r"(?i)(script|javascript|vbscript)",
                r"(?i)(waitfor|delay|benchmark)"
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, value_str):
                    raise SQLSecurityError(
                        f"Potentially dangerous option value for '{key}': contains suspicious pattern"
                    )
            
            # Length check
            if len(value_str) > 1000:  # Reasonable limit
                raise SQLSecurityError(f"Option value for '{key}' exceeds maximum length")
            
            validated_options[key] = value_str
        
        return validated_options
    
    @classmethod
    def build_options_string(cls, options: Dict[str, Any]) -> str:
        """
        Build safe OPTIONS string for FDW statements.
        
        Args:
            options: Dictionary of options
            
        Returns:
            Safe OPTIONS string for SQL interpolation
        """
        if not options:
            return ""
        
        validated_options = cls.validate_options_dict(options)
        
        if not validated_options:
            return ""
        
        option_pairs = []
        for key, value in validated_options.items():
            # Escape single quotes in values by doubling them (PostgreSQL standard)
            escaped_value = value.replace("'", "''")
            option_pairs.append(f"{key} '{escaped_value}'")
        
        return f"OPTIONS ({', '.join(option_pairs)})"
    
    @classmethod
    def validate_column_definition(cls, column_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate column definition for foreign table creation.
        
        Args:
            column_def: Dictionary with 'name', 'type', and optional metadata
            
        Returns:
            Validated column definition
        """
        if not isinstance(column_def, dict):
            raise SQLSecurityError("Column definition must be a dictionary")
        
        if 'name' not in column_def:
            raise SQLSecurityError("Column definition missing 'name'")
            
        if 'type' not in column_def:
            raise SQLSecurityError("Column definition missing 'type'")
        
        validated = {
            'name': cls.validate_identifier(column_def['name'], "column name"),
            'type': cls.validate_data_type(column_def['type'])
        }
        
        # Optional fields
        for field in ['not_null', 'default', 'description']:
            if field in column_def:
                validated[field] = column_def[field]
        
        return validated