"""Comprehensive security tests for all identified vulnerabilities."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

from instant_connector import (
    SecureSerializer, load_data_connector, save_data_connector,
    unsafe_load_data_connector, PickleManager
)
from instant_connector.sql_security import SQLSecurityValidator, SQLSecurityError
from instant_connector.secure_credentials import SecureCredentialManager
from instant_connector.sources.database_source import DatabaseSource
from instant_connector.config_parser import ConfigParser
from instant_connector.lazy_query_builder import LazyQueryBuilder


class TestPickleSecurityFixes:
    """Test fixes for unsafe pickle deserialization vulnerability (CVSS 9.8)."""
    
    def test_secure_serialization_is_default(self):
        """Test that secure serialization is used by default."""
        # Importing should use secure serialization
        assert load_data_connector.__module__ == 'instant_connector.secure_serializer'
        assert save_data_connector.__module__ == 'instant_connector.secure_serializer'
    
    def test_unsafe_pickle_functions_show_warnings(self):
        """Test that unsafe pickle functions show security warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should trigger security warnings
            try:
                unsafe_load_data_connector('/nonexistent/file.pkl')
            except:
                pass  # File doesn't exist, but we should get warnings
            
            # Check that security warnings were issued
            security_warnings = [warning for warning in w if "CRITICAL SECURITY WARNING" in str(warning.message)]
            assert len(security_warnings) > 0
            assert "CVSS 9.8" in str(security_warnings[0].message)
    
    def test_pickle_manager_shows_deprecation_warning(self):
        """Test that PickleManager shows deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            pm = PickleManager()
            
            # Check for deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            assert "critical security vulnerabilities" in str(deprecation_warnings[0].message)
    
    def test_secure_serializer_basic_functionality(self):
        """Test that SecureSerializer works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_data.json.lz4"
            
            # Test data
            test_data = {
                'raw_data': {'key': 'value'},
                'metadata': {'source': 'test'}
            }
            
            # Test save
            serializer = SecureSerializer()
            result = serializer.serialize_datasets(test_data, file_path)
            
            assert result['file_path'] == str(file_path)
            assert result['compression_method'] == 'lz4'
            assert not result['encryption_enabled']  # No encryption key provided
            
            # Test load
            loaded_data = serializer.deserialize_datasets(file_path)
            assert loaded_data['raw_data'] == test_data['raw_data']
    
    def test_restricted_unpickler_blocks_dangerous_modules(self):
        """Test that restricted unpickler blocks dangerous modules."""
        from instant_connector.pickle_manager import RestrictedUnpickler, PickleSecurityError
        import io
        import pickle
        
        # Create a malicious pickle payload that tries to import os
        class MaliciousClass:
            def __reduce__(self):
                import os
                return (os.system, ('echo "This should be blocked"',))
        
        malicious_obj = MaliciousClass()
        pickle_data = pickle.dumps(malicious_obj)
        
        # Try to unpickle with restricted unpickler
        with pytest.raises(PickleSecurityError) as exc_info:
            RestrictedUnpickler(io.BytesIO(pickle_data)).load()
        
        assert "Forbidden module" in str(exc_info.value)


class TestCredentialSecurityFixes:
    """Test fixes for hardcoded credentials and credential management."""
    
    def test_config_parser_detects_hardcoded_credentials(self):
        """Test that ConfigParser detects hardcoded credentials."""
        parser = ConfigParser()
        
        # Configuration with hardcoded password
        config_with_hardcoded = {
            'sources': {
                'db1': {
                    'password': 'supersecret123'  # Hardcoded password
                }
            }
        }
        
        errors = parser.validate_required_credentials(config_with_hardcoded)
        assert len(errors) > 0
        assert any("hardcoded credential" in error.lower() for error in errors)
    
    def test_config_parser_accepts_env_vars(self):
        """Test that ConfigParser accepts environment variables."""
        parser = ConfigParser()
        
        # Configuration with environment variables
        config_with_env_vars = {
            'sources': {
                'db1': {
                    'password': '${DB_PASSWORD}'  # Environment variable
                }
            }
        }
        
        errors = parser.validate_required_credentials(config_with_env_vars)
        # Should only complain about missing env var, not hardcoded credential
        hardcoded_errors = [error for error in errors if "hardcoded credential" in error.lower()]
        assert len(hardcoded_errors) == 0
    
    def test_config_parser_accepts_credential_manager(self):
        """Test that ConfigParser accepts credential manager references."""
        parser = ConfigParser()
        
        # Configuration with credential manager
        config_with_cred_manager = {
            'sources': {
                'db1': {
                    'password': 'credential:db1_password'  # Credential manager
                }
            }
        }
        
        # Mock credential manager to return None (missing credential)
        with patch('instant_connector.config_parser.get_global_credential_manager') as mock_cred_mgr:
            mock_instance = MagicMock()
            mock_instance.get_credential.return_value = None
            mock_cred_mgr.return_value = mock_instance
            
            errors = parser.validate_required_credentials(config_with_cred_manager)
            # Should complain about missing credential, not hardcoded credential
            hardcoded_errors = [error for error in errors if "hardcoded credential" in error.lower()]
            assert len(hardcoded_errors) == 0
    
    def test_database_source_resolves_credentials_securely(self):
        """Test that DatabaseSource resolves credentials securely."""
        # Test environment variable resolution
        with patch.dict(os.environ, {'TEST_DB_PASSWORD': 'env_password'}):
            connection_params = {
                'db_type': 'postgresql',
                'host': 'localhost',
                'port': 5432,
                'database': 'test',
                'username': 'user',
                'password': '${TEST_DB_PASSWORD}'
            }
            
            db_source = DatabaseSource(connection_params)
            resolved_password = db_source._resolve_password()
            assert resolved_password == 'env_password'
    
    def test_database_source_warns_about_plaintext_passwords(self):
        """Test that DatabaseSource warns about plaintext passwords."""
        connection_params = {
            'db_type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'username': 'user',
            'password': 'plaintext_secret'  # Plain text password
        }
        
        # Use logging to capture the security warning since it's logged, not warned
        import logging
        with patch('instant_connector.sources.database_source.logger') as mock_logger:
            db_source = DatabaseSource(connection_params)
            resolved_password = db_source._resolve_password()
            
            assert resolved_password == 'plaintext_secret'
            # Check that warning was logged
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "SECURITY WARNING" in warning_call


class TestSQLInjectionFixes:
    """Test fixes for SQL injection vulnerabilities in FDW operations."""
    
    def test_sql_validator_blocks_dangerous_identifiers(self):
        """Test that SQL validator blocks actual SQL injection attempts."""
        # Test actual SQL injection patterns (not just keyword presence)
        dangerous_identifiers = [
            "users; DROP TABLE users; --",  # SQL injection attempt
            "--comment",  # SQL comment injection
            "table'; DROP TABLE users; --",  # SQL injection with quotes
            "users/*comment*/",  # SQL comment injection
            "users UNION SELECT * FROM passwords",  # UNION injection
        ]
        
        for identifier in dangerous_identifiers:
            with pytest.raises(SQLSecurityError):
                SQLSecurityValidator.validate_identifier(identifier)
    
    def test_sql_validator_allows_legitimate_identifiers_with_keywords(self):
        """Test that SQL validator allows legitimate identifiers that contain keywords."""
        # Test legitimate identifiers that happen to contain SQL keywords
        legitimate_identifiers = [
            "drop_table",  # Legitimate table name
            "exec_command",  # Legitimate column name
            "union_select",  # Legitimate identifier
            "user_table",  # Contains 'table' keyword
            "select_option",  # Contains 'select' keyword
        ]
        
        for identifier in legitimate_identifiers:
            # Should not raise exception
            validated = SQLSecurityValidator.validate_identifier(identifier)
            assert validated == identifier
    
    def test_sql_validator_accepts_safe_identifiers(self):
        """Test that SQL validator accepts safe identifiers."""
        safe_identifiers = [
            "user_table",
            "customer_data",
            "order_items",
            "_private_table",
            "Table123",
        ]
        
        for identifier in safe_identifiers:
            # Should not raise exception
            validated = SQLSecurityValidator.validate_identifier(identifier)
            assert validated == identifier
    
    def test_sql_validator_validates_data_types(self):
        """Test that SQL validator validates data types."""
        # Safe data types
        safe_types = ["text", "integer", "varchar(255)", "timestamp", "jsonb"]
        for data_type in safe_types:
            validated = SQLSecurityValidator.validate_data_type(data_type)
            assert validated == data_type
        
        # Dangerous data types
        dangerous_types = ["exec", "system", "dangerous_type"]
        for data_type in dangerous_types:
            with pytest.raises(SQLSecurityError):
                SQLSecurityValidator.validate_data_type(data_type)
    
    def test_sql_validator_validates_options(self):
        """Test that SQL validator validates FDW options."""
        # Safe options
        safe_options = {
            "host": "localhost",
            "port": "5432",
            "dbname": "test"
        }
        validated = SQLSecurityValidator.validate_options_dict(safe_options)
        assert validated == safe_options
        
        # Dangerous options
        dangerous_options = {
            "host": "localhost; DROP TABLE users; --",  # SQL injection
            "command": "rm -rf /"  # System command
        }
        with pytest.raises(SQLSecurityError):
            SQLSecurityValidator.validate_options_dict(dangerous_options)
    
    def test_sql_validator_builds_safe_options_string(self):
        """Test that SQL validator builds safe OPTIONS strings."""
        options = {
            "host": "localhost",
            "port": "5432",
            "dbname": "test_database"  # Safe database name
        }
        
        options_string = SQLSecurityValidator.build_options_string(options)
        
        # Should build proper OPTIONS string
        assert "OPTIONS (" in options_string
        assert "host 'localhost'" in options_string
        assert "port '5432'" in options_string
        assert "dbname 'test_database'" in options_string
    
    def test_lazy_query_builder_validates_identifiers(self):
        """Test that LazyQueryBuilder validates identifiers."""
        builder = LazyQueryBuilder()
        
        # Test safe table name
        safe_result = builder._escape_table_name("users")
        assert safe_result == '"users"'
        
        # Test qualified table name
        qualified_result = builder._escape_table_name("public.users")
        assert qualified_result == '"public"."users"'
        
        # Test dangerous table name
        with pytest.raises(SQLSecurityError):
            builder._escape_table_name("users; DROP TABLE users; --")
    
    def test_lazy_query_builder_builds_safe_queries(self):
        """Test that LazyQueryBuilder builds safe queries."""
        builder = LazyQueryBuilder()
        
        # Should work with safe parameters
        query_info = builder.build_select_query(
            "users",
            columns=["id", "name"],
            filters={"status": "active"}
        )
        
        assert '"users"' in query_info["sql"]
        assert '"id"' in query_info["sql"]
        assert '"name"' in query_info["sql"]
        assert "WHERE" in query_info["sql"]
        
        # Should fail with dangerous table name
        with pytest.raises(SQLSecurityError):
            builder.build_select_query(
                "users; DROP TABLE users; --",
                columns=["id"]
            )


class TestConnectionSecurityFixes:
    """Test connection security enhancements."""
    
    def test_database_source_uses_secure_connection_args(self):
        """Test that DatabaseSource uses secure connection arguments."""
        connection_params = {
            'db_type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'username': 'user',
            'password': 'password'
        }
        
        db_source = DatabaseSource(connection_params)
        
        # Test secure connection args
        with patch.dict(os.environ, {
            'DB_CONNECT_TIMEOUT': '5',
            'DB_SSL_MODE': 'require'
        }):
            connect_args = db_source._get_secure_connect_args()
            
            assert connect_args['connect_timeout'] == 5
            assert connect_args['sslmode'] == 'require'
    
    def test_database_source_masks_connection_string(self):
        """Test that DatabaseSource masks sensitive info in connection strings."""
        db_source = DatabaseSource({
            'db_type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'username': 'user',
            'password': 'secret123'
        })
        
        connection_string = "postgresql://user:secret123@localhost:5432/test"
        masked = db_source._mask_connection_string(connection_string)
        
        assert "secret123" not in masked
        assert "***" in masked
        assert "user" in masked  # Username should still be visible
        assert "localhost" in masked  # Host should still be visible


class TestSecurityIntegration:
    """Test integration of all security fixes."""
    
    def test_secure_configuration_loading(self):
        """Test that configuration loading validates security."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write configuration with hardcoded credentials
            f.write("""
sources:
  test_db:
    type: database
    connection:
      db_type: postgresql
      host: localhost
      username: user
      password: hardcoded_secret
""")
            config_path = f.name
        
        try:
            parser = ConfigParser()
            
            # Should raise error due to hardcoded credentials
            with pytest.raises(ValueError) as exc_info:
                config = parser.parse_config(config_path)
                parser.validate_required_credentials(config)
                
                # If validation doesn't raise, check for security warnings
                errors = parser.validate_required_credentials(config)
                if any("SECURITY WARNING" in error for error in errors):
                    raise ValueError("Security validation failed")
            
            assert "hardcoded credential" in str(exc_info.value).lower() or "security" in str(exc_info.value).lower()
        
        finally:
            os.unlink(config_path)
    
    def test_end_to_end_security_workflow(self):
        """Test complete security workflow with all fixes."""
        # 1. Test secure serialization
        test_data = {'key': 'value'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "secure_data.json.lz4"
            
            # Save with secure serialization
            result = save_data_connector(test_data, file_path)
            assert result['encryption_enabled'] is False  # No key provided
            
            # Load with secure serialization
            loaded_data = load_data_connector(file_path)
            assert loaded_data['raw_data']['key'] == 'value'
        
        # 2. Test SQL security validation
        validator = SQLSecurityValidator()
        
        # Should validate safe identifiers
        safe_table = validator.validate_and_escape_identifier("users", "table")
        assert safe_table == '"users"'
        
        # Should block dangerous identifiers
        with pytest.raises(SQLSecurityError):
            validator.validate_identifier("users; DROP TABLE users; --")
        
        # 3. Test credential validation
        parser = ConfigParser()
        safe_config = {
            'sources': {
                'db1': {
                    'password': '${DB_PASSWORD}'  # Environment variable
                }
            }
        }
        
        errors = parser.validate_required_credentials(safe_config)
        hardcoded_errors = [e for e in errors if "hardcoded" in e.lower()]
        assert len(hardcoded_errors) == 0  # No hardcoded credentials


if __name__ == "__main__":
    pytest.main([__file__])