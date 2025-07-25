"""Secure credential management system for database and API connections."""

import os
import json
import base64
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import getpass

logger = logging.getLogger(__name__)


class CredentialSecurityError(Exception):
    """Custom exception for credential security errors."""
    pass


class SecureCredentialManager:
    """Secure credential management with encryption and keyring integration."""
    
    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        use_keyring: bool = True,
        credential_file: Optional[Union[str, Path]] = None
    ):
        """
        Initialize secure credential manager.
        
        Args:
            encryption_key: Optional encryption key for credentials
            use_keyring: Whether to use system keyring for storage
            credential_file: Optional path to encrypted credential file
        """
        self.use_keyring = use_keyring
        self.credential_file = Path(credential_file) if credential_file else None
        
        # Initialize encryption
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key)
        else:
            self.cipher_suite = None
        
        # Cache for decrypted credentials (memory only)
        self._credential_cache = {}
    
    @staticmethod
    def generate_encryption_key() -> bytes:
        """Generate a secure encryption key."""
        return Fernet.generate_key()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _validate_credential_name(self, name: str) -> None:
        """Validate credential name for security."""
        if not name or not isinstance(name, str):
            raise CredentialSecurityError("Credential name must be a non-empty string")
        
        # Only allow alphanumeric and safe characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
        if not set(name) <= allowed_chars:
            raise CredentialSecurityError(f"Invalid credential name: {name}")
    
    def _encrypt_credential(self, credential_data: Dict[str, Any]) -> bytes:
        """Encrypt credential data."""
        if not self.cipher_suite:
            raise CredentialSecurityError("No encryption key configured")
        
        json_data = json.dumps(credential_data)
        return self.cipher_suite.encrypt(json_data.encode('utf-8'))
    
    def _decrypt_credential(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt credential data."""
        if not self.cipher_suite:
            raise CredentialSecurityError("No encryption key configured")
        
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode('utf-8'))
        except Exception as e:
            raise CredentialSecurityError(f"Failed to decrypt credential: {e}")
    
    def store_credential(
        self,
        name: str,
        credential_data: Dict[str, Any],
        description: Optional[str] = None
    ) -> None:
        """
        Store encrypted credential securely.
        
        Args:
            name: Unique name for the credential
            credential_data: Dictionary containing credential information
            description: Optional description of the credential
        """
        self._validate_credential_name(name)
        
        # Add metadata
        full_data = {
            'credential_data': credential_data,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        if self.use_keyring:
            # Store in system keyring
            try:
                if self.cipher_suite:
                    encrypted_data = self._encrypt_credential(full_data)
                    keyring.set_password("instant_connector", name, base64.b64encode(encrypted_data).decode())
                else:
                    # Store as JSON (less secure, but functional)
                    keyring.set_password("instant_connector", name, json.dumps(full_data))
                
                logger.info(f"Credential '{name}' stored in system keyring")
            except Exception as e:
                logger.error(f"Failed to store credential in keyring: {e}")
                raise CredentialSecurityError(f"Failed to store credential: {e}")
        
        elif self.credential_file:
            # Store in encrypted file
            try:
                credentials = {}
                if self.credential_file.exists():
                    credentials = self._load_credential_file()
                
                if self.cipher_suite:
                    credentials[name] = base64.b64encode(self._encrypt_credential(full_data)).decode()
                else:
                    credentials[name] = full_data
                
                self._save_credential_file(credentials)
                logger.info(f"Credential '{name}' stored in credential file")
            except Exception as e:
                logger.error(f"Failed to store credential in file: {e}")
                raise CredentialSecurityError(f"Failed to store credential: {e}")
        
        else:
            # Store in memory cache only (not persistent)
            self._credential_cache[name] = full_data
            logger.warning(f"Credential '{name}' stored in memory only (not persistent)")
    
    def retrieve_credential(self, name: str) -> Dict[str, Any]:
        """
        Retrieve and decrypt credential.
        
        Args:
            name: Name of the credential to retrieve
            
        Returns:
            Decrypted credential data
        """
        self._validate_credential_name(name)
        
        # Check memory cache first
        if name in self._credential_cache:
            return self._credential_cache[name]['credential_data']
        
        if self.use_keyring:
            # Retrieve from system keyring
            try:
                stored_data = keyring.get_password("instant_connector", name)
                if not stored_data:
                    raise CredentialSecurityError(f"Credential '{name}' not found in keyring")
                
                if self.cipher_suite:
                    encrypted_data = base64.b64decode(stored_data.encode())
                    full_data = self._decrypt_credential(encrypted_data)
                else:
                    full_data = json.loads(stored_data)
                
                # Cache for future use
                self._credential_cache[name] = full_data
                return full_data['credential_data']
                
            except Exception as e:
                logger.error(f"Failed to retrieve credential from keyring: {e}")
                raise CredentialSecurityError(f"Failed to retrieve credential: {e}")
        
        elif self.credential_file and self.credential_file.exists():
            # Retrieve from encrypted file
            try:
                credentials = self._load_credential_file()
                if name not in credentials:
                    raise CredentialSecurityError(f"Credential '{name}' not found in file")
                
                stored_data = credentials[name]
                if self.cipher_suite and isinstance(stored_data, str):
                    encrypted_data = base64.b64decode(stored_data.encode())
                    full_data = self._decrypt_credential(encrypted_data)
                else:
                    full_data = stored_data
                
                # Cache for future use
                self._credential_cache[name] = full_data
                return full_data['credential_data']
                
            except Exception as e:
                logger.error(f"Failed to retrieve credential from file: {e}")
                raise CredentialSecurityError(f"Failed to retrieve credential: {e}")
        
        else:
            raise CredentialSecurityError(f"Credential '{name}' not found")
    
    def delete_credential(self, name: str) -> None:
        """
        Delete credential from storage.
        
        Args:
            name: Name of the credential to delete
        """
        self._validate_credential_name(name)
        
        # Remove from cache
        self._credential_cache.pop(name, None)
        
        if self.use_keyring:
            try:
                keyring.delete_password("instant_connector", name)
                logger.info(f"Credential '{name}' deleted from keyring")
            except Exception as e:
                logger.warning(f"Failed to delete credential from keyring: {e}")
        
        if self.credential_file and self.credential_file.exists():
            try:
                credentials = self._load_credential_file()
                if name in credentials:
                    del credentials[name]
                    self._save_credential_file(credentials)
                    logger.info(f"Credential '{name}' deleted from file")
            except Exception as e:
                logger.warning(f"Failed to delete credential from file: {e}")
    
    def list_credentials(self) -> list:
        """List available credential names."""
        credential_names = set()
        
        # From memory cache
        credential_names.update(self._credential_cache.keys())
        
        # From credential file
        if self.credential_file and self.credential_file.exists():
            try:
                credentials = self._load_credential_file()
                credential_names.update(credentials.keys())
            except Exception as e:
                logger.warning(f"Failed to list credentials from file: {e}")
        
        return sorted(list(credential_names))
    
    def _load_credential_file(self) -> Dict[str, Any]:
        """Load credentials from encrypted file."""
        if not self.credential_file.exists():
            return {}
        
        # Set secure file permissions
        os.chmod(self.credential_file, 0o600)
        
        with open(self.credential_file, 'r') as f:
            return json.load(f)
    
    def _save_credential_file(self, credentials: Dict[str, Any]) -> None:
        """Save credentials to encrypted file."""
        # Ensure parent directory exists
        self.credential_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write with secure permissions
        with open(self.credential_file, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        # Set restrictive permissions
        os.chmod(self.credential_file, 0o600)
    
    def create_database_credential(
        self,
        name: str,
        db_type: str,
        host: str,
        port: int,
        database: str,
        username: str,
        password: Optional[str] = None,
        schema: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Create and store database credential.
        
        Args:
            name: Unique name for the credential
            db_type: Database type (postgresql, mysql, sqlite)
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password (will prompt if not provided)
            schema: Optional database schema
            **kwargs: Additional connection parameters
        """
        if not password:
            password = getpass.getpass(f"Enter password for {username}@{host}: ")
        
        credential_data = {
            'db_type': db_type,
            'host': host,
            'port': port,
            'database': database,
            'username': username,
            'password': password,
            'schema': schema,
            **kwargs
        }
        
        self.store_credential(
            name,
            credential_data,
            description=f"Database connection to {db_type}://{host}:{port}/{database}"
        )
    
    def create_api_credential(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> None:
        """
        Create and store API credential.
        
        Args:
            name: Unique name for the credential
            base_url: API base URL
            api_key: API key (will prompt if not provided)
            username: Username for basic auth
            password: Password for basic auth (will prompt if username provided but password not)
            headers: Additional headers
            **kwargs: Additional parameters
        """
        if username and not password:
            password = getpass.getpass(f"Enter password for {username}: ")
        
        if not api_key and not username:
            api_key = getpass.getpass("Enter API key: ")
        
        credential_data = {
            'base_url': base_url,
            'api_key': api_key,
            'username': username,
            'password': password,
            'headers': headers or {},
            **kwargs
        }
        
        self.store_credential(
            name,
            credential_data,
            description=f"API connection to {base_url}"
        )
    
    def get_database_connection_params(self, name: str) -> Dict[str, Any]:
        """Get database connection parameters from stored credential."""
        credential = self.retrieve_credential(name)
        
        # Validate required fields
        required_fields = ['db_type', 'database']
        missing_fields = [field for field in required_fields if field not in credential]
        if missing_fields:
            raise CredentialSecurityError(f"Missing required database fields: {missing_fields}")
        
        return credential
    
    def get_api_connection_params(self, name: str) -> Dict[str, Any]:
        """Get API connection parameters from stored credential."""
        credential = self.retrieve_credential(name)
        
        # Validate required fields
        if 'base_url' not in credential:
            raise CredentialSecurityError("Missing required API field: base_url")
        
        return credential
    
    def clear_cache(self) -> None:
        """Clear in-memory credential cache."""
        self._credential_cache.clear()
        logger.info("Credential cache cleared")


# Convenience functions for backward compatibility
def get_secure_credential_manager(
    encryption_password: Optional[str] = None,
    use_keyring: bool = True
) -> SecureCredentialManager:
    """
    Get a configured secure credential manager.
    
    Args:
        encryption_password: Password for deriving encryption key
        use_keyring: Whether to use system keyring
        
    Returns:
        Configured SecureCredentialManager instance
    """
    encryption_key = None
    if encryption_password:
        encryption_key = SecureCredentialManager.derive_key_from_password(encryption_password)
    
    return SecureCredentialManager(
        encryption_key=encryption_key,
        use_keyring=use_keyring
    )


# Global credential manager instance
_global_credential_manager = None

def get_global_credential_manager() -> SecureCredentialManager:
    """Get the global credential manager instance."""
    global _global_credential_manager
    if _global_credential_manager is None:
        _global_credential_manager = SecureCredentialManager(use_keyring=True)
    return _global_credential_manager


def store_database_credential(name: str, **kwargs) -> None:
    """Store database credential using global manager."""
    manager = get_global_credential_manager()
    manager.create_database_credential(name, **kwargs)


def store_api_credential(name: str, **kwargs) -> None:
    """Store API credential using global manager."""
    manager = get_global_credential_manager()
    manager.create_api_credential(name, **kwargs)


def get_database_credentials(name: str) -> Dict[str, Any]:
    """Get database credentials using global manager."""
    manager = get_global_credential_manager()
    return manager.get_database_connection_params(name)


def get_api_credentials(name: str) -> Dict[str, Any]:
    """Get API credentials using global manager."""
    manager = get_global_credential_manager()
    return manager.get_api_connection_params(name)