"""Tests for API source connector."""

import pytest
import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile

from instant_connector.sources.api_source import APISource


class TestAPISource:
    """Test suite for APISource class."""
    
    @pytest.fixture
    def api_source(self):
        """Create APISource instance."""
        return APISource(
            base_url="https://api.example.com",
            headers={"X-API-Key": "test-key"},
            timeout=30,
            max_retries=3
        )
    
    @pytest.fixture
    def mock_response(self):
        """Create mock response."""
        mock = Mock(spec=requests.Response)
        mock.status_code = 200
        mock.headers = {'Content-Type': 'application/json'}
        mock.json.return_value = [
            {'id': 1, 'name': 'Item 1', 'value': 10},
            {'id': 2, 'name': 'Item 2', 'value': 20}
        ]
        mock.raise_for_status = Mock()
        return mock
    
    def test_init(self):
        """Test initialization."""
        source = APISource(
            base_url="https://api.example.com/",
            headers={"Authorization": "Bearer token"},
            auth=("user", "pass"),
            timeout=60,
            max_retries=5,
            rate_limit_delay=0.5,
            cache_enabled=False
        )
        
        assert source.base_url == "https://api.example.com"  # Trailing slash removed
        assert source.headers == {"Authorization": "Bearer token"}
        assert source.auth == ("user", "pass")
        assert source.timeout == 60
        assert source.rate_limit_delay == 0.5
        assert source.cache_enabled is False
    
    def test_session_configuration(self, api_source):
        """Test session configuration with retry strategy."""
        assert api_source.session is not None
        adapters = api_source.session.adapters
        assert "http://" in adapters
        assert "https://" in adapters
        
        # Check retry configuration
        adapter = adapters["https://"]
        assert adapter.max_retries.total == 3
        assert 429 in adapter.max_retries.status_forcelist
    
    @patch('requests.Session.request')
    def test_extract_data_basic(self, mock_request, api_source, mock_response):
        """Test basic data extraction."""
        mock_request.return_value = mock_response
        
        df = api_source.extract_data(
            endpoint="/items",
            method="GET",
            include_metadata=False
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'value']
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_extract_data_with_params(self, mock_request, api_source, mock_response):
        """Test data extraction with query parameters."""
        mock_request.return_value = mock_response
        
        df = api_source.extract_data(
            endpoint="/items",
            params={'category': 'electronics', 'limit': 10},
            include_metadata=False
        )
        
        call_args = mock_request.call_args
        assert call_args[1]['params'] == {'category': 'electronics', 'limit': 10}
    
    @patch('requests.Session.request')
    def test_extract_data_post(self, mock_request, api_source):
        """Test POST request data extraction."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {'result': 'success', 'data': [{'id': 1}]}
        mock_request.return_value = mock_response
        
        df = api_source.extract_data(
            endpoint="/create",
            method="POST",
            json_data={'name': 'New Item'},
            data_path='data',
            include_metadata=False
        )
        
        assert len(df) == 1
        call_args = mock_request.call_args
        assert call_args[1]['json'] == {'name': 'New Item'}
    
    @patch('requests.Session.request')
    def test_extract_data_with_pagination_offset(self, mock_request, api_source):
        """Test pagination with offset strategy."""
        # Mock paginated responses
        page1 = Mock()
        page1.status_code = 200
        page1.headers = {'Content-Type': 'application/json'}
        page1.json.return_value = [{'id': 1}, {'id': 2}]
        
        page2 = Mock()
        page2.status_code = 200
        page2.headers = {'Content-Type': 'application/json'}
        page2.json.return_value = [{'id': 3}, {'id': 4}]
        
        page3 = Mock()
        page3.status_code = 200
        page3.headers = {'Content-Type': 'application/json'}
        page3.json.return_value = []  # Empty response to stop pagination
        
        mock_request.side_effect = [page1, page2, page3]
        
        df = api_source.extract_data(
            endpoint="/items",
            paginate=True,
            pagination_type='offset',
            params={'limit': 2},
            include_metadata=False
        )
        
        assert len(df) == 4
        assert mock_request.call_count == 3
        
        # Check offset parameters
        call_args_list = mock_request.call_args_list
        assert call_args_list[0][1]['params'] == {'limit': 2}
        assert call_args_list[1][1]['params'] == {'limit': 2, 'offset': 2}
        assert call_args_list[2][1]['params'] == {'limit': 2, 'offset': 4}
    
    @patch('requests.Session.request')
    def test_extract_data_with_pagination_page(self, mock_request, api_source):
        """Test pagination with page-based strategy."""
        page1 = Mock()
        page1.status_code = 200
        page1.headers = {'Content-Type': 'application/json'}
        page1.json.return_value = [{'id': 1}, {'id': 2}]
        
        page2 = Mock()
        page2.status_code = 200
        page2.headers = {'Content-Type': 'application/json'}
        page2.json.return_value = []
        
        mock_request.side_effect = [page1, page2]
        
        df = api_source.extract_data(
            endpoint="/items",
            paginate=True,
            pagination_type='page',
            include_metadata=False
        )
        
        assert len(df) == 2
        
        # Check page parameters
        call_args_list = mock_request.call_args_list
        assert 'page' not in call_args_list[0][1].get('params', {})
        assert call_args_list[1][1]['params']['page'] == 2
    
    @patch('requests.Session.request')
    def test_extract_data_with_pagination_next_url(self, mock_request, api_source):
        """Test pagination with next URL strategy."""
        page1 = Mock()
        page1.status_code = 200
        page1.headers = {
            'Content-Type': 'application/json',
            'Link': '<https://api.example.com/items?page=2>; rel="next"'
        }
        page1.json.return_value = {'data': [{'id': 1}], 'next': 'https://api.example.com/items?page=2'}
        
        page2 = Mock()
        page2.status_code = 200
        page2.headers = {'Content-Type': 'application/json'}
        page2.json.return_value = {'data': [{'id': 2}]}
        
        mock_request.side_effect = [page1, page2]
        
        df = api_source.extract_data(
            endpoint="/items",
            paginate=True,
            pagination_type='next_url',
            data_path='data',
            include_metadata=False
        )
        
        assert len(df) == 2
        assert mock_request.call_count == 2
        
        # Check that second request used next URL
        second_call_url = mock_request.call_args_list[1][0][1]
        assert second_call_url == 'https://api.example.com/items?page=2'
    
    @patch('requests.Session.request')
    def test_extract_data_with_transform(self, mock_request, api_source, mock_response):
        """Test data extraction with transformation function."""
        mock_request.return_value = mock_response
        
        def transform(data):
            # Add computed field
            for item in data:
                item['value_squared'] = item['value'] ** 2
            return data
        
        df = api_source.extract_data(
            endpoint="/items",
            transform_func=transform,
            include_metadata=False
        )
        
        assert 'value_squared' in df.columns
        assert df['value_squared'].tolist() == [100, 400]
    
    @patch('requests.Session.request')
    def test_extract_data_with_data_path(self, mock_request, api_source):
        """Test data extraction with nested data path."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {
            'status': 'ok',
            'results': {
                'items': [{'id': 1}, {'id': 2}]
            }
        }
        mock_request.return_value = mock_response
        
        df = api_source.extract_data(
            endpoint="/search",
            data_path='results.items',
            include_metadata=False
        )
        
        assert len(df) == 2
    
    @patch('requests.Session.request')
    def test_extract_data_with_metadata(self, mock_request, api_source, mock_response):
        """Test data extraction with metadata columns."""
        mock_request.return_value = mock_response
        
        df = api_source.extract_data(
            endpoint="/items",
            include_metadata=True
        )
        
        # Check metadata columns
        assert '_source_api' in df.columns
        assert '_endpoint' in df.columns
        assert '_extraction_timestamp' in df.columns
        assert '_from_cache' in df.columns
        assert '_row_hash' in df.columns
        
        assert df['_source_api'].iloc[0] == "https://api.example.com"
        assert df['_endpoint'].iloc[0] == "/items"
        assert df['_from_cache'].iloc[0] is False
    
    def test_normalize_json_columns(self, api_source):
        """Test JSON column normalization."""
        df = pd.DataFrame([
            {'id': 1, 'meta': {'age': 25, 'city': 'NYC'}, 'tags': ['a', 'b']},
            {'id': 2, 'meta': {'age': 30, 'city': 'LA'}, 'tags': ['c', 'd', 'e']}
        ])
        
        normalized = api_source._normalize_json_columns(df)
        
        # Check nested dict was normalized
        assert 'meta.age' in normalized.columns
        assert 'meta.city' in normalized.columns
        assert 'meta' not in normalized.columns
        
        # Check list was converted to string
        assert normalized['tags'].dtype == 'object'
        assert isinstance(normalized['tags'].iloc[0], str)
    
    def test_optimize_dtypes(self, api_source):
        """Test DataFrame dtype optimization."""
        df = pd.DataFrame({
            'numeric_str': ['1', '2', '3'],
            'float_str': ['1.5', '2.5', '3.5'],
            'bool_str': ['true', 'false', 'true'],
            'date_str': ['2023-01-01T00:00:00', '2023-01-02T00:00:00', '2023-01-03T00:00:00'],
            'cat_str': ['A', 'B', 'A', 'B', 'A'],
            '_metadata': ['skip', 'skip', 'skip', 'skip', 'skip']
        })
        
        optimized = api_source._optimize_dtypes(df)
        
        assert pd.api.types.is_integer_dtype(optimized['numeric_str'])
        assert pd.api.types.is_float_dtype(optimized['float_str'])
        assert optimized['bool_str'].dtype == bool
        assert pd.api.types.is_datetime64_any_dtype(optimized['date_str'])
        assert pd.api.types.is_categorical_dtype(optimized['cat_str'])
        assert optimized['_metadata'].dtype == 'object'  # Should skip
    
    def test_is_datetime_column(self, api_source):
        """Test datetime column detection."""
        # ISO format
        iso_series = pd.Series(['2023-01-01T10:30:00', '2023-01-02T11:45:00'])
        assert api_source._is_datetime_column(iso_series) is True
        
        # Unix timestamp
        timestamp_series = pd.Series(['1672531200', '1672617600'])
        assert api_source._is_datetime_column(timestamp_series) is True
        
        # Not datetime
        text_series = pd.Series(['hello', 'world'])
        assert api_source._is_datetime_column(text_series) is False
    
    @patch('requests.Session.request')
    def test_rate_limit_handling(self, mock_request, api_source):
        """Test rate limit detection and handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'Content-Type': 'application/json',
            'X-RateLimit-Limit': '100',
            'X-RateLimit-Remaining': '5',
            'X-RateLimit-Reset': '1234567890'
        }
        mock_response.json.return_value = []
        mock_request.return_value = mock_response
        
        # Store original delay
        original_delay = api_source.rate_limit_delay
        
        api_source.extract_data("/items", include_metadata=False)
        
        # Check rate limit info was stored
        assert api_source._api_metadata['rate_limits']['limit'] == 100
        assert api_source._api_metadata['rate_limits']['remaining'] == 5
        
        # Rate limit delay should increase when approaching limit
        assert api_source.rate_limit_delay > original_delay
    
    @patch('requests.Session.request')
    def test_error_handling_timeout(self, mock_request, api_source):
        """Test timeout error handling."""
        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(requests.exceptions.Timeout):
            api_source.extract_data("/items")
    
    @patch('requests.Session.request')
    def test_error_handling_http_error(self, mock_request, api_source):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_request.return_value = mock_response
        
        with pytest.raises(requests.exceptions.HTTPError):
            api_source.extract_data("/items")
    
    @patch('requests.Session.request')
    def test_error_handling_invalid_json(self, mock_request, api_source):
        """Test invalid JSON response handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_request.return_value = mock_response
        
        with pytest.raises(ValueError, match="Response is not valid JSON"):
            api_source.extract_data("/items")
    
    def test_caching_enabled(self, api_source):
        """Test response caching functionality."""
        # Ensure cache directory exists
        assert api_source.CACHE_DIR.exists()
        
        # Test cache key generation
        cache_key = api_source._get_cache_key(
            "https://api.example.com/items",
            "GET",
            {'page': 1},
            None,
            None
        )
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
    
    @patch('requests.Session.request')
    def test_caching_hit(self, mock_request, api_source):
        """Test cache hit scenario."""
        # Create cached data
        cached_data = [{'id': 1, 'cached': True}]
        cache_key = api_source._get_cache_key(
            f"{api_source.base_url}/items",
            "GET",
            None,
            None,
            None
        )
        api_source._cache_response(cache_key, cached_data)
        
        # Extract data - should use cache
        df = api_source.extract_data("/items", include_metadata=True)
        
        assert len(df) == 1
        assert df['_from_cache'].iloc[0] is True
        mock_request.assert_not_called()  # Should not make request
        
        # Clean up
        api_source.clear_cache()
    
    def test_validate_api_data(self, api_source):
        """Test API data validation."""
        df = pd.DataFrame({
            'id': [1, 2, 2, 3],  # Duplicate
            'value': [10, None, None, None],  # High null percentage
            'constant': [1, 1, 1, 1],  # Constant value
            'empty': [None, None, None, None]  # All null
        })
        
        report = api_source._validate_api_data(df, "/items")
        
        assert report['endpoint'] == "/items"
        assert report['duplicate_rows'] == 1
        assert report['null_percentage'] > 0
        assert 'duplicate_rows: 1' in report['issues']
        assert 'all_null_column: empty' in report['issues']
        assert 'constant_column: constant' in report['issues']
        assert report['quality_score'] < 100
    
    @patch('requests.Session.request')
    def test_batch_extract_sequential(self, mock_request, api_source):
        """Test batch extraction in sequential mode."""
        # Mock responses for different endpoints
        response1 = Mock()
        response1.status_code = 200
        response1.headers = {'Content-Type': 'application/json'}
        response1.json.return_value = [{'id': 1, 'type': 'A'}]
        
        response2 = Mock()
        response2.status_code = 200
        response2.headers = {'Content-Type': 'application/json'}
        response2.json.return_value = [{'id': 2, 'type': 'B'}]
        
        mock_request.side_effect = [response1, response2]
        
        endpoints = [
            {'endpoint': '/type-a'},
            {'endpoint': '/type-b'}
        ]
        
        result = api_source.batch_extract(endpoints, concat=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert mock_request.call_count == 2
    
    @patch('requests.Session.request')
    def test_batch_extract_parallel(self, mock_request, api_source):
        """Test batch extraction in parallel mode."""
        response = Mock()
        response.status_code = 200
        response.headers = {'Content-Type': 'application/json'}
        response.json.return_value = [{'id': 1}]
        mock_request.return_value = response
        
        endpoints = [
            {'endpoint': '/endpoint1'},
            {'endpoint': '/endpoint2'},
            {'endpoint': '/endpoint3'}
        ]
        
        results = api_source.batch_extract(endpoints, concat=False, parallel=True, max_workers=2)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(df, pd.DataFrame) for df in results)
    
    @patch('requests.Session.request')
    def test_test_connection_success(self, mock_request, api_source):
        """Test connection testing - success."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        
        result = api_source.test_connection()
        
        assert result is True
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_test_connection_failure(self, mock_request, api_source):
        """Test connection testing - failure."""
        mock_request.side_effect = Exception("Connection failed")
        
        result = api_source.test_connection()
        
        assert result is False
    
    def test_get_api_metadata(self, api_source):
        """Test API metadata retrieval."""
        # Add some test data
        api_source._api_metadata['response_times'] = [0.1, 0.2, 0.15]
        api_source._api_metadata['rate_limits'] = {'limit': 100, 'remaining': 50}
        
        metadata = api_source.get_api_metadata()
        
        assert 'avg_response_time' in metadata
        assert 'min_response_time' in metadata
        assert 'max_response_time' in metadata
        assert metadata['avg_response_time'] == pytest.approx(0.15)
        assert metadata['min_response_time'] == 0.1
        assert metadata['max_response_time'] == 0.2
    
    def test_update_pagination_params_cursor(self, api_source):
        """Test cursor pagination parameter update."""
        params = {}
        all_data = [{'id': 1, 'cursor': 'abc123'}]
        
        updated = api_source._update_pagination_params(
            params,
            'cursor',
            1,
            all_data,
            None
        )
        
        assert updated['cursor'] == 'abc123'
    
    def test_update_pagination_params_token(self, api_source):
        """Test token pagination parameter update."""
        params = {}
        next_token = 'next_page_token_123'
        
        updated = api_source._update_pagination_params(
            params,
            'token',
            1,
            [],
            next_token
        )
        
        assert updated['page_token'] == 'next_page_token_123'
    
    def test_get_next_url_from_header(self, api_source):
        """Test extracting next URL from Link header."""
        response = Mock()
        response.headers = {
            'Link': '<https://api.example.com/items?page=2>; rel="next", <https://api.example.com/items?page=10>; rel="last"'
        }
        response.json.return_value = {}
        
        next_url = api_source._get_next_url(response)
        assert next_url == 'https://api.example.com/items?page=2'
    
    def test_get_next_url_from_body(self, api_source):
        """Test extracting next URL from response body."""
        response = Mock()
        response.headers = {}
        response.json.return_value = {
            'data': [],
            'next': 'https://api.example.com/items?page=2'
        }
        
        next_url = api_source._get_next_url(response)
        assert next_url == 'https://api.example.com/items?page=2'
    
    def test_get_next_token(self, api_source):
        """Test extracting next token from response."""
        response = Mock()
        response.json.return_value = {
            'items': [{'id': 1}],
            'next_page_token': 'token123'
        }
        
        next_token = api_source._get_next_token(response, [{'id': 1}])
        assert next_token == 'token123'
    
    def test_create_dataframe_mixed_types(self, api_source):
        """Test DataFrame creation with mixed data types."""
        # Test with dict items
        dict_data = [{'id': 1}, {'id': 2}]
        df = api_source._create_dataframe(dict_data)
        assert len(df) == 2
        
        # Test with list items
        list_data = [[1, 'a'], [2, 'b']]
        df = api_source._create_dataframe(list_data)
        assert len(df) == 2
        assert 'field_0' in df.columns
        assert 'field_1' in df.columns
        
        # Test with scalar items
        scalar_data = [1, 2, 3]
        df = api_source._create_dataframe(scalar_data)
        assert len(df) == 3
        assert 'value' in df.columns
    
    def test_clear_cache(self, api_source):
        """Test cache clearing."""
        # Create some cache files
        cache_file = api_source.CACHE_DIR / "test_cache.json"
        cache_file.write_text('{"test": true}')
        
        assert cache_file.exists()
        
        api_source.clear_cache()
        
        assert not cache_file.exists()