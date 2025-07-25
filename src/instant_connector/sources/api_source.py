"""API source connector for REST endpoints with JSON normalization and ML-ready extraction."""

import pandas as pd
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class APISource:
    """Connector for REST API data sources with response normalization and caching."""
    
    # Default cache settings
    CACHE_DIR = Path.home() / '.instant_connector' / 'api_cache'
    CACHE_EXPIRY_HOURS = 24
    
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Union[tuple, requests.auth.AuthBase]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit_delay: float = 0.1,
        cache_enabled: bool = True,
        cache_expiry_hours: Optional[int] = None
    ):
        """
        Initialize API source.
        
        Args:
            base_url: Base URL for the API
            headers: Default headers for requests
            auth: Authentication (tuple of username/password or auth object)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            rate_limit_delay: Delay between requests in seconds
            cache_enabled: Enable response caching
            cache_expiry_hours: Cache expiry time in hours
        """
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.auth = auth
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.cache_enabled = cache_enabled
        self.cache_expiry_hours = cache_expiry_hours or self.CACHE_EXPIRY_HOURS
        
        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        if headers:
            self.session.headers.update(headers)
        if auth:
            self.session.auth = auth
            
        # Initialize cache
        if cache_enabled:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
        # Track API metadata
        self._api_metadata = {
            'endpoints': {},
            'rate_limits': {},
            'response_times': []
        }
    
    def extract_data(
        self,
        endpoint: str,
        method: str = 'GET',
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        paginate: bool = False,
        pagination_type: str = 'offset',
        max_pages: Optional[int] = None,
        data_path: Optional[str] = None,
        transform_func: Optional[Callable] = None,
        normalize_json: bool = True,
        include_metadata: bool = True,
        validate_response: bool = True
    ) -> pd.DataFrame:
        """
        Extract data from API endpoint with JSON normalization.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Form data for POST requests
            json_data: JSON data for POST requests
            paginate: Whether to paginate through results
            pagination_type: Type of pagination ('offset', 'page', 'cursor', 'next_url', 'token')
            max_pages: Maximum number of pages to fetch
            data_path: Path to data in response (e.g., 'results' or 'data.items')
            transform_func: Function to transform response before creating DataFrame
            normalize_json: Normalize nested JSON structures
            include_metadata: Add metadata columns to DataFrame
            validate_response: Validate response data quality
            
        Returns:
            DataFrame with extracted and normalized data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        all_data = []
        page = 0
        next_token = None
        
        # Check cache first
        cache_key = self._get_cache_key(url, method, params, data, json_data)
        if self.cache_enabled and method == 'GET':
            cached_data = self._get_cached_response(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached data for {endpoint}")
                df = pd.DataFrame(cached_data)
                if include_metadata:
                    df = self._add_metadata_columns(df, endpoint, from_cache=True)
                return df
        
        while True:
            # Apply rate limiting
            if page > 0:
                time.sleep(self.rate_limit_delay)
                
            # Prepare pagination parameters
            if paginate and page > 0:
                params = self._update_pagination_params(
                    params or {},
                    pagination_type,
                    page,
                    all_data,
                    next_token
                )
            
            try:
                start_time = time.time()
                response = self._make_request(
                    url,
                    method=method,
                    params=params,
                    data=data,
                    json=json_data
                )
                response_time = time.time() - start_time
                self._api_metadata['response_times'].append(response_time)
                
                # Check for rate limit headers
                self._check_rate_limits(response)
                
                # Extract data from response
                page_data = self._extract_response_data(response, data_path)
                
                # Apply transformation if provided
                if transform_func:
                    page_data = transform_func(page_data)
                
                # Handle different response structures
                if isinstance(page_data, list):
                    all_data.extend(page_data)
                elif isinstance(page_data, dict):
                    # If dict contains a data array, extract it
                    if 'data' in page_data and isinstance(page_data['data'], list):
                        all_data.extend(page_data['data'])
                    else:
                        all_data.append(page_data)
                else:
                    logger.warning(f"Unexpected data type: {type(page_data)}")
                    break
                
                # Check if we should continue pagination
                if not paginate:
                    break
                    
                if max_pages and page >= max_pages - 1:
                    logger.info(f"Reached max pages limit ({max_pages})")
                    break
                    
                # Check for next page
                if pagination_type == 'next_url':
                    next_url = self._get_next_url(response)
                    if not next_url:
                        break
                    url = next_url
                elif pagination_type == 'token':
                    next_token = self._get_next_token(response, page_data)
                    if not next_token:
                        break
                elif not page_data or (isinstance(page_data, list) and len(page_data) == 0):
                    break
                    
                page += 1
                logger.info(f"Fetched page {page} from {endpoint}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch page {page}: {e}")
                if page == 0:
                    raise
                break
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                if page == 0:
                    raise
                break
        
        # Cache successful GET responses
        if self.cache_enabled and method == 'GET' and all_data:
            self._cache_response(cache_key, all_data)
        
        # Convert to DataFrame
        if not all_data:
            logger.warning(f"No data extracted from {endpoint}")
            return pd.DataFrame()
            
        df = self._create_dataframe(all_data, normalize_json)
        
        # Validate response data
        if validate_response:
            validation_report = self._validate_api_data(df, endpoint)
            self._api_metadata['endpoints'][endpoint] = validation_report
        
        # Add metadata
        if include_metadata:
            df = self._add_metadata_columns(df, endpoint)
        
        logger.info(f"Extracted {len(df)} records from API")
        
        return df
    
    def _make_request(
        self,
        url: str,
        method: str = 'GET',
        **kwargs
    ) -> requests.Response:
        """Make HTTP request with error handling and logging."""
        try:
            # Log request details
            logger.debug(f"{method} {url}")
            if 'params' in kwargs and kwargs['params']:
                logger.debug(f"Params: {kwargs['params']}")
                
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs
            )
            
            # Log response status
            logger.debug(f"Response: {response.status_code}")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {self.timeout}s: {url}")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: {url}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text[:200]}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def _extract_response_data(
        self,
        response: requests.Response,
        data_path: Optional[str] = None
    ) -> Union[List[Dict], Dict]:
        """Extract and validate data from response."""
        try:
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                logger.warning(f"Non-JSON response: {content_type}")
                
            data = response.json()
        except ValueError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise ValueError("Response is not valid JSON")
        
        # Navigate to data using path
        if data_path:
            original_data = data
            for key in data_path.split('.'):
                if isinstance(data, dict) and key in data:
                    data = data[key]
                elif isinstance(data, list) and key.isdigit():
                    data = data[int(key)]
                else:
                    logger.warning(f"Data path '{data_path}' not found in response")
                    return original_data
        
        return data
    
    def _create_dataframe(self, data: List[Union[Dict, Any]], normalize: bool = True) -> pd.DataFrame:
        """Create DataFrame from API data with normalization."""
        if not data:
            return pd.DataFrame()
        
        # Handle different data structures
        if all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        else:
            # Try to convert non-dict items
            processed_data = []
            for item in data:
                if isinstance(item, dict):
                    processed_data.append(item)
                elif isinstance(item, (list, tuple)):
                    # Convert to dict with index keys
                    processed_data.append({f'field_{i}': v for i, v in enumerate(item)})
                else:
                    # Wrap scalar values
                    processed_data.append({'value': item})
            
            df = pd.DataFrame(processed_data)
        
        # Normalize nested JSON structures
        if normalize and len(df) > 0:
            df = self._normalize_json_columns(df)
        
        # Optimize data types
        df = self._optimize_dtypes(df)
        
        return df
    
    def _normalize_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize nested JSON structures in DataFrame columns."""
        # Identify columns with nested data
        nested_cols = []
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                nested_cols.append(col)
        
        if not nested_cols:
            return df
        
        # Process each nested column
        for col in nested_cols:
            # Handle dict columns
            if df[col].apply(lambda x: isinstance(x, dict) if pd.notna(x) else False).any():
                # Extract nested data
                nested_data = pd.json_normalize(df[col].dropna())
                if not nested_data.empty:
                    # Prefix column names
                    nested_data.columns = [f"{col}.{subcol}" for subcol in nested_data.columns]
                    nested_data.index = df[col].dropna().index
                    
                    # Join with original DataFrame
                    df = df.drop(columns=[col]).join(nested_data)
            
            # Handle list columns
            elif df[col].apply(lambda x: isinstance(x, list) if pd.notna(x) else False).any():
                # Different strategies for list columns
                list_lengths = df[col].dropna().apply(len)
                
                if list_lengths.max() <= 10 and list_lengths.std() < 2:
                    # Expand lists into separate columns if they're small and consistent
                    max_len = int(list_lengths.max())
                    for i in range(max_len):
                        df[f"{col}_{i}"] = df[col].apply(
                            lambda x: x[i] if isinstance(x, list) and len(x) > i else None
                        )
                    df = df.drop(columns=[col])
                else:
                    # Convert to string representation for large/variable lists
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for ML workflows."""
        for col in df.columns:
            if col.startswith('_'):  # Skip metadata columns
                continue
                
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() / len(df) > 0.9:
                    df[col] = numeric_series
                    # Further optimize numeric type
                    if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
                        df[col] = df[col].astype('Int64')
                    continue
                
                # Try to parse as datetime
                if self._is_datetime_column(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        continue
                    except:
                        pass
                
                # Check for boolean
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2 and all(
                    str(v).lower() in ['true', 'false', '1', '0', 'yes', 'no'] 
                    for v in unique_vals
                ):
                    df[col] = df[col].map({
                        'true': True, 'false': False, 'True': True, 'False': False,
                        '1': True, '0': False, 'yes': True, 'no': False,
                        'Yes': True, 'No': False, 'YES': True, 'NO': False
                    })
                    continue
                
                # Convert low cardinality to categorical
                if df[col].nunique() < 0.5 * len(df) and df[col].nunique() < 100:
                    df[col] = df[col].astype('category')
            
            elif pd.api.types.is_numeric_dtype(col_type):
                # Optimize numeric types
                df[col] = pd.to_numeric(df[col], downcast='integer' if pd.api.types.is_integer_dtype(col_type) else 'float')
        
        return df
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if a column likely contains datetime values."""
        if series.dtype != 'object':
            return False
            
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        # Common datetime patterns in APIs
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # Standard format
            r'\d{4}/\d{2}/\d{2}',                     # Date only
            r'\d{10,13}'                              # Unix timestamp
        ]
        
        for pattern in datetime_patterns:
            if sample.astype(str).str.match(pattern).any():
                return True
                
        return False
    
    def _update_pagination_params(
        self,
        params: Dict[str, Any],
        pagination_type: str,
        page: int,
        all_data: List,
        next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update parameters for pagination based on type."""
        params = params.copy()
        
        if pagination_type == 'offset':
            # Offset-based pagination
            page_size = params.get('limit', params.get('per_page', params.get('page_size', 100)))
            params['offset'] = page * page_size
            if 'limit' not in params and 'per_page' not in params:
                params['limit'] = page_size
                
        elif pagination_type == 'page':
            # Page-based pagination (usually 1-indexed)
            params['page'] = page + 1
            
        elif pagination_type == 'cursor' and all_data:
            # Cursor-based pagination
            last_item = all_data[-1]
            if isinstance(last_item, dict):
                # Look for cursor in common fields
                cursor_fields = ['cursor', 'next_cursor', 'id', '_id', 'after']
                for field in cursor_fields:
                    if field in last_item:
                        params['cursor'] = last_item[field]
                        break
                        
        elif pagination_type == 'token' and next_token:
            # Token-based pagination
            params['page_token'] = next_token
        
        return params
    
    def _get_next_url(self, response: requests.Response) -> Optional[str]:
        """Extract next URL from response for pagination."""
        # Check Link header (RFC 5988)
        link_header = response.headers.get('Link', '')
        if link_header:
            import re
            match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
            if match:
                return match.group(1)
        
        # Check response body
        try:
            data = response.json()
            if isinstance(data, dict):
                # Common fields for next URL
                next_fields = ['next', 'next_url', 'nextPage', 'next_page', 'links.next']
                for field in next_fields:
                    if '.' in field:
                        # Handle nested fields
                        parts = field.split('.')
                        value = data
                        for part in parts:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                value = None
                                break
                        if value:
                            return value
                    elif field in data and data[field]:
                        return data[field]
        except:
            pass
            
        return None
    
    def _get_next_token(self, response: requests.Response, data: Any) -> Optional[str]:
        """Extract next token from response for token-based pagination."""
        try:
            response_data = response.json()
            
            # Common fields for next token
            token_fields = ['next_page_token', 'nextPageToken', 'next_token', 'continuation_token']
            
            # Check in response data
            if isinstance(response_data, dict):
                for field in token_fields:
                    if field in response_data and response_data[field]:
                        return response_data[field]
            
            # Check in last data item
            if isinstance(data, list) and data:
                last_item = data[-1]
                if isinstance(last_item, dict):
                    for field in token_fields:
                        if field in last_item and last_item[field]:
                            return last_item[field]
        except:
            pass
            
        return None
    
    def _check_rate_limits(self, response: requests.Response):
        """Check and store rate limit information from response headers."""
        rate_limit_headers = {
            'X-RateLimit-Limit': 'limit',
            'X-RateLimit-Remaining': 'remaining',
            'X-RateLimit-Reset': 'reset',
            'X-Rate-Limit-Limit': 'limit',
            'X-Rate-Limit-Remaining': 'remaining',
            'X-Rate-Limit-Reset': 'reset'
        }
        
        rate_info = {}
        for header, key in rate_limit_headers.items():
            if header in response.headers:
                try:
                    value = int(response.headers[header])
                    rate_info[key] = value
                except ValueError:
                    pass
        
        if rate_info:
            self._api_metadata['rate_limits'] = rate_info
            
            # Log warning if approaching limit
            if 'remaining' in rate_info and 'limit' in rate_info:
                remaining_pct = rate_info['remaining'] / rate_info['limit']
                if remaining_pct < 0.1:
                    logger.warning(f"API rate limit warning: {rate_info['remaining']} requests remaining")
            
            # Auto-throttle if needed
            if 'remaining' in rate_info and rate_info['remaining'] < 10:
                logger.info("Approaching rate limit, increasing delay")
                self.rate_limit_delay = min(self.rate_limit_delay * 2, 5.0)
    
    def _add_metadata_columns(self, df: pd.DataFrame, endpoint: str, from_cache: bool = False) -> pd.DataFrame:
        """Add metadata columns for data lineage and tracking."""
        df['_source_api'] = self.base_url
        df['_endpoint'] = endpoint
        df['_extraction_timestamp'] = datetime.now()
        df['_from_cache'] = from_cache
        
        # Add response time if available
        if self._api_metadata['response_times']:
            df['_avg_response_time'] = sum(self._api_metadata['response_times']) / len(self._api_metadata['response_times'])
        
        # Add row hash for data integrity
        data_cols = [col for col in df.columns if not col.startswith('_')]
        if data_cols:
            df['_row_hash'] = df[data_cols].apply(
                lambda row: hashlib.md5(
                    json.dumps(row.to_dict(), sort_keys=True).encode()
                ).hexdigest()[:8], axis=1
            )
        
        return df
    
    def _validate_api_data(self, df: pd.DataFrame, endpoint: str) -> Dict[str, Any]:
        """Validate API response data quality."""
        validation_report = {
            'endpoint': endpoint,
            'row_count': len(df),
            'column_count': len([col for col in df.columns if not col.startswith('_')]),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'data_types': df.dtypes.value_counts().to_dict(),
            'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'issues': []
        }
        
        # Check for common API data issues
        if len(df) == 0:
            validation_report['issues'].append('empty_response')
        
        if df.duplicated().any():
            dup_count = df.duplicated().sum()
            validation_report['issues'].append(f'duplicate_rows: {dup_count}')
        
        # Check for suspicious patterns
        for col in df.columns:
            if col.startswith('_'):
                continue
                
            # All null column
            if df[col].isnull().all():
                validation_report['issues'].append(f'all_null_column: {col}')
            
            # Single value column
            elif df[col].nunique() == 1:
                validation_report['issues'].append(f'constant_column: {col}')
            
            # High null percentage
            elif df[col].isnull().sum() / len(df) > 0.9:
                validation_report['issues'].append(f'high_null_column: {col}')
        
        validation_report['quality_score'] = self._calculate_api_quality_score(validation_report)
        
        return validation_report
    
    def _calculate_api_quality_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate quality score for API data (0-100)."""
        score = 100.0
        
        # Deduct for issues
        issue_penalties = {
            'empty_response': 100,
            'duplicate_rows': 20,
            'all_null_column': 15,
            'constant_column': 10,
            'high_null_column': 5
        }
        
        for issue in validation_report['issues']:
            for issue_type, penalty in issue_penalties.items():
                if issue.startswith(issue_type):
                    score -= penalty
                    break
        
        # Deduct for high null percentage
        score -= min(validation_report['null_percentage'], 20)
        
        return max(0, round(score, 2))
    
    def _get_cache_key(self, url: str, method: str, params: Any, data: Any, json_data: Any) -> str:
        """Generate cache key for request."""
        key_parts = [
            url,
            method,
            json.dumps(params, sort_keys=True) if params else '',
            json.dumps(data, sort_keys=True) if data else '',
            json.dumps(json_data, sort_keys=True) if json_data else ''
        ]
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached response if available and not expired."""
        cache_file = self.CACHE_DIR / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is expired
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if file_age_hours > self.cache_expiry_hours:
            logger.debug(f"Cache expired ({file_age_hours:.1f} hours old)")
            cache_file.unlink()
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                logger.debug(f"Cache hit ({file_age_hours:.1f} hours old)")
                return cached_data
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def _cache_response(self, cache_key: str, data: List[Dict]):
        """Cache response data."""
        cache_file = self.CACHE_DIR / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Cached response ({len(data)} items)")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def clear_cache(self):
        """Clear all cached responses."""
        if self.CACHE_DIR.exists():
            for cache_file in self.CACHE_DIR.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared")
    
    def batch_extract(
        self,
        endpoints: List[Dict[str, Any]],
        concat: bool = True,
        parallel: bool = False,
        max_workers: int = 5
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Extract data from multiple endpoints.
        
        Args:
            endpoints: List of endpoint configurations
            concat: Whether to concatenate results
            parallel: Execute requests in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            DataFrame or list of DataFrames
        """
        dataframes = []
        
        if parallel and len(endpoints) > 1:
            # Parallel execution
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_config = {
                    executor.submit(self.extract_data, **config): config
                    for config in endpoints
                }
                
                # Collect results
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        df = future.result()
                        dataframes.append(df)
                    except Exception as e:
                        logger.error(f"Failed to extract from {config.get('endpoint', 'unknown')}: {e}")
        else:
            # Sequential execution
            for config in endpoints:
                try:
                    endpoint = config.pop('endpoint')
                    df = self.extract_data(endpoint, **config)
                    dataframes.append(df)
                except Exception as e:
                    logger.error(f"Failed to extract from endpoint: {e}")
        
        if concat and dataframes:
            return pd.concat(dataframes, ignore_index=True)
        
        return dataframes
    
    def test_connection(self, test_endpoint: str = '/') -> bool:
        """Test API connection and authentication."""
        try:
            url = f"{self.base_url}/{test_endpoint.lstrip('/')}"
            response = self._make_request(url, method='GET')
            logger.info(f"API connection successful: {response.status_code}")
            return True
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            return False
    
    def get_api_metadata(self) -> Dict[str, Any]:
        """Get collected API metadata and statistics."""
        metadata = self._api_metadata.copy()
        
        # Add summary statistics
        if metadata['response_times']:
            metadata['avg_response_time'] = sum(metadata['response_times']) / len(metadata['response_times'])
            metadata['min_response_time'] = min(metadata['response_times'])
            metadata['max_response_time'] = max(metadata['response_times'])
        
        return metadata