"""Tests for file source connector."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import csv
from unittest.mock import Mock, patch, mock_open

from instant_connector.sources.file_source import FileSource


class TestFileSource:
    """Test suite for FileSource class."""
    
    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'value', 'date'])
            writer.writerow(['1', 'Alice', '10.5', '2023-01-01'])
            writer.writerow(['2', 'Bob', '20.3', '2023-01-02'])
            writer.writerow(['3', 'Charlie', '30.1', '2023-01-03'])
            temp_path = f.name
        
        yield Path(temp_path)
        Path(temp_path).unlink()
    
    @pytest.fixture
    def temp_json_file(self):
        """Create a temporary JSON file."""
        data = [
            {'id': 1, 'name': 'Alice', 'meta': {'age': 25, 'city': 'NYC'}},
            {'id': 2, 'name': 'Bob', 'meta': {'age': 30, 'city': 'LA'}}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        yield Path(temp_path)
        Path(temp_path).unlink()
    
    @pytest.fixture
    def temp_excel_file(self):
        """Create a temporary Excel file."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(temp_path) as writer:
            df1 = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
            df2 = pd.DataFrame({'col3': [3, 4], 'col4': ['c', 'd']})
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            df2.to_excel(writer, sheet_name='Sheet2', index=False)
        
        yield Path(temp_path)
        Path(temp_path).unlink()
    
    def test_init_single_file(self, temp_csv_file):
        """Test initialization with single file."""
        source = FileSource(temp_csv_file)
        assert len(source.file_paths) == 1
        assert source.file_paths[0] == temp_csv_file
    
    def test_init_multiple_files(self, temp_csv_file, temp_json_file):
        """Test initialization with multiple files."""
        source = FileSource([temp_csv_file, temp_json_file])
        assert len(source.file_paths) == 2
    
    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            FileSource('/non/existent/file.csv')
    
    def test_init_unsupported_format(self):
        """Test initialization with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                FileSource(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_extract_csv_basic(self, temp_csv_file):
        """Test basic CSV extraction."""
        source = FileSource(temp_csv_file)
        df = source.extract_data(optimize_dtypes=False, include_metadata=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'value', 'date']
    
    def test_extract_csv_with_optimization(self, temp_csv_file):
        """Test CSV extraction with dtype optimization."""
        source = FileSource(temp_csv_file)
        df = source.extract_data(optimize_dtypes=True, include_metadata=False)
        
        # Check optimized types
        assert pd.api.types.is_integer_dtype(df['id'])
        assert pd.api.types.is_float_dtype(df['value'])
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
    
    def test_extract_csv_with_metadata(self, temp_csv_file):
        """Test CSV extraction with metadata columns."""
        source = FileSource(temp_csv_file)
        df = source.extract_data(include_metadata=True)
        
        # Check metadata columns
        assert '_source_file' in df.columns
        assert '_file_name' in df.columns
        assert '_file_type' in df.columns
        assert '_extraction_timestamp' in df.columns
        assert '_file_size_mb' in df.columns
        assert '_row_hash' in df.columns
    
    def test_extract_csv_with_sample(self, temp_csv_file):
        """Test CSV extraction with sampling."""
        source = FileSource(temp_csv_file)
        df = source.extract_data(sample_size=2, include_metadata=False)
        
        assert len(df) == 2
    
    def test_detect_encoding(self, temp_csv_file):
        """Test encoding detection."""
        source = FileSource(temp_csv_file)
        encoding = source._detect_encoding(temp_csv_file)
        assert encoding in ['utf-8', 'ascii', 'UTF-8', 'ASCII']
    
    def test_detect_delimiter(self):
        """Test delimiter detection."""
        # Test comma delimiter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("a,b,c\n")
            f.write("d,e,f\n")
            temp_path = Path(f.name)
        
        source = FileSource([temp_path])
        delimiter = source._detect_delimiter(temp_path)
        assert delimiter == ','
        temp_path.unlink()
        
        # Test tab delimiter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("col1\tcol2\tcol3\n")
            f.write("a\tb\tc\n")
            f.write("d\te\tf\n")
            temp_path = Path(f.name)
        
        delimiter = source._detect_delimiter(temp_path)
        assert delimiter == '\t'
        temp_path.unlink()
    
    def test_refine_column_types(self):
        """Test column type refinement."""
        source = FileSource([])
        
        # Create test DataFrame
        df = pd.DataFrame({
            'numeric_str': ['1', '2', '3'],
            'float_str': ['1.5', '2.5', '3.5'],
            'date_str': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'bool_str': ['true', 'false', 'true'],
            'mixed': ['1', '2', 'abc']
        })
        
        refined = source._refine_column_types(df)
        
        assert pd.api.types.is_numeric_dtype(refined['numeric_str'])
        assert pd.api.types.is_float_dtype(refined['float_str'])
        assert pd.api.types.is_datetime64_any_dtype(refined['date_str'])
        assert refined['bool_str'].dtype == bool
        assert refined['mixed'].dtype == 'object'  # Should remain object due to mixed content
    
    def test_is_datetime_column(self):
        """Test datetime column detection."""
        source = FileSource([])
        
        # Test various date formats
        date_series1 = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        assert source._is_datetime_column(date_series1) is True
        
        date_series2 = pd.Series(['01/15/2023', '02/20/2023', '03/25/2023'])
        assert source._is_datetime_column(date_series2) is True
        
        non_date_series = pd.Series(['abc', 'def', 'ghi'])
        assert source._is_datetime_column(non_date_series) is False
    
    def test_is_boolean_column(self):
        """Test boolean column detection."""
        source = FileSource([])
        
        bool_series1 = pd.Series(['true', 'false', 'true'])
        assert source._is_boolean_column(bool_series1) is True
        
        bool_series2 = pd.Series(['yes', 'no', 'yes', 'no'])
        assert source._is_boolean_column(bool_series2) is True
        
        bool_series3 = pd.Series(['1', '0', '1', '0'])
        assert source._is_boolean_column(bool_series3) is True
        
        non_bool_series = pd.Series(['abc', 'def', 'ghi'])
        assert source._is_boolean_column(non_bool_series) is False
    
    def test_extract_json_array(self, temp_json_file):
        """Test JSON array extraction."""
        source = FileSource(temp_json_file)
        df = source.extract_data(optimize_dtypes=False, include_metadata=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # Check nested JSON was normalized
        assert 'meta.age' in df.columns or 'meta_age' in df.columns
        assert 'meta.city' in df.columns or 'meta_city' in df.columns
    
    def test_extract_ndjson(self):
        """Test newline-delimited JSON extraction."""
        # Create NDJSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"id": 1, "name": "Alice"}\n')
            f.write('{"id": 2, "name": "Bob"}\n')
            f.write('{"id": 3, "name": "Charlie"}\n')
            temp_path = Path(f.name)
        
        source = FileSource(temp_path)
        df = source.extract_data(include_metadata=False)
        
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name']
        
        temp_path.unlink()
    
    def test_extract_excel_single_sheet(self, temp_excel_file):
        """Test Excel extraction with single sheet."""
        source = FileSource(temp_excel_file)
        df = source.extract_data(sheet_name='Sheet1', include_metadata=False)
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['col1', 'col2']
        assert len(df) == 2
    
    def test_extract_excel_all_sheets(self, temp_excel_file):
        """Test Excel extraction with all sheets."""
        source = FileSource(temp_excel_file)
        df = source.extract_data(include_metadata=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 rows from each sheet
        assert '_sheet_name' in df.columns
        assert set(df['_sheet_name'].unique()) == {'Sheet1', 'Sheet2'}
    
    def test_extract_parquet(self):
        """Test Parquet extraction."""
        # Create Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = Path(f.name)
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.5, 20.3, 30.1],
            'category': pd.Categorical(['A', 'B', 'A'])
        })
        df.to_parquet(temp_path)
        
        source = FileSource(temp_path)
        result = source.extract_data(include_metadata=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert pd.api.types.is_categorical_dtype(result['category'])
        
        temp_path.unlink()
    
    def test_extract_text(self):
        """Test text file extraction."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Line 1\n")
            f.write("Line 2\n")
            f.write("Line 3\n")
            temp_path = Path(f.name)
        
        source = FileSource(temp_path)
        df = source.extract_data(include_metadata=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'text' in df.columns
        assert 'line_number' in df.columns
        assert df['line_number'].tolist() == [1, 2, 3]
        
        temp_path.unlink()
    
    def test_optimize_dtypes(self):
        """Test DataFrame dtype optimization."""
        source = FileSource([])
        
        df = pd.DataFrame({
            'int64_col': np.array([1, 2, 3], dtype='int64'),
            'float64_col': np.array([1.0, 2.0, 3.0], dtype='float64'),
            'float_as_int': np.array([1.0, 2.0, 3.0], dtype='float64'),
            'cat_col': ['A', 'B', 'A', 'B', 'A'],
            '_metadata': ['skip', 'skip', 'skip', 'skip', 'skip']
        })
        
        optimized = source._optimize_dtypes(df)
        
        # Check downcasting
        assert optimized['int64_col'].dtype in ['int8', 'int16', 'int32']
        assert optimized['float_as_int'].dtype in ['int8', 'int16', 'int32', 'Int8', 'Int16', 'Int32', 'Int64']
        assert pd.api.types.is_categorical_dtype(optimized['cat_col'])
        assert optimized['_metadata'].dtype == 'object'  # Should skip
    
    def test_detect_schema(self, temp_csv_file):
        """Test schema detection."""
        source = FileSource(temp_csv_file)
        df = source.extract_data(include_metadata=False)
        schema = source._detect_schema(df)
        
        assert 'columns' in schema
        assert 'shape' in schema
        assert 'memory_usage_mb' in schema
        
        # Check column information
        assert 'id' in schema['columns']
        col_info = schema['columns']['id']
        assert 'dtype' in col_info
        assert 'nullable' in col_info
        assert 'unique_count' in col_info
        assert 'sample_values' in col_info
    
    def test_schema_compatibility(self):
        """Test schema compatibility checking."""
        source = FileSource([])
        
        schema1 = {
            'columns': {
                'col1': {'dtype': 'int64'},
                'col2': {'dtype': 'object'}
            }
        }
        
        schema2 = {
            'columns': {
                'col1': {'dtype': 'int32'},  # Compatible numeric type
                'col2': {'dtype': 'object'}
            }
        }
        
        schema3 = {
            'columns': {
                'col1': {'dtype': 'int64'},
                'col3': {'dtype': 'object'}  # Different column
            }
        }
        
        assert source._check_schema_compatibility([schema1, schema2]) is True
        assert source._check_schema_compatibility([schema1, schema3]) is False
    
    def test_reconcile_schemas(self):
        """Test schema reconciliation."""
        source = FileSource([])
        
        df1 = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        df2 = pd.DataFrame({'col1': [3.0, 4.0], 'col3': ['c', 'd']})
        
        reconciled = source._reconcile_schemas([df1, df2])
        
        # Check all DataFrames have same columns
        assert len(reconciled) == 2
        assert set(reconciled[0].columns) == set(reconciled[1].columns)
        assert 'col3' in reconciled[0].columns  # Added missing column
        assert 'col2' in reconciled[1].columns  # Added missing column
    
    def test_validate_data_quality(self, temp_csv_file):
        """Test data quality validation."""
        source = FileSource(temp_csv_file)
        df = source.extract_data(include_metadata=False)
        
        # Add some quality issues
        df.loc[1, 'value'] = None  # Add null
        df.loc[0, 'id'] = df.loc[1, 'id']  # Create duplicate
        
        report = source.validate_data_quality(df, str(temp_csv_file))
        
        assert 'file_name' in report
        assert 'row_count' in report
        assert 'duplicate_rows' in report
        assert 'completeness_score' in report
        assert 'quality_score' in report
        assert 'columns' in report
        
        # Check specific column reports
        assert 'value' in report['columns']
        value_report = report['columns']['value']
        assert value_report['null_count'] == 1
        assert value_report['null_percentage'] > 0
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        source = FileSource([])
        
        # Create series with outliers
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        outliers = source._detect_outliers(series)
        assert outliers > 0
        
        # Create series without outliers
        series_normal = pd.Series([1, 2, 3, 4, 5])
        outliers_normal = source._detect_outliers(series_normal)
        assert outliers_normal == 0
    
    def test_detect_patterns(self):
        """Test pattern detection in string data."""
        source = FileSource([])
        
        # Test email pattern
        email_series = pd.Series(['test@example.com', 'user@domain.org', 'admin@site.net'])
        assert source._detect_patterns(email_series) == 'email'
        
        # Test date pattern
        date_series = pd.Series(['2023-01-01', '2023-02-15', '2023-03-30'])
        assert source._detect_patterns(date_series) == 'date'
        
        # Test phone pattern
        phone_series = pd.Series(['123-456-7890', '987-654-3210', '555-123-4567'])
        assert source._detect_patterns(phone_series) == 'phone'
        
        # Test no pattern
        random_series = pd.Series(['abc', 'def123', 'xyz789'])
        assert source._detect_patterns(random_series) is None
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        source = FileSource([])
        
        # Good quality report
        good_report = {
            'completeness_score': 0.95,
            'duplicate_rows': 0,
            'row_count': 100,
            'columns': {
                'col1': {'quality_issues': []},
                'col2': {'quality_issues': []}
            }
        }
        
        score = source._calculate_quality_score(good_report)
        assert score > 90
        
        # Poor quality report
        poor_report = {
            'completeness_score': 0.5,
            'duplicate_rows': 20,
            'row_count': 100,
            'columns': {
                'col1': {'quality_issues': ['high_null_percentage']},
                'col2': {'quality_issues': ['constant_value']}
            }
        }
        
        score = source._calculate_quality_score(poor_report)
        assert score < 50
    
    def test_get_file_info(self, temp_csv_file, temp_json_file):
        """Test getting file information."""
        source = FileSource([temp_csv_file, temp_json_file])
        
        # Extract data to populate caches
        source.extract_data()
        
        info = source.get_file_info()
        
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 2
        assert 'file_name' in info.columns
        assert 'file_path' in info.columns
        assert 'format' in info.columns
        assert 'size_mb' in info.columns
        assert 'modified' in info.columns
        assert 'created' in info.columns
    
    @patch('builtins.open', side_effect=OSError("File too large"))
    def test_large_file_handling(self, mock_open):
        """Test handling of large files."""
        # Create a mock large file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
        
        # Mock file size to trigger large file warning
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 600 * 1024 * 1024  # 600 MB
            
            source = FileSource(temp_path)
            # Should not raise, just log warning
            assert len(source.file_paths) == 1
        
        temp_path.unlink()
    
    def test_multiple_file_concatenation(self, temp_csv_file):
        """Test concatenating multiple files."""
        # Create second CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'value', 'date'])
            writer.writerow(['4', 'David', '40.5', '2023-01-04'])
            writer.writerow(['5', 'Eve', '50.3', '2023-01-05'])
            temp_path2 = Path(f.name)
        
        source = FileSource([temp_csv_file, temp_path2])
        df = source.extract_data(concat=True, include_metadata=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 3 + 2 rows
        
        temp_path2.unlink()
    
    def test_multiple_file_no_concat(self, temp_csv_file, temp_json_file):
        """Test extracting multiple files without concatenation."""
        source = FileSource([temp_csv_file, temp_json_file])
        dfs = source.extract_data(concat=False, include_metadata=False)
        
        assert isinstance(dfs, list)
        assert len(dfs) == 2
        assert all(isinstance(df, pd.DataFrame) for df in dfs)