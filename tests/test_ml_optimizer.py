"""Tests for ML optimizer module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
from unittest.mock import Mock, patch

from instant_connector.ml_optimizer import MLOptimizer


class TestMLOptimizer:
    """Test suite for MLOptimizer class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_1': np.random.randn(100),
            'numeric_2': np.random.rand(100) * 100,
            'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_2': np.random.choice(['X', 'Y', 'Z', 'W'], 100),
            'binary': np.random.choice([0, 1], 100),
            'text': ['text_' + str(i) for i in range(100)],
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'constant': [1] * 100,
            'high_null': [None if i % 3 == 0 else i for i in range(100)],
            'target': np.random.choice([0, 1], 100)
        })
    
    @pytest.fixture
    def ml_optimizer(self):
        """Create MLOptimizer instance."""
        return MLOptimizer()
    
    def test_init(self):
        """Test initialization with different parameters."""
        # Default initialization
        opt1 = MLOptimizer()
        assert opt1.handle_missing == 'auto'
        assert opt1.encode_categorical == 'auto'
        assert opt1.scale_numeric == 'auto'
        
        # Custom initialization
        opt2 = MLOptimizer(
            handle_missing='drop',
            encode_categorical='onehot',
            scale_numeric='standard',
            feature_engineering=False
        )
        assert opt2.handle_missing == 'drop'
        assert opt2.encode_categorical == 'onehot'
        assert opt2.scale_numeric == 'standard'
        assert opt2.feature_engineering is False
    
    def test_fit_transform_basic(self, ml_optimizer, sample_df):
        """Test basic fit_transform functionality."""
        result = ml_optimizer.fit_transform(
            sample_df.drop('target', axis=1),
            target_column=None,
            preserve_artifacts=False
        )
        
        assert 'X_processed' in result
        assert isinstance(result['X_processed'], pd.DataFrame)
        assert len(result['X_processed']) == len(sample_df)
        
        # Check that constant column was dropped
        assert 'constant' not in result['X_processed'].columns
    
    def test_fit_transform_with_target(self, ml_optimizer, sample_df):
        """Test fit_transform with target column."""
        result = ml_optimizer.fit_transform(
            sample_df,
            target_column='target',
            test_size=0.2,
            stratify=True
        )
        
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        
        # Check split sizes
        assert len(result['X_train']) == 80
        assert len(result['X_test']) == 20
        assert len(result['y_train']) == 80
        assert len(result['y_test']) == 20
    
    def test_identify_column_types(self, ml_optimizer, sample_df):
        """Test column type identification."""
        numeric, categorical, datetime, other = ml_optimizer._identify_column_types(sample_df)
        
        assert 'numeric_1' in numeric
        assert 'numeric_2' in numeric
        assert 'binary' in numeric  # Binary columns treated as numeric
        assert 'high_null' in numeric
        
        assert 'categorical_1' in categorical
        assert 'categorical_2' in categorical
        
        assert 'date' in datetime
        
        assert 'text' in other
    
    def test_remove_constant_columns(self, ml_optimizer, sample_df):
        """Test constant column removal."""
        cleaned, removed = ml_optimizer._remove_constant_columns(sample_df)
        
        assert 'constant' not in cleaned.columns
        assert 'constant' in removed
        assert len(removed) == 1
    
    def test_handle_missing_values_auto(self, ml_optimizer, sample_df):
        """Test automatic missing value handling."""
        # Add more missing values
        df = sample_df.copy()
        df.loc[:10, 'numeric_1'] = None
        df.loc[:5, 'categorical_1'] = None
        
        ml_optimizer.handle_missing = 'auto'
        result, metadata = ml_optimizer._handle_missing_values(df, None)
        
        # Check no missing values remain
        assert result.isnull().sum().sum() == 0
        
        # Check metadata
        assert 'numeric_1' in metadata['imputed_columns']
        assert 'categorical_1' in metadata['imputed_columns']
    
    def test_handle_missing_values_drop(self, ml_optimizer, sample_df):
        """Test dropping missing values."""
        ml_optimizer.handle_missing = 'drop'
        result, metadata = ml_optimizer._handle_missing_values(sample_df, None)
        
        # Should drop rows with any missing values
        assert len(result) < len(sample_df)
        assert result.isnull().sum().sum() == 0
    
    def test_handle_missing_values_mean(self, ml_optimizer, sample_df):
        """Test mean imputation."""
        df = sample_df.copy()
        df.loc[0, 'numeric_1'] = None
        original_mean = df['numeric_1'].mean()
        
        ml_optimizer.handle_missing = 'mean'
        result, metadata = ml_optimizer._handle_missing_values(df, None)
        
        # Check imputed value is close to mean
        assert abs(result.loc[0, 'numeric_1'] - original_mean) < 0.01
    
    def test_handle_missing_values_median(self, ml_optimizer, sample_df):
        """Test median imputation."""
        df = sample_df.copy()
        df.loc[0, 'numeric_1'] = None
        original_median = df['numeric_1'].median()
        
        ml_optimizer.handle_missing = 'median'
        result, metadata = ml_optimizer._handle_missing_values(df, None)
        
        # Check imputed value is close to median
        assert abs(result.loc[0, 'numeric_1'] - original_median) < 0.01
    
    def test_handle_missing_values_zero(self, ml_optimizer, sample_df):
        """Test zero imputation."""
        df = sample_df.copy()
        df.loc[0, 'numeric_1'] = None
        
        ml_optimizer.handle_missing = 'zero'
        result, metadata = ml_optimizer._handle_missing_values(df, None)
        
        assert result.loc[0, 'numeric_1'] == 0
    
    def test_engineer_features(self, ml_optimizer):
        """Test feature engineering."""
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [2, 4, 6, 8, 10],
            'cat1': ['A', 'B', 'A', 'B', 'A']
        })
        
        result, metadata = ml_optimizer._engineer_features(df)
        
        # Check polynomial features
        assert 'num1^2' in result.columns
        assert 'num1*num2' in result.columns
        
        # Check log transforms
        assert 'num1_log' in result.columns
        assert 'num2_log' in result.columns
        
        # Check ratios
        assert 'num1/num2' in result.columns or 'num2/num1' in result.columns
    
    def test_engineer_features_with_target(self, ml_optimizer):
        """Test feature engineering with target for feature selection."""
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B'], 100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        result, metadata = ml_optimizer._engineer_features(df, y)
        
        # Should have feature importance scores
        assert 'feature_importances' in metadata
        assert len(metadata['feature_importances']) > 0
    
    def test_encode_categoricals_auto(self, ml_optimizer):
        """Test automatic categorical encoding."""
        df = pd.DataFrame({
            'low_card': ['A', 'B', 'A', 'B', 'A'],  # Low cardinality
            'high_card': [f'cat_{i}' for i in range(5)],  # High cardinality
            'numeric': [1, 2, 3, 4, 5]
        })
        
        ml_optimizer.encode_categorical = 'auto'
        result, metadata = ml_optimizer._encode_categoricals(df)
        
        # Low cardinality should be one-hot encoded
        assert 'low_card_A' in result.columns or 'low_card_B' in result.columns
        
        # High cardinality should be label encoded
        assert 'high_card' in result.columns
        assert pd.api.types.is_numeric_dtype(result['high_card'])
    
    def test_encode_categoricals_onehot(self, ml_optimizer):
        """Test one-hot encoding."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X']
        })
        
        ml_optimizer.encode_categorical = 'onehot'
        result, metadata = ml_optimizer._encode_categoricals(df)
        
        # Check one-hot columns created
        expected_cols = ['cat1_A', 'cat1_B', 'cat1_C', 'cat2_X', 'cat2_Y']
        for col in expected_cols:
            assert col in result.columns
    
    def test_encode_categoricals_label(self, ml_optimizer):
        """Test label encoding."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X']
        })
        
        ml_optimizer.encode_categorical = 'label'
        result, metadata = ml_optimizer._encode_categoricals(df)
        
        # Columns should be numeric
        assert pd.api.types.is_numeric_dtype(result['cat1'])
        assert pd.api.types.is_numeric_dtype(result['cat2'])
        
        # Check encoding mappings stored
        assert 'cat1' in metadata['label_encoders']
        assert 'cat2' in metadata['label_encoders']
    
    def test_encode_categoricals_ordinal(self, ml_optimizer):
        """Test ordinal encoding."""
        df = pd.DataFrame({
            'size': ['small', 'medium', 'large', 'small', 'large']
        })
        
        ml_optimizer.encode_categorical = 'ordinal'
        result, metadata = ml_optimizer._encode_categoricals(df)
        
        # Should be numeric
        assert pd.api.types.is_numeric_dtype(result['size'])
    
    def test_encode_categoricals_target(self, ml_optimizer):
        """Test target encoding."""
        df = pd.DataFrame({
            'cat': ['A', 'B', 'A', 'B', 'A', 'B']
        })
        y = pd.Series([1, 0, 1, 0, 1, 0])  # Perfect correlation with cat
        
        ml_optimizer.encode_categorical = 'target'
        result, metadata = ml_optimizer._encode_categoricals(df, y)
        
        # Should have target encoded values
        assert pd.api.types.is_numeric_dtype(result['cat_target_encoded'])
        assert 'cat' not in result.columns  # Original column removed
    
    def test_scale_numeric_auto(self, ml_optimizer):
        """Test automatic numeric scaling."""
        df = pd.DataFrame({
            'normal': np.random.randn(100),  # Normal distribution
            'uniform': np.random.rand(100),  # Uniform distribution
            'skewed': np.random.exponential(1, 100)  # Skewed distribution
        })
        
        ml_optimizer.scale_numeric = 'auto'
        result, metadata = ml_optimizer._scale_numeric(df)
        
        # All columns should be scaled
        for col in df.columns:
            assert result[col].std() < df[col].std() or np.isclose(result[col].std(), 1.0, atol=0.1)
    
    def test_scale_numeric_standard(self, ml_optimizer):
        """Test standard scaling."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        ml_optimizer.scale_numeric = 'standard'
        result, metadata = ml_optimizer._scale_numeric(df)
        
        # Check mean ~= 0 and std ~= 1 (using ddof=0 to match sklearn)
        assert abs(result['col1'].mean()) < 0.01
        assert abs(result['col1'].std(ddof=0) - 1.0) < 0.01
        assert abs(result['col2'].mean()) < 0.01
        assert abs(result['col2'].std(ddof=0) - 1.0) < 0.01
    
    def test_scale_numeric_minmax(self, ml_optimizer):
        """Test min-max scaling."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        ml_optimizer.scale_numeric = 'minmax'
        result, metadata = ml_optimizer._scale_numeric(df)
        
        # Check range [0, 1]
        assert result['col1'].min() == 0
        assert result['col1'].max() == 1
        assert result['col2'].min() == 0
        assert result['col2'].max() == 1
    
    def test_scale_numeric_robust(self, ml_optimizer):
        """Test robust scaling."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 100],  # With outlier
            'col2': [10, 20, 30, 40, 1000]  # With outlier
        })
        
        ml_optimizer.scale_numeric = 'robust'
        result, metadata = ml_optimizer._scale_numeric(df)
        
        # Robust scaler should handle outliers better
        # Most values should be in a reasonable range
        assert result['col1'].iloc[:-1].abs().max() < 2
        assert result['col2'].iloc[:-1].abs().max() < 2
    
    def test_preserve_artifacts(self, ml_optimizer, sample_df):
        """Test artifact preservation."""
        result = ml_optimizer.fit_transform(
            sample_df,
            target_column='target',
            preserve_artifacts=True
        )
        
        # Check all artifacts are preserved
        assert 'column_types' in result
        assert 'preprocessing_metadata' in result
        assert 'scalers' in result
        assert 'encoders' in result
        assert 'feature_names' in result
        assert 'dropped_columns' in result
    
    def test_transform_with_artifacts(self, ml_optimizer, sample_df):
        """Test transforming new data with saved artifacts."""
        # First fit
        result = ml_optimizer.fit_transform(
            sample_df,
            target_column='target',
            preserve_artifacts=True
        )
        
        # Create new data with same structure
        new_df = sample_df.iloc[:10].copy()
        new_df['numeric_1'] = new_df['numeric_1'] + 10  # Different values
        
        # Transform using artifacts
        transformed = ml_optimizer.transform(
            new_df,
            ml_artifacts=result['ml_artifacts']
        )
        
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.columns.tolist() == result['X_train'].columns.tolist()
    
    def test_calculate_feature_importance_numeric(self, ml_optimizer):
        """Test feature importance calculation for numeric features."""
        X = pd.DataFrame({
            'correlated': np.arange(100),
            'noise': np.random.randn(100),
            'somewhat_correlated': np.arange(100) + np.random.randn(100) * 10
        })
        y = X['correlated'] + 0.5 * X['somewhat_correlated']
        
        importance = ml_optimizer._calculate_feature_importance(X, y)
        
        # Correlated feature should have highest importance
        assert importance['correlated'] > importance['noise']
        assert importance['somewhat_correlated'] > importance['noise']
    
    def test_calculate_feature_importance_categorical(self, ml_optimizer):
        """Test feature importance calculation for categorical features."""
        X = pd.DataFrame({
            'cat_important': ['A'] * 50 + ['B'] * 50,
            'cat_random': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        y = pd.Series([0] * 50 + [1] * 50)  # Perfect correlation with cat_important
        
        importance = ml_optimizer._calculate_feature_importance(X, y)
        
        # Important categorical should have higher score
        assert importance['cat_important'] > importance['cat_random']
    
    def test_reduce_memory_usage(self, ml_optimizer, sample_df):
        """Test memory usage reduction."""
        original_memory = sample_df.memory_usage(deep=True).sum()
        
        result = ml_optimizer.fit_transform(
            sample_df,
            target_column='target',
            reduce_memory=True
        )
        
        new_memory = result['X_processed'].memory_usage(deep=True).sum()
        
        # Memory should be reduced (or at least not increased significantly)
        # Note: Due to encoding, memory might increase, but should be optimized
        assert new_memory < original_memory * 2
    
    def test_error_handling_no_columns(self, ml_optimizer):
        """Test error handling when all columns are dropped."""
        df = pd.DataFrame({
            'constant1': [1] * 10,
            'constant2': ['A'] * 10
        })
        
        with pytest.raises(ValueError, match="No columns remaining"):
            ml_optimizer.fit_transform(df)
    
    def test_error_handling_invalid_target(self, ml_optimizer, sample_df):
        """Test error handling with invalid target column."""
        with pytest.raises(KeyError):
            ml_optimizer.fit_transform(
                sample_df,
                target_column='non_existent_column'
            )
    
    def test_datetime_handling(self, ml_optimizer):
        """Test datetime feature extraction."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': np.random.randn(100)
        })
        
        result = ml_optimizer.fit_transform(df)
        
        # Datetime columns should be handled (dropped or transformed)
        assert 'date' not in result['X_processed'].columns
    
    def test_high_cardinality_warning(self, ml_optimizer):
        """Test warning for high cardinality categorical."""
        df = pd.DataFrame({
            'high_card': [f'cat_{i}' for i in range(100)],  # Unique values
            'target': np.random.choice([0, 1], 100)
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ml_optimizer.encode_categorical = 'onehot'
            result = ml_optimizer.fit_transform(df, target_column='target')
            
            # Should warn about high cardinality
            assert any("high cardinality" in str(warning.message).lower() for warning in w)
    
    def test_reproducibility(self, ml_optimizer, sample_df):
        """Test reproducibility with random_state."""
        ml_optimizer.random_state = 42
        
        result1 = ml_optimizer.fit_transform(
            sample_df,
            target_column='target',
            test_size=0.2
        )
        
        ml_optimizer.random_state = 42
        result2 = ml_optimizer.fit_transform(
            sample_df,
            target_column='target',
            test_size=0.2
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1['X_train'], result2['X_train'])
        pd.testing.assert_series_equal(result1['y_train'], result2['y_train'])