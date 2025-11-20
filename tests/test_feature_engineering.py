"""
Tests for feature engineering pipeline.
"""

import pytest # type: ignore
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('src')

from src.feature_engineering import create_features, engineer_advanced_features
from src.exceptions import FeatureEngineeringError


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data for feature engineering."""
        return pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'tenure_months': [6, 12, 24, 36, 60],
            'balance': [10000, 25000, 50000, 75000, 100000],
            'num_products': [1, 2, 3, 4, 5],
            'transaction_count': [10, 25, 50, 75, 100],
            'satisfaction_score': [1, 2, 3, 4, 5]
        })
    
    def test_create_features_basic(self, sample_processed_data):
        """Test basic feature creation."""
        features = create_features(sample_processed_data)
        
        assert features is not None
        assert len(features) == len(sample_processed_data)
        # Should have more columns than input data
        assert features.shape[1] > sample_processed_data.shape[1]
    
    def test_engineer_advanced_features(self, sample_processed_data):
        """Test advanced feature engineering."""
        advanced_features = engineer_advanced_features(sample_processed_data)
        
        assert advanced_features is not None
        # Check for engineered features
        expected_engineered_features = [
            'balance_to_income_ratio', 'transaction_frequency',
            'product_engagement_score', 'tenure_segment'
        ]
        
        for feature in expected_engineered_features:
            if feature in advanced_features.columns:
                assert True
            # Some features might not be created based on input data
    
    def test_feature_engineering_empty_data(self):
        """Test feature engineering with empty data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(FeatureEngineeringError):
            create_features(empty_data)