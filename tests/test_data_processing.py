"""
Tests for data processing pipeline.
"""

import pytest # type: ignore
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append('src')

from src.data_processing import load_and_validate_data, preprocess_data
from src.exceptions import DataValidationError, DataProcessingError


class TestDataProcessing:
    """Test data processing functionality."""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for testing."""
        return pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'age': [25, 35, 45],
            'income': [50000, 75000, 100000],
            'region': ['North', 'South', 'East'],
            'tenure_months': [12, 24, 36],
            'balance': [10000.0, 50000.0, 75000.0]
        })
    
    def test_load_and_validate_data_success(self, sample_raw_data):
        """Test successful data loading and validation."""
        with patch('pandas.read_csv') as mock_read:
            mock_read.return_value = sample_raw_data
            
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                
                data = load_and_validate_data("dummy_path.csv")
                assert data is not None
                assert len(data) == 3
    
    def test_load_and_validate_data_file_not_found(self):
        """Test data loading with missing file."""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            with pytest.raises(DataValidationError):
                load_and_validate_data("nonexistent_file.csv")
    
    def test_preprocess_data(self, sample_raw_data):
        """Test data preprocessing."""
        # Add some missing values and outliers for testing
        test_data = sample_raw_data.copy()
        test_data.loc[0, 'age'] = np.nan
        test_data.loc[1, 'balance'] = -1000  # Invalid value
        
        processed_data = preprocess_data(test_data)
        
        assert processed_data is not None
        assert processed_data['age'].isna().sum() == 0  # Missing values handled
        assert (processed_data['balance'] >= 0).all()  # Invalid values handled