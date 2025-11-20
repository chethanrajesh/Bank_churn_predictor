"""
Tests for data validation and fairness checks.
"""

import pytest # type: ignore
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('src')

from src.validation import validate_data_quality, check_fairness_metrics
from src.exceptions import DataValidationError, FairnessCheckError


class TestValidation:
    """Test validation functionality."""
    
    @pytest.fixture
    def sample_data_with_demographics(self):
        """Create sample data with demographic information."""
        return pd.DataFrame({
            'age': [25, 35, 45, 55, 65, 25, 35, 45, 55, 65],
            'income': [50000, 75000, 100000, 125000, 150000] * 2,
            'region': ['North', 'South', 'East', 'West', 'North'] * 2,
            'gender': ['M', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F'],
            'churn_prediction': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'churn_probability': [0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.5, 0.8]
        })
    
    def test_validate_data_quality_success(self):
        """Test successful data quality validation."""
        clean_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'feature3': ['A', 'B', 'C', 'D', 'E']
        })
        
        quality_report = validate_data_quality(clean_data)
        
        assert quality_report is not None
        assert 'missing_values' in quality_report
        assert 'data_types' in quality_report
        assert 'basic_statistics' in quality_report
    
    def test_validate_data_quality_with_issues(self):
        """Test data quality validation with data issues."""
        problematic_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [1.1, -999, 3.3, 4.4, 5.5],  # -999 as missing value indicator
            'feature3': ['A', 'B', 'C', 'D', 'A']
        })
        
        quality_report = validate_data_quality(problematic_data)
        
        assert quality_report is not None
        assert quality_report['missing_values']['total_missing'] > 0
    
    def test_check_fairness_metrics(self, sample_data_with_demographics):
        """Test fairness metrics computation."""
        data = sample_data_with_demographics
        
        fairness_report = check_fairness_metrics(
            data, 
            protected_attribute='gender',
            prediction_col='churn_prediction',
            probability_col='churn_probability'
        )
        
        assert fairness_report is not None
        assert 'demographic_parity' in fairness_report
        assert 'equal_opportunity' in fairness_report
        assert 'predictive_equality' in fairness_report
    
    def test_fairness_check_missing_demographics(self):
        """Test fairness check with missing demographic data."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'churn_prediction': [0, 1, 0]
        })
        
        with pytest.raises(FairnessCheckError):
            check_fairness_metrics(data, protected_attribute='gender')