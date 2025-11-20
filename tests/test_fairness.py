"""
Tests for fairness and bias detection.
"""

import pytest # type: ignore
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('src')

from src.validation import check_fairness_metrics, detect_bias
from src.exceptions import FairnessCheckError


class TestFairness:
    """Test fairness and bias detection."""
    
    @pytest.fixture
    def sample_fairness_data(self):
        """Create sample data for fairness testing."""
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'age': np.random.randint(18, 70, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
            'income': np.random.normal(50000, 20000, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'actual_churn': np.random.randint(0, 2, n_samples),
            'predicted_churn': np.random.randint(0, 2, n_samples),
            'churn_probability': np.random.uniform(0, 1, n_samples)
        })
        
        # Introduce some bias for testing
        data.loc[data['gender'] == 'F', 'predicted_churn'] = np.random.choice(
            [0, 1], sum(data['gender'] == 'F'), p=[0.7, 0.3]
        )
        data.loc[data['gender'] == 'M', 'predicted_churn'] = np.random.choice(
            [0, 1], sum(data['gender'] == 'M'), p=[0.5, 0.5]
        )
        
        return data
    
    def test_detect_bias(self, sample_fairness_data):
        """Test bias detection."""
        data = sample_fairness_data
        
        bias_report = detect_bias(
            data,
            protected_attribute='gender',
            prediction_col='predicted_churn',
            target_col='actual_churn'
        )
        
        assert bias_report is not None
        assert 'disparate_impact' in bias_report
        assert 'statistical_parity' in bias_report
        assert 'bias_detected' in bias_report
    
    def test_fairness_metrics_comprehensive(self, sample_fairness_data):
        """Test comprehensive fairness metrics."""
        data = sample_fairness_data
        
        fairness_report = check_fairness_metrics(
            data,
            protected_attribute='gender',
            prediction_col='predicted_churn',
            probability_col='churn_probability',
            target_col='actual_churn'
        )
        
        expected_metrics = [
            'demographic_parity',
            'equal_opportunity', 
            'predictive_equality',
            'false_positive_balance',
            'auc_parity'
        ]
        
        for metric in expected_metrics:
            assert metric in fairness_report
    
    def test_fairness_missing_target(self):
        """Test fairness check with missing target variable."""
        data = pd.DataFrame({
            'gender': ['M', 'F', 'M', 'F'],
            'prediction': [0, 1, 0, 1]
        })
        
        with pytest.raises(FairnessCheckError):
            check_fairness_metrics(data, 'gender', 'prediction', target_col='missing_target')