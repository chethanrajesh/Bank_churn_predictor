"""
Tests for SHAP explanations computation.
"""

import pytest # type: ignore
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append('src')

from src.shap_explanations import compute_shap_values, generate_global_explanations
from src.exceptions import SHAPExplanationError


class TestSHAPExplanations:
    """Test SHAP explanations functionality."""
    
    @pytest.fixture
    def sample_model_data(self):
        """Create sample model and data for SHAP testing."""
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'feature3': np.random.normal(0, 1, 50)
        })
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.column_stack([
            np.random.uniform(0, 1, 50),
            np.random.uniform(0, 1, 50)
        ])
        
        return mock_model, X
    
    def test_compute_shap_values_success(self, sample_model_data):
        """Test successful SHAP values computation."""
        mock_model, X = sample_model_data
        
        with patch('shap.TreeExplainer') as mock_explainer:
            mock_shap_values = np.random.normal(0, 1, (50, 3))
            mock_explainer_instance = MagicMock()
            mock_explainer_instance.shap_values.return_value = mock_shap_values
            mock_explainer.return_value = mock_explainer_instance
            
            shap_results = compute_shap_values(mock_model, X)
            
            assert shap_results is not None
            assert 'shap_values' in shap_results
            assert 'explainer' in shap_results
    
    def test_generate_global_explanations(self, sample_model_data):
        """Test global explanations generation."""
        mock_model, X = sample_model_data
        
        mock_shap_values = np.random.normal(0, 1, (50, 3))
        
        explanations = generate_global_explanations(mock_shap_values, X)
        
        assert explanations is not None
        assert 'feature_importance' in explanations
        assert 'summary_plot_data' in explanations
    
    def test_shap_computation_memory_error(self, sample_model_data):
        """Test SHAP computation with memory issues."""
        mock_model, X = sample_model_data
        
        with patch('shap.TreeExplainer') as mock_explainer:
            mock_explainer.side_effect = MemoryError("Not enough memory")
            
            with pytest.raises(SHAPExplanationError):
                compute_shap_values(mock_model, X)