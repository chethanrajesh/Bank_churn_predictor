"""
Tests for model training pipeline.
"""

import pytest # type: ignore
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append('src')

from src.model_training import train_models, evaluate_model
from src.exceptions import ModelTrainingError


class TestModelTraining:
    """Test model training functionality."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_train_models_success(self, sample_training_data):
        """Test successful model training."""
        X, y = sample_training_data
        
        with patch('src.model_training.XGBClassifier') as mock_xgb:
            with patch('src.model_training.RandomForestClassifier') as mock_rf:
                with patch('src.model_training.LGBMClassifier') as mock_lgb:
                    # Mock model instances
                    mock_xgb_instance = MagicMock()
                    mock_rf_instance = MagicMock()
                    mock_lgb_instance = MagicMock()
                    
                    mock_xgb.return_value = mock_xgb_instance
                    mock_rf.return_value = mock_rf_instance
                    mock_lgb.return_value = mock_lgb_instance
                    
                    # Mock fit method
                    mock_xgb_instance.fit.return_value = None
                    mock_rf_instance.fit.return_value = None
                    mock_lgb_instance.fit.return_value = None
                    
                    models = train_models(X, y)
                    
                    assert models is not None
                    assert 'xgboost' in models
                    assert 'random_forest' in models
                    assert 'lightgbm' in models
    
    def test_evaluate_model(self, sample_training_data):
        """Test model evaluation."""
        X, y = sample_training_data
        
        with patch('sklearn.metrics.roc_auc_score') as mock_auc:
            with patch('sklearn.metrics.accuracy_score') as mock_acc:
                mock_auc.return_value = 0.85
                mock_acc.return_value = 0.82
                
                mock_model = MagicMock()
                mock_model.predict_proba.return_value = np.column_stack([
                    np.random.uniform(0, 1, len(y)),
                    np.random.uniform(0, 1, len(y))
                ])
                mock_model.predict.return_value = np.random.randint(0, 2, len(y))
                
                metrics = evaluate_model(mock_model, X, y)
                
                assert 'auc' in metrics
                assert 'accuracy' in metrics
                assert 'precision' in metrics
                assert 'recall' in metrics
    
    def test_train_models_insufficient_data(self):
        """Test model training with insufficient data."""
        X = pd.DataFrame({'feature1': [1, 2]})  # Only 2 samples
        y = np.array([0, 1])
        
        with pytest.raises(ModelTrainingError):
            train_models(X, y)