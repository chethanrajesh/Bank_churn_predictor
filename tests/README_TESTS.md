# Test Suite Documentation

## Overview
This directory contains comprehensive tests for the Bank Churn Prediction system.

## Test Files Structure

- `test_data_processing.py` - Data loading, validation, and preprocessing tests
- `test_feature_engineering.py` - Feature creation and transformation tests  
- `test_model_training.py` - Model training and evaluation tests
- `test_shap_explanations.py` - SHAP value computation and explanation tests
- `test_inference.py` - Production inference pipeline tests
- `test_validation.py` - Data quality and validation tests
- `test_fairness.py` - Fairness and bias detection tests
- `test_export_to_powerbi.py` - Power BI export functionality tests

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v