"""
Bank Churn Prediction System
============================

A comprehensive machine learning system for predicting customer churn in banking,
with explainable AI using SHAP values and business-driven insights.

Key Features:
- 52 engineered features across 10 churn drivers
- Multiple ML models (XGBoost, Random Forest, LightGBM)
- SHAP explanations for interpretability
- Fairness and bias detection
- Interactive Streamlit UI
- Power BI dashboard integration

Author: Data Science Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"
__email__ = "datascience@bank.com"

import logging

# Import and export only the functions we currently have
from .utils import (
    ensure_dir,
    get_project_root,
    load_json,
    save_json,
    load_csv_data,
    save_csv_data,
    load_pickle,
    save_pickle,
    get_timestamp,
    setup_logging
)

# Define public API
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Core utilities
    'ensure_dir',
    'get_project_root',
    'load_json',
    'save_json',
    'load_csv_data',
    'save_csv_data',
    'load_pickle',
    'save_pickle',
    'get_timestamp',
    'setup_logging'
]

# Package initialization
try:
    # Setup default logging when package is imported
    setup_logging()
except Exception as e:
    # Fallback basic logging if custom setup fails
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"Failed to setup custom logging: {e}")