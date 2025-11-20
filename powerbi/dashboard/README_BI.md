# Power BI Dashboard Setup Guide

## Overview
This guide helps you set up and configure the Bank Churn Prediction Power BI dashboard with 5 comprehensive pages for churn analysis and customer retention.

## Dashboard Pages

### 📊 Page 1: Executive Overview
- **Purpose**: High-level business metrics and KPIs
- **Key Visuals**:
  - Overall churn rate and prediction accuracy
  - Model performance metrics (AUC, F1 Score, etc.)
  - Customer segmentation overview
  - Top churn drivers summary
- **Audience**: Executives, Business Leaders

### 🔍 Page 2: Risk Analysis
- **Purpose**: Detailed customer risk profiling
- **Key Visuals**:
  - Risk segment distribution (High/Medium/Low)
  - Churn probability distribution
  - Customer demographics by risk
  - Balance and tenure analysis
- **Audience**: Risk Managers, Marketing Teams

### 📈 Page 3: SHAP Drivers
- **Purpose**: Model interpretability and feature importance
- **Key Visuals**:
  - Top 15 churn drivers with SHAP values
  - Driver breakdown by business categories
  - Feature importance waterfall charts
  - Driver impact on churn probability
- **Audience**: Data Scientists, Analysts

### 👤 Page 4: Individual Analysis
- **Purpose**: Customer-level insights and explanations
- **Key Visuals**:
  - Individual customer risk profiles
  - SHAP force plots for specific customers
  - Customer comparison tools
  - Historical trend analysis
- **Audience**: Customer Service, Relationship Managers

### 🎯 Page 5: Retention Actions
- **Purpose**: Actionable insights for customer retention
- **Key Visuals**:
  - Retention cost-benefit analysis
  - Targeted intervention strategies
  - ROI calculations for retention programs
  - Success metrics tracking
- **Audience**: Marketing, Customer Success Teams

## Data Sources

The dashboard connects to these CSV files in `powerbi/data/`:

| File | Description | Refresh Frequency |
|------|-------------|-------------------|
| `model_metrics.csv` | Model performance metrics | Monthly |
| `predictions_data.csv` | Customer predictions + actuals | Weekly |
| `shap_values_summary.csv` | Feature importance scores | Monthly |
| `driver_breakdown.csv` | Churn drivers by category | Monthly |
| `fairness_metrics.csv` | Bias and fairness metrics | Monthly |
| `customer_segments.csv` | Risk segment analysis | Weekly |
| `top_drivers_data.csv` | Top 15 drivers | Monthly |
| `churn_by_tenure.csv` | Tenure-based analysis | Weekly |

## Setup Instructions

### Step 1: Prepare Data Files
```bash
# Run the Power BI export script
python src/export_to_powerbi.py