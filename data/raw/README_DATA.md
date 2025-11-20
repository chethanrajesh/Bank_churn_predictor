# Bank Customer Churn Data Documentation

## Overview

This dataset contains synthetic banking customer data for churn prediction modeling. The data includes 10,000 customers with 52 engineered features across 10 key churn drivers, specifically designed for the Indian banking context.

## Dataset Details

- **Total Records**: 10,000 customers
- **Total Features**: 54 (52 features + 2 target variables)
- **Churn Rate**: ~20% (realistic for banking industry)
- **Time Period**: Current snapshot with historical behavior
- **Geography**: Indian cities and banking context

## Feature Categories

### Base Demographics (7 features)
Core customer identification and demographic information.

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| customer_id | String | Unique identifier | CUST_XXXXXX |
| age | Integer | Customer age | 18-80 years |
| income | Float | Annual income | ₹10K-₹200K |
| region | Categorical | Indian city | 20 major cities |
| occupation_type | Categorical | Job category | 5 categories |
| customer_since | Date | Account opening date | 1990-2024 |
| account_type | Categorical | Bank account type | 9 types |

### Churn Driver 1: Low Engagement (6 features)
Measures customer activity and engagement levels.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| total_transactions | Integer | Monthly transactions | Higher = Lower risk |
| login_frequency | Float | App/branch logins | Higher = Lower risk |
| app_usage_minutes | Float | Mobile app usage | Higher = Lower risk |
| days_since_last_login | Float | Inactivity period | Higher = Higher risk |
| customer_service_calls | Integer | Support calls | Higher = Higher risk |
| activity_score | Float | Composite engagement score | Higher = Lower risk |

### Churn Driver 2: High Fees (5 features)
Tracks fee-related charges and penalties.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| avg_monthly_fees | Float | Monthly service fees | Higher = Higher risk |
| overdraft_frequency | Integer | Overdraft occurrences | Higher = Higher risk |
| penalty_charges | Float | Total penalty amounts | Higher = Higher risk |
| fee_to_balance_ratio | Float | Fees vs balance ratio | Higher = Higher risk |
| overdraft_amount | Float | Average overdraft amount | Higher = Higher risk |

### Churn Driver 3: Poor Service (5 features)
Measures service quality and complaint history.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| num_complaints | Integer | Total complaints | Higher = Higher risk |
| complaint_resolution_days | Float | Resolution time | Higher = Higher risk |
| satisfaction_score | Float | Customer satisfaction | Higher = Lower risk |
| escalation_count | Integer | Escalated complaints | Higher = Higher risk |
| service_quality_rating | Float | Service quality score | Higher = Lower risk |

### Churn Driver 4: Low Product Holding (6 features)
Tracks product diversification and cross-selling.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| num_products | Integer | Total products held | Higher = Lower risk |
| has_credit_card | Binary | Credit card ownership | 1 = Lower risk |
| has_mortgage | Binary | Mortgage loan | 1 = Lower risk |
| has_investment_account | Binary | Investment products | 1 = Lower risk |
| has_insurance | Binary | Insurance products | 1 = Lower risk |
| product_diversity_score | Float | Product mix score | Higher = Lower risk |

### Churn Driver 5: Short Tenure (5 features)
Customer relationship duration and lifecycle stage.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| tenure_months | Integer | Months as customer | Higher = Lower risk |
| balance_growth_rate | Float | Account growth rate | Higher = Lower risk |
| is_new_customer | Binary | New customer flag | 1 = Higher risk |
| tenure_segment | Categorical | Tenure category | 4 segments |
| age_group | Categorical | Age category | 5 groups |

### Churn Driver 6: Balance Fluctuations (5 features)
Account balance stability and transaction patterns.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| avg_account_balance | Float | Average balance | Higher = Lower risk |
| balance_trend | Float | Balance trend | Positive = Lower risk |
| balance_volatility | Float | Balance fluctuations | Higher = Higher risk |
| salary_consistency | Float | Income consistency | Higher = Lower risk |
| withdrawal_ratio | Float | Withdrawal frequency | Higher = Higher risk |

### Churn Driver 7: Demographics (5 features)
Additional demographic and behavioral characteristics.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| income_bracket | Categorical | Income category | 5 brackets |
| digital_adoption | Float | Digital banking usage | Higher = Lower risk |
| family_size | Integer | Household size | Variable impact |
| education_level | Categorical | Education category | 4 levels |
| marital_status | Categorical | Marital status | 4 categories |

### Churn Driver 8: Credit Issues (5 features)
Credit history and debt management.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| credit_score | Integer | Credit score | Higher = Lower risk |
| credit_utilization | Float | Credit usage ratio | Higher = Higher risk |
| num_loan_defaults | Integer | Default history | Higher = Higher risk |
| debt_to_income | Float | Debt burden | Higher = Higher risk |
| credit_inquiry_count | Integer | Credit checks | Higher = Higher risk |

### Churn Driver 9: Personalization & Offers (5 features)
Marketing engagement and personalization.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| promotions_sent | Integer | Marketing offers sent | Higher = Mixed |
| promotion_response_rate | Float | Offer response rate | Higher = Lower risk |
| reward_points | Integer | Loyalty points | Higher = Lower risk |
| personalized_offers_count | Integer | Targeted offers | Higher = Lower risk |
| offer_relevance_score | Float | Offer relevance | Higher = Lower risk |

### Churn Driver 10: Contract Events (5 features)
Account events and contractual milestones.

| Feature | Type | Description | Impact on Churn |
|---------|------|-------------|-----------------|
| account_age_months | Integer | Account age | Higher = Lower risk |
| maturity_proximity_days | Float | Days to maturity | Lower = Higher risk |
| renewal_flag | Binary | Renewal pending | 1 = Higher risk |
| account_closure_requests | Integer | Closure attempts | Higher = Higher risk |
| maturity_status | Categorical | Maturity state | 3 categories |

### Target Variables (2 features)
Model prediction targets.

| Feature | Type | Description | Usage |
|---------|------|-------------|-------|
| churn_probability | Float | Churn probability (0-1) | Continuous target |
| churn_risk | Binary | High churn risk flag | Binary classification |

## Data Generation Methodology

### Synthetic Data Creation
- **Algorithm**: Statistical distributions with realistic correlations
- **Random Seed**: 42 (for reproducibility)
- **Churn Logic**: Weighted combination of feature drivers
- **Realism**: Based on banking industry patterns

### Feature Engineering
- **Normalization**: Min-max scaling where appropriate
- **Interactions**: Realistic feature correlations
- **Distributions**: Industry-standard statistical distributions
- **Indian Context**: Localized for Indian banking market

## Data Quality

### Completeness
- All records have complete demographic information
- Maximum 2% missing values in behavioral features
- No missing target variables

### Validity
- Age range: 18-80 years
- Income range: ₹10,000 - ₹200,000
- Credit score range: 300-850
- All categorical values within defined sets

### Consistency
- Tenure consistent with account opening date
- Age groups match age values
- Logical relationships between related features

## Usage Guidelines

### Model Training
- Use `churn_risk` for binary classification
- Use `churn_probability` for regression tasks
- Consider feature groupings for interpretability

### Feature Selection
- All 52 features are engineered for churn prediction
- Consider driver-level feature importance
- Monitor for multicollinearity within drivers

### Preprocessing
- Handle categorical encoding for region, occupation, etc.
- Scale numerical features appropriately
- Consider temporal features for time-based analysis

## Business Context

### Churn Driver Weights
The churn probability is calculated with the following driver weights:
1. Low Engagement: 25%
2. High Fees: 20%
3. Poor Service: 15%
4. Low Product Holding: 10%
5. Short Tenure: 10%
6. Credit Issues: 10%
7. Balance Fluctuations: 10%

### Risk Segmentation
- **High Risk**: Top 30% of churn probability
- **Medium Risk**: Middle 40%
- **Low Risk**: Bottom 30%

## File Structure
