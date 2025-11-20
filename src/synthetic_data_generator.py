import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticBankDataGenerator:
    """
    Generate synthetic banking customer data for a SINGLE BANK with 52 features across 10 churn drivers
    Indian context with Indian cities and bank account types
    """
    
    def __init__(self, n_customers: int = 100000, random_state: int = 42):  # Changed to 100000
        self.n_customers = n_customers
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
    def generate_customer_base(self) -> pd.DataFrame:
        """Generate base customer demographics with Indian context"""
        logger.info(f"Generating {self.n_customers} synthetic customers for single Indian bank...")
        
        # Indian cities for regions
        indian_cities = [
            'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
            'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
            'Surat', 'Kanpur', 'Nagpur', 'Patna', 'Indore',
            'Thane', 'Bhopal', 'Visakhapatnam', 'Vadodara', 'Ludhiana'
        ]
        
        # Indian bank account types
        indian_account_types = [
            'Savings Account', 'Salary Account', 'Current Account', 
            'Fixed Deposit', 'Recurring Deposit', 'NRI Account',
            'Senior Citizen Account', 'Student Account', 'Basic Savings'
        ]
        
        customers = []
        
        for i in range(self.n_customers):
            customer = {
                'customer_id': f'CUST_{i:06d}',
                'age': np.random.randint(18, 80),
                'income': np.random.normal(60000, 25000),
                'region': random.choice(indian_cities),  # Indian cities instead of directions
                'occupation_type': random.choice(['Professional', 'Service', 'Business', 'Retired', 'Student']),
                'customer_since': datetime.now() - timedelta(days=np.random.randint(30, 3650)),
                'account_type': random.choice(indian_account_types),  # Indian bank account types
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def add_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engagement-related features (Driver 1: Low Engagement) - 6 features"""
        logger.info("Adding engagement features...")
        
        # 6 engagement features
        df['total_transactions'] = np.random.poisson(15, len(df))
        df['login_frequency'] = np.random.exponential(10, len(df))
        df['app_usage_minutes'] = np.random.gamma(5, 10, len(df))
        df['days_since_last_login'] = np.random.exponential(30, len(df))
        df['customer_service_calls'] = np.random.poisson(2, len(df))
        df['activity_score'] = (df['total_transactions'] * 0.3 + 
                               df['login_frequency'] * 0.4 + 
                               df['app_usage_minutes'] * 0.3)
        
        return df
    
    def add_fee_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fee-related features (Driver 2: High Fees) - 5 features"""
        logger.info("Adding fee features...")
        
        # 5 fee features
        df['avg_monthly_fees'] = np.random.exponential(10, len(df))
        df['overdraft_frequency'] = np.random.poisson(1, len(df))
        df['penalty_charges'] = np.random.exponential(5, len(df))
        df['fee_to_balance_ratio'] = df['avg_monthly_fees'] / (df['income'] / 12 + 1000)
        df['overdraft_amount'] = np.random.exponential(50, len(df))
        
        return df
    
    def add_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add service-related features (Driver 3: Poor Service) - 5 features"""
        logger.info("Adding service features...")
        
        # 5 service features
        df['num_complaints'] = np.random.poisson(0.5, len(df))
        df['complaint_resolution_days'] = np.random.exponential(7, len(df))
        df['satisfaction_score'] = np.random.normal(7, 2, len(df))
        df['escalation_count'] = np.random.poisson(0.2, len(df))
        df['service_quality_rating'] = np.random.normal(8, 1.5, len(df))
        
        return df
    
    def add_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add product holding features (Driver 4: Low Product Holding) - 6 features"""
        logger.info("Adding product features...")
        
        # 6 product features - FIXED: Added random mortgage with age-based probability
        df['num_products'] = np.random.poisson(2.5, len(df))
        df['has_credit_card'] = np.random.binomial(1, 0.7, len(df))
        
        # Mortgage probability increases with age and income
        mortgage_prob = np.clip((df['age'] - 25) / 50 * 0.6 + (df['income'] > 80000) * 0.3, 0.05, 0.8)
        df['has_mortgage'] = np.random.binomial(1, mortgage_prob)
        
        # FIXED: has_investment_account now has random values with 40% probability
        df['has_investment_account'] = np.random.binomial(1, 0.4, len(df))
        
        df['has_insurance'] = np.random.binomial(1, 0.5, len(df))
        df['product_diversity_score'] = (df['num_products'] * 0.2 + 
                                        df['has_credit_card'] * 0.3 +
                                        df['has_investment_account'] * 0.3 +
                                        df['has_insurance'] * 0.2)
        
        return df
    
    def add_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add tenure-related features (Driver 5: Short Tenure) - 5 features"""
        logger.info("Adding tenure features...")
        
        # 5 tenure features
        df['tenure_months'] = ((datetime.now() - pd.to_datetime(df['customer_since'])).dt.days / 30).astype(int)
        df['balance_growth_rate'] = np.random.normal(0.05, 0.02, len(df))
        df['is_new_customer'] = (df['tenure_months'] < 6).astype(int)
        df['tenure_segment'] = pd.cut(df['tenure_months'], 
                                    bins=[0, 12, 36, 120, 999],
                                    labels=['New', 'Established', 'Loyal', 'Veteran'])
        df['age_group'] = pd.cut(df['age'], 
                               bins=[0, 25, 35, 50, 65, 100],
                               labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])
        
        return df
    
    def add_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add balance fluctuation features (Driver 6: Balance Fluctuations) - 5 features"""
        logger.info("Adding balance features...")
        
        # 5 balance features
        df['avg_account_balance'] = np.random.normal(5000, 3000, len(df))
        df['balance_trend'] = np.random.normal(0, 100, len(df))
        df['balance_volatility'] = np.random.exponential(500, len(df))
        df['salary_consistency'] = np.random.normal(0.8, 0.2, len(df))
        df['withdrawal_ratio'] = np.random.beta(2, 5, len(df))
        
        return df
    
    def add_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add demographic features (Driver 7: Demographics) - 5 features"""
        logger.info("Adding demographic features...")
        
        # 5 demographic features
        df['income_bracket'] = pd.cut(df['income'], 
                                    bins=[0, 30000, 60000, 100000, 200000, np.inf],
                                    labels=['Low', 'Medium', 'High', 'Very High', 'Wealthy'])
        df['digital_adoption'] = np.random.normal(0.7, 0.2, len(df))
        df['family_size'] = np.random.poisson(2.5, len(df))
        df['education_level'] = random.choices(['High School', 'Bachelor', 'Master', 'PhD'], 
                                             weights=[0.3, 0.4, 0.2, 0.1], k=len(df))
        df['marital_status'] = random.choices(['Single', 'Married', 'Divorced', 'Widowed'], 
                                            weights=[0.3, 0.5, 0.15, 0.05], k=len(df))
        
        return df
    
    def add_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add credit-related features (Driver 8: Credit Issues) - 5 features"""
        logger.info("Adding credit features...")
        
        # 5 credit features
        df['credit_score'] = np.random.normal(700, 100, len(df))
        df['credit_utilization'] = np.random.beta(2, 5, len(df))
        df['num_loan_defaults'] = np.random.poisson(0.1, len(df))
        df['debt_to_income'] = np.random.beta(1, 4, len(df))
        df['credit_inquiry_count'] = np.random.poisson(1, len(df))
        
        return df
    
    def add_personalization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add personalization features (Driver 9: Personalization & Offers) - 5 features"""
        logger.info("Adding personalization features...")
        
        # 5 personalization features
        df['promotions_sent'] = np.random.poisson(5, len(df))
        df['promotion_response_rate'] = np.random.beta(2, 8, len(df))
        df['reward_points'] = np.random.poisson(1000, len(df))
        df['personalized_offers_count'] = np.random.poisson(3, len(df))
        df['offer_relevance_score'] = np.random.normal(0.6, 0.2, len(df))
        
        return df
    
    def add_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contract/account features (Driver 10: Contract Events) - 5 features"""
        logger.info("Adding contract features...")
        
        # 5 contract features
        df['account_age_months'] = np.random.randint(1, 120, len(df))
        df['maturity_proximity_days'] = np.random.exponential(180, len(df))
        df['renewal_flag'] = np.random.binomial(1, 0.3, len(df))
        df['account_closure_requests'] = np.random.poisson(0.1, len(df))
        df['maturity_status'] = random.choices(['Active', 'Near Maturity', 'Matured'], 
                                             weights=[0.7, 0.2, 0.1], k=len(df))
        
        return df
    
    def add_churn_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create realistic churn target variable correlated with features - CORRECTED VERSION"""
        logger.info("Adding churn target variable...")
    
        # Initialize churn probability with baseline
        baseline_churn = 0.05  # 5% baseline churn rate
        churn_probability = np.full(len(df), baseline_churn)
        
        # 1. Low Engagement (25% weight)
        engagement_prob = (
            (df['days_since_last_login'] / df['days_since_last_login'].quantile(0.95)) * 0.15 +
            ((df['activity_score'].max() - df['activity_score']) / df['activity_score'].max()) * 0.10
        )
        churn_probability += np.clip(engagement_prob, 0, 0.25) * 0.25
        
        # 2. High Fees (20% weight)
        fee_prob = (
            (df['avg_monthly_fees'] / df['avg_monthly_fees'].quantile(0.95)) * 0.10 +
            (df['overdraft_frequency'] / df['overdraft_frequency'].quantile(0.95)) * 0.10
        )
        churn_probability += np.clip(fee_prob, 0, 0.20) * 0.20
        
        # 3. Poor Service (15% weight)
        service_prob = (
            (df['num_complaints'] / df['num_complaints'].quantile(0.95)) * 0.08 +
            ((10 - df['satisfaction_score']) / 10) * 0.07
        )
        churn_probability += np.clip(service_prob, 0, 0.15) * 0.15
        
        # 4. Low Product Holding (10% weight)
        product_prob = (
            ((5 - df['num_products']) / 5) * 0.05 +
            (1 - df['has_credit_card']) * 0.05
        )
        churn_probability += np.clip(product_prob, 0, 0.10) * 0.10
        
        # 5. Short Tenure (10% weight)
        tenure_prob = (
            ((24 - np.minimum(df['tenure_months'], 24)) / 24) * 0.05 +
            df['is_new_customer'] * 0.05
        )
        churn_probability += np.clip(tenure_prob, 0, 0.10) * 0.10
        
        # 6. Credit Issues (10% weight)
        credit_prob = (
            ((850 - df['credit_score']) / 350) * 0.05 +
            (df['debt_to_income'] / df['debt_to_income'].quantile(0.95)) * 0.05
        )
        churn_probability += np.clip(credit_prob, 0, 0.10) * 0.10
        
        # 7. Balance Fluctuations (10% weight)
        balance_prob = (
            (df['balance_volatility'] / df['balance_volatility'].quantile(0.95)) * 0.05 +
            ((1 - df['salary_consistency']) / 1) * 0.05
        )
        churn_probability += np.clip(balance_prob, 0, 0.10) * 0.10
        
        # Add some random noise
        churn_probability += np.random.normal(0, 0.02, len(df))
        
        # Clip probabilities to reasonable range [0.01, 0.35]
        churn_probability = np.clip(churn_probability, 0.01, 0.35)
        
        df['churn_probability'] = churn_probability
        
        # Use a more realistic threshold - aim for 10-15% churn rate
        # Find threshold that gives us ~12% churn rate
        target_churn_rate = 0.12
        threshold = np.percentile(churn_probability, (1 - target_churn_rate) * 100)
        df['churn_risk'] = (churn_probability > threshold).astype(int)
        
        actual_churn_rate = df['churn_risk'].mean()
        logger.info(f"Target churn rate: {target_churn_rate:.2%}")
        logger.info(f"Actual churn rate: {actual_churn_rate:.2%}")
        logger.info(f"Churn probability range: {churn_probability.min():.3f} - {churn_probability.max():.3f}")
        
        return df
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete dataset with exactly 52 features across 10 drivers"""
        logger.info("Starting complete dataset generation with 52 features...")
        
        # Generate base data (6 demographic features)
        df = self.generate_customer_base()
        
        # Add all feature categories (46 additional features)
        df = self.add_engagement_features(df)      # +6 features = 12 total
        df = self.add_fee_features(df)             # +5 features = 17 total
        df = self.add_service_features(df)         # +5 features = 22 total
        df = self.add_product_features(df)         # +6 features = 28 total
        df = self.add_tenure_features(df)          # +5 features = 33 total
        df = self.add_balance_features(df)         # +5 features = 38 total
        df = self.add_demographic_features(df)     # +5 features = 43 total
        df = self.add_credit_features(df)          # +5 features = 48 total
        df = self.add_personalization_features(df) # +5 features = 53 total
        df = self.add_contract_features(df)        # +5 features = 58 total
        
        # Add target variable (2 features)
        df = self.add_churn_target(df)             # +2 features = 60 total
        
        # Select exactly 52 features + target as per project specification
        base_features = ['customer_id', 'age', 'income', 'region', 'occupation_type', 'customer_since', 'account_type']
        target_features = ['churn_probability', 'churn_risk']
        
        # Select exactly 52 features + target
        all_features = base_features + [
            # Engagement (6)
            'total_transactions', 'login_frequency', 'app_usage_minutes', 'days_since_last_login', 
            'customer_service_calls', 'activity_score',
            # Fees (5)
            'avg_monthly_fees', 'overdraft_frequency', 'penalty_charges', 'fee_to_balance_ratio', 'overdraft_amount',
            # Service (5)
            'num_complaints', 'complaint_resolution_days', 'satisfaction_score', 'escalation_count', 'service_quality_rating',
            # Products (6)
            'num_products', 'has_credit_card', 'has_mortgage', 'has_investment_account', 'has_insurance', 'product_diversity_score',
            # Tenure (5)
            'tenure_months', 'balance_growth_rate', 'is_new_customer', 'tenure_segment', 'age_group',
            # Balance (5)
            'avg_account_balance', 'balance_trend', 'balance_volatility', 'salary_consistency', 'withdrawal_ratio',
            # Demographics (5)
            'income_bracket', 'digital_adoption', 'family_size', 'education_level', 'marital_status',
            # Credit (5)
            'credit_score', 'credit_utilization', 'num_loan_defaults', 'debt_to_income', 'credit_inquiry_count',
            # Personalization (5)
            'promotions_sent', 'promotion_response_rate', 'reward_points', 'personalized_offers_count', 'offer_relevance_score',
            # Contract (5)
            'account_age_months', 'maturity_proximity_days', 'renewal_flag', 'account_closure_requests', 'maturity_status'
        ]
        
        # Ensure we have exactly these features
        final_features = [f for f in all_features if f in df.columns]
        final_features = final_features[:52]  # Take first 52 features
        final_features.extend(target_features)  # Add target
        
        df_final = df[final_features].copy()
        
        logger.info(f"Dataset generated with {len(df_final)} customers and {len(df_final.columns)} features")
        logger.info(f"Final churn rate: {df_final['churn_risk'].mean():.2%}")
        logger.info(f"Mortgage holders: {df_final['has_mortgage'].mean():.2%}")
        logger.info(f"Investment account holders: {df_final['has_investment_account'].mean():.2%}")
        
        return df_final

def main():
    """Main function to generate and save synthetic data"""
    generator = SyntheticBankDataGenerator(n_customers=10000)  # Changed to 100000
    df = generator.generate_complete_dataset()
    
    # Save to data/raw directory
    output_path = "data/raw/synthetic_customers.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Successfully generated {len(df)} customers for SINGLE INDIAN BANK")
    print(f"📁 Saved to: {output_path}")
    print(f"🎯 Churn rate: {df['churn_risk'].mean():.2%}")
    print(f"🏠 Mortgage holders: {df['has_mortgage'].mean():.2%}")
    print(f"📈 Investment account holders: {df['has_investment_account'].mean():.2%}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Total features: {len(df.columns)} (52 features + target)")
    
    # Show Indian context information
    print(f"\n🇮🇳 INDIAN BANKING CONTEXT:")
    print(f"   Cities: {df['region'].unique()[:10]}")  # Show first 10 cities
    print(f"   Account Types: {df['account_type'].unique()}")
    
    # Show feature breakdown by driver
    feature_breakdown = {
        '1. Low Engagement': 6,
        '2. High Fees': 5,
        '3. Poor Service': 5,
        '4. Low Product Holding': 6,
        '5. Short Tenure': 5,
        '6. Balance Fluctuations': 5,
        '7. Demographics': 5,
        '8. Credit Issues': 5,
        '9. Personalization & Offers': 5,
        '10. Contract Events': 5,
        'Base Demographics': 6,
        'Target Variables': 2
    }
    
    print(f"\n📊 FEATURE BREAKDOWN BY CHURN DRIVER:")
    for driver, count in feature_breakdown.items():
        print(f"   {driver}: {count} features")

if __name__ == "__main__":
    main()