import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, Any
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processing pipeline for bank churn prediction
    Handles loading, validation, cleaning, and splitting of data
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.data_quality_report = {}
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        # Remove the problematic pd.isna() check and replace with a safer alternative
        elif obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return None
        else:
            return obj
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load synthetic customer data from CSV"""
        logger.info(f"Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality validation"""
        logger.info("Performing data quality validation...")
        
        # Convert numpy types to Python native types for JSON serialization
        missing_values = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
        data_types = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        
        validation_report = {
            'missing_values': missing_values,
            'duplicate_rows': int(df.duplicated().sum()),
            'data_types': data_types,
            'shape': {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1])
            },
            'churn_rate': float(df['churn_risk'].mean()),
            'feature_counts': int(len(df.columns))
        }
        
        # Check for critical issues
        critical_issues = []
        if validation_report['duplicate_rows'] > 0:
            critical_issues.append(f"Found {validation_report['duplicate_rows']} duplicate rows")
        
        if df['churn_risk'].isnull().any():
            critical_issues.append("Missing values in target variable")
            
        if len(df) == 0:
            critical_issues.append("Empty dataset")
            
        validation_report['critical_issues'] = critical_issues
        validation_report['is_valid'] = len(critical_issues) == 0
        
        logger.info(f"Data validation completed. Critical issues: {len(critical_issues)}")
        return validation_report
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        df_clean = df.copy()
        
        # Check if there are any missing values first
        missing_before = df.isnull().sum().sum()
        if missing_before == 0:
            logger.info("No missing values found. Skipping imputation.")
            return df_clean
        
        # Numerical columns: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df_clean[col] = df[col].fillna(df[col].median())
                logger.info(f"Filled missing values in {col} with median")
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df_clean[col] = df[col].fillna(mode_value[0])
                    logger.info(f"Filled missing values in {col} with mode: {mode_value[0]}")
                else:
                    df_clean[col] = df[col].fillna('Unknown')
                    logger.info(f"Filled missing values in {col} with 'Unknown'")
        
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
        
        return df_clean
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        logger.info("Splitting data into train/validation/test sets...")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=df['churn_risk']
        )
        
        # Second split: separate validation set from train+val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_val_df['churn_risk']
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train set: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
        logger.info(f"  Validation set: {len(val_df)} samples ({len(val_df)/len(df):.1%})")
        logger.info(f"  Test set: {len(test_df)} samples ({len(test_df)/len(df):.1%})")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                          output_dir: str = "data/processed"):
        """Save processed datasets to files"""
        logger.info(f"Saving processed data to {output_dir}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_df.to_csv(f"{output_dir}/train_set.csv", index=False)
        val_df.to_csv(f"{output_dir}/val_set.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_set.csv", index=False)
        
        # Convert data quality report to JSON serializable format
        serializable_report = {}
        for key, value in self.data_quality_report.items():
            if isinstance(value, dict):
                serializable_report[key] = {str(k): self._convert_to_serializable(v) for k, v in value.items()}
            else:
                serializable_report[key] = self._convert_to_serializable(value)
        
        # Save data quality report
        with open(f"{output_dir}/data_quality_report.json", 'w') as f:
            json.dump(serializable_report, f, indent=2, default=self._convert_to_serializable)
        
        logger.info("Processed data saved successfully")
    
    def process_pipeline(self, input_file: str = "data/raw/synthetic_customers.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Complete data processing pipeline"""
        logger.info("Starting data processing pipeline...")
        
        # Load data
        df = self.load_data(input_file)
        
        # Validate data
        self.data_quality_report = self.validate_data(df)
        
        if not self.data_quality_report['is_valid']:
            logger.warning("Data validation found critical issues:")
            for issue in self.data_quality_report['critical_issues']:
                logger.warning(f"  - {issue}")
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df_clean)
        
        # Save processed data
        self.save_processed_data(train_df, val_df, test_df)
        
        logger.info("Data processing pipeline completed successfully")
        return train_df, val_df, test_df

def main():
    """Main function to run data processing pipeline"""
    processor = DataProcessor()
    train_df, val_df, test_df = processor.process_pipeline()
    
    print("✅ Data processing completed!")
    print(f"📊 Train set: {len(train_df)} samples")
    print(f"📊 Validation set: {len(val_df)} samples") 
    print(f"📊 Test set: {len(test_df)} samples")
    print(f"🎯 Churn rates - Train: {train_df['churn_risk'].mean():.2%}, "
          f"Val: {val_df['churn_risk'].mean():.2%}, "
          f"Test: {test_df['churn_risk'].mean():.2%}")

if __name__ == "__main__":
    main()