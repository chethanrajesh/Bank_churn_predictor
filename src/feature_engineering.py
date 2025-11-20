import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering pipeline for bank churn prediction
    Creates and transforms 52 features across 10 churn drivers
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = []
        self.encoding_mappings = {}
        self.scaler_params = {}
        
    def load_processed_data(self, data_dir: str = "data/processed") -> Dict[str, pd.DataFrame]:
        """Load processed train, validation, and test sets"""
        logger.info("Loading processed datasets...")
        
        try:
            train_df = pd.read_csv(f"{data_dir}/train_set.csv")
            val_df = pd.read_csv(f"{data_dir}/val_set.csv")
            test_df = pd.read_csv(f"{data_dir}/test_set.csv")
            
            datasets = {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }
            
            logger.info(f"Loaded datasets: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify numerical and categorical features"""
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from numerical features
        if 'churn_risk' in numerical_features:
            numerical_features.remove('churn_risk')
        if 'churn_probability' in numerical_features:
            numerical_features.remove('churn_probability')
        
        # Remove customer_id from features
        if 'customer_id' in numerical_features:
            numerical_features.remove('customer_id')
        if 'customer_id' in categorical_features:
            categorical_features.remove('customer_id')
            
        logger.info(f"Identified {len(numerical_features)} numerical features and {len(categorical_features)} categorical features")
        
        return {
            'numerical': numerical_features,
            'categorical': categorical_features
        }
    
    def handle_categorical_features(self, df: pd.DataFrame, feature_types: Dict[str, List[str]], 
                                  fit: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical features using label encoding and one-hot encoding"""
        logger.info("Processing categorical features...")
        
        df_encoded = df.copy()
        categorical_features = feature_types['categorical']
        encoding_mappings = {}
        
        # Features for one-hot encoding (high cardinality)
        onehot_features = ['region', 'occupation_type', 'account_type', 'education_level', 'marital_status']
        
        # Features for label encoding (ordinal or low cardinality)
        label_features = [f for f in categorical_features if f not in onehot_features]
        
        # Label encoding
        for feature in label_features:
            if feature in df_encoded.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    df_encoded[feature] = self.label_encoders[feature].fit_transform(df_encoded[feature].astype(str))
                    
                    # Store label mappings
                    encoding_mappings[feature] = {
                        'encoding_type': 'label',
                        'mappings': dict(zip(
                            self.label_encoders[feature].classes_,
                            range(len(self.label_encoders[feature].classes_))
                        ))
                    }
                else:
                    # Handle unseen labels during transformation
                    known_labels = set(self.label_encoders[feature].classes_)
                    current_labels = set(df_encoded[feature].unique())
                    unseen_labels = current_labels - known_labels
                    
                    if unseen_labels:
                        logger.warning(f"Unseen labels in {feature}: {unseen_labels}")
                        # Replace unseen labels with most frequent
                        df_encoded[feature] = df_encoded[feature].apply(
                            lambda x: x if x in known_labels else self.label_encoders[feature].classes_[0]
                        )
                    
                    df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature].astype(str))
        
        # One-hot encoding
        if onehot_features:
            available_onehot_features = [f for f in onehot_features if f in df_encoded.columns]
            if available_onehot_features:
                if fit:
                    onehot_encoded = self.onehot_encoder.fit_transform(df_encoded[available_onehot_features])
                    
                    # Store one-hot mappings
                    for i, feature in enumerate(available_onehot_features):
                        encoding_mappings[feature] = {
                            'encoding_type': 'onehot',
                            'categories': self.onehot_encoder.categories_[i].tolist(),
                            'encoded_columns': [f"{feature}_{cat}" for cat in self.onehot_encoder.categories_[i]]
                        }
                else:
                    onehot_encoded = self.onehot_encoder.transform(df_encoded[available_onehot_features])
                
                # Create feature names for one-hot encoded columns
                onehot_feature_names = []
                for i, feature in enumerate(available_onehot_features):
                    for category in self.onehot_encoder.categories_[i]:
                        onehot_feature_names.append(f"{feature}_{category}")
                
                # Create DataFrame with one-hot encoded features
                onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_feature_names, index=df_encoded.index)
                
                # Drop original categorical features and add one-hot encoded ones
                df_encoded = df_encoded.drop(columns=available_onehot_features)
                df_encoded = pd.concat([df_encoded, onehot_df], axis=1)
        
        logger.info(f"Categorical feature processing completed. Final shape: {df_encoded.shape}")
        return df_encoded, encoding_mappings
    
    def scale_numerical_features(self, df: pd.DataFrame, feature_types: Dict[str, List[str]], 
                               fit: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features using StandardScaler"""
        logger.info("Scaling numerical features...")
        
        df_scaled = df.copy()
        numerical_features = feature_types['numerical']
        scaler_params = {}
        
        # Remove non-feature numerical columns
        non_feature_cols = ['churn_risk', 'churn_probability', 'customer_id']
        numerical_features = [f for f in numerical_features if f not in non_feature_cols]
        
        if numerical_features:
            if fit:
                df_scaled[numerical_features] = self.scaler.fit_transform(df_scaled[numerical_features])
                
                # Store scaler parameters
                scaler_params = {
                    'mean': self.scaler.mean_.tolist(),
                    'scale': self.scaler.scale_.tolist(),
                    'features': numerical_features,
                    'n_features_in': self.scaler.n_features_in_,
                    'n_samples_seen': int(self.scaler.n_samples_seen_)
                }
            else:
                df_scaled[numerical_features] = self.scaler.transform(df_scaled[numerical_features])
        
        logger.info(f"Numerical feature scaling completed. Scaled {len(numerical_features)} features")
        return df_scaled, scaler_params
    
    def create_feature_matrix(self, df: pd.DataFrame, feature_types: Dict[str, List[str]]) -> pd.DataFrame:
        """Create final feature matrix by selecting relevant features"""
        logger.info("Creating final feature matrix...")
        
        # Get all feature columns (excluding target and ID)
        all_features = feature_types['numerical'] + feature_types['categorical']
        all_features = [f for f in all_features if f not in ['churn_risk', 'churn_probability', 'customer_id']]
        
        # Add one-hot encoded features that might not be in the original lists
        current_columns = set(df.columns)
        feature_columns = [col for col in all_features if col in current_columns]
        
        # Add any additional columns that are features (from one-hot encoding)
        additional_features = [col for col in current_columns 
                             if col not in feature_columns + ['churn_risk', 'churn_probability', 'customer_id']]
        feature_columns.extend(additional_features)
        
        # Create feature matrix
        feature_matrix = df[feature_columns].copy()
        
        # Store feature names
        self.feature_names = feature_columns
        
        logger.info(f"Feature matrix created with {len(feature_columns)} features")
        return feature_matrix
    
    def save_feature_artifacts(self, output_dir: str = "data/processed"):
        """Save feature engineering artifacts"""
        logger.info("Saving feature engineering artifacts...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save feature names
        joblib.dump(self.feature_names, f"{output_dir}/feature_names.pkl")
        
        # Save encoders and scaler
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{output_dir}/label_encoders.pkl")
        joblib.dump(self.onehot_encoder, f"{output_dir}/onehot_encoder.pkl")
        
        # Save encoding mappings as JSON
        with open(f"{output_dir}/encoding_mappings.json", 'w') as f:
            json.dump(self.encoding_mappings, f, indent=2, default=str)
        
        # Save scaler parameters as JSON
        with open(f"{output_dir}/scaler_params.json", 'w') as f:
            json.dump(self.scaler_params, f, indent=2, default=str)
        
        # Save feature schema
        feature_schema = {
            'feature_names': self.feature_names,
            'total_features': len(self.feature_names),
            'feature_categories': {
                'engagement': [f for f in self.feature_names if any(x in f for x in ['transaction', 'login', 'activity', 'usage', 'calls'])],
                'fees': [f for f in self.feature_names if any(x in f for x in ['fee', 'overdraft', 'penalty', 'charge'])],
                'service': [f for f in self.feature_names if any(x in f for x in ['complaint', 'satisfaction', 'service', 'resolution', 'escalation'])],
                'products': [f for f in self.feature_names if any(x in f for x in ['product', 'credit_card', 'mortgage', 'investment', 'insurance'])],
                'tenure': [f for f in self.feature_names if any(x in f for x in ['tenure', 'customer_since', 'new_customer', 'segment'])],
                'balance': [f for f in self.feature_names if any(x in f for x in ['balance', 'salary', 'volatility', 'withdrawal'])],
                'demographics': [f for f in self.feature_names if any(x in f for x in ['age', 'income', 'region', 'occupation', 'education', 'marital'])],
                'credit': [f for f in self.feature_names if any(x in f for x in ['credit', 'debt', 'loan', 'inquiry'])],
                'personalization': [f for f in self.feature_names if any(x in f for x in ['promotion', 'reward', 'offer', 'personalized'])],
                'contract': [f for f in self.feature_names if any(x in f for x in ['account_age', 'maturity', 'renewal', 'closure'])]
            }
        }
        
        with open(f"{output_dir}/feature_schema.json", 'w') as f:
            json.dump(feature_schema, f, indent=2)
        
        logger.info("Feature engineering artifacts saved successfully")
    
    def create_encoded_feature_matrix(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create combined encoded feature matrix for all datasets"""
        logger.info("Creating combined encoded feature matrix...")
        
        # Combine all datasets
        combined_df = pd.concat([datasets['train'], datasets['val'], datasets['test']], ignore_index=True)
        
        # Remove target variables for the encoded matrix
        feature_columns = [col for col in combined_df.columns if col not in ['churn_risk', 'churn_probability', 'customer_id']]
        
        encoded_matrix = combined_df[feature_columns].copy()
        
        # Add metadata
        encoded_matrix['dataset_split'] = (
            ['train'] * len(datasets['train']) + 
            ['val'] * len(datasets['val']) + 
            ['test'] * len(datasets['test'])
        )
        
        # Add customer_id back if available
        if 'customer_id' in combined_df.columns:
            encoded_matrix['customer_id'] = combined_df['customer_id'].values
        
        logger.info(f"Encoded feature matrix created with shape: {encoded_matrix.shape}")
        return encoded_matrix
    
    def feature_engineering_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        # Load processed data
        datasets = self.load_processed_data()
        
        # Identify feature types using training data
        feature_types = self.identify_feature_types(datasets['train'])
        
        processed_datasets = {}
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing {dataset_name} dataset...")
            
            # Handle categorical features
            is_training = (dataset_name == 'train')
            df_encoded, encoding_mappings = self.handle_categorical_features(df, feature_types, fit=is_training)
            
            # Store encoding mappings from training
            if is_training:
                self.encoding_mappings = encoding_mappings
            
            # Scale numerical features
            df_scaled, scaler_params = self.scale_numerical_features(df_encoded, feature_types, fit=is_training)
            
            # Store scaler parameters from training
            if is_training:
                self.scaler_params = scaler_params
            
            # Create feature matrix
            feature_matrix = self.create_feature_matrix(df_scaled, feature_types)
            
            # Add target variable back
            feature_matrix['churn_risk'] = df['churn_risk'].values
            if 'churn_probability' in df.columns:
                feature_matrix['churn_probability'] = df['churn_probability'].values
            
            # Add customer_id back for reference
            if 'customer_id' in df.columns:
                feature_matrix['customer_id'] = df['customer_id'].values
            
            processed_datasets[dataset_name] = feature_matrix
        
        # Create combined encoded feature matrix
        encoded_matrix = self.create_encoded_feature_matrix(processed_datasets)
        
        # Save all artifacts
        self.save_feature_artifacts()
        
        # Save individual feature matrices
        processed_datasets['train'].to_csv("data/processed/feature_matrix_train.csv", index=False)
        processed_datasets['val'].to_csv("data/processed/feature_matrix_val.csv", index=False)
        processed_datasets['test'].to_csv("data/processed/feature_matrix_test.csv", index=False)
        
        # Save combined feature matrix
        combined_matrix = pd.concat([
            processed_datasets['train'],
            processed_datasets['val'],
            processed_datasets['test']
        ])
        combined_matrix.to_csv("data/processed/feature_matrix.csv", index=False)
        
        # Save encoded feature matrix (this is the main one we need)
        encoded_matrix.to_csv("data/processed/feature_matrix_encoded.csv", index=False)
        
        logger.info("Feature engineering pipeline completed successfully")
        return processed_datasets

def main():
    """Main function to run feature engineering pipeline"""
    engineer = FeatureEngineer()
    processed_datasets = engineer.feature_engineering_pipeline()
    
    print("✅ Feature engineering completed!")
    print(f"📊 Final feature count: {len(engineer.feature_names)}")
    print(f"📁 Generated files:")
    print(f"   - feature_matrix_encoded.csv (Final encoded matrix)")
    print(f"   - encoding_mappings.json (Encoding mappings)")
    print(f"   - scaler_params.json (Scaling parameters)")
    print(f"   - feature_matrix_train.csv (Training features)")
    print(f"   - feature_matrix_val.csv (Validation features)")
    print(f"   - feature_matrix_test.csv (Test features)")
    print(f"   - feature_matrix.csv (Combined features)")

if __name__ == "__main__":
    main()