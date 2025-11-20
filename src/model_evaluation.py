"""
Model Evaluation Module - FIXED for Data Type Issues
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from utils import load_config, get_logger, ensure_dir
from exceptions import ModelTrainingError

ModelEvaluationError = ModelTrainingError
logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates model performance and generates predictions.
    Handles feature encoding and data type issues automatically.
    """
    
    def __init__(self, model_version: str = "v1.0"):
        self.model_version = model_version
        self.config = load_config('./config/paths_config.yaml')
        
        # Ensure output directories exist
        self.validation_dir = self.config['models']['validation']
        ensure_dir(self.validation_dir)
    
    def load_model_and_data(self):
        """Load trained model, preprocessing artifacts, and test data."""
        try:
            # Load the best model with explicit error handling
            model_path = os.path.join(
                self.config['models']['production'],
                f'churn_model_best_{self.model_version}.pkl'
            )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                raise ModelEvaluationError(f"Failed to load model - corrupted file: {e}")
                
            print(f"✅ Loaded model: {type(self.model)}")
            
            # Load test data with validation
            test_data_path = os.path.join(
                self.config['data']['processed'],
                'test_set.csv'
            )
            
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"Test data file not found at {test_data_path}")
                
            self.test_data = pd.read_csv(test_data_path)
            if len(self.test_data) == 0:
                raise ModelEvaluationError("Test data is empty")
                
            print(f"✅ Loaded test data: {len(self.test_data)} samples")
            
            # Load feature names with validation
            feature_names_path = os.path.join(
                self.config['models']['preprocessing'],
                f'feature_names_{self.model_version}.pkl'
            )
            
            if not os.path.exists(feature_names_path):
                raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")
                
            try:
                with open(feature_names_path, 'rb') as f:
                    self.expected_feature_names = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                raise ModelEvaluationError(f"Failed to load feature names - corrupted file: {e}")
                
            print(f"✅ Expected features: {len(self.expected_feature_names)}")
            
        except Exception as e:
            print(f"❌ Failed to load model or data: {e}")
            raise ModelEvaluationError(f"Failed to load model or data: {e}")
    
    def clean_and_prepare_data(self):
        """Clean data and convert to proper numeric types."""
        try:
            print("🔄 Cleaning and preparing data...")
            
            # Create a working copy
            working_data = self.test_data.copy()
            
            # Remove non-feature columns
            columns_to_remove = ['customer_id', 'churn_probability', 'churn_risk']
            for col in columns_to_remove:
                if col in working_data.columns:
                    working_data = working_data.drop(columns=[col])
            
            # Identify and handle categorical columns
            categorical_columns = ['region', 'education_level', 'account_type', 
                                 'occupation_type', 'marital_status']
            
            # Convert categorical columns to string to avoid mixed types
            for col in categorical_columns:
                if col in working_data.columns:
                    working_data[col] = working_data[col].astype(str)
                    print(f"   - Converted {col} to string")
            
            # Convert numeric columns - handle potential string values
            numeric_columns = working_data.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_columns = working_data.select_dtypes(exclude=[np.number]).columns.tolist()
            
            print(f"   - Numeric columns: {len(numeric_columns)}")
            print(f"   - Non-numeric columns: {len(non_numeric_columns)}")
            
            # Convert non-numeric columns that should be numeric
            for col in non_numeric_columns:
                if col not in categorical_columns:
                    try:
                        working_data[col] = pd.to_numeric(working_data[col], errors='coerce')
                        print(f"   - Converted {col} to numeric")
                    except:
                        print(f"   - Could not convert {col} to numeric, keeping as is")
            
            return working_data
            
        except Exception as e:
            print(f"❌ Failed to clean data: {e}")
            raise ModelEvaluationError(f"Failed to clean data: {e}")
    
    def encode_categorical_features(self, cleaned_data):
        """Encode categorical features to match training data format."""
        try:
            print("🔄 Encoding categorical features...")
            
            # Identify categorical columns
            categorical_columns = ['region', 'education_level', 'account_type', 
                                 'occupation_type', 'marital_status']
            
            # Create a copy of cleaned data
            encoded_data = cleaned_data.copy()
            
            # One-hot encode categorical variables
            for col in categorical_columns:
                if col in encoded_data.columns:
                    # Create dummy variables
                    dummies = pd.get_dummies(encoded_data[col], prefix=col)
                    
                    # Remove the original column
                    encoded_data = encoded_data.drop(columns=[col])
                    
                    # Add the dummy variables
                    encoded_data = pd.concat([encoded_data, dummies], axis=1)
                    print(f"   - Encoded {col}: {len(dummies.columns)} categories")
            
            print(f"✅ Encoded features shape: {encoded_data.shape}")
            return encoded_data
            
        except Exception as e:
            print(f"❌ Failed to encode categorical features: {e}")
            raise ModelEvaluationError(f"Failed to encode categorical features: {e}")
    
    def align_features(self, encoded_data):
        """Align features with expected feature names."""
        try:
            print("🔄 Aligning features with expected feature set...")
            
            # Create a DataFrame with all expected features, initialized to 0
            aligned_data = pd.DataFrame(0.0,  # Use float to avoid dtype issues
                                      index=encoded_data.index, 
                                      columns=self.expected_feature_names,
                                      dtype=float)
            
            # Copy available features from encoded data
            features_copied = 0
            for feature in self.expected_feature_names:
                if feature in encoded_data.columns:
                    # Ensure the feature is numeric
                    aligned_data[feature] = pd.to_numeric(encoded_data[feature], errors='coerce').fillna(0)
                    features_copied += 1
            
            print(f"✅ Features aligned: {features_copied}/{len(self.expected_feature_names)}")
            print(f"✅ Final feature matrix shape: {aligned_data.shape}")
            
            # Check for any remaining non-numeric values
            non_numeric_cols = aligned_data.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                print(f"⚠️  Non-numeric columns after alignment: {non_numeric_cols}")
                # Convert any remaining non-numeric columns
                for col in non_numeric_cols:
                    aligned_data[col] = pd.to_numeric(aligned_data[col], errors='coerce').fillna(0)
            
            return aligned_data
            
        except Exception as e:
            print(f"❌ Failed to align features: {e}")
            raise ModelEvaluationError(f"Failed to align features: {e}")
    
    def prepare_features(self):
        """Prepare features for prediction."""
        try:
            # Step 1: Clean and prepare data
            cleaned_data = self.clean_and_prepare_data()
            
            # Step 2: Encode categorical features
            encoded_data = self.encode_categorical_features(cleaned_data)
            
            # Step 3: Align with expected features
            self.X_test = self.align_features(encoded_data)
            
            # Ensure we have the right columns in the right order
            self.X_test = self.X_test[self.expected_feature_names]
            
            # Final data type check
            print(f"✅ Final feature matrix: {self.X_test.shape}")
            print(f"📊 Data types: {self.X_test.dtypes.unique()}")
            print(f"📊 Feature range: {self.X_test.min().min():.3f} - {self.X_test.max().max():.3f}")
            
            # Ensure all data is numeric
            self.X_test = self.X_test.astype(float)
            
        except Exception as e:
            print(f"❌ Failed to prepare features: {e}")
            raise ModelEvaluationError(f"Failed to prepare features: {e}")
    
    def generate_predictions(self):
        """Generate predictions for test set."""
        try:
            # Prepare features first
            self.prepare_features()
            
            # Generate predictions
            print("🎯 Generating predictions...")
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(self.X_test)[:, 1]
            else:
                y_prob = self.model.predict(self.X_test)
            
            y_pred = (y_prob > 0.5).astype(int)
            
            # Create predictions DataFrame
            self.predictions_df = pd.DataFrame({
                'customer_id': self.test_data.get('customer_id', 
                    [f'CUST_{i:06d}' for i in range(len(self.test_data))]),
                'churn_probability': y_prob,
                'churn_prediction': y_pred
            })
            
            # Add actual values if available
            if 'churn' in self.test_data.columns:
                self.predictions_df['actual_churn'] = self.test_data['churn']
            elif 'churn_risk' in self.test_data.columns:
                # Use churn_risk as proxy if churn not available
                self.predictions_df['actual_churn'] = (self.test_data['churn_risk'] == 'High').astype(int)
            
            print(f"✅ Generated predictions for {len(self.predictions_df)} samples")
            print(f"📊 Prediction stats:")
            print(f"   - Probability range: {y_prob.min():.3f} - {y_prob.max():.3f}")
            print(f"   - Mean probability: {y_prob.mean():.3f}")
            print(f"   - Predicted churn rate: {y_pred.mean():.3f}")
            
        except Exception as e:
            print(f"❌ Failed to generate predictions: {e}")
            raise ModelEvaluationError(f"Failed to generate predictions: {e}")
    
    def calculate_metrics(self):
        """Calculate model performance metrics."""
        try:
            if 'actual_churn' not in self.predictions_df.columns:
                print("⚠️ No actual churn values found, using prediction-based metrics")
                # Calculate metrics based on predictions only
                y_prob = self.predictions_df['churn_probability']
                y_pred = self.predictions_df['churn_prediction']
                
                metrics = {
                    'accuracy': 'N/A',
                    'precision': 'N/A', 
                    'recall': 'N/A',
                    'f1_score': 'N/A',
                    'auc_score': 'N/A',
                    'positive_rate': 'N/A',
                    'predicted_positive_rate': float(y_pred.mean()),
                    'test_samples': len(self.predictions_df),
                    'avg_probability': float(y_prob.mean()),
                    'min_probability': float(y_prob.min()),
                    'max_probability': float(y_prob.max())
                }
                
                print(f"📊 Prediction-based metrics calculated")
                return metrics
            
            y_true = self.predictions_df['actual_churn']
            y_pred = self.predictions_df['churn_prediction']
            y_prob = self.predictions_df['churn_probability']
            
            # Check if we have both classes
            if len(np.unique(y_true)) < 2:
                print("⚠️ Only one class in actual values, using prediction metrics")
                metrics = {
                    'accuracy': 'N/A',
                    'precision': 'N/A',
                    'recall': 'N/A',
                    'f1_score': 'N/A', 
                    'auc_score': 'N/A',
                    'positive_rate': float(y_true.mean()),
                    'predicted_positive_rate': float(y_pred.mean()),
                    'test_samples': len(y_true),
                    'avg_probability': float(y_prob.mean())
                }
            else:
                metrics = {
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                    'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
                    'auc_score': float(roc_auc_score(y_true, y_prob)),
                    'positive_rate': float(y_true.mean()),
                    'predicted_positive_rate': float(y_pred.mean()),
                    'test_samples': len(y_true)
                }
                
                # Add confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                metrics.update({
                    'true_negatives': int(cm[0, 0]),
                    'false_positives': int(cm[0, 1]),
                    'false_negatives': int(cm[1, 0]),
                    'true_positives': int(cm[1, 1])
                })
                
                print(f"✅ Calculated metrics: AUC = {metrics['auc_score']:.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Failed to calculate metrics: {e}")
            return {'error': str(e), 'test_samples': len(self.predictions_df)}
        
    def export_predictions(self):
        """Export predictions to CSV file."""
        try:
            output_path = os.path.join(
                self.validation_dir,
                f'test_set_predictions_{self.model_version}.csv'
            )
            
            self.predictions_df.to_csv(output_path, index=False)
            print(f"✅ Exported predictions to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to export predictions: {e}")
            raise ModelEvaluationError(f"Failed to export predictions: {e}")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        print("🚀 Starting model evaluation...")
        
        try:
            # Load model and data
            self.load_model_and_data()
            
            # Generate predictions
            self.generate_predictions()
            
            # Calculate metrics
            metrics = self.calculate_metrics()
            
            # Export results
            predictions_path = self.export_predictions()
            
            print("✅ Model evaluation completed successfully!")
            
            return {
                'predictions_path': predictions_path,
                'metrics': metrics,
                'num_samples': len(self.predictions_df)
            }
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")
            raise ModelEvaluationError(f"Model evaluation failed: {e}")


def evaluate_model(model_version: str = "v1.0"):
    """
    Convenience function to run model evaluation.
    """
    evaluator = ModelEvaluator(model_version)
    return evaluator.run_evaluation()


if __name__ == "__main__":
    print("🔍 Running Model Evaluation...")
    
    try:
        results = evaluate_model("v1.0")
        print("\n🎉 Evaluation completed successfully!")
        print(f"📊 Results:")
        print(f"   - Predictions: {results['predictions_path']}")
        
        # Safe formatting for metrics
        metrics = results['metrics']
        auc_score = metrics.get('auc_score', 'N/A')
        
        # Handle different data types safely
        if isinstance(auc_score, (int, float)):
            print(f"   - AUC Score: {auc_score:.3f}")
        else:
            print(f"   - AUC Score: {auc_score}")
            
        print(f"   - Samples: {results['num_samples']}")
        
        # Safe handling for predicted churn rate
        churn_rate = metrics.get('predicted_positive_rate', 'N/A')
        if isinstance(churn_rate, (int, float)):
            print(f"   - Predicted Churn Rate: {churn_rate:.3f}")
        else:
            print(f"   - Predicted Churn Rate: {churn_rate}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")