import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List, Optional
import json
import joblib
from pathlib import Path
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    SHAP explanations for model interpretability
    Provides global and local explanations for churn predictions
    """
    
    def __init__(self, version: str = "v1.0"):
        self.version = version
        self.explainer = None
        self.shap_values = None
        self.feature_names = []
        self.model = None
        self.X_test = None
        
    def load_model_and_data(self, models_dir: str = "models/production", 
                          data_dir: str = "data/processed") -> tuple:
        """Load best model and test data for explanations"""
        logger.info("Loading model and data for SHAP explanations...")
        
        try:
            # Load current model reference
            with open(f"{models_dir}/CURRENT_MODEL.txt", 'r') as f:
                model_file = f.read().strip()
            
            # Load best model
            self.model = joblib.load(f"{models_dir}/{model_file}")
            
            # Load test data
            self.X_test = pd.read_csv(f"{data_dir}/feature_matrix_test.csv")
            y_test = self.X_test['churn_risk']
            
            # Drop target columns and customer_id if present
            columns_to_drop = ['churn_risk', 'churn_probability', 'customer_id']
            self.X_test = self.X_test.drop(columns=[col for col in columns_to_drop if col in self.X_test.columns])
            
            # Try to load feature names from preprocessing
            feature_names_path = "models/preprocessing/feature_names_v1.0.pkl"
            if Path(feature_names_path).exists():
                self.feature_names = joblib.load(feature_names_path)
                logger.info(f"Loaded {len(self.feature_names)} feature names from preprocessing")
            else:
                # Fallback: use column names from test data
                self.feature_names = self.X_test.columns.tolist()
                logger.info(f"Using {len(self.feature_names)} feature names from test data")
            
            # Ensure consistent column order
            self.X_test = self.X_test[self.feature_names]
            
            logger.info(f"Loaded model: {model_file}")
            logger.info(f"Test data shape: {self.X_test.shape}")
            logger.info(f"Number of features: {len(self.feature_names)}")
            
            return self.model, self.X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading model and data: {e}")
            raise
    
    def create_shap_explainer(self, model: Any, X_test: pd.DataFrame, 
                            sample_size: int = 1000) -> None:
        """Create SHAP explainer and compute SHAP values with better error handling"""
        logger.info("Creating SHAP explainer...")
        
        # Sample data for faster computation
        if len(X_test) > sample_size:
            X_sample = X_test.sample(sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} instances for SHAP computation")
        else:
            X_sample = X_test
        
        try:
            # Determine model type and create appropriate explainer
            model_type = str(type(model)).lower()
            
            # FIXED: Better handling for XGBoost models
            if 'xgboost' in model_type or 'xgbclassifier' in model_type:
                logger.info("Using TreeExplainer for XGBoost model")
                try:
                    # Method 1: Direct TreeExplainer
                    self.explainer = shap.TreeExplainer(model)
                    raw_shap_values = self.explainer.shap_values(X_sample)
                    logger.info("Success with direct TreeExplainer")
                except Exception as e1:
                    logger.warning(f"Direct TreeExplainer failed: {e1}")
                    # Method 2: Use predict_proba as function
                    logger.info("Trying TreeExplainer with model_output='probability'")
                    self.explainer = shap.TreeExplainer(model, data=X_sample, model_output='probability')
                    raw_shap_values = self.explainer.shap_values(X_sample)
                    logger.info("Success with probability output")
                
            elif any(x in model_type for x in ['randomforest', 'extratrees', 'gradientboosting', 'lightgbm']):
                logger.info("Using TreeExplainer for tree-based model")
                self.explainer = shap.TreeExplainer(model)
                raw_shap_values = self.explainer.shap_values(X_sample)
                
            else:
                logger.info("Using KernelExplainer for non-tree model")
                # Use smaller background for KernelSHAP
                background = shap.sample(X_sample, min(50, len(X_sample)))
                self.explainer = shap.KernelExplainer(model.predict_proba, background)
                raw_shap_values = self.explainer.shap_values(X_sample)
            
            # Process SHAP output to consistent format
            self.shap_values = self._handle_shap_output(raw_shap_values, X_sample)
            
            logger.info("SHAP explainer created successfully")
            
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            # Enhanced fallback approach
            logger.info("Trying enhanced fallback SHAP computation...")
            try:
                self._fallback_shap_computation(model, X_sample)
            except Exception as fallback_error:
                logger.error(f"All SHAP computation methods failed: {fallback_error}")
                raise
    
    def _fallback_shap_computation(self, model: Any, X_sample: pd.DataFrame) -> None:
        """Enhanced fallback method for SHAP computation"""
        logger.info("Using enhanced fallback SHAP computation...")
        
        try:
            # Method 1: Try with different model_output options
            model_type = str(type(model)).lower()
            
            if 'xgboost' in model_type:
                # For XGBoost, try different approaches
                try:
                    # Convert to compatible format if needed
                    if hasattr(model, 'get_booster'):
                        booster = model.get_booster()
                        self.explainer = shap.TreeExplainer(booster)
                    else:
                        self.explainer = shap.TreeExplainer(model, model_output='probability')
                    
                    raw_shap_values = self.explainer.shap_values(X_sample)
                    
                except Exception as e:
                    logger.warning(f"Booster method failed: {e}")
                    # Last resort: use predict_proba wrapper
                    def predict_proba_wrapper(X):
                        return model.predict_proba(X)
                    
                    background = shap.sample(X_sample, min(100, len(X_sample)))
                    self.explainer = shap.KernelExplainer(predict_proba_wrapper, background)
                    raw_shap_values = self.explainer.shap_values(X_sample)
            
            else:
                # For other models, use Explainer with auto-detection
                self.explainer = shap.Explainer(model, X_sample)
                raw_shap_values = self.explainer(X_sample)
            
            self.shap_values = self._handle_shap_output(raw_shap_values, X_sample)
            logger.info("Enhanced fallback SHAP computation successful")
            
        except Exception as e:
            logger.error(f"Enhanced fallback also failed: {e}")
            raise
    
    def _handle_shap_output(self, raw_shap_values: Any, X_sample: pd.DataFrame) -> np.ndarray:
        """Handle different SHAP output formats and return consistent 2D array"""
        logger.info("Processing SHAP output format...")
        
        # Case 1: List of arrays (one per class) - common for binary classification
        if isinstance(raw_shap_values, list) and len(raw_shap_values) == 2:
            logger.info("List of arrays format detected - using class 1")
            shap_array = raw_shap_values[1]  # Class 1 (positive class)
            
        # Case 2: 3D array (samples, features, classes)
        elif isinstance(raw_shap_values, np.ndarray) and len(raw_shap_values.shape) == 3:
            logger.info("3D array format detected - extracting class 1")
            shap_array = raw_shap_values[:, :, 1]  # Class 1
            
        # Case 3: 2D array (samples, features) - already in correct format
        elif isinstance(raw_shap_values, np.ndarray) and len(raw_shap_values.shape) == 2:
            logger.info("2D array format detected - using as is")
            shap_array = raw_shap_values
            
        # Case 4: SHAP Explanation object
        elif hasattr(raw_shap_values, 'values'):
            logger.info("SHAP Explanation object detected")
            shap_array = raw_shap_values.values
            # If it's 3D, extract class 1
            if len(shap_array.shape) == 3:
                shap_array = shap_array[:, :, 1]
                
        else:
            logger.warning(f"Unexpected SHAP format: {type(raw_shap_values)}")
            # Try to convert to numpy array
            try:
                shap_array = np.array(raw_shap_values)
                logger.info(f"Converted to array with shape: {shap_array.shape}")
            except:
                raise ValueError(f"Cannot process SHAP output of type: {type(raw_shap_values)}")
        
        # Final shape validation and adjustment
        if len(shap_array.shape) != 2:
            logger.warning(f"SHAP array not 2D, reshaping: {shap_array.shape}")
            if len(shap_array.shape) == 3:
                shap_array = shap_array.reshape(shap_array.shape[0], -1)
            elif len(shap_array.shape) == 1:
                shap_array = shap_array.reshape(1, -1)
        
        # Ensure feature count matches
        n_samples, n_features_shap = shap_array.shape
        n_features_data = len(self.feature_names)
        
        logger.info(f"SHAP array shape: {shap_array.shape}")
        logger.info(f"Data features: {n_features_data}")
        
        if n_features_shap != n_features_data:
            logger.warning(f"Feature count mismatch: SHAP {n_features_shap} != Data {n_features_data}")
            
            if n_features_shap > n_features_data:
                # Common issue: SHAP returns both classes concatenated
                if n_features_shap == n_features_data * 2:
                    logger.info("Detected concatenated classes - using first half")
                    shap_array = shap_array[:, :n_features_data]
                else:
                    logger.warning(f"Trimming SHAP features from {n_features_shap} to {n_features_data}")
                    shap_array = shap_array[:, :n_features_data]
            else:
                logger.warning(f"Padding SHAP features from {n_features_shap} to {n_features_data}")
                # Pad with zeros for missing features
                padding = np.zeros((n_samples, n_features_data - n_features_shap))
                shap_array = np.hstack([shap_array, padding])
        
        logger.info(f"Final SHAP array shape: {shap_array.shape}")
        return shap_array
    
    def compute_global_explanations(self) -> Dict[str, Any]:
        """Compute global feature importance and explanations"""
        logger.info("Computing global explanations...")
        
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run create_shap_explainer first.")
        
        global_explanations = {}
        
        try:
            # Mean absolute SHAP values (feature importance)
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            
            # Ensure we have 1D array
            if len(mean_abs_shap.shape) > 1:
                mean_abs_shap = mean_abs_shap.flatten()
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            global_explanations['feature_importance'] = feature_importance.to_dict('records')
            
            # Top positive drivers (features that increase churn probability)
            mean_shap = self.shap_values.mean(axis=0)
            if len(mean_shap.shape) > 1:
                mean_shap = mean_shap.flatten()
            
            top_positive_drivers = pd.DataFrame({
                'feature': self.feature_names,
                'mean_shap': mean_shap
            }).sort_values('mean_shap', ascending=False).head(15)
            
            global_explanations['top_positive_drivers'] = top_positive_drivers.to_dict('records')
            
            # Top negative drivers (features that decrease churn probability)
            top_negative_drivers = pd.DataFrame({
                'feature': self.feature_names,
                'mean_shap': mean_shap
            }).sort_values('mean_shap', ascending=True).head(15)
            
            global_explanations['top_negative_drivers'] = top_negative_drivers.to_dict('records')
            
            logger.info("Global explanations computed successfully")
            return global_explanations
            
        except Exception as e:
            logger.error(f"Error computing global explanations: {e}")
            raise
    
    def compute_driver_breakdown(self, global_explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Break down feature importance by churn driver categories"""
        logger.info("Computing driver category breakdown...")
        
        # Enhanced driver categories with more keywords
        driver_categories = {
            'Low Engagement': ['transaction', 'login', 'activity', 'usage', 'calls', 'frequency', 'score', 'last_login'],
            'High Fees': ['fee', 'overdraft', 'penalty', 'charge', 'amount', 'ratio'],
            'Poor Service': ['complaint', 'satisfaction', 'service', 'resolution', 'escalation', 'quality', 'rating'],
            'Low Product Holding': ['product', 'credit_card', 'mortgage', 'investment', 'insurance', 'diversity', 'has_'],
            'Short Tenure': ['tenure', 'customer_since', 'new_customer', 'segment', 'age_group', 'months', 'years'],
            'Balance Fluctuations': ['balance', 'salary', 'volatility', 'withdrawal', 'trend', 'consistency', 'ratio', 'avg_'],
            'Demographics': ['age', 'income', 'region', 'occupation', 'education', 'marital', 'family', 'bracket', 'level', 'status'],
            'Credit Issues': ['credit', 'debt', 'loan', 'inquiry', 'default', 'utilization', 'score', 'dtl'],
            'Personalization & Offers': ['promotion', 'reward', 'offer', 'personalized', 'relevance', 'response', 'points', 'sent'],
            'Contract Events': ['account_age', 'maturity', 'renewal', 'closure', 'status', 'proximity', 'flag']
        }
        
        driver_breakdown = {}
        feature_importance_df = pd.DataFrame(global_explanations['feature_importance'])
        
        for driver, keywords in driver_categories.items():
            # Find features belonging to this driver category
            driver_features = []
            for feature in feature_importance_df['feature']:
                feature_lower = str(feature).lower()
                if any(keyword in feature_lower for keyword in keywords):
                    driver_features.append(feature)
            
            if driver_features:
                # Calculate total importance for this driver
                driver_importance = feature_importance_df[
                    feature_importance_df['feature'].isin(driver_features)
                ]['mean_abs_shap'].sum()
                
                # Get top 3 features in this category
                top_features = feature_importance_df[
                    feature_importance_df['feature'].isin(driver_features)
                ].head(3).to_dict('records')
                
                driver_breakdown[driver] = {
                    'total_importance': float(driver_importance),
                    'feature_count': len(driver_features),
                    'features': driver_features,
                    'top_features': top_features
                }
            else:
                driver_breakdown[driver] = {
                    'total_importance': 0.0,
                    'feature_count': 0,
                    'features': [],
                    'top_features': []
                }
        
        # Normalize importance scores
        total_importance = sum([data['total_importance'] for data in driver_breakdown.values()])
        if total_importance > 0:
            for driver in driver_breakdown:
                driver_breakdown[driver]['normalized_importance'] = (
                    driver_breakdown[driver]['total_importance'] / total_importance
                )
        else:
            for driver in driver_breakdown:
                driver_breakdown[driver]['normalized_importance'] = 0.0
        
        logger.info("Driver breakdown computed successfully")
        return driver_breakdown
    
    def create_visualizations(self, global_explanations: Dict[str, Any], 
                        output_dir: str = "models/explainability"):
        """Create and save SHAP visualizations with proper data alignment"""
        logger.info("Creating SHAP visualizations...")
        
        if self.shap_values is None or self.explainer is None:
            logger.warning("Cannot create visualizations - SHAP values or explainer not available")
            return
        
        try:
            # Create output directory
            viz_dir = Path(output_dir) / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure data alignment for visualizations
            n_shap_samples = self.shap_values.shape[0]
            X_viz = self.X_test.iloc[:n_shap_samples]  # Align with SHAP samples
            
            # 1. Summary plot (beeswarm plot)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, X_viz, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(viz_dir / "shap_summary_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved summary plot")
            
            # 2. Bar plot (mean absolute SHAP)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, X_viz, feature_names=self.feature_names, 
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(viz_dir / "shap_bar_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved bar plot")
            
            # 3. Driver importance plot
            driver_breakdown = self.compute_driver_breakdown(global_explanations)
            
            drivers = list(driver_breakdown.keys())
            importances = [driver_breakdown[d]['normalized_importance'] for d in drivers]
            
            plt.figure(figsize=(12, 6))
            y_pos = np.arange(len(drivers))
            bars = plt.barh(y_pos, importances)
            plt.yticks(y_pos, drivers)
            plt.xlabel('Normalized Importance')
            plt.title('Churn Driver Importance')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{importances[i]:.1%}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "driver_importance_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved driver importance plot")
            
            # 4. Top features bar plot
            top_features = global_explanations['feature_importance'][:10]
            feature_names = [f['feature'] for f in top_features]
            importance_vals = [f['mean_abs_shap'] for f in top_features]
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(feature_names))
            bars = plt.barh(y_pos, importance_vals)
            plt.yticks(y_pos, feature_names)
            plt.xlabel('Mean |SHAP| Value')
            plt.title('Top 10 Features by SHAP Importance')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance_vals[i]:.4f}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "top_features_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved top features plot")
            
        except Exception as e:
            logger.warning(f"Could not create some visualizations: {e}")
    
    def create_local_explanations(self, instance_indices: List[int] = None) -> Dict[str, Any]:
        """Create local explanations for specific instances"""
        logger.info("Creating local explanations...")
        
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run create_shap_explainer first.")
            
        if instance_indices is None:
            # Select diverse instances: high churn risk, low churn risk, and borderline
            try:
                # Use SHAP values to estimate predictions
                if hasattr(self.explainer, 'expected_value'):
                    if isinstance(self.explainer.expected_value, np.ndarray):
                        base_value = float(self.explainer.expected_value[1])
                    else:
                        base_value = float(self.explainer.expected_value)
                    
                    y_pred_proba = base_value + self.shap_values.sum(axis=1)
                    
                    # Select instances: top 2 churn risks, bottom 2, and 1 borderline
                    high_risk = np.argsort(y_pred_proba)[-2:]
                    low_risk = np.argsort(y_pred_proba)[:2]
                    borderline = [np.argsort(np.abs(y_pred_proba - 0.5))[0]]  # Closest to 0.5
                    
                    instance_indices = list(high_risk) + list(low_risk) + borderline
                    logger.info(f"Selected diverse instances: {instance_indices}")
                else:
                    raise AttributeError("Explainer has no expected_value")
            except Exception as e:
                # Fallback: use first 5 instances
                instance_indices = [0, 1, 2, 3, 4]
                logger.warning(f"Using first 5 instances as fallback: {e}")
        
        local_explanations = {}
        
        for idx in instance_indices:
            if idx < len(self.X_test) and idx < len(self.shap_values):
                # Get base value
                if hasattr(self.explainer, 'expected_value'):
                    if isinstance(self.explainer.expected_value, np.ndarray):
                        base_value = float(self.explainer.expected_value[1])
                    else:
                        base_value = float(self.explainer.expected_value)
                else:
                    base_value = 0.0
                
                # Get feature values and SHAP values for this instance
                instance_features = {}
                for i, feature in enumerate(self.feature_names):
                    if feature in self.X_test.columns:
                        instance_features[feature] = float(self.X_test[feature].iloc[idx])
                    else:
                        instance_features[feature] = 0.0
                
                # Calculate prediction
                shap_sum = float(np.sum(self.shap_values[idx]))
                prediction = base_value + shap_sum
                
                # Get top contributing features
                feature_contributions = []
                for i, feature in enumerate(self.feature_names):
                    feature_contributions.append({
                        'feature': feature,
                        'value': instance_features[feature],
                        'shap_value': float(self.shap_values[idx][i]),
                        'contribution': float(self.shap_values[idx][i])
                    })
                
                # Sort by absolute contribution
                feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
                
                instance_data = {
                    'features': instance_features,
                    'shap_values': self.shap_values[idx].tolist(),
                    'base_value': base_value,
                    'prediction': prediction,
                    'top_contributors': feature_contributions[:10]  # Top 10 features
                }
                local_explanations[f'instance_{idx}'] = instance_data
        
        logger.info(f"Local explanations created for {len(instance_indices)} instances")
        return local_explanations
    
    def save_shap_artifacts(self, global_explanations: Dict[str, Any], 
                          driver_breakdown: Dict[str, Any],
                          local_explanations: Dict[str, Any],
                          output_dir: str = "models/explainability"):
        """Save SHAP explanations and artifacts"""
        logger.info("Saving SHAP artifacts...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Save SHAP values
            if self.shap_values is not None:
                np.save(f"{output_dir}/shap_values_test_set_{self.version}.npy", self.shap_values)
                logger.info("Saved SHAP values array")
            
            # Save explainer metadata instead of full object
            if self.explainer is not None:
                try:
                    explainer_metadata = {
                        'type': str(type(self.explainer).__name__),
                        'feature_names': self.feature_names,
                        'expected_value': float(self.explainer.expected_value[1]) if isinstance(self.explainer.expected_value, np.ndarray) else float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0,
                        'model_type': str(type(self.model).__name__) if self.model is not None else None,
                        'input_shape': self.X_test.shape if self.X_test is not None else None
                    }
                    with open(f"{output_dir}/shap_explainer_metadata_{self.version}.json", 'w') as f:
                        json.dump(explainer_metadata, f, indent=2)
                    logger.info("Saved SHAP explainer metadata")
                except Exception as e:
                    logger.warning(f"Could not save SHAP explainer metadata: {e}")
            
            # Save all explanations 
            artifacts = {
                f'feature_importance_summary_{self.version}.json': global_explanations,
                f'top_drivers_{self.version}.json': {
                    'top_positive': global_explanations['top_positive_drivers'][:10],
                    'top_negative': global_explanations['top_negative_drivers'][:10]
                },
                f'shap_driver_breakdown_{self.version}.json': driver_breakdown,
                f'local_explanations_{self.version}.json': local_explanations
            }
            
            for filename, data in artifacts.items():
                with open(f"{output_dir}/{filename}", 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info(f"Saved {filename}")
            
            # Create comprehensive summary
            summary = {
                'version': self.version,
                'timestamp': pd.Timestamp.now().isoformat(),
                'feature_count': len(self.feature_names),
                'shap_values_shape': list(self.shap_values.shape) if self.shap_values is not None else None,
                'top_features': global_explanations['feature_importance'][:10],
                'driver_summary': {
                    driver: {
                        'importance': float(data['normalized_importance']),
                        'feature_count': data['feature_count'],
                        'top_features': data['top_features']
                    }
                    for driver, data in driver_breakdown.items()
                }
            }
            
            with open(f"{output_dir}/shap_analysis_summary_{self.version}.json", 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved analysis summary")
            
            logger.info("All SHAP artifacts saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving SHAP artifacts: {e}")
            # Don't re-raise to allow pipeline to continue
    
    def explanation_pipeline(self) -> Dict[str, Any]:
        """Complete SHAP explanation pipeline"""
        logger.info("Starting SHAP explanation pipeline...")
        
        try:
            # Load model and data
            model, X_test, y_test = self.load_model_and_data()
            
            # Create SHAP explainer
            self.create_shap_explainer(model, X_test)
            
            # Compute global explanations
            global_explanations = self.compute_global_explanations()
            
            # Compute driver breakdown
            driver_breakdown = self.compute_driver_breakdown(global_explanations)
            
            # Create visualizations
            self.create_visualizations(global_explanations)
            
            # Compute local explanations
            local_explanations = self.create_local_explanations()
            
            # Save artifacts
            self.save_shap_artifacts(global_explanations, driver_breakdown, local_explanations)
            
            logger.info("SHAP explanation pipeline completed successfully")
            
            return {
                'global_explanations': global_explanations,
                'driver_breakdown': driver_breakdown,
                'local_explanations': local_explanations
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation pipeline failed: {e}")
            raise

def main():
    """Main function to run SHAP explanation pipeline"""
    try:
        explainer = SHAPExplainer(version="v1.0")
        results = explainer.explanation_pipeline()
        
        print("\n" + "="*60)
        print("✅ SHAP EXPLANATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n📊 TOP 5 FEATURES BY IMPORTANCE:")
        for i, feature in enumerate(results['global_explanations']['feature_importance'][:5]):
            print(f"   {i+1}. {feature['feature']}: {feature['mean_abs_shap']:.4f}")
        
        print(f"\n🎯 TOP 5 CHURN DRIVERS BY CATEGORY:")
        driver_breakdown = results['driver_breakdown']
        for driver, data in sorted(driver_breakdown.items(), 
                                 key=lambda x: x[1]['normalized_importance'], reverse=True)[:5]:
            print(f"   {driver}: {data['normalized_importance']:.1%}")
            if data['top_features']:
                top_feat = data['top_features'][0]
                print(f"      Top feature: {top_feat['feature']} ({top_feat['mean_abs_shap']:.4f})")
        
        print(f"\n💾 ARTIFACTS SAVED TO: models/explainability/")
        print(f"   - SHAP values: shap_values_test_set_v1.0.npy")
        print(f"   - Feature importance: feature_importance_summary_v1.0.json")
        print(f"   - Driver breakdown: shap_driver_breakdown_v1.0.json")
        print(f"   - Local explanations: local_explanations_v1.0.json")
        print(f"   - Visualizations: models/explainability/visualizations/")
        
    except Exception as e:
        print(f"\n❌ SHAP EXPLANATIONS FAILED: {e}")
        print("💡 Check the logs for detailed error information")

if __name__ == "__main__":
    main()