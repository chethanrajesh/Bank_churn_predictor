import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import logging
from typing import Dict, Any, Tuple
import json
import joblib
from pathlib import Path
import sys

# Add src to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

# Import required packages with proper error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  SMOTE not available. Install with: pip install imbalanced-learn")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model training pipeline for bank churn prediction
    Trains and evaluates multiple models including XGBoost, Random Forest, and LightGBM
    Uses SMOTE for handling class imbalance
    """
    
    def __init__(self, random_state: int = 42, version: str = "v1.0"):
        self.random_state = random_state
        self.version = version
        self.models = {}
        self.model_metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.smote = None
        self.feature_names = []
        
    def load_feature_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load feature matrices for train, validation, and test sets"""
        logger.info("Loading feature matrices...")
        
        try:
            # Load feature names first
            feature_names_path = f"{data_dir}/feature_names.pkl"
            if Path(feature_names_path).exists():
                self.feature_names = joblib.load(feature_names_path)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            else:
                logger.warning("Feature names file not found. Will extract from data.")
            
            X_train = pd.read_csv(f"{data_dir}/feature_matrix_train.csv")
            X_val = pd.read_csv(f"{data_dir}/feature_matrix_val.csv")
            X_test = pd.read_csv(f"{data_dir}/feature_matrix_test.csv")
            
            # Store original shapes for reference
            self.original_shapes = {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
            
            # Separate features and target - handle customer_id if present
            columns_to_drop = ['churn_risk', 'churn_probability', 'customer_id']
            
            y_train = X_train['churn_risk']
            y_val = X_val['churn_risk']
            y_test = X_test['churn_risk']
            
            # Drop target columns and customer_id
            X_train = X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns])
            X_val = X_val.drop(columns=[col for col in columns_to_drop if col in X_val.columns])
            X_test = X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns])
            
            # Ensure feature names are set
            if not self.feature_names:
                self.feature_names = X_train.columns.tolist()
                logger.info(f"Extracted {len(self.feature_names)} feature names from data")
            
            # Ensure consistent column order
            X_train = X_train[self.feature_names]
            X_val = X_val[self.feature_names]
            X_test = X_test[self.feature_names]
            
            logger.info(f"Loaded feature matrices: Train{X_train.shape}, Val{X_val.shape}, Test{X_test.shape}")
            logger.info(f"Target distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error loading feature data: {e}")
            raise
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle class imbalance using SMOTE"""
        logger.info("Handling class imbalance...")
        
        original_distribution = np.bincount(y_train)
        logger.info(f"Original class distribution: {dict(zip(range(len(original_distribution)), original_distribution))}")
        
        if not SMOTE_AVAILABLE:
            logger.warning("SMOTE not available. Using class weights instead.")
            # Calculate class weights for fallback
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            n_classes = len(class_counts)
            
            class_weights = {}
            for i in range(n_classes):
                class_weights[i] = total_samples / (n_classes * class_counts[i])
            
            logger.info(f"Using class weights: {class_weights}")
            self.class_weights = class_weights
            return X_train, y_train
        
        try:
            # Apply SMOTE only if there's significant imbalance
            minority_class_ratio = min(original_distribution) / max(original_distribution)
            if minority_class_ratio > 0.3:  # If minority class is more than 30% of majority
                logger.info("Class imbalance is moderate. Using class weights instead of SMOTE.")
                return X_train, y_train
            
            # Apply SMOTE for severe imbalance
            self.smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
            
            new_distribution = np.bincount(y_train_resampled)
            logger.info(f"After SMOTE - Class distribution: {dict(zip(range(len(new_distribution)), new_distribution))}")
            logger.info(f"Dataset size: {X_train.shape} -> {X_train_resampled.shape}")
            
            return X_train_resampled, y_train_resampled
            
        except Exception as e:
            logger.error(f"Error applying SMOTE: {e}. Using original data.")
            return X_train, y_train
    
    def initialize_models(self, y_train: pd.DataFrame) -> Dict[str, Any]:
        """Initialize multiple models for training including XGBoost and LightGBM"""
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = 1.0
        if len(y_train) > 0:
            class_counts = np.bincount(y_train)
            if len(class_counts) == 2:
                scale_pos_weight = class_counts[0] / class_counts[1]
                logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Enhanced model configurations for better convergence and performance
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=5000,  # Significantly increased for convergence
                solver='saga',  # More robust solver
                class_weight='balanced',
                n_jobs=-1,
                tol=1e-3,  # Relaxed tolerance for faster convergence
                C=0.1,  # Added regularization
                penalty='l2'  # Explicit L2 regularization
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight='balanced_subsample',  # Better for imbalanced data
                n_jobs=-1,
                max_depth=10,
                min_samples_split=10,  # Increased to reduce overfitting
                min_samples_leaf=5,    # Increased to reduce overfitting
                max_features='sqrt'    # Better generalization
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                n_jobs=-1,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,  # Added for better generalization
                colsample_bytree=0.8,  # Added for better generalization
            )
            logger.info("✅ XGBoost model initialized")
        else:
            logger.warning("❌ XGBoost not available - skipping XGBoost model")
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1,
                verbosity=-1,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1
            )
            logger.info("✅ LightGBM model initialized")
        else:
            logger.warning("❌ LightGBM not available - skipping LightGBM model")
        
        logger.info(f"Initialized {len(models)} models for training")
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                    X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Any]:
        """Train all models and evaluate on validation set"""
        logger.info("Training models...")
        
        self.models = self.initialize_models(y_train)
        model_metrics = {}
        
        for model_name, model in self.models.items():
            logger.info(f"🏋️ Training {model_name}...")
            
            try:
                # Train model with progress tracking
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba, model_name)
                model_metrics[model_name] = metrics
                
                # Cross-validation for additional validation (use smaller folds for speed)
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
                metrics['cv_auc_mean'] = float(cv_scores.mean())
                metrics['cv_auc_std'] = float(cv_scores.std())
                
                logger.info(f"✅ {model_name} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
                model_metrics[model_name] = {
                    'error': str(e),
                    'auc': 0.0,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'accuracy': 0.0,
                    'cv_auc_mean': 0.0,
                    'cv_auc_std': 0.0
                }
        
        self.model_metrics = model_metrics
        return model_metrics
    
    def calculate_metrics(self, y_true: pd.DataFrame, y_pred: pd.DataFrame, 
                         y_pred_proba: pd.DataFrame, model_name: str) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        try:
            # Calculate AUC
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            # Calculate F1 score
            f1 = f1_score(y_true, y_pred)
            
            # Get classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Calculate additional metrics
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # ✅ ADD PRECISION-RECALL METRICS HERE
            from sklearn.metrics import precision_recall_curve, average_precision_score
        
             # Calculate Precision-Recall AUC
            pr_auc = average_precision_score(y_true, y_pred_proba)

             # Calculate Precision-Recall curve for threshold analysis
            precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            # Find optimal threshold (maximizing F1 score)
            f1_scores = []
            for i in range(len(precision_vals)):
                if precision_vals[i] + recall_vals[i] > 0:
                    f1_scores.append(2 * (precision_vals[i] * recall_vals[i]) / (precision_vals[i] + recall_vals[i]))
                else:
                    f1_scores.append(0)
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            metrics = {
                'auc': float(auc_score),
                'f1': float(f1),
                'precision': float(report['weighted avg']['precision']),
                'recall': float(report['weighted avg']['recall']),
                'accuracy': float(report['accuracy']),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp),
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                # ✅ NEW METRICS ADDED
                'pr_auc': float(pr_auc),  # Precision-Recall AUC
                'optimal_threshold': float(optimal_threshold),  # Optimal threshold for max F1
                'precision_at_optimal': float(precision_vals[optimal_idx]),
                'recall_at_optimal': float(recall_vals[optimal_idx]),
                'f1_at_optimal': float(f1_scores[optimal_idx])
            }
            
            # Add class-specific metrics
            if '0' in report:
                metrics['precision_0'] = float(report['0']['precision'])
                metrics['recall_0'] = float(report['0']['recall'])
                metrics['f1_0'] = float(report['0']['f1-score'])
                metrics['support_0'] = int(report['0']['support'])
            if '1' in report:
                metrics['precision_1'] = float(report['1']['precision'])
                metrics['recall_1'] = float(report['1']['recall'])
                metrics['f1_1'] = float(report['1']['f1-score'])
                metrics['support_1'] = int(report['1']['support'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            return {
                'auc': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'error': str(e)
            }
    
    def select_best_model(self) -> Tuple[Any, str]:
        """Select the best model based on validation performance (prioritizing AUC)"""
        logger.info("Selecting best model...")
        
        best_auc = 0
        best_model_name = None
        best_model = None
        
        for model_name, metrics in self.model_metrics.items():
            if ('auc' in metrics and metrics['auc'] > best_auc and 
                'error' not in metrics and metrics['auc'] > 0.5):  # Basic sanity check
                best_auc = metrics['auc']
                best_model_name = model_name
                best_model = self.models[model_name]
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        
        if best_model_name:
            logger.info(f"🎯 Best model: {best_model_name} with AUC: {best_auc:.4f}")
        else:
            logger.warning("❌ No best model selected - all models may have failed")
            # Fallback to first available model without errors
            for model_name, model in self.models.items():
                model_metrics = self.model_metrics.get(model_name, {})
                if 'error' not in model_metrics and model_metrics.get('auc', 0) > 0.5:
                    self.best_model = model
                    self.best_model_name = model_name
                    logger.info(f"🔄 Fallback to: {model_name}")
                    break
        
        return self.best_model, self.best_model_name
    
    def evaluate_best_model(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the best model on test set"""
        logger.info("Evaluating best model on test set...")
    
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model first.")
        
        try:
            # Predict on test set
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
            
            # ✅ ADD THRESHOLD-BASED PREDICTIONS
            from sklearn.metrics import precision_recall_curve
            
            # Get optimal threshold from validation performance
            best_model_metrics = self.model_metrics.get(self.best_model_name, {})
            optimal_threshold = best_model_metrics.get('optimal_threshold', 0.5)
            
            # Make predictions with default threshold (0.5)
            y_pred_default = (y_pred_proba >= 0.5).astype(int)
            
            # Make predictions with optimal threshold
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Calculate metrics for both thresholds
            default_metrics = self.calculate_metrics(y_test, y_pred_default, y_pred_proba, f"{self.best_model_name}_default")
            optimal_metrics = self.calculate_metrics(y_test, y_pred_optimal, y_pred_proba, f"{self.best_model_name}_optimal")
            
            # Use optimal threshold metrics as primary
            test_metrics = optimal_metrics
            
            # Add comparison information
            test_metrics['threshold_used'] = optimal_threshold
            test_metrics['default_threshold_metrics'] = {
                'f1': default_metrics['f1'],
                'precision': default_metrics['precision'],
                'recall': default_metrics['recall'],
                'accuracy': default_metrics['accuracy']
            }
            
            # Create detailed confusion matrix
            cm = confusion_matrix(y_test, y_pred_optimal)
            test_metrics['confusion_matrix'] = cm.tolist()
            
            logger.info(f"📊 Test performance - AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")
            logger.info(f"📊 PR AUC: {test_metrics['pr_auc']:.4f}, Optimal Threshold: {optimal_threshold:.4f}")
            logger.info(f"📈 Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
            
            # ✅ LOG THRESHOLD COMPARISON
            logger.info(f"🔧 Threshold Comparison:")
            logger.info(f"   Default (0.5)  - F1: {default_metrics['f1']:.4f}, Precision: {default_metrics['precision']:.4f}, Recall: {default_metrics['recall']:.4f}")
            logger.info(f"   Optimal ({optimal_threshold:.3f}) - F1: {optimal_metrics['f1']:.4f}, Precision: {optimal_metrics['precision']:.4f}, Recall: {optimal_metrics['recall']:.4f}")
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating best model: {e}")
            return {'error': str(e)}
    
    def save_preprocessing_artifacts(self, output_dir: str = "models/preprocessing"):
        """Save preprocessing artifacts including feature names"""
        logger.info("Saving preprocessing artifacts...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Save feature names
            joblib.dump(self.feature_names, f"{output_dir}/feature_names_{self.version}.pkl")
            
            # Save SMOTE if used
            if self.smote is not None:
                joblib.dump(self.smote, f"{output_dir}/smote_{self.version}.pkl")
            
            logger.info("✅ Preprocessing artifacts saved")
        except Exception as e:
            logger.error(f"Error saving preprocessing artifacts: {e}")
    
    def save_models_and_artifacts(self, output_dir: str = "models/production"):
        """Save trained models and training artifacts"""
        logger.info("Saving models and artifacts...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save preprocessing artifacts first
        self.save_preprocessing_artifacts()
        
        # Save all models
        for model_name, model in self.models.items():
            try:
                model_path = f"{output_dir}/churn_model_{model_name}_{self.version}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"💾 Saved model: {model_path}")
            except Exception as e:
                logger.error(f"Error saving model {model_name}: {e}")
        
        # Save best model separately
        if self.best_model is not None:
            try:
                best_model_path = f"{output_dir}/churn_model_best_{self.version}.pkl"
                joblib.dump(self.best_model, best_model_path)
                logger.info(f"💾 Saved best model: {best_model_path}")
                
                # Update current model reference
                current_model_file = f"{output_dir}/CURRENT_MODEL.txt"
                with open(current_model_file, 'w') as f:
                    f.write(f"churn_model_{self.best_model_name}_{self.version}.pkl")
                logger.info(f"📝 Current model set to: churn_model_{self.best_model_name}_{self.version}.pkl")
                
            except Exception as e:
                logger.error(f"Error saving best model: {e}")
        
        # Save comprehensive model metrics
        metrics_summary = {
            'version': self.version,
            'best_model': self.best_model_name,
            'validation_metrics': self.model_metrics,
            'test_metrics': getattr(self, 'test_metrics', {}),
            'training_info': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'feature_count': len(self.feature_names),
                'random_state': self.random_state,
                'original_data_shapes': self.original_shapes,
                'smote_used': SMOTE_AVAILABLE and self.smote is not None,
                'models_available': {
                    'xgboost': XGBOOST_AVAILABLE,
                    'lightgbm': LIGHTGBM_AVAILABLE,
                    'smote': SMOTE_AVAILABLE
                }
            }
        }
        
        try:
            metrics_path = f"{output_dir}/model_performance_summary_{self.version}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_summary, f, indent=2, default=str)
            logger.info(f"💾 Saved model performance summary: {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
        
        # Save feature importance for tree-based models
        feature_importance = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                try:
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Convert to serializable format
                    records = []
                    for _, row in importance_df.iterrows():
                        records.append({
                            'feature': row['feature'],
                            'importance': float(row['importance'])
                        })
                    
                    feature_importance[model_name] = records
                    logger.info(f"💾 Saved feature importance for {model_name}")
                    
                except Exception as e:
                    logger.error(f"Error saving feature importance for {model_name}: {e}")
        
        if feature_importance:
            try:
                importance_path = f"{output_dir}/feature_importance_ranks_{self.version}.json"
                with open(importance_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                logger.info(f"💾 Saved feature importance ranks: {importance_path}")
            except Exception as e:
                logger.error(f"Error saving feature importance: {e}")
        
        # Save hyperparameters
        hyperparameters = {}
        for model_name, model in self.models.items():
            try:
                hyperparameters[model_name] = {
                    str(k): (str(v) if not isinstance(v, (int, float, bool, str)) else v)
                    for k, v in model.get_params().items()
                }
            except Exception as e:
                logger.error(f"Error getting hyperparameters for {model_name}: {e}")
        
        if hyperparameters:
            try:
                hyperparams_path = f"{output_dir}/hyperparameters_{self.version}.json"
                with open(hyperparams_path, 'w') as f:
                    json.dump(hyperparameters, f, indent=2)
                logger.info(f"💾 Saved hyperparameters: {hyperparams_path}")
            except Exception as e:
                logger.error(f"Error saving hyperparameters: {e}")
        
        # Update model registry
        self.update_model_registry()
        
        logger.info("✅ All models and artifacts saved successfully")
    
    def update_model_registry(self):
        """Update the model registry with current training information"""
        registry_path = "models/metadata/model_registry.json"
        Path("models/metadata").mkdir(parents=True, exist_ok=True)
        
        registry_entry = {
            'version': self.version,
            'timestamp': pd.Timestamp.now().isoformat(),
            'best_model': self.best_model_name,
            'test_auc': self.test_metrics.get('auc', 0) if hasattr(self, 'test_metrics') else 0,
            'test_f1': self.test_metrics.get('f1', 0) if hasattr(self, 'test_metrics') else 0,
            'feature_count': len(self.feature_names),
            'models_trained': list(self.models.keys())
        }
        
        try:
            # Initialize empty registry by default
            registry = []
            
            # Try to load existing registry if it exists
            if Path(registry_path).exists():
                try:
                    with open(registry_path, 'r') as f:
                        content = f.read().strip()
                        if content:  # Only try to parse if file is not empty
                            registry = json.loads(content)
                        else:
                            registry = []
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Invalid JSON in registry file. Starting fresh. Error: {e}")
                    registry = []
            
            # Add new entry
            registry.append(registry_entry)
            
            # Keep only last 10 entries
            if len(registry) > 10:
                registry = registry[-10:]
            
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            logger.info(f"📝 Updated model registry: {registry_path}")
        except Exception as e:
            logger.error(f"Error updating model registry: {e}")
    
    def training_pipeline(self) -> Dict[str, Any]:  # FIXED: Proper indentation
        """Complete model training pipeline"""
        logger.info("🚀 Starting model training pipeline...")
        
        # Load feature data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_feature_data()
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self.handle_class_imbalance(X_train, y_train)
        
        # Train models
        validation_metrics = self.train_models(X_train_resampled, y_train_resampled, X_val, y_val)
        
        # Select best model
        best_model, best_model_name = self.select_best_model()
        
        # Evaluate on test set
        test_metrics = self.evaluate_best_model(X_test, y_test)
        self.test_metrics = test_metrics
        
        # Save models and artifacts
        self.save_models_and_artifacts()
        
        # Generate metadata after training
        try:
            from metadata_generator import MetadataGenerator
            metadata_gen = MetadataGenerator(version=self.version)
            metadata_gen.generate_all_metadata()
            logger.info("✅ Metadata generation completed")
        except ImportError:
            logger.warning("Metadata generator not available, skipping metadata generation")
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
        
        logger.info("✅ Model training pipeline completed successfully")
        
        return {
            'best_model': best_model_name,
            'validation_metrics': validation_metrics,
            'test_metrics': test_metrics
        }

def main():
    """Main function to run model training pipeline"""
    print("🔧 Model Training Pipeline")
    print("=" * 50)
    
    # Check package availability
    print(f"📦 XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"📦 LightGBM available: {LIGHTGBM_AVAILABLE}")
    print(f"📦 SMOTE available: {SMOTE_AVAILABLE}")
    print("=" * 50)
    
    trainer = ModelTrainer(version="v1.0")
    results = trainer.training_pipeline()
    
    print("\n" + "=" * 50)
    print("✅ MODEL TRAINING COMPLETED!")
    print("=" * 50)
    
    # In the main() function, update the output section:
    if results['best_model']:
        best_model_metrics = results['validation_metrics'][results['best_model']]
        test_metrics = results['test_metrics']
        
        print(f"🎯 Best Model: {results['best_model']}")
        print(f"📊 Validation AUC: {best_model_metrics['auc']:.4f}")
        print(f"📊 Validation F1: {best_model_metrics['f1']:.4f}")
        print(f"📊 Validation PR AUC: {best_model_metrics.get('pr_auc', 0):.4f}")
        
        if 'auc' in test_metrics:
            print(f"📊 Test AUC: {test_metrics['auc']:.4f}")
            print(f"📊 Test F1: {test_metrics['f1']:.4f}")
            print(f"📊 Test PR AUC: {test_metrics.get('pr_auc', 0):.4f}")
            print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"🔧 Optimal Threshold: {test_metrics.get('threshold_used', 0.5):.4f}")
            
            # Show threshold comparison
            if 'default_threshold_metrics' in test_metrics:
                default = test_metrics['default_threshold_metrics']
                print(f"📈 Threshold Comparison:")
                print(f"   Default (0.5)  - F1: {default['f1']:.4f}, Precision: {default['precision']:.4f}, Recall: {default['recall']:.4f}")
                print(f"   Optimal ({test_metrics.get('threshold_used', 0.5):.3f}) - F1: {test_metrics['f1']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
                
if __name__ == "__main__":
    main()