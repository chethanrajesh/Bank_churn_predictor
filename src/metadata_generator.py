import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, List
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetadataGenerator:
    """
    Generate metadata files for the Bank Churn Prediction system
    Creates driver mappings, feature schema, and baseline statistics
    """
    
    def __init__(self, version: str = "v1.0"):
        self.version = version
        
    def load_feature_data(self, data_dir: str = "data/processed") -> pd.DataFrame:
        """Load feature matrix for metadata generation"""
        logger.info("Loading feature data for metadata generation...")
        
        try:
            # Try to load encoded feature matrix first
            encoded_path = f"{data_dir}/feature_matrix_encoded.csv"
            if Path(encoded_path).exists():
                df = pd.read_csv(encoded_path)
                logger.info(f"Loaded encoded feature matrix: {df.shape}")
            else:
                # Fallback to combined feature matrix
                df = pd.read_csv(f"{data_dir}/feature_matrix.csv")
                logger.info(f"Loaded combined feature matrix: {df.shape}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading feature data: {e}")
            raise
    
    def load_feature_names(self, preprocessing_dir: str = "models/preprocessing") -> List[str]:
        """Load feature names from preprocessing artifacts"""
        logger.info("Loading feature names...")
        
        try:
            feature_names_path = f"{preprocessing_dir}/feature_names_{self.version}.pkl"
            if Path(feature_names_path).exists():
                feature_names = joblib.load(feature_names_path)
                logger.info(f"Loaded {len(feature_names)} feature names")
                return feature_names
            else:
                raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
        except Exception as e:
            logger.error(f"Error loading feature names: {e}")
            raise
    
    def create_driver_to_features_mapping(self, feature_names: List[str]) -> Dict[str, Any]:
        """Create mapping of churn drivers to their corresponding features"""
        logger.info("Creating driver to features mapping...")
        
        # Enhanced driver categories with comprehensive feature matching
        driver_mappings = {
            "Low Engagement": {
                "description": "Customer activity and engagement levels",
                "keywords": ["transaction", "login", "activity", "usage", "calls", "frequency", "score", "last_login", "service_calls"],
                "features": [],
                "business_impact": "High",
                "mitigation_strategy": "Increase engagement through personalized communication and rewards"
            },
            "High Fees": {
                "description": "Fee-related charges and penalties",
                "keywords": ["fee", "overdraft", "penalty", "charge", "amount", "ratio", "monthly_fees"],
                "features": [],
                "business_impact": "High", 
                "mitigation_strategy": "Review fee structure, offer fee waivers for loyal customers"
            },
            "Poor Service": {
                "description": "Service quality and complaint history",
                "keywords": ["complaint", "satisfaction", "service", "resolution", "escalation", "quality", "rating"],
                "features": [],
                "business_impact": "Medium",
                "mitigation_strategy": "Improve service quality, reduce resolution time"
            },
            "Low Product Holding": {
                "description": "Product diversification and cross-selling",
                "keywords": ["product", "credit_card", "mortgage", "investment", "insurance", "diversity", "has_"],
                "features": [],
                "business_impact": "Medium",
                "mitigation_strategy": "Cross-selling campaigns, product bundles"
            },
            "Short Tenure": {
                "description": "Customer relationship duration and lifecycle",
                "keywords": ["tenure", "customer_since", "new_customer", "segment", "age_group", "months", "years"],
                "features": [],
                "business_impact": "Medium",
                "mitigation_strategy": "Onboarding programs, early engagement initiatives"
            },
            "Balance Fluctuations": {
                "description": "Account balance stability and transaction patterns", 
                "keywords": ["balance", "salary", "volatility", "withdrawal", "trend", "consistency", "ratio", "avg_", "account_balance"],
                "features": [],
                "business_impact": "Medium",
                "mitigation_strategy": "Balance alerts, savings products"
            },
            "Demographics": {
                "description": "Customer demographic characteristics",
                "keywords": ["age", "income", "region", "occupation", "education", "marital", "family", "bracket", "level", "status"],
                "features": [],
                "business_impact": "Low",
                "mitigation_strategy": "Segmented marketing, personalized offers"
            },
            "Credit Issues": {
                "description": "Credit history and debt management",
                "keywords": ["credit", "debt", "loan", "inquiry", "default", "utilization", "score", "dtl", "debt_to_income"],
                "features": [],
                "business_impact": "High",
                "mitigation_strategy": "Credit counseling, debt restructuring"
            },
            "Personalization & Offers": {
                "description": "Marketing engagement and personalization",
                "keywords": ["promotion", "reward", "offer", "personalized", "relevance", "response", "points", "sent"],
                "features": [],
                "business_impact": "Medium", 
                "mitigation_strategy": "Improved targeting, personalized communication"
            },
            "Contract Events": {
                "description": "Account events and contractual milestones",
                "keywords": ["account_age", "maturity", "renewal", "closure", "status", "proximity", "flag", "maturity_proximity"],
                "features": [],
                "business_impact": "Medium",
                "mitigation_strategy": "Proactive renewal reminders, retention offers"
            }
        }
        
        # Map features to drivers
        for feature in feature_names:
            feature_lower = feature.lower()
            matched = False
            
            for driver, driver_info in driver_mappings.items():
                keywords = driver_info["keywords"]
                if any(keyword in feature_lower for keyword in keywords):
                    driver_info["features"].append(feature)
                    matched = True
            
            if not matched:
                # Add to uncategorized
                if "Uncategorized" not in driver_mappings:
                    driver_mappings["Uncategorized"] = {
                        "description": "Features not categorized into specific drivers",
                        "keywords": [],
                        "features": [],
                        "business_impact": "Unknown",
                        "mitigation_strategy": "Needs analysis"
                    }
                driver_mappings["Uncategorized"]["features"].append(feature)
        
        # Add statistics
        for driver, driver_info in driver_mappings.items():
            driver_info["feature_count"] = len(driver_info["features"])
            driver_info["features"] = sorted(driver_info["features"])
        
        logger.info(f"Created mappings for {len(driver_mappings)} driver categories")
        return driver_mappings
    
    def create_feature_schema(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Create comprehensive feature schema with statistics"""
        logger.info("Creating feature schema...")
        
        feature_schema = {
            "version": self.version,
            "total_features": len(feature_names),
            "feature_categories": {},
            "feature_details": {},
            "data_statistics": {
                "total_samples": len(df),
                "numerical_features": 0,
                "categorical_features": 0,
                "binary_features": 0
            }
        }
        
        # Define feature categories
        categories = {
            "engagement": ["transaction", "login", "activity", "usage", "calls", "frequency"],
            "fees": ["fee", "overdraft", "penalty", "charge", "amount", "ratio"],
            "service": ["complaint", "satisfaction", "service", "resolution", "escalation", "quality"],
            "products": ["product", "credit_card", "mortgage", "investment", "insurance", "diversity"],
            "tenure": ["tenure", "customer_since", "new_customer", "segment", "age_group"],
            "balance": ["balance", "salary", "volatility", "withdrawal", "trend", "consistency"],
            "demographics": ["age", "income", "region", "occupation", "education", "marital", "family"],
            "credit": ["credit", "debt", "loan", "inquiry", "default", "utilization"],
            "personalization": ["promotion", "reward", "offer", "personalized", "relevance", "response"],
            "contract": ["account_age", "maturity", "renewal", "closure", "status", "proximity"]
        }
        
        # Initialize categories
        for category in categories:
            feature_schema["feature_categories"][category] = []
        
        # Analyze each feature
        for feature in feature_names:
            if feature in df.columns:
                feature_data = df[feature]
                
                # Determine feature type
                if feature_data.dtype in ['int64', 'float64']:
                    if feature_data.nunique() == 2:
                        feature_type = "binary"
                        feature_schema["data_statistics"]["binary_features"] += 1
                    else:
                        feature_type = "numerical"
                        feature_schema["data_statistics"]["numerical_features"] += 1
                else:
                    feature_type = "categorical"
                    feature_schema["data_statistics"]["categorical_features"] += 1
                
                # Calculate statistics
                stats = {
                    "type": feature_type,
                    "dtype": str(feature_data.dtype),
                    "missing_values": int(feature_data.isnull().sum()),
                    "missing_percentage": float(feature_data.isnull().mean()),
                    "unique_values": int(feature_data.nunique())
                }
                
                if feature_type == "numerical":
                    stats.update({
                        "min": float(feature_data.min()),
                        "max": float(feature_data.max()),
                        "mean": float(feature_data.mean()),
                        "std": float(feature_data.std()),
                        "median": float(feature_data.median())
                    })
                elif feature_type == "categorical":
                    stats["top_categories"] = feature_data.value_counts().head(5).to_dict()
                
                feature_schema["feature_details"][feature] = stats
                
                # Categorize feature
                feature_lower = feature.lower()
                categorized = False
                for category, keywords in categories.items():
                    if any(keyword in feature_lower for keyword in keywords):
                        feature_schema["feature_categories"][category].append(feature)
                        categorized = True
                        break
                
                if not categorized:
                    if "other" not in feature_schema["feature_categories"]:
                        feature_schema["feature_categories"]["other"] = []
                    feature_schema["feature_categories"]["other"].append(feature)
        
        logger.info("Feature schema created successfully")
        return feature_schema
    
    def create_data_statistics_baseline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create baseline statistics for drift detection"""
        logger.info("Creating data statistics baseline...")
        
        baseline_stats = {
            "version": self.version,
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_info": {
                "total_samples": len(df),
                "total_features": len(df.columns),
                "churn_rate": float(df['churn_risk'].mean()) if 'churn_risk' in df.columns else None
            },
            "feature_statistics": {},
            "correlation_analysis": {},
            "data_quality": {
                "missing_values_total": int(df.isnull().sum().sum()),
                "missing_values_percentage": float(df.isnull().sum().sum() / (len(df) * len(df.columns))),
                "duplicate_rows": int(df.duplicated().sum())
            }
        }
        
        # Feature-level statistics
        numerical_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=['object']).columns
        
        for feature in df.columns:
            if feature in numerical_features:
                baseline_stats["feature_statistics"][feature] = {
                    "type": "numerical",
                    "mean": float(df[feature].mean()),
                    "std": float(df[feature].std()),
                    "min": float(df[feature].min()),
                    "max": float(df[feature].max()),
                    "median": float(df[feature].median()),
                    "q1": float(df[feature].quantile(0.25)),
                    "q3": float(df[feature].quantile(0.75)),
                    "missing_count": int(df[feature].isnull().sum())
                }
            elif feature in categorical_features:
                value_counts = df[feature].value_counts()
                baseline_stats["feature_statistics"][feature] = {
                    "type": "categorical",
                    "unique_categories": int(df[feature].nunique()),
                    "top_category": value_counts.index[0] if len(value_counts) > 0 else None,
                    "top_category_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "missing_count": int(df[feature].isnull().sum())
                }
        
        # Correlation matrix (for numerical features only, top 20)
        if len(numerical_features) > 1:
            corr_matrix = df[numerical_features].corr()
            # Get top correlations (absolute value)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
            
            # Sort by absolute correlation and take top 20
            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            baseline_stats["correlation_analysis"]["top_correlations"] = corr_pairs[:20]
        
        logger.info("Data statistics baseline created successfully")
        return baseline_stats
    
    def create_model_performance_log(self, models_dir: str = "models/production") -> pd.DataFrame:
        """Create and update model performance log"""
        logger.info("Creating model performance log...")
        
        try:
            # Load current performance summary
            performance_path = f"{models_dir}/model_performance_summary_{self.version}.json"
            if Path(performance_path).exists():
                with open(performance_path, 'r') as f:
                    performance_data = json.load(f)
                
                # Create log entry
                log_entry = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'version': self.version,
                    'best_model': performance_data.get('best_model', 'unknown'),
                    'test_auc': performance_data.get('test_metrics', {}).get('auc', 0),
                    'test_f1': performance_data.get('test_metrics', {}).get('f1', 0),
                    'test_accuracy': performance_data.get('test_metrics', {}).get('accuracy', 0),
                    'feature_count': performance_data.get('training_info', {}).get('feature_count', 0),
                    'training_date': performance_data.get('training_info', {}).get('timestamp', '')
                }
                
                # Create or update log file
                log_path = "models/metadata/model_performance_log.csv"
                Path("models/metadata").mkdir(parents=True, exist_ok=True)
                
                if Path(log_path).exists():
                    # Append to existing log
                    log_df = pd.read_csv(log_path)
                    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
                else:
                    # Create new log
                    log_df = pd.DataFrame([log_entry])
                
                # Save log
                log_df.to_csv(log_path, index=False)
                logger.info(f"Model performance log updated: {log_path}")
                
                return log_df
            else:
                logger.warning("Performance summary not found, creating empty log")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error creating performance log: {e}")
            return pd.DataFrame()
    
    def generate_all_metadata(self):
        """Generate all metadata files"""
        logger.info("Starting metadata generation...")
        
        try:
            # Load required data
            df = self.load_feature_data()
            feature_names = self.load_feature_names()
            
            # Ensure metadata directory exists
            metadata_dir = "models/metadata"
            Path(metadata_dir).mkdir(parents=True, exist_ok=True)
            
            # 1. Generate driver to features mapping
            driver_mapping = self.create_driver_to_features_mapping(feature_names)
            with open(f"{metadata_dir}/driver_to_features_mapping_{self.version}.json", 'w') as f:
                json.dump(driver_mapping, f, indent=2)
            logger.info("✅ Driver to features mapping saved")
            
            # 2. Generate feature schema
            feature_schema = self.create_feature_schema(df, feature_names)
            with open(f"{metadata_dir}/feature_schema_{self.version}.json", 'w') as f:
                json.dump(feature_schema, f, indent=2)
            logger.info("✅ Feature schema saved")
            
            # 3. Generate data statistics baseline
            data_stats = self.create_data_statistics_baseline(df)
            with open(f"{metadata_dir}/data_statistics_baseline_{self.version}.json", 'w') as f:
                json.dump(data_stats, f, indent=2)
            logger.info("✅ Data statistics baseline saved")
            
            # 4. Update model performance log
            performance_log = self.create_model_performance_log()
            if not performance_log.empty:
                logger.info("✅ Model performance log updated")
            else:
                logger.warning("❌ Model performance log not updated")
            
            logger.info("🎉 All metadata files generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            raise

def main():
    """Main function to run metadata generation"""
    print("📊 Metadata Generation Pipeline")
    print("=" * 50)
    
    generator = MetadataGenerator(version="v1.0")
    generator.generate_all_metadata()
    
    print("\n" + "=" * 50)
    print("✅ METADATA GENERATION COMPLETED!")
    print("=" * 50)
    print("📁 Generated files:")
    print("   - models/metadata/driver_to_features_mapping_v1.0.json")
    print("   - models/metadata/feature_schema_v1.0.json") 
    print("   - models/metadata/data_statistics_baseline_v1.0.json")
    print("   - models/metadata/model_performance_log.csv")
    print("\n💡 These files provide comprehensive documentation and baseline for monitoring")

if __name__ == "__main__":
    main()