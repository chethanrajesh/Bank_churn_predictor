import os
import glob
from pathlib import Path

print("🔍 File System Diagnostic (Fixed Paths)")
print("=" * 50)

# Get the project root directory (where the script is located)
PROJECT_ROOT = Path(__file__).parent.parent

# Check key directories with absolute paths
directories_to_check = [
    PROJECT_ROOT / 'data/processed/',
    PROJECT_ROOT / 'models/validation/',
    PROJECT_ROOT / 'models/explainability/',
    PROJECT_ROOT / 'models/production/',
    PROJECT_ROOT / 'models/preprocessing/',
    PROJECT_ROOT / 'models/metadata/'
]

for directory in directories_to_check:
    print(f"\n📁 {directory}:")
    if directory.exists():
        files = list(directory.glob('*'))
        if files:
            for file in files[:5]:  # Show first 5 files
                size = file.stat().st_size
                print(f"   {file.name} ({size} bytes)")
        else:
            print("   ❌ No files found")
    else:
        print("   ❌ Directory doesn't exist")

# Check specific files
print(f"\n🔎 Specific file check:")
specific_files = [
    PROJECT_ROOT / 'data/processed/test_set.csv',
    PROJECT_ROOT / 'models/validation/test_set_predictions_v1.0.csv',
    PROJECT_ROOT / 'models/explainability/shap_values_test_set_v1.0.npy',
    PROJECT_ROOT / 'models/preprocessing/feature_names_v1.0.pkl',
    PROJECT_ROOT / 'models/production/model_performance_summary_v1.0.json',
    PROJECT_ROOT / 'models/explainability/feature_importance_summary_v1.0.json',
    PROJECT_ROOT / 'models/metadata/driver_to_features_mapping_v1.0.json'
]

for file_path in specific_files:
    exists = file_path.exists()
    if exists:
        size = file_path.stat().st_size
        status = "✅" if size > 0 else "⚠️ EMPTY"
        print(f"   {file_path}: {status} ({size} bytes)")
    else:
        print(f"   {file_path}: ❌ NOT FOUND")