import sys
import os
import joblib
from pathlib import Path

print("🔍 DIAGNOSING DRIVER STYLE ANALYZER")
print("=" * 50)

base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
models_path = base_path / 'models'

# Check if driver style model exists
style_model_path = models_path / 'driver_style_cluster_model.joblib'
print(f"Model path: {style_model_path}")
print(f"Model exists: {style_model_path.exists()}")

if style_model_path.exists():
    try:
        data = joblib.load(style_model_path)
        print("✅ Model loaded successfully")
        print(f"Keys in model: {list(data.keys())}")
        
        if 'driver_profiles' in data:
            driver_profiles = data['driver_profiles']
            print(f"Driver profiles shape: {driver_profiles.shape}")
            print(f"Columns: {list(driver_profiles.columns)}")
            print(f"Drivers in model: {driver_profiles['Driver'].tolist() if 'Driver' in driver_profiles.columns else 'No Driver column'}")
            if 'style_label' in driver_profiles.columns:
                print(f"Style labels: {driver_profiles['style_label'].unique()}")
        else:
            print("❌ No driver_profiles in model")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("❌ No pre-trained model found - you need to run the driver style analyzer first!")
