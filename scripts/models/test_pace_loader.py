"""
Test if the trained LSTM model and preprocessing objects load correctly.
"""

import sys
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.models.pace_forecaster import PaceForecaster

def test_loading():
    print("🔄 Loading Pace Forecaster...")
    forecaster = PaceForecaster()
    forecaster.load_model()
    print("✅ Model loaded successfully.")
    print(f"   Model type: {type(forecaster.model).__name__}")
    print(f"   Scaler: {forecaster.scaler}")
    print(f"   Encoders: {list(forecaster.label_encoders.keys())}")

if __name__ == "__main__":
    test_loading()