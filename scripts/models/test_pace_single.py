"""
Test the LSTM model on a single next-lap prediction using a real stint.
Merges driver style profiles to ensure all required features are present.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.models.pace_forecaster import PaceForecaster

def load_driver_profiles(models_path):
    """Load driver style profiles from saved clustering model."""
    style_path = models_path / 'driver_style_cluster_model.joblib'
    if style_path.exists():
        style_data = joblib.load(style_path)
        driver_profiles = style_data['driver_profiles']
        print(f"✅ Loaded driver profiles for {len(driver_profiles)} drivers")
        return driver_profiles
    else:
        print("⚠️ Driver style model not found. Using default values.")
        return None

def test_single_prediction():
    # 1. Load the trained model and preprocessing
    print("🔄 Loading Pace Forecaster...")
    forecaster = PaceForecaster()
    forecaster.load_model()
    print("✅ Model loaded.\n")

    # 2. Load lap data
    df = pd.read_csv(forecaster.data_path / 'modern_with_historical_context.csv')
    print(f"📊 Loaded {len(df)} laps.")

    # 3. Load driver profiles
    driver_profiles = load_driver_profiles(forecaster.models_path)

    # 4. Find a stint with at least 11 laps
    groups = df.groupby(['event_name', 'year', 'round', 'DriverNumber', 'Stint'])
    sample_stint = None
    for _, group in groups:
        if len(group) >= 11:
            sample_stint = group.sort_values('LapNumber').iloc[:11].copy()
            break

    if sample_stint is None:
        print("❌ No stint with 11 laps found.")
        return

    # 5. Merge driver profiles into the stint
    if driver_profiles is not None and 'Driver' in sample_stint.columns:
        sample_stint = sample_stint.merge(driver_profiles, on='Driver', how='left')
        # Fill missing style values with default 0.5
        style_cols = ['AggressionScore', 'ConsistencyScore', 'BrakingIntensity',
                      'TyrePreservation', 'OvertakingAbility']
        for col in style_cols:
            if col in sample_stint.columns:
                sample_stint[col] = sample_stint[col].fillna(0.5)
            else:
                sample_stint[col] = 0.5
    else:
        # If no profiles or no Driver column, create default style columns
        print("⚠️ Driver column missing or profiles not available. Using default style values.")
        for col in ['AggressionScore', 'ConsistencyScore', 'BrakingIntensity',
                    'TyrePreservation', 'OvertakingAbility']:
            sample_stint[col] = 0.5

    # 6. Display stint info
    first_row = sample_stint.iloc[0]
    print(f"\n🏁 Stint: {first_row.get('Driver', 'Unknown')} at {first_row['circuit']}, laps {first_row['LapNumber']:.0f}–{sample_stint.iloc[10]['LapNumber']:.0f}")

    # 7. Extract first 10 laps as input, and the 11th as target
    input_laps = sample_stint.iloc[:10].copy()
    actual_next = sample_stint.iloc[10]['lap_time_seconds']

    # 8. Use the model to predict the next lap
    predicted_next = forecaster.predict_next_lap(input_laps)

    # 9. Print results
    print(f"\n🔮 Predicted next lap: {predicted_next:.3f}s")
    print(f"📊 Actual next lap:    {actual_next:.3f}s")
    print(f"📈 Error: {abs(predicted_next - actual_next):.3f}s")

if __name__ == "__main__":
    test_single_prediction()