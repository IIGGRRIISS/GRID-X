# scripts/preprocessing/verify_fixed_data.py
import pandas as pd
from pathlib import Path

def verify_fixed_data():
    base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
    fixed_file = base_path / 'data' / 'processed' / 'modern_races_processed_FIXED.csv'
    
    df = pd.read_csv(fixed_file)
    print(f"✅ Verified fixed data: {len(df):,} records")
    print(f"📊 Unique events: {df['event_name'].nunique()}")
    print(f"📊 Unique drivers: {df['DriverNumber'].nunique()}")
    print(f"📊 Seasons: {sorted(df['year'].unique())}")
    
    # Check for duplicates
    key_cols = ['DriverNumber', 'LapNumber', 'event_name', 'year', 'round']
    duplicates = df.duplicated(subset=key_cols).sum()
    print(f"🔍 Remaining duplicates: {duplicates}")

if __name__ == "__main__":
    verify_fixed_data()