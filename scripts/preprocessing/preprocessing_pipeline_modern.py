# scripts/preprocessing/fix_modern_data.py

import pandas as pd
import numpy as np
from pathlib import Path

def fix_modern_data():
    base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
    processed_path = base_path / 'data' / 'processed'
    
    print("🔧 FIXING MODERN DATA MERGE ISSUE...")
    
    # Load the problematic merged data
    modern_file = processed_path / 'modern_races_processed.csv'
    df = pd.read_csv(modern_file)
    
    print(f"📊 Current record count: {len(df):,} (TOO HIGH!)")
    print(f"📊 Expected: ~96,385 records")
    
    # The issue: We merged on ['year','round','event_name'] which created duplicates
    # Let's find the actual unique lap identifiers
    
    # Check for unique combinations that should identify a single lap
    potential_keys = ['DriverNumber', 'LapNumber', 'event_name', 'year', 'round']
    if all(col in df.columns for col in potential_keys):
        print("🔍 Identifying unique laps...")
        
        # Count duplicates
        duplicate_count = df.duplicated(subset=potential_keys).sum()
        print(f"🔍 Duplicate laps found: {duplicate_count:,}")
        
        # Remove duplicates
        df_fixed = df.drop_duplicates(subset=potential_keys)
        print(f"✅ Fixed record count: {len(df_fixed):,}")
        
        # Save fixed data
        fixed_file = processed_path / 'modern_races_processed_FIXED.csv'
        df_fixed.to_csv(fixed_file, index=False)
        print(f"💾 Saved fixed dataset: {fixed_file}")
        
        # Also save sample
        sample_file = processed_path / 'modern_races_sample_FIXED.csv'
        df_fixed.sample(min(10000, len(df_fixed))).to_csv(sample_file, index=False)
        print(f"📝 Sample saved: {sample_file}")
        
        return df_fixed
    else:
        print("❌ Cannot fix - missing required columns")
        return None

if __name__ == "__main__":
    fix_modern_data()