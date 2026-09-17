# Save as: C:\Users\Faiz Ahmed\OneDrive\Desktop\GRID-X\verify_modern_fixed.py
import pandas as pd
import os

def verify_modern_fixed():
    modern_path = "C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X/data/raw/fastf1/"
    
    print("🔍 VERIFYING MODERN DATA FIX...")
    print(f"Checking: {modern_path}")
    
    if not os.path.exists(modern_path):
        print("❌ Modern path doesn't exist!")
        return
    
    files = os.listdir(modern_path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files:")
    
    for file in csv_files:
        full_path = os.path.join(modern_path, file)
        try:
            df = pd.read_csv(full_path)
            size_kb = os.path.getsize(full_path) / 1024
            print(f"✅ {file}: {len(df):,} rows, {len(df.columns)} cols, {size_kb:.1f} KB")
            
            # Show sample
            if 'fastf1_lap_times' in file:
                print(f"   Sample drivers: {df['Driver'].unique()[:3] if 'Driver' in df.columns else 'N/A'}")
                
        except Exception as e:
            print(f"❌ {file}: Error - {e}")
    
    if csv_files:
        print(f"\n🎉 MODERN DATA FIXED SUCCESSFULLY!")
    else:
        print(f"\n❌ NO MODERN DATA FOUND")

if __name__ == "__main__":
    verify_modern_fixed()