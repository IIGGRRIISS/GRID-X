# Save as: src/data_collection/definitive_verification.py
import pandas as pd
import os

def definitive_verification():
    print("🎯 DEFINITIVE GRID-X DATA VERIFICATION")
    print("=" * 60)
    
    # Use the exact path from your project
    base_path = "C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X/"
    
    # Historical data (we know this works)
    historical_path = base_path + "data/raw/kaggle/historical/"
    
    # Modern data - let's check multiple possible locations
    modern_paths = [
        base_path + "data/raw/fastf1/",
        base_path + "data/raw/fastf1/",  # Double check
        base_path + "data/raw/fastf1/",  # Triple check
    ]
    
    print("📚 HISTORICAL DATA CONFIRMED:")
    hist_laps = pd.read_csv(historical_path + "lap_times.csv")
    hist_races = pd.read_csv(historical_path + "races.csv")
    print(f"   Lap times: {len(hist_laps):,} records")
    print(f"   Races: {len(hist_races):,} races")
    print(f"   Date range: {hist_races['year'].min()} - {hist_races['year'].max()}")
    
    print(f"\n🔍 CHECKING MODERN DATA LOCATIONS:")
    modern_data_found = False
    
    for i, modern_path in enumerate(modern_paths):
        print(f"   Checking location {i+1}: {modern_path}")
        if os.path.exists(modern_path):
            files = os.listdir(modern_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"      Found {len(csv_files)} CSV files: {csv_files}")
            
            if csv_files:
                modern_data_found = True
                for file in csv_files:
                    try:
                        df = pd.read_csv(modern_path + file)
                        print(f"      ✅ {file}: {len(df):,} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"      ❌ {file}: Error - {e}")
        else:
            print(f"      ❌ Path does not exist")
    
    if not modern_data_found:
        print(f"\n❌ MODERN DATA NOT FOUND IN EXPECTED LOCATIONS")
        print(f"   Let's search the entire project...")
        search_entire_project(base_path)
    
    # Final summary
    print(f"\n📈 FINAL SUMMARY:")
    print(f"   Historical lap records: {len(hist_laps):,}")
    if modern_data_found:
        # We'll calculate modern records if found
        pass
    print(f"   Total historical races: {len(hist_races):,}")
    print(f"   Project status: {'READY FOR ML' if len(hist_laps) > 500000 else 'NEEDS DATA'}")

def search_entire_project(base_path):
    """Search entire project for modern data files"""
    print(f"\n🔍 SEARCHING ENTIRE PROJECT FOR MODERN DATA...")
    
    modern_file_patterns = [
        "fastf1_lap_times",
        "fastf1_sessions", 
        "fastf1_weather"
    ]
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(pattern in file for pattern in modern_file_patterns) and file.endswith('.csv'):
                full_path = os.path.join(root, file)
                print(f"   ✅ FOUND: {file}")
                print(f"      Location: {full_path}")
                try:
                    df = pd.read_csv(full_path)
                    print(f"      Rows: {len(df):,}, Columns: {len(df.columns)}")
                except Exception as e:
                    print(f"      Error reading: {e}")

if __name__ == "__main__":
    definitive_verification()