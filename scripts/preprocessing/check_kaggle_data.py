# scripts/preprocessing/check_kaggle_structure.py
import pandas as pd
from pathlib import Path

def check_kaggle_structure():
    base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
    kaggle_path = base_path / 'data' / 'raw' / 'kaggle'
    
    print("🔍 Checking Kaggle data structure...")
    print(f"📍 Kaggle path: {kaggle_path}")
    
    if not kaggle_path.exists():
        print("❌ Kaggle directory doesn't exist!")
        return
    
    print("📁 Contents of kaggle directory:")
    for item in kaggle_path.iterdir():
        if item.is_dir():
            print(f"   📂 {item.name}/")
            # Show CSV files in subdirectories
            for csv_file in item.rglob('*.csv'):
                print(f"      📄 {csv_file.relative_to(item)}")
        else:
            print(f"   📄 {item.name}")
    
    # Check for required files in any subdirectory
    required_files = ['circuits.csv', 'constructors.csv', 'drivers.csv', 'races.csv', 'results.csv']
    found_files = {}
    
    for file in required_files:
        matches = list(kaggle_path.rglob(file))
        if matches:
            found_files[file] = matches[0]
            print(f"✅ Found {file}: {matches[0]}")
        else:
            print(f"❌ Missing {file}")
    
    return found_files

if __name__ == "__main__":
    check_kaggle_structure()