import pandas as pd
from pathlib import Path

print("📊 CHECKING DRIVER STYLE PROFILES CSV")
print("=" * 50)

csv_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X/models/driver_style_profiles.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 10 drivers:")
    print(df[['Driver', 'style_label', 'ConsistencyScore', 'AggressionScore']].head(10))
    print(f"\nUnique style labels: {df['style_label'].unique()}")
else:
    print("❌ CSV file not found")
