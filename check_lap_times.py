# Quick fix for consistency scores - add this to diagnose the issue
import pandas as pd
import numpy as np
from pathlib import Path

base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
data_path = base_path / 'data' / 'processed'

# Load the telemetry data to check lap time distributions
modern_df = pd.read_csv(data_path / 'modern_with_historical_context.csv')
print("LapTime statistics:")
print(modern_df['LapTime'].describe())
print("\nSample LapTime values:")
print(modern_df['LapTime'].head())

# Check if lap times are properly converted
def check_lap_time_conversion(series):
    print(f"Data type: {series.dtype}")
    print(f"Sample values: {series.head(3).tolist()}")
    print(f"Min: {series.min()}, Max: {series.max()}, Mean: {series.mean()}")
    
check_lap_time_conversion(modern_df['LapTime'])
