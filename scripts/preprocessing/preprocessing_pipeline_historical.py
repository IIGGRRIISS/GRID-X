# scripts/preprocessing/preprocessing_pipeline_historical.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HistoricalDataPreprocessor:
    def __init__(self):
        # ABSOLUTE PATH - GUARANTEED TO WORK
        self.base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
        self.raw_data_path = self.base_path / 'data' / 'raw' / 'kaggle'
        self.processed_path = self.base_path / 'data' / 'processed'
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📍 Base path: {self.base_path}")
        print(f"📍 Raw data: {self.raw_data_path}")
        print(f"📍 Processed: {self.processed_path}")
        
    def find_kaggle_files(self):
        """Find Kaggle CSV files in subdirectories"""
        print("🔍 Searching for Kaggle CSV files...")
        
        csv_files = {}
        required_files = ['circuits.csv', 'constructors.csv', 'drivers.csv', 'races.csv', 'results.csv']
        
        # Search recursively in kaggle directory
        for file_path in self.raw_data_path.rglob('*.csv'):
            if file_path.name in required_files:
                csv_files[file_path.name] = file_path
                print(f"✅ Found {file_path.name}: {file_path}")
        
        # Check if we found all required files
        missing_files = [f for f in required_files if f not in csv_files]
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return None
        
        return csv_files
    
    def load_historical_data(self):
        """Load all historical Kaggle datasets"""
        print("📥 Loading historical Kaggle data...")
        
        # Check if kaggle directory exists
        if not self.raw_data_path.exists():
            print(f"❌ Kaggle directory not found: {self.raw_data_path}")
            return None, None, None, None, None, None
        
        # Find CSV files
        csv_files = self.find_kaggle_files()
        if csv_files is None:
            return None, None, None, None, None, None
        
        try:
            # Load main tables
            circuits = pd.read_csv(csv_files['circuits.csv'])
            constructors = pd.read_csv(csv_files['constructors.csv'])
            drivers = pd.read_csv(csv_files['drivers.csv'])
            races = pd.read_csv(csv_files['races.csv'])
            results = pd.read_csv(csv_files['results.csv'])
            
            print("✅ Loaded historical tables:")
            print(f"   - Circuits: {len(circuits):,} records")
            print(f"   - Constructors: {len(constructors):,} records") 
            print(f"   - Drivers: {len(drivers):,} records")
            print(f"   - Races: {len(races):,} records")
            print(f"   - Results: {len(results):,} records")
            
            # Try to load qualifying if available
            qualifying = None
            qualifying_files = list(self.raw_data_path.rglob('qualifying.csv'))
            if qualifying_files:
                qualifying = pd.read_csv(qualifying_files[0])
                print(f"✅ Loaded qualifying: {len(qualifying):,} records")
            else:
                print("ℹ️  Qualifying data not available (optional)")
                
            return circuits, constructors, drivers, races, results, qualifying
            
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return None, None, None, None, None, None
    
    def create_historical_races_dataset(self, circuits, constructors, drivers, races, results, qualifying):
        """Create comprehensive historical races dataset"""
        print("🔄 Building historical races dataset...")
        
        # Start with results and join all related data
        historical_df = results.copy()
        
        # Join race information
        historical_df = historical_df.merge(races, on='raceId', suffixes=('', '_race'))
        
        # Join circuit information
        historical_df = historical_df.merge(circuits, on='circuitId', suffixes=('', '_circuit'))
        
        # Join driver information
        historical_df = historical_df.merge(drivers, on='driverId', suffixes=('', '_driver'))
        
        # Join constructor information
        historical_df = historical_df.merge(constructors, on='constructorId', suffixes=('', '_constructor'))
        
        # Join qualifying if available
        if qualifying is not None:
            # Qualifying might have multiple entries per race/driver (Q1, Q2, Q3)
            # We take the best qualifying position
            qualifying_best = qualifying.groupby(['raceId', 'driverId'])['position'].min().reset_index()
            historical_df = historical_df.merge(
                qualifying_best,
                on=['raceId', 'driverId'], 
                how='left',
                suffixes=('', '_qualifying')
            )
            historical_df.rename(columns={'position_qualifying': 'qualifying_position'}, inplace=True)
        else:
            historical_df['qualifying_position'] = historical_df['grid']  # Use grid as fallback
        
        print(f"✅ Historical dataset: {len(historical_df):,} records")
        print(f"📊 Historical columns: {len(historical_df.columns)}")
        return historical_df
    
    def clean_historical_data(self, df):
        """Clean historical data"""
        print("🧹 Cleaning historical data...")
        
        initial_count = len(df)
        
        # Handle missing values in critical columns
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
        df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)
        
        # Remove records with critical missing data
        df = df[df['year'].notna()]
        df = df[df['name'].notna()]
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()]
        
        # Fill missing qualifying positions with grid positions
        df['qualifying_position'] = df['qualifying_position'].fillna(df['grid'])
        
        print(f"✅ Final record count: {len(df):,} (removed {initial_count - len(df):,})")
        return df
    
    def feature_engineering_historical(self, df):
        """Create features for historical analysis"""
        print("⚙️ Engineering historical features...")
        
        # Sort by driver and date for cumulative features
        df = df.sort_values(['driverId', 'date'])
        
        # Driver experience
        df['driver_debut_year'] = df.groupby('driverId')['year'].transform('min')
        df['driver_experience_years'] = df['year'] - df['driver_debut_year']
        df['driver_experience_races'] = df.groupby('driverId').cumcount() + 1
        
        # Team experience at track
        df['team_track_appearances'] = df.groupby(['constructorId', 'circuitId'])['raceId'].transform('count')
        
        # Performance metrics
        df['finished_race'] = df['positionText'].isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']).astype(int)
        df['podium_finish'] = (df['position'] <= 3).astype(int)
        df['points_finish'] = (df['points'] > 0).astype(int)
        df['win'] = (df['position'] == 1).astype(int)
        
        # Career statistics (cumulative)
        df['career_points'] = df.groupby('driverId')['points'].cumsum()
        df['career_wins'] = df.groupby('driverId')['win'].cumsum()
        df['career_podiums'] = df.groupby('driverId')['podium_finish'].cumsum()
        
        # Season statistics
        df['season_points'] = df.groupby(['driverId', 'year'])['points'].cumsum()
        df['season_wins'] = df.groupby(['driverId', 'year'])['win'].cumsum()
        
        # Grid to position change
        df['grid_position_change'] = df['grid'] - df['position']
        
        print("✅ Historical feature engineering complete")
        return df
    
    def validate_historical_data(self, df):
        """Validate historical data quality"""
        print("🔍 Validating historical data quality...")
        
        validation_checks = {
            'total_records': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'seasons_covered': f"{df['year'].min()} to {df['year'].max()}",
            'unique_drivers': df['driverId'].nunique(),
            'unique_constructors': df['constructorId'].nunique(),
            'unique_circuits': df['circuitId'].nunique(),
            'total_races': df['raceId'].nunique()
        }
        
        for check, result in validation_checks.items():
            print(f"   {check}: {result}")
        
        return validation_checks
    
    def save_historical_dataset(self, df):
        """Save processed historical dataset"""
        output_file = self.processed_path / 'historical_races_processed.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 Saved historical dataset: {output_file}")
        print(f"📊 Final dataset shape: {df.shape}")
        
        # Save sample
        sample_file = self.processed_path / 'historical_races_sample.csv'
        sample_size = min(10000, len(df))
        df.sample(sample_size).to_csv(sample_file, index=False)
        print(f"📝 Sample saved ({sample_size} records): {sample_file}")
    
    def run_pipeline(self):
        """Execute the complete historical data pipeline"""
        print("🚀 STARTING HISTORICAL DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Load data
        circuits, constructors, drivers, races, results, qualifying = self.load_historical_data()
        if circuits is None:
            print("❌ Failed to load historical data. Pipeline stopped.")
            print("💡 Please ensure Kaggle F1 dataset is downloaded and extracted to:")
            print(f"   {self.raw_data_path}")
            print("   The CSV files should be in subdirectories under this path.")
            return
        
        # Process data
        merged_df = self.create_historical_races_dataset(circuits, constructors, drivers, races, results, qualifying)
        cleaned_df = self.clean_historical_data(merged_df)
        featured_df = self.feature_engineering_historical(cleaned_df)
        
        # Validate and save
        self.validate_historical_data(featured_df)
        self.save_historical_dataset(featured_df)
        
        print("=" * 60)
        print("✅ HISTORICAL DATA PIPELINE COMPLETE!")

if __name__ == "__main__":
    processor = HistoricalDataPreprocessor()
    processor.run_pipeline()