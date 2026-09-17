# scripts/preprocessing/integration_pipeline_FIXED.py

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataIntegrationPipeline:
    def __init__(self):
        # ABSOLUTE PATH - GUARANTEED TO WORK
        self.base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
        self.processed_path = self.base_path / 'data' / 'processed'
        
        print(f"📍 Base path: {self.base_path}")
        print(f"📍 Processed data: {self.processed_path}")
        
    def check_prerequisites(self):
        """Check if required files exist"""
        print("🔍 Checking prerequisites...")
        
        modern_file = self.processed_path / 'modern_races_processed_FIXED.csv'  # Use FIXED file
        historical_file = self.processed_path / 'historical_races_processed.csv'
        
        prerequisites_met = True
        
        if not modern_file.exists():
            print(f"❌ Modern data not found: {modern_file}")
            print("   Make sure fix_modern_data.py was run successfully")
            prerequisites_met = False
        else:
            print(f"✅ Modern data found: {modern_file}")
            
        if not historical_file.exists():
            print(f"❌ Historical data not found: {historical_file}")
            print("   Run preprocessing_pipeline_historical.py first")
            prerequisites_met = False
        else:
            print(f"✅ Historical data found: {historical_file}")
            
        return prerequisites_met
    
    def load_processed_datasets(self):
        """Load both processed datasets"""
        print("📥 Loading processed datasets...")
        
        try:
            # Load the FIXED modern data (90,901 records)
            modern_df = pd.read_csv(self.processed_path / 'modern_races_processed_FIXED.csv')
            historical_df = pd.read_csv(self.processed_path / 'historical_races_processed.csv')
            
            print(f"✅ Modern data: {len(modern_df):,} records")
            print(f"✅ Historical data: {len(historical_df):,} records")
            
            # Show modern data columns for debugging
            print("📋 Modern data columns:", modern_df.columns.tolist()[:10])  # First 10 columns
            
            return modern_df, historical_df
            
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return None, None
    
    def create_cross_reference_mappings(self, modern_df, historical_df):
        """Create mappings between modern and historical entities"""
        print("🔗 Creating cross-reference mappings...")
        
        # For now, create placeholder mappings
        driver_mapping = {}
        team_mapping = {}
        track_mapping = {}
        
        print("💡 Note: Manual mapping files required for full integration")
        print("   Create: driver_mapping.csv, team_mapping.csv, track_mapping.csv")
        
        return driver_mapping, team_mapping, track_mapping
    
    def create_integrated_features(self, modern_df, historical_df, driver_mapping, team_mapping, track_mapping):
        """Create features that integrate historical context"""
        print("🔄 Creating integrated features...")
        
        modern_enriched = modern_df.copy()
        
        # Add basic historical context even without mappings
        modern_enriched = self._add_basic_historical_context(modern_enriched, historical_df)
        
        return modern_enriched
    
    def _add_basic_historical_context(self, modern_df, historical_df):
        """Add basic historical context that doesn't require mappings"""
        print("   Adding basic historical context...")
        
        # Add decade/era information
        modern_df['era'] = 'Modern (2021+)'
        
        # Add some basic historical stats that don't require exact mappings
        # Calculate general historical trends by year
        if 'year' in modern_df.columns:
            modern_df['historical_seasons_count'] = modern_df['year'] - 1950
        
        print("   ✅ Added basic historical context")
        return modern_df
    
    def create_era_analysis_dataset(self, modern_df, historical_df):
        """Create dataset for cross-era analysis with correct column names"""
        print("📊 Creating era analysis dataset...")
        
        # Use ACTUAL column names from modern data
        modern_columns_available = modern_df.columns.tolist()
        print(f"   Available modern columns: {modern_columns_available}")
        
        # Select compatible columns that actually exist
        modern_comparable_cols = ['year']
        
        # Map modern column names to what we expect
        if 'DriverNumber' in modern_df.columns:
            modern_comparable_cols.append('DriverNumber')
        elif 'driver_number' in modern_df.columns:
            modern_comparable_cols.append('driver_number')
            
        if 'Position' in modern_df.columns:
            modern_comparable_cols.append('Position')
        elif 'position' in modern_df.columns:
            modern_comparable_cols.append('position')
        
        print(f"   Using modern columns: {modern_comparable_cols}")
        
        modern_comparable = modern_df[modern_comparable_cols].copy()
        modern_comparable['era'] = 'Modern (2021+)'
        modern_comparable['data_source'] = 'FastF1'
        
        # For historical, use available columns
        historical_comparable = historical_df[[
            'year', 'driverId', 'position', 'points'
        ]].copy()
        historical_comparable['era'] = 'Historical'
        historical_comparable['data_source'] = 'Kaggle'
        
        # Rename columns to match (use whatever columns we have)
        if len(modern_comparable_cols) >= 3:  # We have year, driver, position
            # Modern has: ['year', 'DriverNumber', 'Position']
            modern_comparable.columns = ['year', 'driver_id', 'position', 'era', 'data_source']
            historical_comparable.columns = ['year', 'driver_id', 'position', 'points', 'era', 'data_source']
        else:
            # Fallback: just use year
            modern_comparable.columns = ['year', 'era', 'data_source']
            historical_comparable = historical_df[['year', 'era', 'data_source']].copy()
            historical_comparable['era'] = 'Historical'
            historical_comparable['data_source'] = 'Kaggle'
        
        # Combine datasets
        era_analysis_df = pd.concat([modern_comparable, historical_comparable], ignore_index=True)
        
        print(f"✅ Era analysis dataset: {len(era_analysis_df):,} records")
        return era_analysis_df
    
    def save_integrated_datasets(self, modern_enriched, era_analysis_df):
        """Save all integrated datasets"""
        print("💾 Saving integrated datasets...")
        
        # Save modern data with historical context
        modern_enriched.to_csv(self.processed_path / 'modern_with_historical_context.csv', index=False)
        print(f"✅ Saved modern with historical context: {len(modern_enriched):,} records")
        
        # Save era analysis dataset
        if era_analysis_df is not None:
            era_analysis_df.to_csv(self.processed_path / 'cross_era_analysis.csv', index=False)
            print(f"✅ Saved cross-era analysis: {len(era_analysis_df):,} records")
        
        # Save integration report
        self._save_integration_report(modern_enriched, era_analysis_df)
    
    def _save_integration_report(self, modern_enriched, era_analysis_df):
        """Save integration summary report"""
        report = f"""
        DATA INTEGRATION REPORT
        ======================
        Generated on: {pd.Timestamp.now()}
        
        Dataset Sizes:
        - Modern Records with Historical Context: {len(modern_enriched):,}
        - Cross-Era Analysis Records: {len(era_analysis_df) if era_analysis_df is not None else 'N/A':,}
        
        Integration Status: BASIC INTEGRATION COMPLETE
        
        Data Quality:
        - Modern data: {len(modern_enriched):,} clean laps (2021-2024)
        - Historical data: Available (1950-2024)
        
        Next Steps for Model Development:
        1. Use 'modern_with_historical_context.csv' for all 8 GRID-X models
        2. Use 'historical_races_processed.csv' for trend analysis
        3. Models can be built immediately with current data
        
        Manual Integration (Optional):
        - Create driver_mapping.csv for full driver career analysis
        - Create team_mapping.csv for constructor legacy analysis  
        - Create track_mapping.csv for circuit evolution analysis
        
        MODEL DEVELOPMENT CAN NOW BEGIN!
        """
        
        report_file = self.processed_path / 'integration_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📋 Integration report saved: {report_file}")
    
    def run_pipeline(self):
        """Execute the complete integration pipeline"""
        print("🚀 STARTING DATA INTEGRATION PIPELINE (FIXED)")
        print("=" * 60)
        
        # Check prerequisites first
        if not self.check_prerequisites():
            print("❌ Prerequisites not met. Pipeline stopped.")
            return
        
        # Load processed data
        modern_df, historical_df = self.load_processed_datasets()
        if modern_df is None or historical_df is None:
            return
        
        # Create mappings (will be empty initially)
        driver_mapping, team_mapping, track_mapping = self.create_cross_reference_mappings(modern_df, historical_df)
        
        # Create integrated features
        modern_enriched = self.create_integrated_features(modern_df, historical_df, driver_mapping, team_mapping, track_mapping)
        
        # Create era analysis dataset
        era_analysis_df = self.create_era_analysis_dataset(modern_df, historical_df)
        
        # Save everything
        self.save_integrated_datasets(modern_enriched, era_analysis_df)
        
        print("=" * 60)
        print("🎉 DATA INTEGRATION PIPELINE COMPLETE!")
        print("🚀 ALL DATA IS NOW READY FOR MODEL DEVELOPMENT!")
        print("📁 Check integration_report.txt for next steps")

if __name__ == "__main__":
    integrator = DataIntegrationPipeline()
    integrator.run_pipeline()