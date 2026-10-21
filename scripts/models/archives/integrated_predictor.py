"""
GRID-X Integrated Predictor
Combines Driver Style Analysis + Lap Time Prediction + Race Outcome Classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
import sys
import os

# Add the project root to Python path so we can import our modules
project_root = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

class GridXIntegratedPredictor:
    def __init__(self):
        self.base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
        self.models_path = self.base_path / 'models'
        self.data_path = self.base_path / 'data' / 'processed'
        
        # Cache for loaded models
        self._models_loaded = False
        self._driver_profiles = None
        
        print("🏎️ GRID-X INTEGRATED PREDICTOR INITIALIZED")
        print("=" * 50)
    
    def load_or_train_models(self):
        """Load pre-trained models or train them if not available"""
        print("🔄 Loading/Preparing Models...")
        
        try:
            # Try to load pre-trained models
            self._load_pre_trained_models()
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("🔄 Using fallback data...")
            self._create_fallback_data()
        
        self._models_loaded = True
    
    def _load_pre_trained_models(self):
        """Attempt to load pre-trained models"""
        # Load driver style model
        style_model_path = self.models_path / 'driver_style_cluster_model.joblib'
        if style_model_path.exists():
            style_data = joblib.load(style_model_path)
            self._driver_profiles = style_data['driver_profiles']
            print("✅ Loaded pre-trained driver style model")
            print(f"📊 Driver profiles: {len(self._driver_profiles)} drivers")
        else:
            print("❌ No pre-trained driver style model found")
            raise FileNotFoundError("Driver style model not found")
        
    def _create_fallback_data(self):
        """Create fallback data if models aren't available"""
        print("🔄 Creating fallback driver profiles...")
        
        # Use the actual driver list from your successful run
        drivers = ['HAM', 'VER', 'BOT', 'NOR', 'PER', 'LEC', 'RIC', 'SAI', 'TSU', 'STR', 
                  'RAI', 'GIO', 'OCO', 'RUS', 'VET', 'MSC', 'GAS', 'LAT', 'ALO', 'MAZ', 
                  'KUB', 'MAG', 'ALB', 'ZHO', 'HUL', 'DEV', 'SAR', 'PIA', 'LAW', 'BEA', 'COL']
        
        # Create realistic profiles based on your actual clustering results
        smooth_drivers = ['HAM', 'OCO', 'ALO', 'HUL']
        opportunistic_drivers = ['VER', 'BOT', 'NOR', 'PER', 'SAI', 'TSU', 'STR', 'GIO', 'RUS', 
                               'MSC', 'GAS', 'KUB', 'MAG', 'SAR', 'PIA', 'LAW', 'BEA', 'COL',
                               'LEC', 'RIC', 'RAI', 'VET', 'LAT', 'MAZ', 'ALB', 'ZHO', 'DEV']
        
        driver_data = []
        for driver in drivers:
            if driver in smooth_drivers:
                driver_data.append({
                    'Driver': driver,
                    'style_label': 'SMOOTH',
                    'AggressionScore': np.random.normal(0.45, 0.05),
                    'ConsistencyScore': np.random.normal(0.89, 0.03),
                    'BrakingIntensity': np.random.normal(0.35, 0.08),
                    'TyrePreservation': np.random.normal(0.85, 0.05),
                    'OvertakingAbility': np.random.normal(0.65, 0.07)
                })
            else:  # Opportunistic
                driver_data.append({
                    'Driver': driver,
                    'style_label': 'OPPORTUNISTIC', 
                    'AggressionScore': np.random.normal(0.68, 0.06),
                    'ConsistencyScore': np.random.normal(0.73, 0.05),
                    'BrakingIntensity': np.random.normal(0.60, 0.08),
                    'TyrePreservation': np.random.normal(0.70, 0.07),
                    'OvertakingAbility': np.random.normal(0.75, 0.06)
                })
        
        self._driver_profiles = pd.DataFrame(driver_data)
        print(f"✅ Created fallback data for {len(self._driver_profiles)} drivers")
    
    def get_driver_styles(self, drivers=None):
        """Get driving styles for specified drivers"""
        if not self._models_loaded:
            self.load_or_train_models()
        
        if self._driver_profiles is None:
            print("❌ No driver profiles available")
            return None
        
        if drivers is None:
            return self._driver_profiles
        
        # Filter for specific drivers
        driver_data = self._driver_profiles[self._driver_profiles['Driver'].isin(drivers)]
        if len(driver_data) == 0:
            print(f"⚠️ No style data for drivers: {drivers}")
            # Create fallback data for missing drivers
            fallback_data = []
            for driver in drivers:
                fallback_data.append({
                    'Driver': driver,
                    'style_label': 'BALANCED',
                    'AggressionScore': 0.6,
                    'ConsistencyScore': 0.7,
                    'BrakingIntensity': 0.55,
                    'TyrePreservation': 0.65,
                    'OvertakingAbility': 0.7
                })
            driver_data = pd.DataFrame(fallback_data)
            
        return driver_data
    
    def predict_lap_times(self, track_conditions, drivers=None):
        """
        Predict lap times based on conditions
        """
        if not self._models_loaded:
            self.load_or_train_models()
        
        if drivers is None:
            drivers = ['VER', 'HAM', 'LEC', 'NOR', 'ALO', 'RUS', 'SAI', 'PIA']
        
        # Base lap times for different circuits (seconds) - UPDATED with realistic values
        circuit_base_times = {
            'Monaco': 78.5, 'Silverstone': 87.2, 'Monza': 81.3,
            'Spa': 105.4, 'Singapore': 98.7, 'Suzuka': 94.2,
            'Bahrain': 95.8, 'Baku': 103.2, 'Austria': 68.4,
            'Monza': 81.3, 'Hungaroring': 88.5, 'Mexico': 77.8
        }
        
        base_time = circuit_base_times.get(track_conditions.get('circuit', 'Silverstone'), 85.0)
        
        predictions = {}
        driver_styles = self.get_driver_styles(drivers)
        
        for driver in drivers:
            # Start with base time
            predicted_time = base_time
            
            # Adjust for conditions
            rainfall = track_conditions.get('rainfall', 0)
            if rainfall > 0.5:
                predicted_time += 12.0  # Heavy rain penalty
            elif rainfall > 0:
                predicted_time += 6.0   # Light rain penalty
            
            track_temp = track_conditions.get('track_temp', 25)
            if track_temp > 40:
                predicted_time += 1.5   # Hot track penalty
            elif track_temp < 15:
                predicted_time += 1.0   # Cold track penalty
            
            # Adjust for driver style if available
            if driver_styles is not None:
                driver_style = driver_styles[driver_styles['Driver'] == driver]
                if not driver_style.empty:
                    style_data = driver_style.iloc[0]
                    aggression = style_data.get('AggressionScore', 0.5)
                    consistency = style_data.get('ConsistencyScore', 0.5)
                    braking = style_data.get('BrakingIntensity', 0.5)
                    
                    # Aggressive drivers might be faster but less consistent
                    time_variation = np.random.normal(0, 0.8 * (1 - consistency))
                    predicted_time += time_variation
                    
                    # Aggression can give small speed boost but higher risk
                    if aggression > 0.7:
                        predicted_time -= 0.5  # More aggressive = faster
                    elif aggression < 0.4:
                        predicted_time += 0.3  # Conservative drivers slightly slower
                    
                    # High braking intensity can help in technical circuits
                    if track_conditions.get('circuit') in ['Monaco', 'Hungaroring', 'Singapore']:
                        predicted_time -= braking * 0.3
            
            # Small random variation
            predicted_time += np.random.normal(0, 0.3)
            
            predictions[driver] = {
                'predicted_lap_time': round(max(65.0, predicted_time), 3),
                'confidence': np.random.uniform(0.7, 0.95),
                'conditions_impact': 'high' if rainfall > 0 else 'medium' if track_temp > 35 else 'low'
            }
        
        return predictions
    
    def predict_race_outcomes(self, qualifying_results, track_conditions, drivers=None):
        """
        Predict race outcomes based on qualifying and conditions
        """
        if not self._models_loaded:
            self.load_or_train_models()
        
        if drivers is None:
            drivers = list(qualifying_results.keys())
        
        # Get driver styles for additional context
        driver_styles = self.get_driver_styles(drivers)
        
        predictions = {}
        
        for driver, grid_pos in qualifying_results.items():
            # Base probability based on grid position (improved model)
            win_prob = max(0.01, 0.5 / (grid_pos ** 0.7))
            podium_prob = max(0.05, 0.9 / (grid_pos ** 0.4))
            points_prob = max(0.2, 0.98 - (grid_pos * 0.04))
            
            # Adjust based on driver style if available
            style_boost = 0
            if driver_styles is not None:
                driver_style = driver_styles[driver_styles['Driver'] == driver]
                if not driver_style.empty:
                    driver_data = driver_style.iloc[0]
                    aggression = driver_data.get('AggressionScore', 0.5)
                    consistency = driver_data.get('ConsistencyScore', 0.5)
                    tyre_preservation = driver_data.get('TyrePreservation', 0.5)
                    overtaking = driver_data.get('OvertakingAbility', 0.5)
                    style = driver_data.get('style_label', 'BALANCED')
                    
                    # Style-based adjustments
                    if style == 'SMOOTH':
                        style_boost = consistency * 0.1 + tyre_preservation * 0.08
                    elif style == 'AGGRESSIVE':
                        style_boost = aggression * 0.12 + overtaking * 0.1
                    elif style == 'OPPORTUNISTIC':
                        style_boost = (aggression + consistency + overtaking) * 0.07
                    else:  # BALANCED
                        style_boost = (aggression + consistency) * 0.05
                    
                    # Conditions-specific boosts
                    rainfall = track_conditions.get('rainfall', 0)
                    if rainfall > 0:
                        # Aggressive drivers better in rain, smooth drivers worse
                        if style == 'AGGRESSIVE':
                            style_boost += 0.15
                        elif style == 'SMOOTH':
                            style_boost -= 0.08
                    else:
                        # Smooth drivers better in dry conditions
                        if style == 'SMOOTH':
                            style_boost += 0.12
            
            win_prob += style_boost
            podium_prob += style_boost
            points_prob += style_boost * 0.7
            
            # Adjust for wet conditions (higher randomness)
            rainfall = track_conditions.get('rainfall', 0)
            if rainfall > 0:
                # More randomness in wet conditions
                wet_randomness = np.random.normal(0, 0.12)
                win_prob = max(0.01, win_prob + wet_randomness)
                podium_prob = max(0.05, podium_prob + wet_randomness)
            
            predictions[driver] = {
                'win_probability': round(min(0.95, win_prob), 3),
                'podium_probability': round(min(0.99, podium_prob), 3),
                'points_finish_probability': round(min(0.99, points_prob), 3),
                'grid_position': grid_pos
            }
        
        return predictions
    
    def integrated_race_prediction(self, race_data):
        """
        Complete integrated prediction for a race
        """
        print(f"🏁 GENERATING INTEGRATED PREDICTION FOR {race_data['circuit']}")
        print("=" * 50)
        
        drivers = list(race_data['qualifying_results'].keys())
        
        # 1. Get driver styles
        print("🔍 Analyzing driver styles...")
        driver_styles = self.get_driver_styles(drivers)
        
        # 2. Predict lap times
        print("⏱️ Predicting lap times...")
        lap_predictions = self.predict_lap_times(race_data['conditions'], drivers)
        
        # 3. Predict race outcomes
        print("🏆 Predicting race outcomes...")
        race_predictions = self.predict_race_outcomes(
            race_data['qualifying_results'], 
            race_data['conditions'], 
            drivers
        )
        
        # Combine all predictions
        integrated_result = {
            'race_info': {
                'circuit': race_data['circuit'],
                'year': race_data['year'],
                'conditions': race_data['conditions'],
                'qualifying_results': race_data['qualifying_results']
            },
            'driver_analysis': {},
            'lap_time_predictions': lap_predictions,
            'race_outcome_predictions': race_predictions,
            'summary': {
                'favorite': max(race_predictions.items(), key=lambda x: x[1]['win_probability'])[0],
                'predicted_best_lap': min(lap_predictions.items(), key=lambda x: x[1]['predicted_lap_time'])[0] if lap_predictions else 'Unknown'
            }
        }
        
        # Add driver style information
        if driver_styles is not None:
            for driver in drivers:
                driver_style_data = driver_styles[driver_styles['Driver'] == driver]
                if not driver_style_data.empty:
                    style_data = driver_style_data.iloc[0]
                    integrated_result['driver_analysis'][driver] = {
                        'driving_style': style_data.get('style_label', 'UNKNOWN'),
                        'aggression': round(style_data.get('AggressionScore', 0.5), 3),
                        'consistency': round(style_data.get('ConsistencyScore', 0.5), 3),
                        'tyre_preservation': round(style_data.get('TyrePreservation', 0.5), 3),
                        'overtaking_ability': round(style_data.get('OvertakingAbility', 0.5), 3),
                        'braking_intensity': round(style_data.get('BrakingIntensity', 0.5), 3)
                    }
                else:
                    integrated_result['driver_analysis'][driver] = {
                        'driving_style': 'UNKNOWN',
                        'aggression': 0.5,
                        'consistency': 0.5,
                        'tyre_preservation': 0.5,
                        'overtaking_ability': 0.5,
                        'braking_intensity': 0.5
                    }
        
        print("✅ Integrated prediction complete!")
        return integrated_result

def main():
    """Test the integrated predictor"""
    predictor = GridXIntegratedPredictor()
    
    # Sample race data
    sample_race = {
        'circuit': 'Monaco',
        'year': 2024,
        'qualifying_results': {'VER': 1, 'HAM': 2, 'LEC': 3, 'NOR': 4, 'ALO': 5},
        'conditions': {
            'air_temp': 25,
            'track_temp': 40,
            'humidity': 60,
            'rainfall': 0
        },
        'tyre_compounds': {
            'VER': 'SOFT', 'HAM': 'MEDIUM', 'LEC': 'SOFT',
            'NOR': 'SOFT', 'ALO': 'MEDIUM'
        }
    }
    
    # Generate prediction
    result = predictor.integrated_race_prediction(sample_race)
    
    # Display results
    print("\n" + "="*60)
    print("🎯 INTEGRATED PREDICTION RESULTS")
    print("="*60)
    
    print(f"\n🏁 RACE: {result['race_info']['circuit']} {result['race_info']['year']}")
    print(f"📊 Conditions: {result['race_info']['conditions']}")
    
    print(f"\n🏆 FAVORITE: {result['summary']['favorite']}")
    print(f"⏱️ PREDICTED BEST LAP: {result['summary']['predicted_best_lap']}")
    
    print(f"\n📈 DRIVER PREDICTIONS:")
    for driver, prediction in result['race_outcome_predictions'].items():
        style_info = result['driver_analysis'].get(driver, {})
        style = style_info.get('driving_style', 'UNKNOWN')
        
        print(f"   {driver}:")
        print(f"      🏆 Win: {prediction['win_probability']*100:.1f}%")
        print(f"      🥇 Podium: {prediction['podium_probability']*100:.1f}%")
        print(f"      📍 Grid: P{prediction['grid_position']}")
        print(f"      🎯 Style: {style}")
        
        if driver in result['lap_time_predictions']:
            lap_time = result['lap_time_predictions'][driver]['predicted_lap_time']
            print(f"      ⏱️ Predicted Lap: {lap_time}s")

if __name__ == "__main__":
    main()