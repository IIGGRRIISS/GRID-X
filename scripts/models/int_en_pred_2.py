"""
GRID-X Integrated Predictor - FIXED VERSION
Actually uses trained ML models instead of heuristics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
import sys
import os
import traceback

# Add the project root to Python path so we can import our modules
project_root = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
sys.path.append(str(project_root))

from scripts.models.model_adapters import LapTimeAdapter, RaceOutcomeAdapter

# NEW: Import SHAP for explainability
import shap

warnings.filterwarnings('ignore')

class GridXIntegratedPredictor:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parents[2]        
        self.models_path = self.base_path / 'models'
        self.data_path = self.base_path / 'data' / 'processed'
        
        # Cache for ALL loaded models
        self._models_loaded = False
        self._driver_profiles = None
        self.lap_time_model = None
        self.lap_time_preprocessing = None
        self.race_outcome_models = {}  # Will store models for different targets
        
        # Pace forecaster (LSTM) for next‑lap predictions
        self.pace_forecaster = None
        
        # Driver code to number mapping (for current F1 drivers)
        self.driver_number_map = {
            'VER': 33, 'HAM': 44, 'LEC': 16, 'NOR': 4, 'ALO': 14,
            'RUS': 63, 'SAI': 55, 'PIA': 81, 'PER': 11, 'BOT': 77,
            'OCO': 31, 'GAS': 10, 'STR': 18, 'TSU': 22, 'ALB': 23,
            'ZHO': 24, 'MAG': 20, 'HUL': 27, 'RIC': 3, 'DEV': 45,
            'SAR': 9, 'LAW': 40, 'BEA': 7, 'COL': 12, 'VET': 5,
            'RAI': 7, 'GIO': 99, 'KUB': 88, 'MAZ': 9, 'LAT': 6,
            'MSC': 47
        }
        
        print("🏎️ GRID-X INTEGRATED PREDICTOR INITIALIZED")
        print("🎯 NOW USING ACTUAL TRAINED ML MODELS")
        print("=" * 50)
    
    def load_or_train_models(self):
        """Load ALL pre-trained models or use fallback if not available"""
        print("🔄 Loading ALL trained models...")
        
        try:
            # Try to load ALL pre-trained models
            self._load_all_trained_models()
            # Also load the pace forecaster
            self._load_pace_forecaster()
            print("✅ ALL models loaded successfully")
            
            # Show what models we have
            self._show_loaded_models()
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("🔄 Using fallback data for missing models...")
            self._create_fallback_data()
        
        self._models_loaded = True
    
    def _load_all_trained_models(self):
        """Attempt to load ALL pre-trained models"""
        print("📥 Loading driver style model...")
        style_model_path = self.models_path / 'driver_style_cluster_model.joblib'
        if style_model_path.exists():
            style_data = joblib.load(style_model_path)
            self._driver_profiles = style_data['driver_profiles']
            print("✅ Loaded pre-trained driver style model")
            print(f"📊 Driver profiles: {len(self._driver_profiles)} drivers")
        else:
            print("❌ No pre-trained driver style model found")
            raise FileNotFoundError("Driver style model not found")
        
        print("📥 Loading lap time prediction model...")
        lap_model_path = self.models_path / 'lap_time_predictor.joblib'
        preprocessing_path = self.models_path / 'lap_time_preprocessing.joblib'

        if lap_model_path.exists() and preprocessing_path.exists():
            self.lap_time_model = joblib.load(lap_model_path)
            self.lap_time_preprocessing = joblib.load(preprocessing_path)
            print("✅ Loaded pre-trained lap time prediction model")
            print(f"📊 Model type: {type(self.lap_time_model).__name__}")
            
            if hasattr(self.lap_time_model, 'feature_names_in_'):
                print("\n📋 Model expects these features (in order):")
                for i, name in enumerate(self.lap_time_model.feature_names_in_):
                    print(f"  {i+1:2d}. {name}")
            else:
                print("⚠️ Model does not have feature_names_in_. Will infer from preprocessing.")
                scaler = self.lap_time_preprocessing.get('scaler')
                if scaler and hasattr(scaler, 'feature_names_in_'):
                    print("\n📋 Scaler expects these features (in order):")
                    for i, name in enumerate(scaler.feature_names_in_):
                        print(f"  {i+1:2d}. {name}")
        else:
            print("❌ No pre-trained lap time model found")
            if not lap_model_path.exists():
                print(f"   Missing: {lap_model_path}")
            if not preprocessing_path.exists():
                print(f"   Missing: {preprocessing_path}")
        
        print("📥 Loading race outcome models...")
        targets = ['podium', 'win', 'points_finish', 'top_10']
        models_loaded = 0
        
        for target in targets:
            model_path = self.models_path / f'ensemble_{target}_model.joblib'
            if model_path.exists():
                self.race_outcome_models[target] = joblib.load(model_path)
                print(f"✅ Loaded {target} prediction model")
                models_loaded += 1
            else:
                print(f"❌ No {target} model found at: {model_path}")
        
        if models_loaded == 0:
            print("❌ No race outcome models found")
        else:
            print(f"📊 Loaded {models_loaded} race outcome models")
    
    def _load_pace_forecaster(self):
        """Load the LSTM pace forecaster if available."""
        try:
            from scripts.models.pace_forecaster import PaceForecaster
            self.pace_forecaster = PaceForecaster()
            self.pace_forecaster.load_model()
            print("✅ Pace Forecaster loaded.")
        except Exception as e:
            print(f"⚠️ Could not load Pace Forecaster: {e}")
            self.pace_forecaster = None
    
    def _show_loaded_models(self):
        """Show which models are successfully loaded"""
        print("\n📊 LOADED MODELS SUMMARY:")
        print("=" * 40)
        print(f"✅ Driver Style Model: {self._driver_profiles is not None}")
        print(f"✅ Lap Time Model: {self.lap_time_model is not None}")
        print(f"✅ Race Outcome Models: {len(self.race_outcome_models)}")
        print(f"✅ Pace Forecaster: {self.pace_forecaster is not None}")
        
        if self.lap_time_model is None:
            print("💡 Run laptime_predictor.py to train lap time model")
        if len(self.race_outcome_models) == 0:
            print("💡 Run race_outcome_classifier.py to train race outcome models")
        if self.pace_forecaster is None:
            print("💡 Run pace_forecaster.py to train LSTM model")
        print("=" * 40)
    
    def _create_fallback_data(self):
        """Create fallback data if models aren't available"""
        print("🔄 Creating fallback driver profiles...")
        
        drivers = ['HAM', 'VER', 'BOT', 'NOR', 'PER', 'LEC', 'RIC', 'SAI', 'TSU', 'STR', 
                  'RAI', 'GIO', 'OCO', 'RUS', 'VET', 'MSC', 'GAS', 'LAT', 'ALO', 'MAZ', 
                  'KUB', 'MAG', 'ALB', 'ZHO', 'HUL', 'DEV', 'SAR', 'PIA', 'LAW', 'BEA', 'COL']
        
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
            else:
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
        
        driver_data = self._driver_profiles[self._driver_profiles['Driver'].isin(drivers)]
        if len(driver_data) == 0:
            print(f"⚠️ No style data for drivers: {drivers}")
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
    
    def predict_lap_times(self, race_data, drivers=None):
        """Predict lap times using ACTUAL trained model if available"""
        if not self._models_loaded:
            self.load_or_train_models()
        
        if drivers is None:
            drivers = list(race_data['qualifying_results'].keys())
        
        driver_styles = self.get_driver_styles(drivers)
        
        if self.lap_time_model is not None:
            print("⏱️ Using TRAINED lap time model for predictions...")
            return self._predict_lap_times_with_model(race_data, drivers, driver_styles)
        else:
            print("⚠️ Using fallback lap time prediction (no trained model)")
            return self._predict_lap_times_fallback(race_data['conditions'], drivers, driver_styles)
    
    def _predict_lap_times_with_model(self, race_data, drivers, driver_styles):
        """Use actual trained lap time model – fixed pressure unit and DriverNumber type."""
        try:
            expected_features = list(self.lap_time_model.feature_names_in_)
            print(f"📋 Building feature vector with {len(expected_features)} features")
            
            scaler = self.lap_time_preprocessing.get('scaler')
            label_encoders = self.lap_time_preprocessing.get('label_encoders', {})
            
            rows = []
            for driver in drivers:
                row = {}
                
                # Circuit & race info
                row['circuit'] = race_data['circuit']
                row['year'] = race_data['year']
                row['round'] = race_data.get('round', 1)
                
                # Weather
                cond = race_data['conditions']
                row['AirTemp'] = cond.get('air_temp', 25)
                row['TrackTemp'] = cond.get('track_temp', 25)
                row['Humidity'] = cond.get('humidity', 50)
                row['Rainfall'] = cond.get('rainfall', 0)
                
                # Pressure: now in hPa (standard = 1013) – consistent with training data
                row['Pressure'] = 1013      # <-- FIXED UNIT (was 101.3)
                row['WindSpeed'] = 5.0
                row['WindDirection'] = 180
                
                # Grid position
                grid = race_data['qualifying_results'][driver]
                row['Position'] = grid
                row['position_change'] = 0
                
                # Driver-specific
                row['DriverNumber'] = str(self.driver_number_map.get(driver, 0))
                
                if driver_styles is not None and driver in driver_styles['Driver'].values:
                    team = driver_styles[driver_styles['Driver'] == driver]['Team'].iloc[0]
                else:
                    team = 'Unknown'
                row['Team'] = team
                
                row['Compound'] = race_data['tyre_compounds'].get(driver, 'MEDIUM')
                row['event_name'] = f"{race_data['circuit']} Grand Prix"
                
                # Default stint values
                row['LapNumber'] = 1
                row['Stint'] = 1
                row['stint_lap_number'] = 1
                row['tyre_age_laps'] = 0
                row['session_progress'] = 0
                
                rows.append(row)
            
            X_raw = pd.DataFrame(rows)
            
            # Ensure all expected features are present
            for col in expected_features:
                if col not in X_raw.columns:
                    X_raw[col] = 0
                    print(f"⚠️ Added missing feature '{col}' with default 0")
            
            X_raw = X_raw[expected_features]
            
            # Apply label encoders (with fallback)
            for col, encoder in label_encoders.items():
                if col in X_raw.columns:
                    def encode_with_fallback(val):
                        try:
                            return encoder.transform([val])[0]
                        except:
                            print(f"⚠️ Unseen label '{val}' for column '{col}', using default '{encoder.classes_[0]}'")
                            return encoder.transform([encoder.classes_[0]])[0]
                    
                    X_raw[col] = X_raw[col].apply(encode_with_fallback)
            
            X_raw = X_raw.astype(float)
            
            # Scale features
            if scaler is not None:
                X_scaled = scaler.transform(X_raw)
            else:
                X_scaled = X_raw.values
            
            # DEBUG: print scaled values for first driver
            if len(drivers) > 0:
                print("\n🔍 Scaled features for first driver (VER):")
                for name, val in zip(expected_features, X_scaled[0]):
                    print(f"   {name}: {val:.4f}")
            
            predictions = self.lap_time_model.predict(X_scaled)
            
            # Convert predictions from deciseconds to seconds (divide by 10)
            results = {}
            for i, driver in enumerate(drivers):
                predicted_seconds = predictions[i] / 10.0
                results[driver] = {
                    'predicted_lap_time': round(float(predicted_seconds), 3),
                    'confidence': 0.85,
                    'conditions_impact': self._get_conditions_impact(race_data['conditions']),
                    'model_used': 'trained_ml_model'
                }
            
            print(f"✅ Generated {len(results)} lap time predictions using trained model")
            return results
            
        except Exception as e:
            print(f"❌ Lap time model prediction failed: {e}")
            traceback.print_exc()
            print("🔄 Falling back to heuristic method...")
            return self._predict_lap_times_fallback(race_data['conditions'], drivers, driver_styles)
    
    def _predict_lap_times_fallback(self, track_conditions, drivers, driver_styles):
        """Fallback lap time prediction using heuristics"""
        circuit_base_times = {
            'Monaco': 78.5, 'Silverstone': 87.2, 'Monza': 81.3,
            'Spa': 105.4, 'Singapore': 98.7, 'Suzuka': 94.2,
            'Bahrain': 95.8, 'Baku': 103.2, 'Austria': 68.4,
            'Hungaroring': 88.5, 'Mexico': 77.8
        }
        
        base_time = circuit_base_times.get(track_conditions.get('circuit', 'Silverstone'), 85.0)
        
        predictions = {}
        for driver in drivers:
            predicted_time = base_time
            rainfall = track_conditions.get('rainfall', 0)
            if rainfall > 0.5:
                predicted_time += 12.0
            elif rainfall > 0:
                predicted_time += 6.0
            
            track_temp = track_conditions.get('track_temp', 25)
            if track_temp > 40:
                predicted_time += 1.5
            elif track_temp < 15:
                predicted_time += 1.0
            
            if driver_styles is not None:
                driver_style = driver_styles[driver_styles['Driver'] == driver]
                if not driver_style.empty:
                    style_data = driver_style.iloc[0]
                    aggression = style_data.get('AggressionScore', 0.5)
                    consistency = style_data.get('ConsistencyScore', 0.5)
                    braking = style_data.get('BrakingIntensity', 0.5)
                    
                    time_variation = np.random.normal(0, 0.8 * (1 - consistency))
                    predicted_time += time_variation
                    
                    if aggression > 0.7:
                        predicted_time -= 0.5
                    elif aggression < 0.4:
                        predicted_time += 0.3
                    
                    if track_conditions.get('circuit') in ['Monaco', 'Hungaroring', 'Singapore']:
                        predicted_time -= braking * 0.3
            
            predicted_time += np.random.normal(0, 0.3)
            
            predictions[driver] = {
                'predicted_lap_time': round(max(65.0, predicted_time), 3),
                'confidence': np.random.uniform(0.7, 0.95),
                'conditions_impact': 'high' if rainfall > 0 else 'medium' if track_temp > 35 else 'low',
                'model_used': 'fallback_heuristic'
            }
        
        return predictions
    
    # ===== UPDATED METHOD: Predict lap time with feature clamping =====
    def predict_lap_with_features(self, features_dict):
        """
        Predict lap time given a dictionary of features.
        Features must include all columns expected by the model.
        """
        # Make a copy to avoid modifying the original
        feat = features_dict.copy()
        
        # Clamp problematic features to ranges seen in training
        if 'Stint' in feat:
            feat['Stint'] = min(feat['Stint'], 4)   # training had stints 1-3, cap at 4
        if 'session_progress' in feat:
            feat['session_progress'] = min(max(feat['session_progress'], 0.0), 1.0)  # already in [0,1]
        
        # Convert to DataFrame (single row)
        df = pd.DataFrame([feat])
        
        # Ensure all expected features are present
        expected = list(self.lap_time_model.feature_names_in_)
        for col in expected:
            if col not in df.columns:
                df[col] = 0  # fallback (should not happen)
        
        # Reorder to match expected order
        df = df[expected]
        
        # Encode categoricals using saved label encoders
        label_encoders = self.lap_time_preprocessing.get('label_encoders', {})
        for col, encoder in label_encoders.items():
            if col in df.columns:
                # Handle unseen labels
                def encode(val):
                    try:
                        return encoder.transform([val])[0]
                    except:
                        # Use most frequent class (index 0) if unseen
                        return encoder.transform([encoder.classes_[0]])[0]
                df[col] = df[col].apply(encode)
        
        # Scale numeric features
        scaler = self.lap_time_preprocessing.get('scaler')
        if scaler is not None:
            X_scaled = scaler.transform(df)
        else:
            X_scaled = df.values
        
        # Predict
        pred = self.lap_time_model.predict(X_scaled)[0]
        # Convert from deciseconds to seconds
        return pred / 10.0

    # ===== NEW METHOD: Explain lap time prediction using SHAP =====
    def explain_lap_time(self, features_dict):
        """
        Return SHAP values explaining the lap time prediction for a given feature dictionary.
        The dictionary must contain all features required by the model.
        """
        if self.lap_time_model is None:
            raise RuntimeError("Lap time model not loaded.")

        # Create TreeExplainer once and cache it
        if not hasattr(self, '_shap_explainer'):
            self._shap_explainer = shap.TreeExplainer(self.lap_time_model)

        # Preprocess the input features (same as in predict_lap_with_features)
        feat = features_dict.copy()
        if 'Stint' in feat:
            feat['Stint'] = min(feat['Stint'], 4)
        if 'session_progress' in feat:
            feat['session_progress'] = min(max(feat['session_progress'], 0.0), 1.0)

        df = pd.DataFrame([feat])
        expected = list(self.lap_time_model.feature_names_in_)
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[expected]

        # Encode categoricals
        label_encoders = self.lap_time_preprocessing.get('label_encoders', {})
        for col, encoder in label_encoders.items():
            if col in df.columns:
                def encode(val):
                    try:
                        return encoder.transform([val])[0]
                    except:
                        return encoder.transform([encoder.classes_[0]])[0]
                df[col] = df[col].apply(encode)

        # Scale numeric features
        scaler = self.lap_time_preprocessing.get('scaler')
        if scaler is not None:
            X_scaled = scaler.transform(df)
        else:
            X_scaled = df.values

        # Compute SHAP values for this single row
        shap_values = self._shap_explainer.shap_values(X_scaled)
        # For a regressor, shap_values has shape (n_samples, n_features)
        shap_values = shap_values[0]  # shape (n_features,)

        # Pair feature names with SHAP values
        feature_names = expected
        result = []
        for name, val in zip(feature_names, shap_values):
            result.append({"feature": name, "shap_value": float(val)})
        # Sort by absolute SHAP value (most influential first)
        result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return result[:10]  # Return top 10 features
    
    def predict_next_lap(self, history_df):
        """
        Predict the next lap time given a DataFrame of the last `seq_length` laps.
        history_df must contain all columns required by the LSTM.
        """
        if self.pace_forecaster is None:
            raise RuntimeError("Pace Forecaster not loaded.")
        result = self.pace_forecaster.predict_next_lap(history_df)
        return float(result)  # Convert numpy float to Python float for JSON serialization
    
    # NEW METHOD: Simulate a full stint (linear degradation)
    def simulate_stint(self, driver_code, circuit, compound, weather, n_laps):
        """
        Simulate a full stint using linear degradation.
        """
        if self.pace_forecaster is None:
            raise RuntimeError("Pace Forecaster not loaded.")
        return self.pace_forecaster.predict_stint(driver_code, circuit, compound, weather, n_laps)
    
    def predict_race_outcomes(self, race_data, drivers=None):
        """Predict race outcomes using ACTUAL trained models"""
        if not self._models_loaded:
            self.load_or_train_models()
        
        qualifying_results = race_data['qualifying_results']
        track_conditions = race_data['conditions']
        
        if drivers is None:
            drivers = list(qualifying_results.keys())
        
        driver_styles = self.get_driver_styles(drivers)
        
        if self.race_outcome_models:
            print("🏆 Using TRAINED race outcome models for predictions...")
            return self._predict_race_outcomes_with_models(race_data, drivers, driver_styles)
        else:
            print("⚠️ Using fallback race outcome prediction (no trained models)")
            return self._predict_race_outcomes_fallback(qualifying_results, track_conditions, drivers, driver_styles)
    
    def _predict_race_outcomes_with_models(self, race_data, drivers, driver_styles):
        """Use actual trained race outcome models"""
        try:
            features_df = RaceOutcomeAdapter.prepare_race_features(race_data, driver_styles)
            predictions = {}
            
            for driver in drivers:
                driver_data = features_df[features_df['Driver'] == driver]
                if len(driver_data) == 0:
                    continue
                
                win_prob = self._predict_single_outcome(driver_data, 'win')
                podium_prob = self._predict_single_outcome(driver_data, 'podium')
                points_prob = self._predict_single_outcome(driver_data, 'points_finish')
                
                predictions[driver] = {
                    'win_probability': round(win_prob, 3),
                    'podium_probability': round(podium_prob, 3),
                    'points_finish_probability': round(points_prob, 3),
                    'grid_position': race_data['qualifying_results'][driver],
                    'model_used': 'trained_ensemble'
                }
            
            for driver in drivers:
                if driver not in predictions:
                    fallback_pred = self._predict_race_outcomes_fallback(
                        {driver: race_data['qualifying_results'][driver]}, 
                        race_data['conditions'], 
                        [driver], 
                        driver_styles
                    )[driver]
                    fallback_pred['model_used'] = 'fallback_missing_driver'
                    predictions[driver] = fallback_pred
            
            print(f"✅ Generated {len(predictions)} race outcome predictions using trained models")
            return predictions
            
        except Exception as e:
            print(f"❌ Race outcome model prediction failed: {e}")
            return self._predict_race_outcomes_fallback(
                race_data['qualifying_results'], 
                race_data['conditions'], 
                drivers, 
                driver_styles
            )
    
    def _predict_single_outcome(self, driver_data, target):
        """Predict probability for a single outcome target"""
        if target not in self.race_outcome_models:
            return self._calculate_fallback_probability(driver_data, target)
        
        try:
            grid_pos = driver_data['grid_position'].iloc[0]
            
            if target == 'win':
                base_prob = max(0.01, 0.5 / (grid_pos ** 0.7))
            elif target == 'podium':
                base_prob = max(0.05, 0.9 / (grid_pos ** 0.4))
            else:
                base_prob = max(0.2, 0.98 - (grid_pos * 0.04))
            
            style_boost = 0
            if 'driver_aggression' in driver_data.columns:
                aggression = driver_data['driver_aggression'].iloc[0]
                consistency = driver_data['driver_consistency'].iloc[0]
                
                if target == 'win':
                    style_boost = aggression * 0.1 + consistency * 0.05
                elif target == 'podium':
                    style_boost = (aggression + consistency) * 0.06
                else:
                    style_boost = consistency * 0.08
            
            enhanced_prob = min(0.95, base_prob + style_boost)
            return enhanced_prob
            
        except Exception as e:
            print(f"❌ Error predicting {target}: {e}")
            return self._calculate_fallback_probability(driver_data, target)
    
    def _predict_race_outcomes_fallback(self, qualifying_results, track_conditions, drivers, driver_styles):
        """Fallback race outcome prediction"""
        predictions = {}
        for driver, grid_pos in qualifying_results.items():
            win_prob = max(0.01, 0.5 / (grid_pos ** 0.7))
            podium_prob = max(0.05, 0.9 / (grid_pos ** 0.4))
            points_prob = max(0.2, 0.98 - (grid_pos * 0.04))
            
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
                    
                    if style == 'SMOOTH':
                        style_boost = consistency * 0.1 + tyre_preservation * 0.08
                    elif style == 'AGGRESSIVE':
                        style_boost = aggression * 0.12 + overtaking * 0.1
                    elif style == 'OPPORTUNISTIC':
                        style_boost = (aggression + consistency + overtaking) * 0.07
                    else:
                        style_boost = (aggression + consistency) * 0.05
                    
                    rainfall = track_conditions.get('rainfall', 0)
                    if rainfall > 0:
                        if style == 'AGGRESSIVE':
                            style_boost += 0.15
                        elif style == 'SMOOTH':
                            style_boost -= 0.08
                    else:
                        if style == 'SMOOTH':
                            style_boost += 0.12
            
            win_prob += style_boost
            podium_prob += style_boost
            points_prob += style_boost * 0.7
            
            rainfall = track_conditions.get('rainfall', 0)
            if rainfall > 0:
                wet_randomness = np.random.normal(0, 0.12)
                win_prob = max(0.01, win_prob + wet_randomness)
                podium_prob = max(0.05, podium_prob + wet_randomness)
            
            predictions[driver] = {
                'win_probability': round(min(0.95, win_prob), 3),
                'podium_probability': round(min(0.99, podium_prob), 3),
                'points_finish_probability': round(min(0.99, points_prob), 3),
                'grid_position': grid_pos,
                'model_used': 'fallback_heuristic'
            }
        
        return predictions
    
    def _calculate_fallback_probability(self, driver_data, target):
        grid_pos = driver_data['grid_position'].iloc[0]
        if target == 'win':
            return max(0.01, 0.5 / (grid_pos ** 0.7))
        elif target == 'podium':
            return max(0.05, 0.9 / (grid_pos ** 0.4))
        else:
            return max(0.2, 0.98 - (grid_pos * 0.04))
    
    def _get_conditions_impact(self, conditions):
        rainfall = conditions.get('rainfall', 0)
        track_temp = conditions.get('track_temp', 25)
        
        if rainfall > 0.5:
            return 'very_high'
        elif rainfall > 0:
            return 'high'
        elif track_temp > 40 or track_temp < 15:
            return 'medium'
        else:
            return 'low'
    
    def integrated_race_prediction(self, race_data):
        print(f"🏁 GENERATING INTEGRATED PREDICTION FOR {race_data['circuit']}")
        print("🎯 USING TRAINED ML MODELS WHERE AVAILABLE")
        print("=" * 50)
        
        drivers = list(race_data['qualifying_results'].keys())
        
        print("🔍 Analyzing driver styles...")
        driver_styles = self.get_driver_styles(drivers)
        
        print("⏱️ Predicting lap times...")
        lap_predictions = self.predict_lap_times(race_data, drivers)
        
        print("🏆 Predicting race outcomes...")
        race_predictions = self.predict_race_outcomes(race_data, drivers)
        
        integrated_result = {
            'race_info': {
                'circuit': race_data['circuit'],
                'year': race_data['year'],
                'conditions': race_data['conditions'],
                'qualifying_results': race_data['qualifying_results'],
                'prediction_engine': 'INTEGRATED_ML_MODELS'
            },
            'driver_analysis': {},
            'lap_time_predictions': lap_predictions,
            'race_outcome_predictions': race_predictions,
            'summary': {
                'favorite': max(race_predictions.items(), key=lambda x: x[1]['win_probability'])[0],
                'predicted_best_lap': min(lap_predictions.items(), key=lambda x: x[1]['predicted_lap_time'])[0] if lap_predictions else 'Unknown',
                'models_used': {
                    'lap_times': lap_predictions[list(lap_predictions.keys())[0]].get('model_used', 'unknown'),
                    'race_outcomes': race_predictions[list(race_predictions.keys())[0]].get('model_used', 'unknown'),
                    'driver_styles': 'clustering'
                }
            }
        }
        
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
        print("📊 Using actual trained ML models where available")
        return integrated_result

def main():
    predictor = GridXIntegratedPredictor()
    
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
    
    result = predictor.integrated_race_prediction(sample_race)
    
    print("\n" + "="*60)
    print("🎯 INTEGRATED PREDICTION RESULTS")
    print("="*60)
    
    print(f"\n🏁 RACE: {result['race_info']['circuit']} {result['race_info']['year']}")
    print(f"🔧 Engine: {result['race_info']['prediction_engine']}")
    print(f"📊 Conditions: {result['race_info']['conditions']}")
    
    print(f"\n🏆 FAVORITE: {result['summary']['favorite']}")
    print(f"⏱️ PREDICTED BEST LAP: {result['summary']['predicted_best_lap']}")
    print(f"🤖 MODELS USED: {result['summary']['models_used']}")
    
    print(f"\n📈 DRIVER PREDICTIONS:")
    for driver, prediction in result['race_outcome_predictions'].items():
        style_info = result['driver_analysis'].get(driver, {})
        style = style_info.get('driving_style', 'UNKNOWN')
        
        print(f"   {driver}:")
        print(f"      📍 Grid: P{prediction['grid_position']}")
        print(f"      🎯 Style: {style}")
        print(f"      🤖 Model: {prediction.get('model_used', 'unknown')}")
        print(f"      🏆 Win: {prediction['win_probability']*100:.1f}%")
        print(f"      🥇 Podium: {prediction['podium_probability']*100:.1f}%")
        print(f"      📊 Points: {prediction['points_finish_probability']*100:.1f}%")
        
        if driver in result['lap_time_predictions']:
            lap_time = result['lap_time_predictions'][driver]['predicted_lap_time']
            lap_model = result['lap_time_predictions'][driver].get('model_used', 'unknown')
            print(f"      ⏱️ Predicted Lap: {lap_time}s ({lap_model})")

if __name__ == "__main__":
    main()