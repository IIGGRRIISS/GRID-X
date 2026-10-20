"""
GRID-X Pace Forecaster (LSTM)
Predicts lap‑by‑lap times over a stint using historical lap data.
Includes driver style features from the clustering model.
Handles missing columns gracefully.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML / DL libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class PaceForecaster:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent  # GRID-X root
        self.data_path = self.base_path / 'data' / 'processed'
        self.models_path = self.base_path / 'models'
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.seq_length = 10  # number of past laps to use for prediction
        
        # Load lap time predictor for generating initial laps (loaded on demand)
        self.lap_time_predictor = None
        
        print("🏁 Initializing Pace Forecaster (LSTM)")
        print("=" * 50)
    
    def load_data(self):
        """Load modern lap data with stint context."""
        print("📥 Loading lap data...")
        df = pd.read_csv(self.data_path / 'modern_with_historical_context.csv')
        print(f"✅ Loaded {len(df):,} laps")
        return df
    
    def load_driver_styles(self, df):
        """
        Load driver style profiles from saved clustering model and merge into dataframe.
        Also adds 'Team' from profiles if available, otherwise sets default.
        """
        style_path = self.models_path / 'driver_style_cluster_model.joblib'
        if style_path.exists():
            style_data = joblib.load(style_path)
            driver_profiles = style_data['driver_profiles']
            print(f"✅ Loaded driver profiles for {len(driver_profiles)} drivers")
            
            # Check if 'Driver' column exists in the lap data
            if 'Driver' in df.columns:
                # Merge on driver code
                df = df.merge(driver_profiles, on='Driver', how='left')
                # Fill missing style values (in case some drivers not in profiles)
                style_cols = ['AggressionScore', 'ConsistencyScore', 'BrakingIntensity',
                              'TyrePreservation', 'OvertakingAbility']
                for col in style_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(0.5)  # default balanced value
                # Ensure 'Team' column exists (from profiles, or create default)
                if 'Team' not in df.columns:
                    df['Team'] = 'UNKNOWN'
                else:
                    df['Team'] = df['Team'].fillna('UNKNOWN')
                print("✅ Merged driver style features.")
            else:
                print("⚠️ 'Driver' column not found in lap data. Cannot merge styles.")
                # Create placeholder columns with default values
                for col in ['AggressionScore','ConsistencyScore','BrakingIntensity',
                            'TyrePreservation','OvertakingAbility']:
                    df[col] = 0.5
                df['Team'] = 'UNKNOWN'
        else:
            print("⚠️ Driver style model not found. Using default values for style features.")
            for col in ['AggressionScore','ConsistencyScore','BrakingIntensity',
                        'TyrePreservation','OvertakingAbility']:
                df[col] = 0.5
            df['Team'] = 'UNKNOWN'
        return df
    
    def prepare_sequences(self, df):
        """
        Create sequences of laps for LSTM training.
        Each sample: sequence of past `seq_length` laps → next lap time.
        """
        print("🔄 Creating sequences...")
        
        # First merge driver styles
        df = self.load_driver_styles(df)
        
        # Ensure data is sorted by race, driver, stint, lap
        sort_cols = ['event_name', 'year', 'round', 'DriverNumber', 'Stint', 'LapNumber']
        # Check if all sort columns exist; if not, adjust
        existing_sort = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(existing_sort)
        
        # Features to use (excluding target and identifiers)
        feature_cols = [
            'stint_lap_number', 'tyre_age_laps', 'LapNumber',  # progression
            'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall',    # weather
            'AggressionScore', 'ConsistencyScore', 'BrakingIntensity',  # driver style
            'TyrePreservation', 'OvertakingAbility',
            'year', 'round'
        ]
        # Categoricals that need encoding:
        cat_cols = ['DriverNumber', 'Compound', 'Team', 'circuit']
        
        # Ensure all categorical columns exist; if not, create with default
        for col in cat_cols:
            if col not in df.columns:
                print(f"⚠️ Column '{col}' not found. Creating with default 'UNKNOWN'.")
                df[col] = 'UNKNOWN'
        
        # Encode categoricals
        for col in cat_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            # Fill any NaN with 'UNKNOWN' and convert to string
            df[col] = df[col].fillna('UNKNOWN').astype(str)
            df[col + '_enc'] = self.label_encoders[col].fit_transform(df[col])
            feature_cols.append(col + '_enc')
        
        # Target
        target = 'lap_time_seconds'
        
        # Remove rows with missing target
        df = df[df[target].notna()].copy()
        
        # Scale numerical features
        X_all = df[feature_cols].fillna(0).values
        self.scaler.fit(X_all)
        X_scaled = self.scaler.transform(X_all)
        
        # Group by (race, driver, stint) to build sequences
        sequences_X = []
        sequences_y = []
        
        group_keys = ['event_name', 'year', 'round', 'DriverNumber', 'Stint']
        # Use only existing group keys
        existing_group = [k for k in group_keys if k in df.columns]
        for _, group in df.groupby(existing_group):
            if len(group) < self.seq_length + 1:
                continue  # not enough laps for a full sequence
            # Get scaled features for this stint
            indices = group.index
            stint_X = X_scaled[df.index.get_indexer(indices)]
            stint_y = group[target].values
            # Slide a window
            for i in range(len(stint_X) - self.seq_length):
                seq_x = stint_X[i : i + self.seq_length]
                seq_y = stint_y[i + self.seq_length]  # next lap time
                sequences_X.append(seq_x)
                sequences_y.append(seq_y)
        
        X_seq = np.array(sequences_X)
        y_seq = np.array(sequences_y)
        print(f"✅ Created {len(X_seq)} sequences of length {self.seq_length}")
        print(f"   X shape: {X_seq.shape}, y shape: {y_seq.shape}")
        return X_seq, y_seq, feature_cols
    
    def build_model(self, input_shape):
        """Define LSTM architecture."""
        model = Sequential([
            Masking(mask_value=0., input_shape=input_shape),  # handle variable lengths (optional)
            LSTM(64, return_sequences=True, dropout=0.2),
            LSTM(32, dropout=0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1)  # linear activation for regression
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        model.summary()
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the LSTM."""
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(str(self.models_path / 'pace_forecaster.h5'), save_best_only=True)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def save_preprocessing(self):
        """Save scaler and label encoders."""
        joblib.dump(self.scaler, self.models_path / 'pace_scaler.joblib')
        joblib.dump(self.label_encoders, self.models_path / 'pace_encoders.joblib')
        print("💾 Preprocessing objects saved.")
    
    def load_model(self):
        """Load trained model and preprocessing. Handles loss serialization issue."""
        model_path = self.models_path / 'pace_forecaster.h5'
        try:
            self.model = load_model(model_path)
        except TypeError as e:
            print("⚠️ Loading model without compiling due to serialization issue. Recompiling...")
            self.model = load_model(model_path, compile=False)
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        self.scaler = joblib.load(self.models_path / 'pace_scaler.joblib')
        self.label_encoders = joblib.load(self.models_path / 'pace_encoders.joblib')
        print("✅ Model and preprocessing loaded.")

    def predict_next_lap(self, sequence_df):
        """
        Given a DataFrame containing the last `seq_length` laps with all required columns,
        predict the next lap time.
        """
        # Prepare features as during training
        feature_cols = [
            'stint_lap_number', 'tyre_age_laps', 'LapNumber',
            'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall',
            'AggressionScore', 'ConsistencyScore', 'BrakingIntensity',
            'TyrePreservation', 'OvertakingAbility',
            'year', 'round'
        ]
        cat_cols = ['DriverNumber', 'Compound', 'Team', 'circuit']
        
        # Apply label encoders to categorical columns
        X_cats = []
        for col in cat_cols:
            if col in sequence_df.columns:
                le = self.label_encoders.get(col)
                if le is not None:
                    # Handle unseen labels (use 0)
                    encoded = sequence_df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
                    X_cats.append(encoded.values.reshape(-1, 1))
                else:
                    # If encoder missing, use 0
                    X_cats.append(np.zeros((len(sequence_df), 1)))
            else:
                X_cats.append(np.zeros((len(sequence_df), 1)))
        
        # Select numeric features
        X_num = sequence_df[feature_cols].fillna(0).values
        
        # Combine numeric and encoded categorical
        X_raw = np.hstack([X_num] + X_cats)
        
        # Scale
        X_scaled = self.scaler.transform(X_raw)
        
        # Reshape for LSTM: (1, seq_length, n_features)
        X_input = X_scaled.reshape(1, self.seq_length, -1)
        
        # Predict
        pred = self.model.predict(X_input, verbose=0)[0, 0]
        return float(pred)  # Convert to Python float for JSON serialization

    def predict_stint(self, driver_code, circuit, compound, weather, n_laps):
        """
        Simulate a full stint using linear degradation.
        Uses lap time predictor for first lap, then increases linearly by 0.1s per lap.
        """
        # Load driver profile (for consistency, but not used in linear)
        profile = self._get_driver_profile(driver_code)
        if profile is None:
            profile = {
                'AggressionScore': 0.5,
                'ConsistencyScore': 0.5,
                'BrakingIntensity': 0.5,
                'TyrePreservation': 0.5,
                'OvertakingAbility': 0.5,
                'Team': 'UNKNOWN'
            }

        # Load lap time predictor if needed
        if self.lap_time_predictor is None:
            from scripts.models.int_en_pred_2 import GridXIntegratedPredictor
            self.lap_time_predictor = GridXIntegratedPredictor()
            self.lap_time_predictor.load_or_train_models()

        # Get first lap time from lap time predictor
        driver_num_str = str(self.lap_time_predictor.driver_number_map.get(driver_code, 0))
        feat = {
            'DriverNumber': driver_num_str,
            'LapNumber': 1,
            'Stint': 1,
            'Compound': compound,
            'Team': profile.get('Team', 'UNKNOWN'),
            'event_name': f"{circuit} Grand Prix",
            'circuit': circuit,
            'year': 2024,
            'round': 1,
            'AirTemp': weather['air_temp'],
            'Humidity': weather['humidity'],
            'Pressure': 1013.0,
            'Rainfall': weather['rainfall'],
            'TrackTemp': weather['track_temp'],
            'WindSpeed': 5.0,
            'WindDirection': 180,
            'stint_lap_number': 1,
            'tyre_age_laps': 1,
            'session_progress': 1 / n_laps,
            'Position': 1,
            'position_change': 0
        }
        first_lap = self.lap_time_predictor.predict_lap_with_features(feat)

        # Linear degradation (0.1s per lap – you can adjust this rate)
        degradation_rate = 0.1
        predicted_laps = [first_lap + (lap_num - 1) * degradation_rate for lap_num in range(1, n_laps + 1)]

        return predicted_laps

    def _get_driver_profile(self, driver_code):
        """Helper to load a single driver profile from saved model."""
        style_path = self.models_path / 'driver_style_cluster_model.joblib'
        if style_path.exists():
            style_data = joblib.load(style_path)
            profiles = style_data['driver_profiles']
            match = profiles[profiles['Driver'] == driver_code]
            if not match.empty:
                return match.iloc[0].to_dict()
        return None

    def run_pipeline(self):
        """Complete training pipeline. Do NOT run this again unless you want to retrain."""
        print("🚀 Starting Pace Forecaster training pipeline")
        df = self.load_data()
        X_seq, y_seq, feature_cols = self.prepare_sequences(df)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        self.train(X_train, y_train, X_val, y_val)
        self.save_preprocessing()
        print("🎉 Training complete. Model saved.")

if __name__ == "__main__":
    forecaster = PaceForecaster()
    forecaster.run_pipeline()