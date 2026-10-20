"""
Tire Safety & Degradation Risk Predictor
Uses engineered features to predict risk score (0-100).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap

class TireSafetyPredictor:
    def __init__(self, models_path=None):
        self.base_path = Path(__file__).parent.parent.parent
        if models_path is None:
            self.models_path = self.base_path / 'models'
        else:
            self.models_path = Path(models_path)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self._shap_explainer = None

    def engineer_features(self, df):
        """
        Engineer features and create risk score labels from raw lap data.
        Uses 'lap_time_seconds' which is already numeric.
        """
        # Make a copy
        data = df.copy()
        print(f"Initial rows: {len(data)}")
        print(f"Columns present: {list(data.columns)}")

        # ----- Ensure we have the required columns -----
        required_cols = ['lap_time_seconds', 'tyre_age_laps', 'Compound', 'TrackTemp', 'AirTemp',
                         'LapNumber', 'Stint', 'Driver', 'event_name', 'year', 'round',
                         'Humidity', 'Rainfall', 'Position', 'session_progress']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Convert numeric columns to proper numeric (already numeric, but ensure)
        numeric_cols = ['lap_time_seconds', 'tyre_age_laps', 'TrackTemp', 'AirTemp',
                        'LapNumber', 'Stint', 'Humidity', 'Rainfall', 'Position', 'session_progress']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            print(f"After converting {col}, non-null: {data[col].notna().sum()}")

        # Drop rows with missing essential features
        essential = ['lap_time_seconds', 'tyre_age_laps', 'Compound', 'TrackTemp', 'AirTemp']
        for col in essential:
            data = data[data[col].notna()]
            print(f"After filtering {col}, rows: {len(data)}")

        if len(data) == 0:
            raise ValueError("No valid rows after essential filtering.")

        # Compute stint baseline lap time (first lap of stint)
        stint_keys = ['event_name', 'year', 'round', 'Driver', 'Stint']
        existing_keys = [k for k in stint_keys if k in data.columns]
        if existing_keys:
            first_laps = data.sort_values('LapNumber').groupby(existing_keys)['lap_time_seconds'].first().reset_index()
            first_laps.rename(columns={'lap_time_seconds': 'stint_baseline'}, inplace=True)
            data = data.merge(first_laps, on=existing_keys, how='left')
            # Fill missing stint_baseline with overall median per compound
            compound_medians = data.groupby('Compound')['lap_time_seconds'].median().to_dict()
            data['stint_baseline'] = data['stint_baseline'].fillna(data['Compound'].map(compound_medians))
        else:
            # Fallback: use overall median per compound
            compound_medians = data.groupby('Compound')['lap_time_seconds'].median().to_dict()
            data['stint_baseline'] = data['Compound'].map(compound_medians)

        # Degradation from baseline
        data['deg_from_baseline'] = data['lap_time_seconds'] - data['stint_baseline']

        # Degradation rate per lap (rolling within stint)
        data = data.sort_values(['event_name', 'year', 'round', 'Driver', 'Stint', 'LapNumber'])
        data['prev_laptime'] = data.groupby(['event_name', 'year', 'round', 'Driver', 'Stint'])['lap_time_seconds'].shift(1)
        data['deg_rate'] = data['lap_time_seconds'] - data['prev_laptime']

        # Acceleration of degradation (second derivative)
        data['prev_deg'] = data.groupby(['event_name', 'year', 'round', 'Driver', 'Stint'])['deg_rate'].shift(1)
        data['deg_acceleration'] = data['deg_rate'] - data['prev_deg']

        # Compound-specific max safe laps (heuristic)
        compound_max = {'SOFT': 15, 'MEDIUM': 25, 'HARD': 35, 'INTERMEDIATE': 30, 'WET': 30}
        data['compound_max_age'] = data['Compound'].map(compound_max).fillna(20)

        # Age ratio (normalized)
        data['age_ratio'] = data['tyre_age_laps'] / data['compound_max_age']
        data['age_ratio'] = data['age_ratio'].clip(0, 2)

        # Time degradation ratio (baseline +2s is critical)
        data['time_ratio'] = data['deg_from_baseline'] / 2.0
        data['time_ratio'] = data['time_ratio'].clip(0, 2)

        # Engineered risk score (0-100)
        data['risk_score'] = (data['age_ratio'] * 60 + data['time_ratio'] * 40).clip(0, 100)

        # Fill any remaining NaN values (e.g., first lap of stint where deg_rate is NaN)
        fill_cols = ['deg_rate', 'deg_acceleration', 'age_ratio', 'time_ratio', 'risk_score']
        for col in fill_cols:
            if col in data.columns:
                data[col] = data[col].fillna(0)

        # Final check
        if data['risk_score'].isna().all():
            raise ValueError("All risk_score values are NaN. Check input data.")

        print(f"✅ Features engineered. Risk score range: {data['risk_score'].min():.1f}-{data['risk_score'].max():.1f}")
        return data

    def prepare_training_data(self, df):
        """
        From engineered dataframe, select features and target.
        """
        feature_cols = [
            'tyre_age_laps', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall',
            'deg_rate', 'deg_acceleration', 'age_ratio', 'time_ratio',
            'Position', 'session_progress', 'Compound'
        ]
        # Ensure all exist
        available = [c for c in feature_cols if c in df.columns]
        # One-hot encode compound
        X = pd.get_dummies(df[available], columns=['Compound'], drop_first=True)
        y = df['risk_score']
        return X, y

    def train(self, X, y):
        """
        Train the RandomForest regressor.
        """
        self.model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        print(f"✅ Tire safety model trained. R² on training: {self.model.score(X, y):.3f}")
        return self.model

    def save_model(self, path=None):
        """
        Save model, scaler, label encoders, feature names.
        """
        if path is None:
            path = self.models_path / 'tire_safety_model.joblib'
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'compound_categories': list(self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else [])
        }, path)
        print(f"💾 Tire safety model saved to {path}")

    def load_model(self, path=None):
        """
        Load trained model.
        """
        if path is None:
            path = self.models_path / 'tire_safety_model.joblib'
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        print(f"✅ Tire safety model loaded from {path}")

    def predict(self, features_dict):
        """
        Predict risk score (0-100) from a single sample dictionary.
        """
        # Convert to DataFrame (single row)
        df = pd.DataFrame([features_dict])
        # One-hot encode compound similarly to training
        X = pd.get_dummies(df, columns=['Compound'], drop_first=True)
        # Ensure all columns present (add missing with 0)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        pred = self.model.predict(X)[0]
        return float(np.clip(pred, 0, 100))

    def explain(self, features_dict):
        """
        Return SHAP values for a single prediction.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)

        df = pd.DataFrame([features_dict])
        X = pd.get_dummies(df, columns=['Compound'], drop_first=True)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        shap_values = self._shap_explainer.shap_values(X)[0]
        result = []
        for name, val in zip(self.feature_names, shap_values):
            result.append({"feature": name, "shap_value": float(val)})
        result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return result[:5]  # top 5