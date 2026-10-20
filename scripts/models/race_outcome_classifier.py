# scripts/models/race_outcome_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib

class EraSpecificClassifier:
    def __init__(self, era_name):
        self.era_name = era_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.imputer = SimpleImputer(strategy='median')
        self.features_used = []
        
    def train(self, X, y, target_name):
        """Train era-specific model"""
        print(f"   🤖 Training {self.era_name} model for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Use XGBoost as primary model for each era
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=42,
            n_jobs=-1,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"   ✅ {self.era_name} model - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.features_used = X.columns.tolist()
        
        return accuracy, f1
    
    def predict(self, X):
        """Make predictions with era-specific model"""
        if self.model is None:
            raise ValueError(f"{self.era_name} model not trained")
        
        # Handle missing features
        missing_features = set(self.features_used) - set(X.columns)
        extra_features = set(X.columns) - set(self.features_used)
        
        if missing_features:
            print(f"   ⚠️ Adding missing features for {self.era_name}: {list(missing_features)}")
            for feature in missing_features:
                X[feature] = 0
        
        if extra_features:
            print(f"   ⚠️ Removing extra features for {self.era_name}: {list(extra_features)}")
            X = X[self.features_used]
        
        return self.model.predict_proba(X)[:, 1]  # Return probability for class 1

class RaceOutcomeClassifier:
    def __init__(self):
        self.base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
        self.data_path = self.base_path / 'data' / 'processed'
        self.models_path = self.base_path / 'models'
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Separate models for each era
        self.historical_model = EraSpecificClassifier("Historical")
        self.modern_model = EraSpecificClassifier("Modern")
        self.ensemble_model = None
        
        print("🏁 INITIALIZING DUAL-ERA RACE OUTCOME CLASSIFIER")
        print("=" * 50)
    
    def load_data(self):
        """Load historical and modern race data"""
        print("📥 Loading race outcome data...")
        
        try:
            # Load historical race data
            historical_df = pd.read_csv(self.data_path / 'historical_races_processed.csv')
            print(f"✅ Loaded {len(historical_df):,} historical race records")
            
            # Load modern data for additional features
            modern_df = pd.read_csv(self.data_path / 'modern_with_historical_context.csv')
            print(f"✅ Loaded {len(modern_df):,} modern lap records")
            
            return historical_df, modern_df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def prepare_historical_features(self, historical_df):
        """Prepare features specifically for historical data"""
        print("🔄 Preparing historical era features...")
        
        df = historical_df.copy()
        
        # Basic identifiers
        df['era'] = 'historical'
        
        # Target variables
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df = df[df['position'].notna() & (df['position'] > 0)]
        
        df['win'] = (df['position'] == 1).astype(int)
        df['podium'] = (df['position'] <= 3).astype(int)
        df['points_finish'] = (df['position'] <= 10).astype(int)
        df['top_10'] = (df['position'] <= 10).astype(int)
        
        # Historical-specific features
        features = []
        
        # Grid/Qualifying features
        if 'grid' in df.columns:
            df['grid_position'] = df['grid']
            features.append('grid_position')
        
        if 'qualifying_position' in df.columns:
            features.append('qualifying_position')
        
        # Driver experience features
        if 'driver_experience_races' in df.columns:
            features.append('driver_experience_races')
        
        if 'driver_experience_years' in df.columns:
            features.append('driver_experience_years')
        
        # Career statistics
        career_features = ['career_wins', 'career_podiums', 'career_points', 
                          'season_wins', 'season_points']
        for feature in career_features:
            if feature in df.columns:
                features.append(feature)
        
        # Team features
        if 'team_track_appearances' in df.columns:
            features.append('team_track_appearances')
        
        # Temporal features
        if 'year' in df.columns:
            features.append('year')
        
        if 'round' in df.columns:
            features.append('round')
        
        # Circuit features
        if 'circuitId' in df.columns:
            # Encode circuitId
            le = LabelEncoder()
            df['circuit_encoded'] = le.fit_transform(df['circuitId'].astype(str))
            features.append('circuit_encoded')
        
        print(f"✅ Historical features: {len(features)} features")
        print(f"   Features: {features}")
        
        return df, features
    
    def prepare_modern_features(self, modern_df):
        """Prepare features specifically for modern data"""
        print("🔄 Preparing modern era features...")
        
        # Aggregate modern data to race level
        modern_race = modern_df.sort_values(['event_name', 'year', 'round', 'DriverNumber', 'LapNumber'])\
                              .groupby(['event_name', 'year', 'round', 'DriverNumber'])\
                              .last()\
                              .reset_index()
        
        df = modern_race.copy()
        df['era'] = 'modern'
        
        # Target variables
        if 'Position' in df.columns:
            df['position'] = pd.to_numeric(df['Position'], errors='coerce')
            df = df[df['position'].notna() & (df['position'] > 0)]
            
            df['win'] = (df['position'] == 1).astype(int)
            df['podium'] = (df['position'] <= 3).astype(int)
            df['points_finish'] = (df['position'] <= 10).astype(int)
            df['top_10'] = (df['position'] <= 10).astype(int)
        
        # Modern-specific features
        features = []
        
        # Team performance
        if 'Team' in df.columns:
            # Calculate team strength based on historical performance
            team_strength = df.groupby('Team')['position'].mean().to_dict()
            df['team_strength'] = df['Team'].map(team_strength)
            features.append('team_strength')
        
        # Driver performance metrics
        if 'DriverNumber' in df.columns:
            # Calculate driver form from lap data
            driver_form = modern_df.groupby('DriverNumber')['lap_time_seconds'].mean().to_dict()
            df['driver_avg_pace'] = df['DriverNumber'].map(driver_form)
            features.append('driver_avg_pace')
        
        # Temporal features
        if 'year' in df.columns:
            features.append('year')
        
        if 'round' in df.columns:
            features.append('round')
        
        # Circuit features
        if 'circuit' in df.columns:
            le = LabelEncoder()
            df['circuit_encoded'] = le.fit_transform(df['circuit'].astype(str))
            features.append('circuit_encoded')
        
        # Weather features (modern data has detailed weather)
        weather_features = ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']
        for feature in weather_features:
            if feature in df.columns:
                features.append(feature)
        
        # Tyre performance features
        if 'Compound' in df.columns:
            compound_strength = {'SOFT': 3, 'MEDIUM': 2, 'HARD': 1, 'INTERMEDIATE': 2, 'WET': 1}
            df['compound_strength'] = df['Compound'].map(compound_strength).fillna(2)
            features.append('compound_strength')
        
        print(f"✅ Modern features: {len(features)} features")
        print(f"   Features: {features}")
        
        return df, features
    
    def preprocess_era_data(self, df, features, target):
        """Preprocess data for a specific era"""
        # Create working copy
        data = df[features + [target]].copy()
        
        # Remove missing target
        data = data[data[target].notna()]
        
        # Handle missing values in features
        for feature in features:
            if data[feature].isna().any():
                if data[feature].dtype == 'object':
                    data[feature].fillna('UNKNOWN', inplace=True)
                else:
                    data[feature].fillna(data[feature].median(), inplace=True)
        
        # Encode categorical features
        categorical_features = data[features].select_dtypes(include=['object']).columns
        for feature in categorical_features:
            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature].astype(str))
        
        X = data[features]
        y = data[target]
        
        print(f"   📊 {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_era_models(self, historical_df, modern_df, target):
        """Train separate models for historical and modern data"""
        print(f"🎯 Training dual-era models for {target}...")
        
        # Prepare historical features
        historical_processed, historical_features = self.prepare_historical_features(historical_df)
        X_hist, y_hist = self.preprocess_era_data(historical_processed, historical_features, target)
        
        # Prepare modern features  
        modern_processed, modern_features = self.prepare_modern_features(modern_df)
        X_mod, y_mod = self.preprocess_era_data(modern_processed, modern_features, target)
        
        # Train historical model
        if len(X_hist) > 100:
            hist_accuracy, hist_f1 = self.historical_model.train(X_hist, y_hist, target)
        else:
            print("❌ Not enough historical data for training")
            hist_accuracy, hist_f1 = 0, 0
        
        # Train modern model
        if len(X_mod) > 50:
            mod_accuracy, mod_f1 = self.modern_model.train(X_mod, y_mod, target)
        else:
            print("❌ Not enough modern data for training")
            mod_accuracy, mod_f1 = 0, 0
        
        return {
            'historical': {'accuracy': hist_accuracy, 'f1': hist_f1, 'samples': len(X_hist)},
            'modern': {'accuracy': mod_accuracy, 'f1': mod_f1, 'samples': len(X_mod)}
        }
    
    def create_ensemble_features(self, historical_df, modern_df, target):
        """Create features for ensemble model that combines both eras"""
        print("🔄 Creating ensemble features...")
        
        # Process both eras
        historical_processed, historical_features = self.prepare_historical_features(historical_df)
        modern_processed, modern_features = self.prepare_modern_features(modern_df)
        
        # Get predictions from era-specific models
        ensemble_data = []
        
        # Historical predictions
        if len(historical_processed) > 0:
            X_hist, y_hist = self.preprocess_era_data(historical_processed, historical_features, target)
            if self.historical_model.model is not None:
                hist_proba = self.historical_model.predict(X_hist)
                for i, (idx, row) in enumerate(historical_processed.iterrows()):
                    ensemble_data.append({
                        'era': 'historical',
                        'era_model_pred': hist_proba[i] if i < len(hist_proba) else 0.5,
                        'target': y_hist.iloc[i] if i < len(y_hist) else 0,
                        'year': row.get('year', 0),
                        'round': row.get('round', 0)
                    })
        
        # Modern predictions
        if len(modern_processed) > 0:
            X_mod, y_mod = self.preprocess_era_data(modern_processed, modern_features, target)
            if self.modern_model.model is not None:
                mod_proba = self.modern_model.predict(X_mod)
                for i, (idx, row) in enumerate(modern_processed.iterrows()):
                    ensemble_data.append({
                        'era': 'modern',
                        'era_model_pred': mod_proba[i] if i < len(mod_proba) else 0.5,
                        'target': y_mod.iloc[i] if i < len(y_mod) else 0,
                        'year': row.get('year', 0),
                        'round': row.get('round', 0)
                    })
        
        ensemble_df = pd.DataFrame(ensemble_data)
        
        if len(ensemble_df) == 0:
            return None, None
        
        # Ensemble features
        ensemble_features = ['era_model_pred', 'year', 'round']
        X_ensemble = ensemble_df[ensemble_features]
        
        # Encode era
        le = LabelEncoder()
        X_ensemble['era_encoded'] = le.fit_transform(ensemble_df['era'])
        ensemble_features.append('era_encoded')
        
        y_ensemble = ensemble_df['target']
        
        return X_ensemble[ensemble_features], y_ensemble
    
    def train_ensemble_model(self, X_ensemble, y_ensemble, target):
        """Train ensemble model that combines era-specific predictions"""
        print("🤖 Training ensemble model...")
        
        if X_ensemble is None or len(X_ensemble) == 0:
            print("❌ No ensemble data available")
            return 0, 0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_ensemble, y_ensemble, test_size=0.2, random_state=42, stratify=y_ensemble
        )
        
        # Train ensemble model (simple logistic regression)
        self.ensemble_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        print(f"✅ Ensemble model - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {roc_auc:.3f}")
        
        return accuracy, f1
    
    def predict_final(self, historical_df, modern_df, target):
        """Make final predictions using the ensemble approach"""
        print("🔮 Making final predictions...")
        
        # Get era-specific predictions
        historical_processed, historical_features = self.prepare_historical_features(historical_df)
        modern_processed, modern_features = self.prepare_modern_features(modern_df)
        
        predictions = []
        
        # Historical predictions
        if len(historical_processed) > 0 and self.historical_model.model is not None:
            X_hist, _ = self.preprocess_era_data(historical_processed, historical_features, target)
            hist_proba = self.historical_model.predict(X_hist)
            
            for i, (idx, row) in enumerate(historical_processed.iterrows()):
                era_pred = hist_proba[i] if i < len(hist_proba) else 0.5
                
                # Create ensemble feature vector
                ensemble_features = pd.DataFrame([{
                    'era_model_pred': era_pred,
                    'year': row.get('year', 0),
                    'round': row.get('round', 0),
                    'era_encoded': 0  # historical
                }])
                
                # Get final prediction
                if self.ensemble_model is not None:
                    final_pred = self.ensemble_model.predict_proba(ensemble_features)[0, 1]
                else:
                    final_pred = era_pred
                
                predictions.append({
                    'era': 'historical',
                    'driver': row.get('driverRef', 'unknown'),
                    'year': row.get('year', 0),
                    'round': row.get('round', 0),
                    'era_specific_pred': era_pred,
                    'final_pred': final_pred,
                    'actual': row.get(target, 0)
                })
        
        # Modern predictions
        if len(modern_processed) > 0 and self.modern_model.model is not None:
            X_mod, _ = self.preprocess_era_data(modern_processed, modern_features, target)
            mod_proba = self.modern_model.predict(X_mod)
            
            for i, (idx, row) in enumerate(modern_processed.iterrows()):
                era_pred = mod_proba[i] if i < len(mod_proba) else 0.5
                
                # Create ensemble feature vector
                ensemble_features = pd.DataFrame([{
                    'era_model_pred': era_pred,
                    'year': row.get('year', 0),
                    'round': row.get('round', 0),
                    'era_encoded': 1  # modern
                }])
                
                # Get final prediction
                if self.ensemble_model is not None:
                    final_pred = self.ensemble_model.predict_proba(ensemble_features)[0, 1]
                else:
                    final_pred = era_pred
                
                predictions.append({
                    'era': 'modern',
                    'driver': row.get('Driver', 'unknown'),
                    'year': row.get('year', 0),
                    'round': row.get('round', 0),
                    'era_specific_pred': era_pred,
                    'final_pred': final_pred,
                    'actual': row.get(target, 0)
                })
        
        return pd.DataFrame(predictions)
    
    def save_models(self, target_name):
        """Save all models"""
        print("💾 Saving dual-era models...")
        
        # Save historical model
        if self.historical_model.model is not None:
            hist_path = self.models_path / f'historical_{target_name}_model.joblib'
            joblib.dump(self.historical_model, hist_path)
            print(f"   ✅ Historical model saved: {hist_path}")
        
        # Save modern model
        if self.modern_model.model is not None:
            mod_path = self.models_path / f'modern_{target_name}_model.joblib'
            joblib.dump(self.modern_model, mod_path)
            print(f"   ✅ Modern model saved: {mod_path}")
        
        # Save ensemble model
        if self.ensemble_model is not None:
            ensemble_path = self.models_path / f'ensemble_{target_name}_model.joblib'
            joblib.dump(self.ensemble_model, ensemble_path)
            print(f"   ✅ Ensemble model saved: {ensemble_path}")
    
    def evaluate_dual_era_performance(self, predictions_df, target_name):
        """Comprehensive evaluation of dual-era approach"""
        if predictions_df is None or len(predictions_df) == 0:
            print("❌ No predictions to evaluate")
            return
        
        print(f"📊 DUAL-ERA PERFORMANCE EVALUATION - {target_name.upper()}")
        print("=" * 50)
        
        # Overall performance
        overall_accuracy = accuracy_score(
            predictions_df['actual'], 
            (predictions_df['final_pred'] > 0.5).astype(int)
        )
        
        overall_f1 = f1_score(
            predictions_df['actual'], 
            (predictions_df['final_pred'] > 0.5).astype(int),
            average='weighted'
        )
        
        print(f"🎯 OVERALL PERFORMANCE:")
        print(f"   Accuracy: {overall_accuracy:.3f}")
        print(f"   F1 Score: {overall_f1:.3f}")
        
        # Era-specific performance
        for era in ['historical', 'modern']:
            era_data = predictions_df[predictions_df['era'] == era]
            if len(era_data) > 0:
                era_accuracy = accuracy_score(
                    era_data['actual'], 
                    (era_data['final_pred'] > 0.5).astype(int)
                )
                era_f1 = f1_score(
                    era_data['actual'], 
                    (era_data['final_pred'] > 0.5).astype(int),
                    average='weighted'
                )
                
                print(f"📈 {era.upper()} ERA PERFORMANCE:")
                print(f"   Accuracy: {era_accuracy:.3f}")
                print(f"   F1 Score: {era_f1:.3f}")
                print(f"   Samples: {len(era_data):,}")
    
    def run_dual_era_pipeline(self, target='podium'):
        """Execute complete dual-era classification pipeline"""
        print(f"🚀 DUAL-ERA RACE OUTCOME CLASSIFIER - {target.upper()}")
        print("=" * 60)
        
        # Load data
        historical_df, modern_df = self.load_data()
        if historical_df is None:
            return
        
        # Step 1: Train era-specific models
        era_results = self.train_era_models(historical_df, modern_df, target)
        
        # Step 2: Create and train ensemble model
        X_ensemble, y_ensemble = self.create_ensemble_features(historical_df, modern_df, target)
        ensemble_accuracy, ensemble_f1 = self.train_ensemble_model(X_ensemble, y_ensemble, target)
        
        # Step 3: Make final predictions
        predictions_df = self.predict_final(historical_df, modern_df, target)
        
        # Step 4: Evaluate performance
        self.evaluate_dual_era_performance(predictions_df, target)
        
        # Step 5: Save models
        self.save_models(target)
        
        print("=" * 60)
        print(f"🎉 DUAL-ERA CLASSIFIER COMPLETE!")
        print(f"🏆 Final ensemble performance saved")
        print("🚀 Ready for integration with other GRID-X modules!")

if __name__ == "__main__":
    classifier = RaceOutcomeClassifier()
    
    # Train models for different targets
    targets = ['podium', 'win', 'points_finish', 'top_10']
    
    for target in targets:
        print(f"\n{'='*80}")
        classifier.run_dual_era_pipeline(target=target)
        print(f"{'='*80}\n")