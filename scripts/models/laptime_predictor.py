# scripts/models/lap_time_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib

class LapTimePredictor:
    def __init__(self):
        self.base_path = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
        self.data_path = self.base_path / 'data' / 'processed'
        self.models_path = self.base_path / 'models'
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
        print("🏎️ INITIALIZING LAP-TIME PREDICTOR")
        print("=" * 50)
    
    def load_data(self):
        """Load and prepare the modern dataset for lap time prediction"""
        print("📥 Loading lap time data...")
        
        try:
            # Load the modern dataset with historical context
            df = pd.read_csv(self.data_path / 'modern_with_historical_context.csv')
            print(f"✅ Loaded {len(df):,} lap records")
            
            # Display dataset info
            print(f"📊 Dataset shape: {df.shape}")
            print(f"🔢 Available columns: {len(df.columns)}")
            print("📋 First 20 columns:", df.columns.tolist()[:20])
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def select_features(self, df):
        """Select relevant features for lap time prediction"""
        print("🎯 Selecting features for lap time prediction...")
        
        # Target variable
        target = 'lap_time_seconds'
        
        # Feature categories
        base_features = [
            'DriverNumber', 'LapNumber', 'Stint', 'Compound', 'Team',
            'event_name', 'circuit', 'year', 'round'
        ]
        
        weather_features = [
            'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 
            'WindSpeed', 'WindDirection'
        ]
        
        stint_features = [
            'stint_lap_number', 'tyre_age_laps', 'session_progress'
        ]
        
        position_features = [
            'Position', 'position_change'
        ]
        
        # Combine all features
        all_features = base_features + weather_features + stint_features + position_features
        
        # Check which features actually exist in our data
        available_features = [f for f in all_features if f in df.columns]
        missing_features = [f for f in all_features if f not in df.columns]
        
        print(f"✅ Available features: {len(available_features)}")
        print(f"❌ Missing features: {missing_features}")
        
        return target, available_features
    
    def preprocess_data(self, df, features, target):
        """Preprocess data for model training"""
        print("🧹 Preprocessing data...")
        
        # Create a copy for preprocessing
        data = df[features + [target]].copy()
        
        # Remove rows with missing target
        initial_count = len(data)
        data = data[data[target].notna()]
        print(f"✅ Removed {initial_count - len(data)} rows with missing target")
        
        # Handle missing values in features
        for feature in features:
            if data[feature].isna().any():
                if data[feature].dtype == 'object':
                    data[feature].fillna('UNKNOWN', inplace=True)
                else:
                    data[feature].fillna(data[feature].median(), inplace=True)
        
        # Encode categorical variables
        categorical_features = data[features].select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                data[feature] = self.label_encoders[feature].fit_transform(data[feature].astype(str))
        
        # Separate features and target
        X = data[features]
        y = data[target]
        
        print(f"📊 Final feature set: {X.shape[1]} features")
        print(f"🎯 Target variable: {target}")
        print(f"📈 Data shape: {X.shape}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("🤖 Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"📚 Training set: {X_train.shape[0]:,} samples")
        print(f"🧪 Test set: {X_test.shape[0]:,} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to try
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        # Train and evaluate models
        results = {}
        best_score = float('inf')
        best_model = None
        best_model_name = None
        
        for name, model in models.items():
            print(f"🔧 Training {name}...")
            
            if name in ['Linear Regression', 'Ridge Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'model': model
            }
            
            print(f"   ✅ {name}: RMSE = {rmse:.3f}s, R² = {r2:.3f}")
            
            # Track best model
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_model_name = name
        
        print(f"🏆 Best model: {best_model_name} (RMSE: {best_score:.3f}s)")
        
        # Store feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.model = best_model
        return results, X_test, y_test
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        print(f"📊 Evaluating {model_name}...")
        
        if model_name in ['Linear Regression', 'Ridge Regression']:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate percentage error (average lap time ~90s)
        avg_lap_time = y_test.mean()
        percentage_error = (rmse / avg_lap_time) * 100
        
        print(f"   📈 RMSE: {rmse:.3f} seconds")
        print(f"   📈 MAE: {mae:.3f} seconds") 
        print(f"   📈 R² Score: {r2:.3f}")
        print(f"   📈 Error: {percentage_error:.1f}% of average lap time")
        
        return y_pred, rmse, r2
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance"""
        if self.feature_importance is not None:
            plt.figure(figsize=(10, 8))
            top_features = self.feature_importance.head(top_n)
            
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top {top_n} Most Important Features for Lap Time Prediction')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.models_path / 'lap_time_feature_importance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Feature importance plot saved: {plot_path}")
            
            # Display top features
            print("\n🎯 Top 10 Most Important Features:")
            for i, row in self.feature_importance.head(10).iterrows():
                print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.3f}")
    
    def plot_predictions_vs_actual(self, y_test, y_pred, model_name):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_test, y_pred, alpha=0.5, s=1)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        
        plt.xlabel('Actual Lap Time (seconds)')
        plt.ylabel('Predicted Lap Time (seconds)')
        plt.title(f'Lap Time Predictions vs Actual - {model_name}')
        
        # Add metrics to plot
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}s\nR²: {r2:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plot_path = self.models_path / 'lap_time_predictions_vs_actual.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Predictions vs actual plot saved: {plot_path}")
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        if self.model is not None:
            # Save model
            model_path = self.models_path / 'lap_time_predictor.joblib'
            joblib.dump(self.model, model_path)
            
            # Save preprocessing objects
            preprocessing_path = self.models_path / 'lap_time_preprocessing.joblib'
            joblib.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_importance': self.feature_importance
            }, preprocessing_path)
            
            print(f"💾 Model saved: {model_path}")
            print(f"💾 Preprocessing objects saved: {preprocessing_path}")
        else:
            print("❌ No model to save")
    
    def run_pipeline(self):
        """Execute complete lap time prediction pipeline"""
        print("🚀 STARTING LAP-TIME PREDICTOR PIPELINE")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Select features
        target, features = self.select_features(df)
        
        # Preprocess data
        X, y = self.preprocess_data(df, features, target)
        
        # Train models
        results, X_test, y_test = self.train_models(X, y)
       
       # ✅ FIXED: Evaluate ACTUAL best model (lowest RMSE)
        best_model_name = min(results, key=lambda x: results[x]['RMSE'])
        best_model = results[best_model_name]['model']
        print(f"🎯 CORRECTED: Actual best model is {best_model_name}")
        y_pred, rmse, r2 = self.evaluate_model(best_model, X_test, y_test, best_model_name)

        # Visualizations
        self.plot_feature_importance()
        self.plot_predictions_vs_actual(y_test, y_pred, best_model_name)
        
        # Save model
        self.save_model()
        
        print("=" * 60)
        print("🎉 LAP-TIME PREDICTOR PIPELINE COMPLETE!")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"📊 Final RMSE: {rmse:.3f} seconds")
        print(f"📊 Final R²: {r2:.3f}")
        print("🚀 Ready for integration with other GRID-X modules!")

if __name__ == "__main__":
    predictor = LapTimePredictor()
    predictor.run_pipeline()