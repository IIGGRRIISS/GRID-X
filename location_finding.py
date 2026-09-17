from pathlib import Path
import joblib

base = Path("C:/Users/Faiz Ahmed/OneDrive/Desktop/GRID-X")
models_path = base / 'models'

print(f"Models path: {models_path}")
print(f"Exists? {models_path.exists()}\n")

# List of expected model files
model_files = [
    'driver_style_cluster_model.joblib',
    'lap_time_predictor.joblib',
    'lap_time_preprocessing.joblib',
    'ensemble_win_model.joblib',
    'ensemble_podium_model.joblib',
    'ensemble_points_finish_model.joblib',
]

for fname in model_files:
    full_path = models_path / fname
    print(f"\nChecking: {full_path}")
    print(f"  File exists? {full_path.exists()}")
    if full_path.exists():
        try:
            data = joblib.load(full_path)
            print(f"  Loaded successfully. Type: {type(data)}")
        except Exception as e:
            print(f"  ERROR loading: {e}")
    else:
        print("  File not found.")