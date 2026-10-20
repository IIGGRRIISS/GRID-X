
from pathlib import Path
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ID = "faizprofesh/GRID-X_assets"

MODEL_FILES = [
    "models/circuit_classifier.pth",
    "models/circuit_embeddings.npy",
    "models/circuit_labels.npy",
    "models/class_names.txt",
    "models/crash_feature_columns.pkl",
    "models/crash_risk_classifier.pkl",
    "models/crash_shap_explainer.pkl",
    "models/crash_statistics.json",
    "models/dqn_agent.pth",
    "models/driver_style_cluster_model.joblib",
    "models/ensemble_podium_model.joblib",
    "models/ensemble_points_finish_model.joblib",
    "models/ensemble_top_10_model.joblib",
    "models/ensemble_win_model.joblib",
    "models/lap_time_predictor.joblib",
    "models/lap_time_preprocessing.joblib",
    "models/pace_encoders.joblib",
    "models/pace_forecaster.h5",
    "models/pace_scaler.joblib",
    "models/tire_safety_model.joblib",
]

def main():
    for filename in MODEL_FILES:
        destination = PROJECT_ROOT / filename

        if destination.exists():
            print(f"Already exists: {filename}")
            continue

        print(f"Downloading: {filename}")

        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(PROJECT_ROOT),
        )

        print(f"Downloaded: {filename}")

    print("Model download process completed.")

if __name__ == "__main__":
    main()
