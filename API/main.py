"""
GRID-X Prediction API
Uses the integrated predictor int_en_pred_2.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import io
import json
import numpy as np
import pandas as pd

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
from torchvision import transforms
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).parent.parent  # GRID-X folder
sys.path.append(str(project_root))

# Import your integrated predictor and strategy optimiser
from scripts.models.int_en_pred_2 import GridXIntegratedPredictor
from scripts.models.strategy_optimiser import StrategyOptimiser

# ---------- CNN model imports (dynamic, to avoid package issues) ----------
cnn_model_path = project_root / 'scripts' / 'models' / 'cnn' / 'model.py'
if cnn_model_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cnn_model", cnn_model_path)
    cnn_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnn_model)
    get_model = cnn_model.get_model
else:
    get_model = None
    print("⚠️ CNN model module not found – /analyze-circuit disabled.")

# ---------- Tire Safety Model Import ----------
from scripts.models.tire_safety import TireSafetyPredictor

# ---------- Crash Predictor Imports ----------
import joblib
import xgboost as xgb
import shap
from scripts.models.crash_predictor.crash_risk_models import PreRaceCrashModel

# ========== Pydantic Models ==========
class Conditions(BaseModel):
    air_temp: float = Field(..., description="Air temperature in °C")
    track_temp: float = Field(..., description="Track temperature in °C")
    humidity: float = Field(..., description="Humidity in %")
    rainfall: float = Field(..., description="Rainfall intensity (0 = dry, 0.5 = light, 1 = heavy)")

class RaceInput(BaseModel):
    circuit: str
    year: int
    qualifying_results: Dict[str, int] = Field(..., description="Driver code -> grid position")
    conditions: Conditions
    tyre_compounds: Dict[str, str] = Field(..., description="Driver code -> compound (SOFT, MEDIUM, HARD, etc.)")
    round: Optional[int] = 1

class StintRequest(BaseModel):
    driver_code: str
    circuit: str
    compound: str
    weather: Conditions
    n_laps: int = Field(20, ge=1, le=50, description="Number of laps to simulate (default 20)")

class StrategyRequest(BaseModel):
    driver: str
    circuit: str
    weather: Conditions
    total_laps: int = Field(..., ge=10, le=100, description="Total race laps")
    start_compound: str = Field("SOFT", description="Starting tyre compound")

class LapFeatures(BaseModel):
    DriverNumber: str
    LapNumber: int
    Stint: int
    Compound: str
    Team: str
    event_name: str
    circuit: str
    year: int
    round: int
    AirTemp: float
    Humidity: float
    Pressure: float = 1013.0
    Rainfall: float
    TrackTemp: float
    WindSpeed: float = 5.0
    WindDirection: float = 180.0
    stint_lap_number: int
    tyre_age_laps: int
    session_progress: float
    Position: int
    position_change: int = 0
    AggressionScore: float
    ConsistencyScore: float
    BrakingIntensity: float
    TyrePreservation: float
    OvertakingAbility: float

    @validator('session_progress')
    def session_progress_range(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('session_progress must be between 0 and 1')
        return v

class NextLapRequest(BaseModel):
    laps: List[LapFeatures] = Field(..., min_items=10, max_items=10, description="Exactly 10 laps of history")

class LapExplanationRequest(BaseModel):
    driver: str
    circuit: str
    weather: Conditions
    compound: str = Field(..., description="Tyre compound (SOFT, MEDIUM, HARD)")
    lap_number: int = Field(1, ge=1, description="Lap number")
    stint: int = Field(1, ge=1, description="Stint number")
    tyre_age: int = Field(1, ge=1, description="Laps on current tyres")
    session_progress: float = Field(0.05, ge=0, le=1, description="Race progress (0-1)")
    total_laps: int = Field(20, ge=1, description="Total race laps")

class TireSafetyRequest(BaseModel):
    driver: str
    circuit: str
    compound: str = Field(..., description="SOFT/MEDIUM/HARD/INTERMEDIATE/WET")
    tyre_age: int = Field(..., ge=1, description="Laps on current tyres")
    current_lap_time: float = Field(..., gt=0, description="Lap time in seconds")
    track_temp: float = Field(..., description="Track temperature in °C")
    air_temp: float = Field(..., description="Air temperature in °C")
    humidity: float = Field(..., description="Humidity in %")
    rainfall: float = Field(0, ge=0, le=1, description="Rainfall intensity")
    position: int = Field(..., ge=1, description="Current race position")
    stint_lap_number: Optional[int] = Field(None, description="Lap within current stint (same as tyre_age if not provided)")
    session_progress: float = Field(..., ge=0, le=1, description="Race progress (0-1)")
    total_laps: int = Field(..., ge=1, description="Total race laps")

# Crash Risk Request Model
class CrashRiskRequest(BaseModel):
    circuit: str
    weather_wet: bool = False
    track_temp: float = Field(25, description="Forecast track temperature (°C)")
    grid_positions: List[str] = Field(..., description="List of driver codes in grid order (pole first)")
    championship_standings: Optional[Dict[str, float]] = Field(None, description="Driver -> points")

# ========== FastAPI App ==========
app = FastAPI(
    title="GRID-X Prediction API",
    description="Formula 1 race prediction using trained ML models",
    version="1.6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
predictor = None
circuit_model = None
class_names = []
circuit_metadata = None
circuit_embeddings = None
circuit_labels = None
similarity_transform = None
tire_safety_predictor = None

# Crash predictor globals
crash_model = None
crash_feature_columns = None
crash_explainer = None
crash_stats = None

@app.on_event("startup")
async def load_predictor():
    global predictor, circuit_model, class_names, circuit_metadata
    global circuit_embeddings, circuit_labels, similarity_transform
    global tire_safety_predictor
    global crash_model, crash_feature_columns, crash_explainer, crash_stats

    print("🔄 Loading GRID-X models...")
    predictor = GridXIntegratedPredictor()
    predictor.load_or_train_models()
    print("✅ Main models loaded.")

    model_path = project_root / 'models' / 'circuit_classifier.pth'
    class_names_path = project_root / 'models' / 'class_names.txt'
    if get_model is not None and model_path.exists() and class_names_path.exists():
        with open(class_names_path) as f:
            class_names = [line.strip() for line in f]
        circuit_model = get_model(num_classes=len(class_names), pretrained=False)
        circuit_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        circuit_model.eval()
        print(f"✅ CNN circuit classifier loaded ({len(class_names)} classes).")
        emb_path = project_root / 'models' / 'circuit_embeddings.npy'
        lbl_path = project_root / 'models' / 'circuit_labels.npy'
        if emb_path.exists() and lbl_path.exists():
            circuit_embeddings = np.load(emb_path)
            circuit_labels = np.load(lbl_path)
            print(f"✅ Circuit embeddings loaded: {circuit_embeddings.shape}")
        else:
            print("⚠️ Embeddings not found – similarity search disabled.")
    else:
        circuit_model = None
        class_names = []
        print("⚠️ CNN circuit classifier not found – /analyze-circuit disabled.")

    csv_path = project_root / 'frontend' / 'pages' / 'circuit_metadata.csv'
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            col_names = first_line.split()
            circuit_metadata = pd.read_csv(csv_path, skiprows=1, header=None, names=col_names, encoding='utf-8')
        circuit_metadata.columns = circuit_metadata.columns.str.strip()
        if 'circuit_name' not in circuit_metadata.columns:
            possible = ['circuit name', 'CircuitName', 'circuit']
            for p in possible:
                if p in circuit_metadata.columns:
                    circuit_metadata.rename(columns={p: 'circuit_name'}, inplace=True)
                    break
        circuit_metadata = circuit_metadata.set_index('circuit_name')
        print("✅ Circuit metadata loaded.")
    else:
        circuit_metadata = None
        print("⚠️ Circuit metadata not found.")

    tire_model_path = project_root / 'models' / 'tire_safety_model.joblib'
    if tire_model_path.exists():
        tire_safety_predictor = TireSafetyPredictor()
        tire_safety_predictor.load_model(tire_model_path)
        print("✅ Tire safety model loaded.")
    else:
        tire_safety_predictor = None
        print("⚠️ Tire safety model not found – run train_tire_safety.py")

    crash_model_path = project_root / 'models' / 'crash_risk_classifier.pkl'
    crash_cols_path = project_root / 'models' / 'crash_feature_columns.pkl'
    crash_explainer_path = project_root / 'models' / 'crash_shap_explainer.pkl'
    crash_stats_path = project_root / 'models' / 'crash_statistics.json'
    if crash_model_path.exists() and crash_cols_path.exists():
        crash_model = joblib.load(crash_model_path)
        crash_feature_columns = joblib.load(crash_cols_path)
        if crash_explainer_path.exists():
            crash_explainer = joblib.load(crash_explainer_path)
        if crash_stats_path.exists():
            with open(crash_stats_path) as f:
                crash_stats = json.load(f)
        print("✅ Crash risk predictor loaded.")
    else:
        crash_model = None
        print("⚠️ Crash risk predictor not found – run train_crash_predictor.py")

    similarity_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

@app.get("/")
async def root():
    return {"message": "GRID-X Prediction API", "status": "running", "docs": "/docs"}

@app.post("/predict")
async def predict_race(race: RaceInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        race_dict = race.dict()
        result = predictor.integrated_race_prediction(race_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stint-simulate")
async def stint_simulate(request: StintRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        laps = predictor.simulate_stint(
            driver_code=request.driver_code,
            circuit=request.circuit,
            compound=request.compound,
            weather=request.weather.dict(),
            n_laps=request.n_laps
        )
        return {
            "driver": request.driver_code,
            "circuit": request.circuit,
            "compound": request.compound,
            "weather": request.weather.dict(),
            "n_laps": request.n_laps,
            "laps": laps
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/next-lap")
async def predict_next_lap(request: NextLapRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        history_df = pd.DataFrame([lap.dict() for lap in request.laps])
        next_lap = predictor.predict_next_lap(history_df)
        return {"predicted_next_lap": round(next_lap, 3)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategy-optimize")
async def strategy_optimize(request: StrategyRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        optimiser = StrategyOptimiser()
        optimiser.load()
        result = optimiser.optimize(
            driver=request.driver,
            circuit=request.circuit,
            weather=request.weather.dict(),
            total_laps=request.total_laps,
            start_compound=request.start_compound
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-circuit")
async def analyze_circuit(file: UploadFile = File(...)):
    if circuit_model is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = similarity_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = circuit_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, 3)
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            circuit_name = class_names[idx]
            meta = {}
            if circuit_metadata is not None and circuit_name in circuit_metadata.index:
                meta = circuit_metadata.loc[circuit_name].to_dict()
            predictions.append({"circuit": circuit_name, "confidence": round(prob.item(), 3), "metadata": meta})
        return {"success": True, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain-lap")
async def explain_lap(request: LapExplanationRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        driver_num = predictor.driver_number_map.get(request.driver, 0)
        feat = {
            'DriverNumber': str(driver_num),
            'LapNumber': request.lap_number,
            'Stint': request.stint,
            'Compound': request.compound,
            'Team': 'UNKNOWN',
            'event_name': f"{request.circuit} Grand Prix",
            'circuit': request.circuit,
            'year': 2024,
            'round': 1,
            'AirTemp': request.weather.air_temp,
            'Humidity': request.weather.humidity,
            'Pressure': 1013.0,
            'Rainfall': request.weather.rainfall,
            'TrackTemp': request.weather.track_temp,
            'WindSpeed': 5.0,
            'WindDirection': 180,
            'stint_lap_number': request.tyre_age,
            'tyre_age_laps': request.tyre_age,
            'session_progress': request.lap_number / request.total_laps,
            'Position': 1,
            'position_change': 0
        }
        explanation = predictor.explain_lap_time(feat)
        return {"success": True, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tire-safety-check")
async def tire_safety_check(request: TireSafetyRequest):
    if tire_safety_predictor is None:
        raise HTTPException(status_code=503, detail="Tire safety model not loaded")
    try:
        feat = {
            'tyre_age_laps': request.tyre_age,
            'AirTemp': request.air_temp,
            'TrackTemp': request.track_temp,
            'Humidity': request.humidity,
            'Rainfall': request.rainfall,
            'deg_rate': 0.0,
            'deg_acceleration': 0.0,
            'age_ratio': request.tyre_age / 20.0,
            'time_ratio': 0.0,
            'Position': request.position,
            'session_progress': request.session_progress,
            'Compound': request.compound
        }
        risk_score = tire_safety_predictor.predict(feat)
        if risk_score < 30:
            category, action, safe_laps = "SAFE", "Continue", max(0, int(20 - request.tyre_age))
        elif risk_score < 70:
            category, action, safe_laps = "CAUTION", "Pit within 5 laps", max(0, int(15 - request.tyre_age))
        else:
            category, action, safe_laps = "CRITICAL", "PIT IMMEDIATELY", 0
        explanation = tire_safety_predictor.explain(feat)
        return {
            "success": True,
            "risk_score": round(risk_score, 1),
            "risk_category": category,
            "recommended_action": action,
            "safe_remaining_laps": safe_laps,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Crash Risk Endpoint ==========
@app.post("/crash-risk-predict")
async def crash_risk_predict(request: CrashRiskRequest):
    if crash_model is None:
        raise HTTPException(status_code=503, detail="Crash risk model not loaded")

    try:
        # Map circuit name to features
        circuit_info = circuit_metadata.loc[request.circuit] if circuit_metadata is not None and request.circuit in circuit_metadata.index else None
        if circuit_info is not None:
            circuit_type_encoded = 2 if 'street' in str(circuit_info['track_type']).lower() else 1
            corner_density = circuit_info['corners'] / circuit_info['length_km']
        else:
            circuit_type_encoded = 1
            corner_density = 4.0

        # Driver aggression from top 5 grid positions
        driver_profiles = pd.read_csv(project_root / 'models' / 'driver_style_profiles.csv')
        driver_agg_dict = driver_profiles.set_index('Driver')['AggressionScore'].to_dict()
        top5 = request.grid_positions[:5]
        avg_aggression = np.mean([driver_agg_dict.get(d, 0.5) for d in top5]) if top5 else 0.5

        # Historical crash rates
        circuit_crash_rate = crash_stats['circuit_crash_rates'].get('1', crash_stats['crash_rate']) if crash_stats else 0.05
        avg_driver_crash_rate = np.mean([crash_stats['driver_crash_rates'].get(d, crash_stats['crash_rate']) for d in top5]) if crash_stats else 0.05

        # Driver experience placeholder
        driver_experience = 5.0

        # Average grid position (risk indicator)
        avg_grid = np.mean([i+1 for i, _ in enumerate(request.grid_positions[:5])]) if request.grid_positions else 10.0

        # Championship pressure (max points gap among top5)
        if request.championship_standings:
            points = [request.championship_standings.get(d, 0) for d in top5]
            max_gap = max(points) - min(points) if points else 20.0
        else:
            max_gap = 20.0

        weather_wet = 1 if request.weather_wet else 0

        # Build feature vector in the same order as training
        features = np.array([[
            circuit_type_encoded,
            circuit_crash_rate,
            corner_density,
            avg_aggression,
            avg_driver_crash_rate,
            driver_experience,
            avg_grid,
            max_gap,
            weather_wet,
            request.track_temp
        ]])

        # Predict probability
        prob = crash_model.predict_proba(features)[0, 1]
        crash_prob = float(prob)

        # Manual adjustment since model underweights weather/temp
        if request.weather_wet:
            crash_prob = min(1.0, crash_prob * 1.35)
        if request.track_temp < 20:
            crash_prob = min(1.0, crash_prob * 1.10)
        elif request.track_temp > 45:
            crash_prob = min(1.0, crash_prob * 1.08)

        if crash_prob > 0.3:
            risk_level = "HIGH"
        elif crash_prob > 0.15:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Dynamic risk factors based on actual input
        risk_factors = [
            {
                "factor": "weather_conditions",
                "contribution": "+30%" if request.weather_wet else "+5%",
                "explanation": "Wet track" if request.weather_wet else "Dry track"
            },
            {
                "factor": "track_temperature",
                "contribution": "+15%" if request.track_temp < 20 else ("+8%" if request.track_temp > 45 else "+2%"),
                "explanation": f"Track temp: {request.track_temp}°C"
            },
            {
                "factor": "circuit_historical_crash_rate",
                "contribution": f"+{round(circuit_crash_rate * 100)}%",
                "explanation": f"Historical rate: {circuit_crash_rate:.2f}"
            },
            {
                "factor": "driver_aggression",
                "contribution": f"+{round(avg_aggression * 20)}%",
                "explanation": f"Avg aggression score: {avg_aggression:.2f}"
            },
            {
                "factor": "championship_pressure",
                "contribution": f"+{min(25, round(max_gap / 10))}%",
                "explanation": f"Points gap: {max_gap:.0f}"
            }
        ]

        # Identify high-risk drivers (aggressive + midfield)
        high_risk = []
        for idx, driver in enumerate(request.grid_positions):
            agg = driver_agg_dict.get(driver, 0.5)
            grid = idx + 1
            if agg > 0.7 and grid > 10:
                high_risk.append(driver)
        high_risk = high_risk[:3]

        # Recommendations
        recommendations = []
        if crash_prob > 0.3:
            recommendations.append("HIGH RISK: Safety car likely, consider conservative strategy")
        if request.weather_wet:
            recommendations.append("Wet conditions increase crash risk")
        if circuit_info is not None and 'street' in str(circuit_info['track_type']).lower():
            recommendations.append("Street circuit – tight barriers, high incident probability")
        if not recommendations:
            recommendations.append("Normal risk level")

        safety_car_prob = round(crash_prob * 0.8, 3)

        return {
            "crash_probability": round(crash_prob, 3),
            "risk_level": risk_level,
            "safety_car_probability": safety_car_prob,
            "high_risk_drivers": high_risk,
            "risk_factors": risk_factors,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static assets (CSS, JS, images, videos)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Serve index.html at root URL
@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")