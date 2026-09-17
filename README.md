# 🏎️ GRID-X: AI-Powered Formula 1 Prediction & Strategy Platform

GRID-X (Global Race Intelligence and Data Exchange) is an end-to-end AI-powered Formula 1 platform designed to simulate, analyze, and predict race performance. Developed as a degree capstone project, GRID-X combines data engineering, machine learning, deep learning, reinforcement learning, and computer vision into a unified full-stack system.

The platform uses a hybrid cloud approach: source code and application logic are hosted on GitHub, while large datasets and trained model binaries are hosted separately on Hugging Face Hub.

---

## 🚀 Key Features

### 🏁 Lap-Time Prediction

- Random Forest Regressor for lap-time prediction.
- Uses 21 telemetry and environmental features.
- **97% accuracy**
- **RMSE:** 2.787 seconds
- **R² Score:** 0.993

### 🏆 Race Outcome Prediction

- Dual-era XGBoost ensemble.
- Predicts:
  - Race Winner
  - Podium Finish
  - Points Finish
  - Top-10 Finish
- **96% win classification accuracy**

### 🧠 Driver Style Analysis

Uses K-Means clustering to analyze driver behavior and classify driving styles into:

- Aggressive
- Smooth
- Opportunistic
- Balanced

### 📈 Pace Forecaster

- LSTM-based time-series model.
- Uses a 10-lap sliding window.
- Predicts upcoming lap performance based on telemetry, tyre, driver, circuit, and session data.

### 🤖 Strategy Optimizer

- Reinforcement Learning based strategy system.
- Uses a DQN agent with a Gymnasium environment.
- Optimizes:
  - Pit-stop windows
  - Tyre compounds
  - Race strategy decisions

### 🏎️ Circuit Recognition

- CNN-based circuit recognition system.
- Uses ResNet50 transfer learning.
- Identifies Formula 1 circuits from schematic track images.

### ⚠️ Crash Risk Prediction

- XGBoost-based pre-race crash risk prediction.
- Considers factors such as:
  - Circuit
  - Driver aggression
  - Weather conditions
  - Racing conditions

### 🛞 Tyre Safety Monitoring

Analyzes tyre degradation and assigns safety categories:

- 🟢 SAFE
- 🟡 CAUTION
- 🔴 CRITICAL

Also provides pit-stop recommendations based on tyre conditions.

### 🔍 Explainable AI

- SHAP-based model explainability.
- Helps understand which features contribute to model predictions.

---

## 🏗️ Architecture & Storage

### GitHub

The GitHub repository contains:

- Core application logic
- FastAPI backend
- API endpoints
- Data preprocessing
- Model training scripts
- Machine learning pipelines
- Frontend
- Documentation

### Hugging Face Hub

Large files are hosted separately to keep the GitHub repository lightweight:

- **7.7GB+** Formula 1 telemetry datasets
- **1.4GB+** Serialized trained models

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Full race prediction (RaceInput JSON) |
| `/stint-simulate` | POST | Linear stint simulation & tyre degradation |
| `/crash-risk-predict` | POST | Crash probability analysis |
| `/tire-safety-predict` | POST | Tire degradation risk assessment |
| `/next-lap` | POST | LSTM-based pace forecasting |
| `/strategy-optimize` | POST | RL-driven pit stop optimization |
| `/analyze-circuit` | POST | CNN track recognition from images |
| `/explain-lap` | POST | SHAP feature importance analysis |

---

## 📂 Project Structure

```text
GRID-X/
├── API/
│   ├── main.py
│   └── ...
│
├── frontend/
│   ├── index.html
│   ├── pitwall.html
│   └── pages/
│
├── contributors/
│   ├── faiz.jpg
│   └── ibrahim.jpg
│
├── data/
│
├── models/
│
├── scripts/
│   └── models/
│
├── requirements/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🖥️ Frontend Dashboard

GRID-X includes a responsive web-based dashboard designed to provide an interactive Formula 1 analytics experience.

### 📊 Race Analytics

- Lap-time predictions
- Race outcome predictions
- Driver performance analysis
- Driver style visualization
- Circuit analysis

### 🛡️ Safety & Strategy

- Crash risk prediction
- Tyre degradation monitoring
- Tyre safety status
- Pit-stop recommendations
- Strategy optimization

### 👁️ Vision & Explainability

- AI-powered circuit recognition
- Circuit similarity analysis
- SHAP-based prediction explanations
- Interactive visualizations

The frontend consists of **20+ responsive pages** built using modern HTML, CSS, and JavaScript and communicates with the FastAPI backend through REST APIs.

---

## 📊 Datasets

- **Modern Telemetry (2021–2024):** ~91,000 lap records collected via **FastF1**.
- **Historical Data (1950–2020):** ~27,000 entries sourced from the **Ergast/Kaggle F1 Dataset**.
- **Vision Data:** Manually curated collection of high-resolution Formula 1 circuit schematics used for circuit recognition.

---

## 🤝 Contributors

| Profile | Contributor | GitHub |
|:---:|---|---|
| <img src="./contributors/Faiz.png" width="80px" alt="Faiz Ahmed"> | **Faiz Ahmed** | [@Faiz-ahmed-13](https://github.com/Faiz-ahmed-13) |
| <img src="./contributors/Ibrahim.jpeg" width="80px" alt="Syed Ibrahim Ali"> | **Syed Ibrahim Ali** | [@IIGGRRIISS](https://github.com/IIGGRRIISS) |

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/Faiz-ahmed-13/GRID-X/blob/main/LICENSE) file for details.

---

**GRID-X is a complete full-stack Formula 1 AI platform combining machine learning, deep learning, race analytics, strategy optimization, safety analysis, and an interactive web interface. Production deployment coming soon. 🏁**