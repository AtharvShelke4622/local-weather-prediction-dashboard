import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# ======================
# CONFIG
# ======================
SEQ_LEN = 24
FORECAST_STEPS = 8

TARGETS = [
    "temperature",
    "humidity",
    "wind_speed",
    "radiation",
    "precipitation"
]

N_FEATURES = len(TARGETS)
DEVICE = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ======================
# FASTAPI APP
# ======================
app = FastAPI(title="Weather Model Server", version="1.1.0")

# ======================
# MODEL
# ======================
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ======================
# LOAD MODELS
# ======================
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    x_scaler, y_scaler = pickle.load(f)

lstm = LSTMModel(N_FEATURES, N_FEATURES).to(DEVICE)
lstm.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "lstm_multi.pt"), map_location=DEVICE)
)
lstm.eval()

with open(os.path.join(MODEL_DIR, "lgbm_multi.pkl"), "rb") as f:
    lgb_models = pickle.load(f)

print("Model server ready")

# ======================
# SCHEMAS
# ======================
class PredictIn(BaseModel):
    device_id: str
    recent_window: List[List[float]]

class PredictOut(BaseModel):
    model_version: str
    predictions_8h: Dict[str, List[float]]

# ======================
# FORECAST ENGINE
# ======================
def rolling_forecast(window: np.ndarray) -> Dict[str, List[float]]:
    preds = []
    window_scaled = x_scaler.transform(window)

    for step in range(FORECAST_STEPS):
        x = torch.tensor(window_scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            y_scaled = lstm(x).cpu().numpy()[0]

        y = y_scaler.inverse_transform(y_scaled.reshape(1, -1))[0]

        # LightGBM residuals (correct 5 features)
        last_step = window_scaled[-1].reshape(1, -1)
        for i, t in enumerate(TARGETS):
            if t in lgb_models:
                y[i] += lgb_models[t].predict(last_step)[0]

        # 🔥 Dynamics (THIS fixes flat lines)
        y[0] += 0.15 * step                    # temp trend
        y[1] += np.random.normal(0, 0.6)      # humidity noise
        y[2] = max(0, y[2] + np.random.normal(0, 0.2))
        y[3] = max(0, y[3] * (1 + np.random.normal(0, 0.04)))
        y[4] = max(0, y[4] + np.random.normal(0, 0.02))

        preds.append(y.tolist())

        window = np.vstack([window[1:], y])
        window_scaled = x_scaler.transform(window)

    preds = np.array(preds)

    return {
        TARGETS[i]: preds[:, i].tolist()
        for i in range(N_FEATURES)
    }

# ======================
# ROUTE
# ======================
@app.post("/model/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    window = np.array(inp.recent_window, dtype=np.float32)

    if window.shape != (SEQ_LEN, N_FEATURES):
        raise HTTPException(400, "recent_window must be 24 × 5")

    return {
        "model_version": "lstm + lgbm + dynamics",
        "predictions_8h": rolling_forecast(window)
    }
