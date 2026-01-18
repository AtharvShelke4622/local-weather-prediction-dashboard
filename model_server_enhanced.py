import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import random

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
# MODEL ARCHITECTURES
# ======================
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ======================
# ENHANCED MODEL SERVER
# ======================
app = FastAPI(title="Enhanced Weather Model Server", version="2.0.0")

# Load all models and scalers
print("Loading enhanced models...")
loaded_models = {}
model_scores = {}

def load_model_safely(model_path, model_name):
    """Safely load a model with error handling"""
    try:
        if os.path.exists(model_path):
            if model_path.endswith('.pt'):
                model = EnhancedLSTMModel(N_FEATURES, N_FEATURES) if 'enhanced_lstm' in model_path else GRUModel(N_FEATURES, N_FEATURES)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                return model, f"Loaded {model_name}"
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model, f"Loaded {model_name}"
        return None, f"Model file not found: {model_path}"
    except Exception as e:
        return None, f"Error loading {model_name}: {str(e)}"

# Load scaler
try:
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        x_scaler, y_scaler = pickle.load(f)
except:
    print("Warning: Scaler not found, using default scaling")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

# Load deep learning models
lstm_model, lstm_status = load_model_safely(os.path.join(MODEL_DIR, "enhanced_lstm.pt"), "Enhanced LSTM")
gru_model, gru_status = load_model_safely(os.path.join(MODEL_DIR, "gru.pt"), "GRU")

# Load ensemble models
rf_models, rf_status = load_model_safely(os.path.join(MODEL_DIR, "random_forest_models.pkl"), "Random Forest")
xgb_models, xgb_status = load_model_safely(os.path.join(MODEL_DIR, "xgboost_models.pkl"), "XGBoost")
lgb_enhanced, lgb_status = load_model_safely(os.path.join(MODEL_DIR, "lgbm_multi_enhanced.pkl"), "Enhanced LightGBM")

# Load training results for model selection
try:
    with open(os.path.join(MODEL_DIR, "training_results.pkl"), "rb") as f:
        model_scores = pickle.load(f)
except:
    model_scores = {}

# Load original models as fallback
try:
    with open(os.path.join(MODEL_DIR, "lstm_multi.pt"), "rb") as f:
        lstm_original = torch.load(f, map_location=DEVICE)
        original_lstm = EnhancedLSTMModel(N_FEATURES, N_FEATURES)
        original_lstm.load_state_dict(lstm_original)
        original_lstm.eval()
        loaded_models['lstm_original'] = original_lstm
except:
    loaded_models['lstm_original'] = None

try:
    with open(os.path.join(MODEL_DIR, "lgbm_multi.pkl"), "rb") as f:
        lgb_original = pickle.load(f)
        loaded_models['lgbm_original'] = lgb_original
except:
    loaded_models['lgbm_original'] = None

print(f"Model loading status:")
print(f"  {lstm_status}")
print(f"  {gru_status}") 
print(f"  {rf_status}")
print(f"  {xgb_status}")
print(f"  {lgb_status}")

# ======================
# SCHEMAS
# ======================
class PredictIn(BaseModel):
    device_id: str
    recent_window: List[List[float]]
    model_preference: Optional[str] = None  # "lstm", "gru", "rf", "xgb", "lgbm", "ensemble"

class PredictOut(BaseModel):
    model_version: str
    predictions_8h: Dict[str, List[float]]
    confidence_scores: Optional[Dict[str, float]] = None
    ensemble_weights: Optional[Dict[str, float]] = None

# ======================
# PREDICTION METHODS
# ======================
def predict_with_lstm(window: np.ndarray, model_type: str = "enhanced"):
    """Predict with LSTM model"""
    if model_type == "enhanced" and lstm_model:
        x_scaled = x_scaler.transform(window.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred_scaled = lstm_model(x_tensor)
            pred = y_scaler.inverse_transform(pred_scaled.numpy())
        return pred.flatten()
    elif model_type == "original" and loaded_models.get('lstm_original'):
        x_scaled = x_scaler.transform(window.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred_scaled = loaded_models['lstm_original'](x_tensor)
            pred = y_scaler.inverse_transform(pred_scaled.numpy())
        return pred.flatten()
    return None

def predict_with_gru(window: np.ndarray):
    """Predict with GRU model"""
    if not gru_model:
        return None
        
    x_scaled = x_scaler.transform(window.reshape(1, -1))
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        pred_scaled = gru_model(x_tensor)
        pred = y_scaler.inverse_transform(pred_scaled.numpy())
    return pred.flatten()

def predict_with_random_forest(window: np.ndarray):
    """Predict with Random Forest"""
    if not rf_models:
        return None
        
    # Use last timestep for RF
    last_timestep = window[-1:].flatten()
    pred = {}
    for target in TARGETS:
        if target in rf_models:
            pred[target] = rf_models[target].predict([last_timestep])[0]
    
    return [pred.get(t, 0) for t in TARGETS]

def predict_with_xgboost(window: np.ndarray):
    """Predict with XGBoost"""
    if not xgb_models:
        return None
        
    # Use last timestep for XGB
    last_timestep = window[-1:].flatten()
    pred = {}
    for target in TARGETS:
        if target in xgb_models:
            pred[target] = xgb_models[target].predict([last_timestep])[0]
    
    return [pred.get(t, 0) for t in TARGETS]

def predict_with_enhanced_lgbm(window: np.ndarray, model_type: str = "lstm"):
    """Predict with enhanced LightGBM models"""
    if not lgb_enhanced:
        return None
    
    # Get LSTM or GRU residuals
    if model_type == "lstm" and lstm_model:
        x_scaled = x_scaler.transform(window.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred_scaled = lstm_model(x_tensor)
            lstm_pred = y_scaler.inverse_transform(pred_scaled.numpy())
        
        # Use enhanced features
        last_timestep = window[-1:].flatten()
        X_tab = np.concatenate([last_timestep, lstm_pred])
        
    elif model_type == "gru" and gru_model:
        x_scaled = x_scaler.transform(window.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred_scaled = gru_model(x_tensor)
            gru_pred = y_scaler.inverse_transform(pred_scaled.numpy())
            
        # Use enhanced features
        last_timestep = window[-1:].flatten()
        X_tab = np.concatenate([last_timestep, gru_pred])
    else:
        return None
    
    pred = {}
    for target in TARGETS:
        model_key = f'{model_type}_{target}'
        if model_key in lgb_enhanced:
            lgb_pred = lgb_enhanced[model_key].predict(X_tab)[0]
            pred[target] = lgb_pred
        else:
            # Fallback to original LGBM
            if loaded_models.get('lgbm_original') and target in loaded_models['lgbm_original']:
                pred[target] = loaded_models['lgbm_original'][target].predict(X_tab)[0]
            else:
                pred[target] = 0
    
    return [pred.get(t, 0) for t in TARGETS]

def create_ensemble_predictions(predictions_list, weights=None):
"""Create ensemble predictions from multiple models"""
    all_predictions = [lstm_enhanced, lstm_original, gru_pred, rf_pred, xgb_pred, lgb_lstm, lgb_gru]
    ensemble_pred, confidence = create_ensemble_predictions(all_predictions)
    
    # Handle case where all predictions might be None
    if ensemble_pred is None:
        ensemble_pred = np.zeros(len(TARGETS))
        confidence = {target: 0.5 for target in TARGETS}
    
    # Select best prediction based on user preference or fallback to best performing model
    if inp.model_preference:
        if inp.model_preference == "lstm":
            final_pred = lstm_enhanced or lstm_original
            model_version = "enhanced_lstm"
        elif inp.model_preference == "gru":
            final_pred = gru_pred
            model_version = "gru"
        elif inp.model_preference == "rf":
            final_pred = rf_pred
            model_version = "random_forest"
        elif inp.model_preference == "xgb":
            final_pred = xgb_pred
            model_version = "xgboost"
        elif inp.model_preference == "lgbm":
            final_pred = lgb_lstm or lgb_gru
            model_version = "enhanced_lgbm"
        elif inp.model_preference == "ensemble":
            final_pred = ensemble_pred
            model_version = "weighted_ensemble"
        else:
            # Default to best performing model based on training scores
            best_model = min(model_scores.items(), key=lambda x: x[1] if x[1] else float('inf'))[0]
            if "lstm" in best_model:
                final_pred = lstm_enhanced or lstm_original
                model_version = "best_lstm"
            else:
                final_pred = ensemble_pred
                model_version = "auto_ensemble"
    else:
        # Auto-select best model
        # Try enhanced LSTM first
        if lstm_enhanced is not None:
            final_pred = lstm_enhanced
            model_version = "enhanced_lstm_auto"
        elif ensemble_pred is not None:
            final_pred = ensemble_pred
            model_version = "auto_ensemble"
        elif rf_pred is not None:
            final_pred = rf_pred
            model_version = "random_forest_auto"
        else:
            # Fallback to original models
            if lstm_original is not None:
                final_pred = lstm_original
                model_version = "lstm_original"
            elif loaded_models.get('lgbm_original'):
                pred = {}
                for target in TARGETS:
                    if target in loaded_models['lgbm_original']:
                        pred[target] = loaded_models['lgbm_original'][target].predict(window[-1:].flatten())[0]
                    else:
                        pred[target] = 0
                final_pred = [pred.get(t, 0) for t in TARGETS]
                model_version = "lgbm_original"
            else:
                # Simple fallback
                final_pred = np.zeros(len(TARGETS))
                model_version = "fallback"
    
    # Format response
    result = {
        "model_version": model_version,
        "predictions_8h": {
            "temperature": final_pred.tolist(),
            "humidity": final_pred.tolist(),
            "wind_speed": final_pred.tolist(), 
            "radiation": final_pred.tolist(),
            "precipitation": final_pred.tolist()
        }
    }
    
    # Add confidence scores if available
    if confidence is not None:
        result["confidence_scores"] = {
            "temperature": float(confidence[0]) if len(confidence) > 0 else 0.5,
            "humidity": float(confidence[1]) if len(confidence) > 1 else 0.5,
            "wind_speed": float(confidence[2]) if len(confidence) > 2 else 0.5,
            "radiation": float(confidence[3]) if len(confidence) > 3 else 0.5,
            "precipitation": float(confidence[4]) if len(confidence) > 4 else 0.5,
        }
    
    return result

# ======================
# MODEL INFO ENDPOINT
# ======================
@app.get("/models/info")
def get_model_info():
    """Get information about loaded models"""
    return {
        "loaded_models": {
            "enhanced_lstm": lstm_model is not None,
            "gru": gru_model is not None,
            "random_forest": rf_models is not None,
            "xgboost": xgb_models is not None,
            "enhanced_lgbm": lgb_enhanced is not None,
            "lstm_original": loaded_models.get('lstm_original') is not None,
            "lgbm_original": loaded_models.get('lgbm_original') is not None
        },
        "model_scores": model_scores,
        "available_preferences": ["lstm", "gru", "rf", "xgb", "lgbm", "ensemble", "auto"],
        "default_model": "enhanced_lstm_auto"
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len([m for m in [lstm_model, gru_model, rf_models, xgb_models, lgb_enhanced] if m is not None])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)