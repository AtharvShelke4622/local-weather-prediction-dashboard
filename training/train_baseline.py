import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ======================
# CONFIG
# ======================
SEQ_LEN = 24
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = ["T2M", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "POWER_Point_Hourly_2001_2025_combined.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# LOAD DATA
# ======================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df[TARGETS].dropna()

# ======================
# SCALE (FIT HERE!)
# ======================
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(df.values)
y_scaled = y_scaler.fit_transform(df.values)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump((x_scaler, y_scaler), f)

# ======================
# SEQUENCES
# ======================
def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LEN)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

loader = DataLoader(
    TensorDataset(X_tensor, y_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ======================
# MULTI-TARGET LSTM
# ======================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel(len(TARGETS), len(TARGETS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ======================
# TRAIN
# ======================
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    print(f"LSTM Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_multi.pt"))

# ======================
# LIGHTGBM RESIDUALS
# ======================
print("Training LightGBM residuals...")

model.eval()
with torch.no_grad():
    lstm_preds = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        lstm_preds.append(model(xb).cpu().numpy())

lstm_preds = np.vstack(lstm_preds)
residuals = y_seq - lstm_preds

X_tab = X_scaled[SEQ_LEN:SEQ_LEN + len(residuals)]

lgb_models = {}

for i, target in enumerate(TARGETS):
    lgbm = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )
    lgbm.fit(X_tab, residuals[:, i])
    lgb_models[target] = lgbm

with open(os.path.join(MODEL_DIR, "lgbm_multi.pkl"), "wb") as f:
    pickle.dump(lgb_models, f)

print("\nMULTI-TARGET MODELS SAVED")
