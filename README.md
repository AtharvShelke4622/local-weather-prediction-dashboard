# Local Weather Prediction Dashboard

End-to-end stack for ingesting IoT weather data, storing it, running an ML forecast (TorchScript + LightGBM residuals), and visualizing an 8-hour forecast in a React dashboard.

## Architecture
- **Backend API**: FastAPI (async SQLAlchemy), endpoints under `/api/v1/*`
- **Model server**: Separate FastAPI process loading TorchScript model and LightGBM residuals from `/models`
- **DB**: PostgreSQL / TimescaleDB in Docker; SQLite fallback in dev
- **Frontend**: React + Vite, CSS Modules, Recharts. Polls every 5 minutes (configurable).

## Quickstart (without Docker)

Open two terminals for the backend:

```bash
# 1) Start model server (port 8001)
python backend/model_server.py

# 2) Start API (port 8000)
uvicorn backend.main:app --reload --port 8000
```

Then start the frontend:
```bash
cd frontend
npm install
npm run dev
```

### Environment
Copy `.env.example` to `.env` and adjust if needed. Defaults use SQLite and model server on `http://localhost:8001`.

## With Docker

```bash
docker compose up --build
```
This brings up:
- `db` TimescaleDB (port 5432)
- `backend` API (port 8000)
- `modelserver` (port 8001)

A named volume `models` stores your models shared by API and model server.

## API

- `POST /api/v1/ingest`
- `GET  /api/v1/latest?device_id=...`
- `GET  /api/v1/devices`
- `GET  /api/v1/predict?device_id=...`

### Example ESP32 JSON Payload
```json
{
  "device_id": "esp32-001",
  "ts": "2025-09-06T06:30:00Z",
  "temperature": 30.2,
  "humidity": 62.1,
  "wind_speed": 3.4,
  "radiation": 512.0,
  "precipitation": 0.0,
  "lat": 12.9716,
  "lon": 77.5946
}
```

POST this to `http://<backend-host>:8000/api/v1/ingest` every few minutes.

## Training (Baseline)
A simple baseline trainer is provided. It reads your CSV (e.g. NASA POWER combined hourly file) and writes a minimal TorchScript model and empty LightGBM residuals to `/models`:

```bash
python backend/training/train_baseline.py --csv /path/to/POWER_Point_Hourly_2001_2025_combined.csv --out /models
```
> In Docker, you can copy your dataset into the container or mount it, then run the command with `docker compose exec backend ...`.  
> The model server hot-loads from the shared `/models` volume. If models are missing, it falls back to “repeat last value”.

## Frontend
- Title: **🌤 Local Weather Dashboard**
- Device list with last seen
- Latest sensors cards (Temp, Humidity, Wind, Radiation, Precipitation)
- ForecastCard with 8-hour multi-line Recharts graph and tabular preview
- Polling interval is `POLL_MS` (default 300000 ms), configurable via `VITE_POLL_MS`.

### Configure Frontend
Create a `.env` in `frontend/` if you need non-defaults:
```
VITE_API_BASE=
VITE_POLL_MS=300000
```
By default, Vite dev server proxies `/api` to `http://localhost:8000`.

## Notes
- DB schema is created automatically on startup. For production, add proper migrations and Timescale hypertables/indexes.
- CORS origins can be set via `CORS_ORIGINS` env (comma-separated).

## Folder Layout
```
backend/
  app_config.py
  main.py
  database.py
  db_models.py
  schemas.py
  crud.py
  model_client.py
  model_server.py
  training/train_baseline.py
frontend/
  index.html
  vite.config.ts
  package.json
  src/
    main.tsx
    App.tsx
    api.ts
    config.ts
    components/
      DeviceList.tsx
      SensorCard.tsx
      ForecastCard.tsx
    styles/
      *.module.css
docker-compose.yml
backend/Dockerfile
.env.example
README.md
```
