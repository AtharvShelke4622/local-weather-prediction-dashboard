from fastapi import FastAPI, Depends, BackgroundTasks, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, AsyncGenerator
from datetime import datetime, timezone, timedelta
import httpx
from sqlalchemy import select
import structlog
from pydantic_settings import BaseSettings

# Configure basic logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Basic configuration
class Settings(BaseSettings):
    APP_NAME: str = "Weather Prediction Dashboard"
    DATABASE_URL: str = "sqlite:///./weather_dashboard.db"
    SECRET_KEY: str = "your-secret-key-change-in-production"

settings = Settings()
logger = structlog.get_logger()

# Simple FastAPI app for demo
app = FastAPI(title=settings.APP_NAME, version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic database setup (simplified)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, DateTime, JSON, ForeignKey, Boolean, text

class Base(DeclarativeBase):
    pass

class Device(Base):
    __tablename__ = "devices"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    device_key: Mapped[str] = mapped_column(String(128), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    temperature: Mapped[Optional[float]] = mapped_column(Float)
    humidity: Mapped[Optional[float]] = mapped_column(Float)
    wind_speed: Mapped[Optional[float]] = mapped_column(Float)
    radiation: Mapped[Optional[float]] = mapped_column(Float)
    precipitation: Mapped[Optional[float]] = mapped_column(Float)

# Database setup
engine = create_async_engine(settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://"))
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Application started successfully")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc)}

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint."""
    return {
        "http_requests_total": 0,
        "active_users_total": 1,
        "devices_total": 0
    }

# Basic endpoints for testing
@app.get("/api/v1/devices")
async def list_devices():
    """List all devices."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Device))
        devices = result.scalars().all()
        return [{"device_id": d.device_id, "last_seen": d.last_seen} for d in devices]

@app.post("/api/v1/ingest")
async def ingest_data(payload: dict):
    """Simple data ingestion endpoint."""
    logger.info("Data received", payload=payload)
    
    async with AsyncSessionLocal() as session:
        # Get or create device
        device_result = await session.execute(select(Device).where(Device.device_id == payload["device_id"]))
        device = device_result.scalar_one_or_none()
        
        if not device:
            device = Device(
                device_id=payload["device_id"],
                lat=payload.get("lat"),
                lon=payload.get("lon"),
                last_seen=datetime.now(timezone.utc)
            )
            session.add(device)
            await session.flush()  # Get the device ID
        else:
            device.last_seen = datetime.now(timezone.utc)
        
        # Create sensor reading
        reading = SensorReading(
            device_id=device.id,
            device_key=payload["device_id"],
            ts=datetime.fromisoformat(payload["ts"].replace('Z', '+00:00')),
            temperature=payload.get("temperature"),
            humidity=payload.get("humidity"),
            wind_speed=payload.get("wind_speed"),
            radiation=payload.get("radiation"),
            precipitation=payload.get("precipitation")
        )
        session.add(reading)
        await session.commit()
        
    return {"status": "ingested"}

@app.get("/api/v1/latest")
async def get_latest(device_id: str = Query(...)):
    """Get latest reading for a device."""
    async with AsyncSessionLocal() as session:
        # Get device
        device_result = await session.execute(select(Device).where(Device.device_id == device_id))
        device = device_result.scalar_one_or_none()
        
        if not device:
            return {"error": "Device not found"}
        
        # Get latest reading
        reading_result = await session.execute(
            select(SensorReading)
            .where(SensorReading.device_id == device.id)
            .order_by(SensorReading.ts.desc())
            .limit(1)
        )
        reading = reading_result.scalar_one_or_none()
        
        if not reading:
            return {"error": "No readings found"}
        
        return {
            "device_id": device_id,
            "ts": reading.ts,
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "wind_speed": reading.wind_speed,
            "radiation": reading.radiation,
            "precipitation": reading.precipitation,
        }

@app.get("/api/v1/predict")
async def predict(device_id: str = Query(...)):
    """Real prediction endpoint using model server."""
    import httpx
    
    # Get recent readings for device
    async with AsyncSessionLocal() as session:
        # Get device
        device_result = await session.execute(select(Device).where(Device.device_id == device_id))
        device = device_result.scalar_one_or_none()
        
        if not device:
            return {"error": "Device not found"}
        
        # Get last 24 readings
        readings_result = await session.execute(
            select(SensorReading)
            .where(SensorReading.device_id == device.id)
            .order_by(SensorReading.ts.desc())
            .limit(24)
        )
        readings = readings_result.scalars().all()
        
        if len(readings) < 24:
            # Fallback to mock predictions for demo
            import random
            current_time = datetime.now(timezone.utc)
            
            # Generate mock predictions
            mock_preds = {
                "temperature": [20 + random.uniform(-5, 5) for _ in range(8)],
                "humidity": [60 + random.uniform(-10, 10) for _ in range(8)],
                "wind_speed": [5 + random.uniform(-2, 3) for _ in range(8)],
                "radiation": [100 + random.uniform(-50, 50) for _ in range(8)],
                "precipitation": [max(0, random.uniform(-1, 2)) for _ in range(8)]
            }
            
            # Generate prediction text
            from prediction_text import generate_prediction_text
            prediction_text = generate_prediction_text(mock_preds)
            
            return {
                "device_id": device_id,
                "pred_ts": current_time.isoformat(),
                "for_ts": [(current_time + timedelta(hours=i)).isoformat() for i in range(1, 9)],
                "predictions": mock_preds,
                "prediction_text": prediction_text,
                "model_version": "demo_v2.0_fallback"
            }
        
        # Prepare data for model server (reverse to chronological order)
        window = [
            [
                reading.temperature or 0,
                reading.humidity or 0,
                reading.wind_speed or 0,
                reading.radiation or 0,
                reading.precipitation or 0
            ]
            for reading in reversed(readings)
        ]
    
    # Call model server
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8001/model/predict",
                json={"device_id": device_id, "recent_window": window},
                timeout=10.0
            )
            response.raise_for_status()
            model_result = response.json()
        
        # Convert model server response to API format
        current_time = datetime.now(timezone.utc)
        predictions = model_result["predictions_8h"]
        
        # Convert model server response to API format
        current_time = datetime.now(timezone.utc)
        predictions = model_result["predictions_8h"]
        
        # Generate prediction text
        from prediction_text import generate_prediction_text
        prediction_text = generate_prediction_text(predictions)
        
        forecast = {
            "device_id": device_id,
            "pred_ts": current_time.isoformat(),
            "for_ts": [(current_time + timedelta(hours=i)).isoformat() for i in range(1, 9)],
            "predictions": {
                "temperature": predictions.get("temperature", []),
                "humidity": predictions.get("humidity", []),
                "wind_speed": predictions.get("wind_speed", []),
                "radiation": predictions.get("radiation", []),
                "precipitation": predictions.get("precipitation", [])
            },
            "prediction_text": prediction_text,
            "model_version": model_result["model_version"]
        }
        
        logger.info("Prediction generated", device_id=device_id, model_version=model_result["model_version"])
        return forecast
        
    except httpx.RequestError as e:
        logger.error("Model server request failed", error=str(e))
        return {"error": "Model server unavailable"}
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        return {"error": "Prediction failed"}

# =========================================================
# PREDICTION TEXT ENDPOINT
# =========================================================
@app.get("/api/v1/prediction-text")
async def get_prediction_text(device_id: str = Query(...)):
    """Generate human-readable weather insights."""
    from prediction_text import generate_prediction_text
    
    # Generate simple mock predictions for text
    import random
    predictions = {
        "temperature": [20 + random.uniform(-5, 5) for _ in range(8)],
        "humidity": [60 + random.uniform(-10, 10) for _ in range(8)],
        "wind_speed": [5 + random.uniform(-2, 3) for _ in range(8)],
        "radiation": [100 + random.uniform(-50, 50) for _ in range(8)],
        "precipitation": [max(0, random.uniform(-1, 2)) for _ in range(8)]
    }
    
    # Generate prediction text
    prediction_text = generate_prediction_text(predictions)
    
    return {
        "device_id": device_id,
        "model_version": "demo_v2.0_text",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_text": prediction_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)