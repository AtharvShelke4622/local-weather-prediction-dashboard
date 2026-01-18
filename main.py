from fastapi import FastAPI, Depends, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, AsyncGenerator
from datetime import datetime, timezone
from sqlalchemy import select
from logging_config import logger, audit_logger, setup_logging, error_boundary, database_transaction
from app_config import settings
from database import AsyncSessionLocal, init_models, ping_db
import crud
from schemas import IngestPayload, DeviceOut, LatestOut, ForecastOut
from model_client import predict_8h
from db_models import Device, SensorReading, User
from prediction_text import generate_prediction_text
from auth_routes import router as auth_router
from auth import get_current_active_user, user_required
from rate_limiter import limiter, rate_limit_exceeded_handler
# from cache import cache
cache = None
from metrics import metrics_endpoint

# Configure structured logging
import structlog
logger = structlog.get_logger()

app = FastAPI(
    title=settings.APP_NAME,
    description="Local Weather Prediction Dashboard API",
    version="2.0.0"
)

# =========================================================
# RATE LIMITING
# =========================================================
app.state.limiter = limiter
app.add_exception_handler(429, rate_limit_exceeded_handler)

# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DB SESSION
# =========================================================
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

# =========================================================
# INCLUDE ROUTES
# =========================================================
app.include_router(auth_router)

# =========================================================
# STARTUP
# =========================================================
@app.on_event("startup")
async def on_startup():
    setup_logging()
    await init_models()
    await ping_db()
    if cache:
        await cache.connect()
    logger.info("Application started successfully")
    audit_logger.log_system_event(
        event_type="application_startup",
        severity="info",
        component="main_api",
        details={"version": "2.0.0"}
    )

# =========================================================
# SHUTDOWN
# =========================================================
@app.on_event("shutdown")
async def on_shutdown():
    if cache:
        await cache.disconnect()
    logger.info("Application shutdown complete")

# =========================================================
# HEALTH
# =========================================================
@app.get("/healthz")
@limiter.limit("100/minute")
async def healthz(request: Request):
    return {"status": "ok", "timestamp": datetime.now(timezone.utc)}

# =========================================================
# METRICS
# =========================================================
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return await metrics_endpoint()

# =========================================================
# INGEST
# =========================================================
@app.post("/api/v1/ingest")
@limiter.limit("60/minute")
@error_boundary
async def ingest(
    request: Request,
    payload: IngestPayload,
    bg: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(user_required),
):
    device = await crud.upsert_device(
        session,
        payload.device_id,
        payload.lat,
        payload.lon,
    )

    reading = SensorReading(
        device_id=device.id,
        device_key=device.device_id,
        ts=payload.ts or datetime.now(timezone.utc),
        temperature=payload.temperature,
        humidity=payload.humidity,
        wind_speed=payload.wind_speed,
        radiation=payload.radiation,
        precipitation=payload.precipitation,
        raw=payload.model_dump(),
    )

    async with database_transaction(session, "ingest_sensor_reading"):
        session.add(reading)
        await session.flush()
        
        # Log data access
        audit_logger.log_data_access(
            user_id=current_user.id,
            resource_type="sensor_reading",
            resource_id=payload.device_id,
            action="create",
            ip_address=request.client.host if request.client else "unknown",
            details={"payload": payload.dict()}
        )

    async def _bg_task(dev_id: str):
        async with AsyncSessionLocal() as s:
            window = await crud.last_n_readings(s, dev_id, n=24)

            if not window:
                return

            for_ts, preds, model_version = await predict_8h(dev_id, window)

            result = await s.execute(
                select(Device).where(Device.device_id == dev_id)
            )
            dev = result.scalar_one_or_none()

            if dev:
                await crud.store_forecast(
                    s,
                    dev,
                    for_ts,
                    preds,
                    model_version,
                )
                await s.commit()

    bg.add_task(_bg_task, payload.device_id)
    return {"status": "ingested"}

# =========================================================
# LATEST
# =========================================================
@app.get("/api/v1/latest", response_model=Optional[LatestOut])
@limiter.limit("100/minute")
async def latest(
    request: Request,
    device_id: str = Query(...),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(user_required),
):
    return await crud.get_latest_reading(session, device_id)

# =========================================================
# DEVICES
# =========================================================
@app.get("/api/v1/devices", response_model=List[DeviceOut])
@limiter.limit("50/minute")
async def devices(
    request: Request,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(user_required),
):
    items = await crud.list_devices(session)
    return [DeviceOut(**i) for i in items]

# =========================================================
# NUMERIC FORECAST
# =========================================================
@app.get("/api/v1/predict", response_model=Optional[ForecastOut])
@limiter.limit("30/minute")
async def predict(
    request: Request,
    device_id: str = Query(...),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(user_required),
):
    window = await crud.last_n_readings(session, device_id, n=24)

    if not window:
        return {
            "device_id": device_id,
            "pred_ts": datetime.now(timezone.utc),
            "for_ts": [],
            "predictions": {},
            "model_version": "fallback",
        }

    for_ts, preds, model_version = await predict_8h(device_id, window)

    result = await session.execute(
        select(Device).where(Device.device_id == device_id)
    )
    dev = result.scalar_one_or_none()

    if not dev:
        return None

    await crud.store_forecast(
        session,
        dev,
        for_ts,
        preds,
        model_version,
    )
    await session.commit()

    return {
        "device_id": device_id,
        "pred_ts": datetime.now(timezone.utc),
        "for_ts": for_ts,
        "predictions": preds,
        "model_version": model_version,
    }

# =========================================================
# HUMAN READABLE TEXT
# =========================================================
@app.get("/api/v1/prediction-text")
@limiter.limit("30/minute")
async def prediction_text(
    request: Request,
    device_id: str = Query(...),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(user_required),
):
    window = await crud.last_n_readings(session, device_id, n=24)

    if not window:
        return {
            "device_id": device_id,
            "status": "insufficient_data",
            "message": "At least one sensor reading is required.",
            "prediction_text": {},
        }

    for_ts, preds, model_version = await predict_8h(device_id, window)
    readable_text = generate_prediction_text(preds)

    return {
        "device_id": device_id,
        "model_version": model_version,
        "generated_at": datetime.now(timezone.utc),
        "prediction_text": readable_text,
    }
