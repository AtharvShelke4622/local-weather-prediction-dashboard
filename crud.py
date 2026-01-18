from typing import Optional, List
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from db_models import Device, SensorReading, Forecast8h, User


# =====================================================
# DEVICE UPSERT
# =====================================================
async def upsert_device(
    session: AsyncSession,
    device_id: str,
    lat: Optional[float],
    lon: Optional[float],
) -> Device:
    res = await session.execute(
        select(Device).where(Device.device_id == device_id)
    )
    device = res.scalar_one_or_none()
    now = datetime.now(timezone.utc)

    if device is None:
        device = Device(
            device_id=device_id,
            lat=lat,
            lon=lon,
            last_seen=now,
        )
        session.add(device)
        await session.flush()
    else:
        device.last_seen = now
        if lat is not None:
            device.lat = lat
        if lon is not None:
            device.lon = lon
        await session.flush()

    return device


# =====================================================
# LATEST SENSOR READING
# =====================================================
async def get_latest_reading(
    session: AsyncSession,
    device_id: str,
) -> Optional[dict]:

    q = (
        select(SensorReading, Device)
        .join(Device, SensorReading.device_id == Device.id)
        .where(Device.device_id == device_id)
        .order_by(desc(SensorReading.ts))
        .limit(1)
    )

    res = await session.execute(q)
    row = res.first()
    if not row:
        return None

    sr, dev = row
    return {
        "device_id": dev.device_id,
        "ts": sr.ts,
        "temperature": sr.temperature,
        "humidity": sr.humidity,
        "wind_speed": sr.wind_speed,
        "radiation": sr.radiation,
        "precipitation": sr.precipitation,
    }


# =====================================================
# DEVICE LIST
# =====================================================
async def list_devices(session: AsyncSession) -> List[dict]:
    q = select(Device.device_id, Device.last_seen).order_by(Device.device_id)
    res = await session.execute(q)
    return [{"device_id": d, "last_seen": ls} for d, ls in res.all()]


# =====================================================
# LAST N READINGS (🔥 CRITICAL FIX 🔥)
# =====================================================
async def last_n_readings(
    session: AsyncSession,
    device_id: str,
    n: int = 24,
) -> List[List[float]]:
    """
    Returns latest N readings in chronological order (old → new)
    """

    q = (
        select(
            SensorReading.temperature,
            SensorReading.humidity,
            SensorReading.wind_speed,
            SensorReading.radiation,
            SensorReading.precipitation,
        )
        .where(SensorReading.device_key == device_id)
        .order_by(desc(SensorReading.ts))   # 🔥 latest first
        .limit(n)
    )

    res = await session.execute(q)
    rows = res.all()

    if not rows:
        return []

    # reverse → chronological order
    rows = list(reversed(rows))

    return [list(map(float, r)) for r in rows]


# =====================================================
# STORE FORECAST
# =====================================================
async def store_forecast(
    session: AsyncSession,
    device: Device,
    for_ts: List[str],
    predictions: dict,
    model_version: str,
):
    forecast = Forecast8h(
        device_id=device.id,
        for_ts=for_ts,
        predictions=predictions,
        model_version=model_version,
    )

    session.add(forecast)


# =========================================================
# USER CRUD OPERATIONS
# =========================================================
async def create_user(
    session: AsyncSession,
    email: str,
    username: str,
    password: str,
    role: str = "user"
) -> User:
    """Create a new user."""
    from auth import get_password_hash
    
    user = User(
        id=username,  # Use username as ID for simplicity
        email=email,
        username=username,
        hashed_password=get_password_hash(password),
        role=role
    )
    session.add(user)
    await session.flush()
    return user

async def get_user_by_username(
    session: AsyncSession,
    username: str
) -> Optional[User]:
    """Get user by username."""
    result = await session.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()

async def get_user_by_email(
    session: AsyncSession,
    email: str
) -> Optional[User]:
    """Get user by email."""
    result = await session.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()

async def update_user_last_login(
    session: AsyncSession,
    user_id: str
):
    """Update user's last login timestamp."""
    from datetime import datetime, timezone
    
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user:
        user.last_login = datetime.now(timezone.utc)
        await session.flush()

async def list_users(session: AsyncSession) -> List[User]:
    """List all users."""
    result = await session.execute(
        select(User).order_by(User.created_at.desc())
    )
    return result.scalars().all()
