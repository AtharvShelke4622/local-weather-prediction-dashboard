from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, DateTime, JSON, ForeignKey, Index, text, Boolean
from typing import Optional
from datetime import datetime


class Base(DeclarativeBase):
    pass


# =====================================================
# USER
# =====================================================
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String(50), default="user")
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )


# =====================================================
# DEVICE
# =====================================================
class Device(Base):
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    last_seen: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    readings = relationship(
        "SensorReading",
        back_populates="device",
        cascade="all, delete-orphan"
    )


# =====================================================
# SENSOR READING
# =====================================================
class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    device_id: Mapped[int] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"),
        index=True
    )

    # 🔑 ML-friendly denormalized key
    device_key: Mapped[str] = mapped_column(String(128), index=True)

    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP")
    )

    temperature: Mapped[Optional[float]] = mapped_column(Float)
    humidity: Mapped[Optional[float]] = mapped_column(Float)
    wind_speed: Mapped[Optional[float]] = mapped_column(Float)
    radiation: Mapped[Optional[float]] = mapped_column(Float)
    precipitation: Mapped[Optional[float]] = mapped_column(Float)

    raw: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    device = relationship("Device", back_populates="readings")

    __table_args__ = (
        Index("idx_device_ts", "device_id", "ts"),
        Index("idx_devicekey_ts", "device_key", "ts"),
    )


# =====================================================
# FORECAST
# =====================================================
class Forecast8h(Base):
    __tablename__ = "forecast_8h"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    device_id: Mapped[int] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"),
        index=True
    )

    pred_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP")
    )

    for_ts: Mapped[list] = mapped_column(JSON)
    predictions: Mapped[dict] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(255))
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
