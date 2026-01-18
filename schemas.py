from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

class IngestPayload(BaseModel):
    device_id: str
    ts: Optional[datetime] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    radiation: Optional[float] = None
    precipitation: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

class DeviceOut(BaseModel):
    device_id: str
    last_seen: Optional[datetime]

class LatestOut(BaseModel):
    device_id: str
    ts: datetime
    temperature: Optional[float]
    humidity: Optional[float]
    wind_speed: Optional[float]
    radiation: Optional[float]
    precipitation: Optional[float]

class ForecastOut(BaseModel):
    device_id: str
    pred_ts: datetime
    for_ts: List[str]
    predictions: Dict[str, List[float]]
    model_version: str

class PredictRequestToModel(BaseModel):
    device_id: str
    recent_window: List[List[float]] = Field(..., description="Rows ordered oldest->newest with [temperature, humidity, wind_speed, radiation, precipitation]")

# =========================================================
# AUTHENTICATION SCHEMAS
# =========================================================
class UserBase(BaseModel):
    email: EmailStr
    username: str
    role: Optional[str] = "user"

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

class UserOut(UserBase):
    id: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    user_id: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(UserCreate):
    confirm_password: str
