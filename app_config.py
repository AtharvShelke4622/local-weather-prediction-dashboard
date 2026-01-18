from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "Local Weather Prediction Dashboard API"
    ENV: str = "development"
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    DATABASE_URL: str
    MODEL_SERVER_URL: str = "http://localhost:8001"
    REDIS_URL: str = "redis://localhost:6379"
    SECRET_KEY: str = "your-secret-key-change-in-production"

    POLL_MS: int = 300_000

    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://frontend-five-nu-11.vercel.app",
        "https://*.vercel.app",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )

settings = Settings()
