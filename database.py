from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from app_config import settings
from db_models import Base
from fastapi import Depends

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def ping_db():
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

async def get_session() -> AsyncSession:
    """Get database session dependency."""
    async with AsyncSessionLocal() as session:
        yield session
