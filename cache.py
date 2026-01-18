import json
import pickle
from typing import Optional, Any, Union
from datetime import datetime, timedelta
import aioredis
import structlog
from functools import wraps

logger = structlog.get_logger()

class RedisCache:
    """Redis caching wrapper with serialization support."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
    
    async def connect(self):
        """Initialize Redis connection."""
        try:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False
            )
            await self._redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            self._redis = None
    
    async def disconnect(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._redis:
            return None
        
        try:
            data = await self._redis.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value in cache."""
        if not self._redis:
            return False
        
        try:
            data = pickle.dumps(value)
            if isinstance(expire, timedelta):
                expire = int(expire.total_seconds())
            
            await self._redis.set(key, data, ex=expire)
            return True
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._redis:
            return False
        
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._redis:
            return False
        
        try:
            return bool(await self._redis.exists(key))
        except Exception as e:
            logger.error("Cache exists error", key=key, error=str(e))
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        if not self._redis:
            return 0
        
        try:
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error("Cache clear pattern error", pattern=pattern, error=str(e))
            return 0

# Global cache instance
cache = RedisCache()

def cached(
    key_prefix: str,
    expire: Union[int, timedelta] = timedelta(minutes=5),
    key_builder: Optional[callable] = None
):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = f"{key_prefix}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit", key=cache_key)
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, expire)
            logger.debug("Cache set", key=cache_key)
            
            return result
        return wrapper
    return decorator

# Cache key builders
def device_cache_key(device_id: str, suffix: str = "") -> str:
    """Build cache key for device-specific data."""
    return f"device:{device_id}{suffix}"

def forecast_cache_key(device_id: str, model_version: str = "") -> str:
    """Build cache key for forecast data."""
    return f"forecast:{device_id}:{model_version}"

def user_cache_key(user_id: str, suffix: str = "") -> str:
    """Build cache key for user-specific data."""
    return f"user:{user_id}{suffix}"