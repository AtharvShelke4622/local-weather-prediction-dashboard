from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
import structlog

logger = structlog.get_logger()

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom rate limit exceeded handler."""
    logger.warning(
        "Rate limit exceeded",
        client_ip=get_remote_address(request),
        endpoint=request.url.path,
        limit=exc.detail
    )
    return Response(
        content={"error": "Rate limit exceeded", "detail": str(exc.detail)},
        status_code=429,
        media_type="application/json"
    )