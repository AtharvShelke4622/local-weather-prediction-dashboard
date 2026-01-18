from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
import time
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
)

REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

# Application metrics
ACTIVE_USERS = Gauge(
    'active_users_total',
    'Number of currently active users'
)

DEVICE_COUNT = Gauge(
    'devices_total',
    'Total number of registered devices'
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_version', 'status']
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference time in seconds',
    ['model_version'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate (0-1)',
    ['cache_type']
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections'
)

# Error metrics
ERROR_COUNT = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

class MetricsMiddleware:
    """Middleware to collect request metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        request = Request(scope, receive)
        
        # Get request size
        request_size = 0
        if "content-length" in request.headers:
            try:
                request_size = int(request.headers["content-length"])
            except ValueError:
                pass
        
        # Process request
        response_sent = False
        
        async def send_wrapper(message):
            nonlocal response_sent
            if message["type"] == "http.response.start":
                response_sent = True
                status_code = message["status"]
                
                # Record metrics
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=status_code
                ).inc()
                
                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(time.time() - start_time)
                
                REQUEST_SIZE.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(request_size)
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

def update_prediction_metrics(model_version: str, status: str, inference_time: float):
    """Update prediction-related metrics."""
    PREDICTION_COUNT.labels(model_version=model_version, status=status).inc()
    MODEL_INFERENCE_TIME.labels(model_version=model_version).observe(inference_time)

def update_cache_metrics(cache_type: str, hits: int, total: int):
    """Update cache metrics."""
    if total > 0:
        hit_rate = hits / total
        CACHE_HIT_RATE.labels(cache_type=cache_type).set(hit_rate)

def update_device_metrics(device_count: int):
    """Update device metrics."""
    DEVICE_COUNT.set(device_count)

def update_active_users(user_count: int):
    """Update active users metrics."""
    ACTIVE_USERS.set(user_count)

def update_database_connections(connection_count: int):
    """Update database connection metrics."""
    DATABASE_CONNECTIONS.set(connection_count)

def record_error(error_type: str, endpoint: str):
    """Record an error metric."""
    ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint).inc()

async def metrics_endpoint() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    try:
        metrics_data = generate_latest()
        return PlainTextResponse(
            metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        return PlainTextResponse(
            "Error generating metrics",
            status_code=500
        )