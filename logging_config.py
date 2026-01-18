import structlog
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from functools import wraps
from fastapi import Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Get logger instance
logger = structlog.get_logger()

class AuditLogger:
    """Audit logging for security and compliance."""
    
    def __init__(self):
        self.audit_logger = structlog.get_logger("audit")
    
    def log_authentication(
        self,
        user_id: str,
        action: str,
        success: bool,
        ip_address: str,
        user_agent: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log authentication events."""
        self.audit_logger.info(
            "authentication_event",
            user_id=user_id,
            action=action,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details or {}
        )
    
    def log_api_access(
        self,
        user_id: Optional[str],
        endpoint: str,
        method: str,
        status_code: int,
        ip_address: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log API access events."""
        self.audit_logger.info(
            "api_access",
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            ip_address=ip_address,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details or {}
        )
    
    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        ip_address: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log data access events."""
        self.audit_logger.info(
            "data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details or {}
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        user_id: Optional[str],
        ip_address: str,
        details: Dict[str, Any]
    ):
        """Log security events."""
        self.audit_logger.warning(
            "security_event",
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details
        )
    
    def log_system_event(
        self,
        event_type: str,
        severity: str,
        component: str,
        details: Dict[str, Any]
    ):
        """Log system events."""
        self.audit_logger.info(
            "system_event",
            event_type=event_type,
            severity=severity,
            component=component,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details
        )

# Global audit logger instance
audit_logger = AuditLogger()

class ErrorHandler:
    """Centralized error handling and logging."""
    
    @staticmethod
    def log_error(
        error: Exception,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        request: Optional[Request] = None
    ):
        """Log error with context."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if request:
            error_info.update({
                "endpoint": request.url.path,
                "method": request.method,
                "ip_address": ErrorHandler._get_client_ip(request)
            })
        
        logger.error("application_error", **error_info)
        
        # Log security-related errors
        if isinstance(error, (HTTPException, PermissionError)):
            audit_logger.log_security_event(
                event_type="application_error",
                severity="medium",
                user_id=user_id,
                ip_address=ErrorHandler._get_client_ip(request) if request else "unknown",
                details=error_info
            )
    
    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded IP
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to client IP
        return request.client.host if request.client else "unknown"

def error_boundary(func):
    """Decorator for error handling and logging."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now(timezone.utc)
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            # Extract context information
            context = {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs": list(kwargs.keys())
            }
            
            # Try to extract user information
            user_id = None
            request = None
            for arg in args:
                if hasattr(arg, 'user') and hasattr(arg.user, 'id'):
                    user_id = arg.user.id
                if hasattr(arg, 'url') and hasattr(arg, 'method'):
                    request = arg
            
            # Log the error
            ErrorHandler.log_error(e, context, user_id, request)
            
            # Re-raise the exception
            raise
        finally:
            # Log performance metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                "function_performance",
                function=func.__name__,
                duration_ms=duration,
                success=e is None if 'e' in locals() else True
            )
    
    return wrapper

@asynccontextmanager
async def database_transaction(session: AsyncSession, operation_name: str):
    """Context manager for database transactions with logging."""
    start_time = datetime.now(timezone.utc)
    try:
        logger.info("database_transaction_start", operation=operation_name)
        yield session
        await session.commit()
        logger.info(
            "database_transaction_success",
            operation=operation_name,
            duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    except Exception as e:
        await session.rollback()
        logger.error(
            "database_transaction_error",
            operation=operation_name,
            error=str(e),
            duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
        raise

def setup_logging():
    """Setup logging configuration."""
    # Configure standard library logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("weather_dashboard.log")
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)