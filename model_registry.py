import enum
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Float, DateTime, JSON, ForeignKey, Boolean, Integer, Index

class Base:
    pass

class ModelVersion(Base):
    """Model version tracking and management."""
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(String(1000))
    
    # Model metadata
    algorithm: Mapped[str] = mapped_column(String(100))  # LSTM, LightGBM, etc.
    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSON)
    training_metrics: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    
    # Versioning
    parent_version: Mapped[Optional[str]] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="CURRENT_TIMESTAMP"
    )
    trained_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # File paths
    model_path: Mapped[Optional[str]] = mapped_column(String(500))
    config_path: Mapped[Optional[str]] = mapped_column(String(500))

class Experiment(Base):
    """A/B testing experiment management."""
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[Optional[str]] = mapped_column(String(1000))
    
    # Experiment configuration
    control_model_version: Mapped[str] = mapped_column(String(128))
    test_model_version: Mapped[str] = mapped_column(String(128))
    traffic_split: Mapped[float] = mapped_column(Float, default=0.5)  # 0.0 to 1.0
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    target_metrics: Mapped[List[str]] = mapped_column(JSON)  # List of metrics to track
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="CURRENT_TIMESTAMP"
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Results
    results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    winner: Mapped[Optional[str]] = mapped_column(String(128))  # "control", "test", or "inconclusive"

class PredictionLog(Base):
    """Prediction logging for A/B testing and analytics."""
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Request info
    device_id: Mapped[str] = mapped_column(String(128), index=True)
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("experiments.id"))
    variant: Mapped[Optional[str]] = mapped_column(String(50))  # "control" or "test"
    
    # Prediction details
    input_features: Mapped[Dict[str, Any]] = mapped_column(JSON)
    predictions: Mapped[Dict[str, List[float]]] = mapped_column(JSON)
    prediction_time_ms: Mapped[float] = mapped_column(Float)
    
    # Metadata
    request_id: Mapped[Optional[str]] = mapped_column(String(128))
    user_id: Mapped[Optional[str]] = mapped_column(String(128))
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="CURRENT_TIMESTAMP"
    )
    
    __table_args__ = (
        Index("idx_device_model_ts", "device_id", "model_version", "created_at"),
        Index("idx_experiment_variant", "experiment_id", "variant"),
    )

class ModelMetrics(Base):
    """Model performance metrics tracking."""
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    metric_name: Mapped[str] = mapped_column(String(100))
    metric_value: Mapped[float] = mapped_column(Float)
    
    # Context
    device_id: Mapped[Optional[str]] = mapped_column(String(128))
    time_window: Mapped[str] = mapped_column(String(50))  # "1h", "24h", "7d", etc.
    
    # Timestamps
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="CURRENT_TIMESTAMP"
    )
    
    __table_args__ = (
        Index("idx_model_metric_time", "model_version", "metric_name", "calculated_at"),
    )