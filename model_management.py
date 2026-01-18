import asyncio
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
import json
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from model_registry import ModelVersion, Experiment, PredictionLog, ModelMetrics
from cache import cache, experiment_cache_key
from metrics import update_prediction_metrics

logger = structlog.get_logger()

class ModelRegistry:
    """Centralized model management and versioning."""
    
    def __init__(self):
        self._active_models = {}
        self._model_cache = {}
    
    async def register_model(
        self,
        session: AsyncSession,
        version: str,
        name: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        model_path: str,
        description: Optional[str] = None
    ) -> ModelVersion:
        """Register a new model version."""
        
        model_version = ModelVersion(
            version=version,
            name=name,
            description=description,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            model_path=model_path,
            created_at=datetime.now(timezone.utc)
        )
        
        session.add(model_version)
        await session.flush()
        
        # Cache model info
        cache_key = f"model:{version}"
        await cache.set(cache_key, {
            'version': version,
            'name': name,
            'algorithm': algorithm,
            'hyperparameters': hyperparameters,
            'model_path': model_path
        }, expire=timedelta(hours=24))
        
        logger.info("Model registered", version=version, name=name, algorithm=algorithm)
        return model_version
    
    async def activate_model(
        self,
        session: AsyncSession,
        version: str,
        is_production: bool = False
    ):
        """Activate a model version."""
        
        # Deactivate current active model
        await session.execute(
            update(ModelVersion)
            .where(ModelVersion.algorithm == (
                select(ModelVersion.algorithm)
                .where(ModelVersion.version == version)
                .scalar_subquery()
            ))
            .where(ModelVersion.is_active == True)
            .values(is_active=False)
        )
        
        # Activate new model
        await session.execute(
            update(ModelVersion)
            .where(ModelVersion.version == version)
            .values(
                is_active=True,
                is_production=is_production,
                deployed_at=datetime.now(timezone.utc)
            )
        )
        
        await session.flush()
        
        # Update cache
        cache_key = f"active_model:{algorithm}" if 'algorithm' in locals() else f"active_model:{version}"
        await cache.set(cache_key, version, expire=timedelta(hours=24))
        
        logger.info("Model activated", version=version, is_production=is_production)
    
    async def get_active_model(
        self,
        session: AsyncSession,
        algorithm: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Get the active model for an algorithm."""
        
        result = await session.execute(
            select(ModelVersion)
            .where(ModelVersion.is_active == True)
            .where(ModelVersion.algorithm == algorithm if algorithm else True)
            .order_by(ModelVersion.deployed_at.desc())
            .limit(1)
        )
        
        return result.scalar_one_or_none()

class ABTestManager:
    """A/B testing framework for models."""
    
    def __init__(self):
        self._experiments = {}
    
    async def create_experiment(
        self,
        session: AsyncSession,
        name: str,
        control_model: str,
        test_model: str,
        traffic_split: float = 0.5,
        target_metrics: List[str] = None
    ) -> Experiment:
        """Create a new A/B test experiment."""
        
        experiment = Experiment(
            name=name,
            control_model_version=control_model,
            test_model_version=test_model,
            traffic_split=traffic_split,
            target_metrics=target_metrics or ["accuracy", "latency"],
            created_at=datetime.now(timezone.utc)
        )
        
        session.add(experiment)
        await session.flush()
        
        # Cache experiment config
        cache_key = experiment_cache_key(name)
        await cache.set(cache_key, {
            'control_model': control_model,
            'test_model': test_model,
            'traffic_split': traffic_split
        }, expire=timedelta(days=30))
        
        logger.info("A/B test created", name=name, traffic_split=traffic_split)
        return experiment
    
    async def get_variant(
        self,
        session: AsyncSession,
        experiment_name: str,
        device_id: str
    ) -> Tuple[str, str]:
        """Get variant for a device in an experiment."""
        
        # Check cache first
        cache_key = f"variant:{experiment_name}:{device_id}"
        cached_variant = await cache.get(cache_key)
        if cached_variant:
            return cached_variant['variant'], cached_variant['model_version']
        
        # Get experiment from database
        result = await session.execute(
            select(Experiment)
            .where(Experiment.name == experiment_name)
            .where(Experiment.is_active == True)
        )
        
        experiment = result.scalar_one_or_none()
        if not experiment:
            return "control", None
        
        # Use consistent hashing for device assignment
        hash_value = hash(device_id) % 100
        traffic_split_percentage = int(experiment.traffic_split * 100)
        
        if hash_value < traffic_split_percentage:
            variant = "test"
            model_version = experiment.test_model_version
        else:
            variant = "control"
            model_version = experiment.control_model_version
        
        # Cache the assignment
        await cache.set(cache_key, {
            'variant': variant,
            'model_version': model_version
        }, expire=timedelta(hours=24))
        
        return variant, model_version
    
    async def log_prediction(
        self,
        session: AsyncSession,
        device_id: str,
        model_version: str,
        input_features: Dict[str, Any],
        predictions: Dict[str, List[float]],
        prediction_time_ms: float,
        experiment_id: Optional[int] = None,
        variant: Optional[str] = None
    ):
        """Log prediction for analytics."""
        
        prediction_log = PredictionLog(
            device_id=device_id,
            model_version=model_version,
            experiment_id=experiment_id,
            variant=variant,
            input_features=input_features,
            predictions=predictions,
            prediction_time_ms=prediction_time_ms,
            created_at=datetime.now(timezone.utc)
        )
        
        session.add(prediction_log)
        
        # Update metrics
        update_prediction_metrics(model_version, "success", prediction_time_ms)
    
    async def calculate_experiment_results(
        self,
        session: AsyncSession,
        experiment_id: int,
        window_days: int = 7
    ) -> Dict[str, Any]:
        """Calculate experiment results."""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=window_days)
        
        # Get predictions for the experiment
        result = await session.execute(
            select(PredictionLog)
            .where(PredictionLog.experiment_id == experiment_id)
            .where(PredictionLog.created_at >= cutoff_date)
        )
        
        predictions = result.scalars().all()
        
        # Group by variant
        control_predictions = [p for p in predictions if p.variant == "control"]
        test_predictions = [p for p in predictions if p.variant == "test"]
        
        # Calculate metrics
        control_metrics = self._calculate_metrics(control_predictions)
        test_metrics = self._calculate_metrics(test_predictions)
        
        # Determine winner (simplified - in reality would be more complex)
        winner = "inconclusive"
        if test_metrics.get('accuracy', 0) > control_metrics.get('accuracy', 0) + 0.02:  # 2% improvement threshold
            winner = "test"
        elif control_metrics.get('accuracy', 0) > test_metrics.get('accuracy', 0) + 0.02:
            winner = "control"
        
        results = {
            'control': control_metrics,
            'test': test_metrics,
            'winner': winner,
            'sample_size': {
                'control': len(control_predictions),
                'test': len(test_predictions)
            }
        }
        
        # Update experiment
        await session.execute(
            update(Experiment)
            .where(Experiment.id == experiment_id)
            .values(results=results, winner=winner)
        )
        
        return results
    
    def _calculate_metrics(self, predictions: List[PredictionLog]) -> Dict[str, float]:
        """Calculate metrics for a set of predictions."""
        if not predictions:
            return {}
        
        # Simplified metrics calculation
        avg_latency = sum(p.prediction_time_ms for p in predictions) / len(predictions)
        
        # In reality, you'd calculate accuracy based on actual vs predicted values
        # For now, returning placeholder
        accuracy = random.uniform(0.85, 0.95)  # Placeholder
        
        return {
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency,
            'prediction_count': len(predictions)
        }

class ModelPerformanceTracker:
    """Track model performance over time."""
    
    async def record_metrics(
        self,
        session: AsyncSession,
        model_version: str,
        metrics: Dict[str, float],
        device_id: Optional[str] = None,
        time_window: str = "1h"
    ):
        """Record model performance metrics."""
        
        timestamp = datetime.now(timezone.utc)
        
        for metric_name, metric_value in metrics.items():
            model_metric = ModelMetrics(
                model_version=model_version,
                metric_name=metric_name,
                metric_value=metric_value,
                device_id=device_id,
                time_window=time_window,
                calculated_at=timestamp
            )
            
            session.add(model_metric)
        
        logger.info("Metrics recorded", model_version=model_version, metrics=metrics)
    
    async def get_model_metrics(
        self,
        session: AsyncSession,
        model_version: str,
        metric_name: str,
        hours: int = 24
    ) -> List[ModelMetrics]:
        """Get model metrics over time."""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        result = await session.execute(
            select(ModelMetrics)
            .where(ModelMetrics.model_version == model_version)
            .where(ModelMetrics.metric_name == metric_name)
            .where(ModelMetrics.calculated_at >= cutoff_time)
            .order_by(ModelMetrics.calculated_at)
        )
        
        return result.scalars().all()

# Global instances
model_registry = ModelRegistry()
ab_test_manager = ABTestManager()
performance_tracker = ModelPerformanceTracker()