import asyncio
import schedule
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import structlog
import os
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from database import AsyncSessionLocal
from db_models import SensorReading, Device, Forecast8h
from model_management import model_registry, performance_tracker
from training.train_baseline import (
    create_sequences,
    load_lstm,
    train_lstm,
    train_lightgbm_residuals,
    post_process_dynamics
)
from logging_config import logger

class RetrainingPipeline:
    """Automated ML model retraining pipeline."""
    
    def __init__(self):
        self.min_data_points = 1000  # Minimum data points for retraining
        self.retraining_interval_days = 7  # Retrain weekly
        self.performance_threshold = 0.1  # 10% degradation threshold
        self.models_dir = "backend/models"
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
    
    async def check_retraining_triggers(self) -> bool:
        """Check if retraining should be triggered."""
        async with AsyncSessionLocal() as session:
            # Check if enough new data
            week_ago = datetime.now(timezone.utc) - timedelta(days=self.retraining_interval_days)
            
            result = await session.execute(
                select(SensorReading)
                .where(SensorReading.ts >= week_ago)
            )
            new_data_count = len(result.scalars().all())
            
            if new_data_count < self.min_data_points:
                logger.info("Insufficient data for retraining", 
                          new_data_count=new_data_count, 
                          required=self.min_data_points)
                return False
            
            # Check model performance
            latest_forecast = await session.execute(
                select(Forecast8h)
                .order_by(Forecast8h.pred_ts.desc())
                .limit(1)
            )
            forecast = latest_forecast.scalar_one_or_none()
            
            if forecast and forecast.metrics:
                mae = forecast.metrics.get('mae', 0)
                if mae > self.performance_threshold:
                    logger.info("Performance threshold exceeded", mae=mae)
                    return True
            
            # Check if it's time for scheduled retraining
            last_training = await self._get_last_training_date(session)
            if last_training:
                days_since_training = (datetime.now(timezone.utc) - last_training).days
                if days_since_training >= self.retraining_interval_days:
                    logger.info("Scheduled retraining due", days_since=training)
                    return True
            
            return True  # Default to retraining for demo purposes
    
    async def _get_last_training_date(self, session: AsyncSession) -> Optional[datetime]:
        """Get the date of last successful training."""
        # This would typically be stored in a separate training log table
        # For now, return None to trigger training
        return None
    
    async def collect_training_data(self, session: AsyncSession) -> pd.DataFrame:
        """Collect and preprocess training data."""
        # Get all sensor readings
        result = await session.execute(
            select(SensorReading, Device)
            .join(Device, SensorReading.device_id == Device.id)
            .order_by(SensorReading.ts)
        )
        
        data = []
        for reading, device in result:
            data.append({
                'device_id': device.device_id,
                'ts': reading.ts,
                'temperature': reading.temperature,
                'humidity': reading.humidity,
                'wind_speed': reading.wind_speed,
                'radiation': reading.radiation,
                'precipitation': reading.precipitation,
                'lat': device.lat,
                'lon': device.lon
            })
        
        df = pd.DataFrame(data)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        logger.info("Training data collected", 
                  rows=len(df), 
                  devices=df['device_id'].nunique())
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better model performance."""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['ts'].dt.hour
        df['day_of_week'] = df['ts'].dt.dayofweek
        df['month'] = df['ts'].dt.month
        df['season'] = df['month'].apply(self._get_season)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        for col in ['temperature', 'humidity', 'wind_speed', 'radiation', 'precipitation']:
            for lag in [1, 3, 6, 12]:  # 1, 3, 6, 12 hours ago
                df[f'{col}_lag_{lag}'] = df.groupby('device_id')[col].shift(lag)
        
        # Rolling statistics
        for col in ['temperature', 'humidity', 'wind_speed']:
            for window in [3, 6, 12]:  # 3, 6, 12 hour windows
                df[f'{col}_rolling_mean_{window}'] = (
                    df.groupby('device_id')[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                df[f'{col}_rolling_std_{window}'] = (
                    df.groupby('device_id')[col]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
        
        # Weather interaction features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        df['wind_radiation_interaction'] = df['wind_speed'] * df['radiation']
        
        # Weather indices
        df['heat_index'] = self._calculate_heat_index(
            df['temperature'], df['humidity']
        )
        df['wind_chill'] = self._calculate_wind_chill(
            df['temperature'], df['wind_speed']
        )
        
        # Fill NaN values
        df = df.fillna(method='forward').fillna(0)
        
        return df
    
    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index (simplified)."""
        # Simplified heat index calculation
        return temp + (0.5 * humidity) - 10
    
    def _calculate_wind_chill(self, temp: pd.Series, wind: pd.Series) -> pd.Series:
        """Calculate wind chill (simplified)."""
        # Simplified wind chill calculation
        return temp - (0.2 * wind)
    
    async def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM and LightGBM models."""
        logger.info("Starting model training...")
        
        # Prepare data
        feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'radiation', 'precipitation',
            'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
            'month_sin', 'month_cos', 'temp_humidity_interaction',
            'wind_radiation_interaction', 'heat_index', 'wind_chill'
        ]
        
        # Add lag and rolling features if they exist
        lag_cols = [col for col in df.columns if 'lag_' in col]
        rolling_cols = [col for col in df.columns if 'rolling_' in col]
        feature_columns.extend(lag_cols + rolling_cols)
        
        # Create sequences for LSTM
        X, y = create_sequences(df[feature_columns + ['device_id']], sequence_length=24)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train LSTM
        lstm_model = await self._train_lstm_async(X_train, y_train)
        
        # Train LightGBM for residuals
        lgb_model = await self._train_lightgbm_async(X_train, y_train, lstm_model)
        
        # Evaluate models
        lstm_metrics = self._evaluate_model(lstm_model, X_test, y_test)
        lgb_metrics = self._evaluate_lightgbm(lgb_model, X_test, y_test, lstm_model)
        
        # Save models
        model_version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        lstm_path = f"{self.models_dir}/lstm_{model_version}.pt"
        lgb_path = f"{self.models_dir}/lgb_{model_version}.txt"
        
        torch.save(lstm_model.state_dict(), lstm_path)
        lgb_model.booster_.save_model(lgb_path)
        
        training_metrics = {
            'lstm': lstm_metrics,
            'lightgbm': lgb_metrics,
            'combined': {
                'mae': min(lstm_metrics['mae'], lgb_metrics['mae']),
                'rmse': min(lstm_metrics['rmse'], lgb_metrics['rmse'])
            }
        }
        
        logger.info("Model training completed", 
                  model_version=model_version,
                  metrics=training_metrics)
        
        return {
            'model_version': model_version,
            'lstm_path': lstm_path,
            'lgb_path': lgb_path,
            'metrics': training_metrics,
            'hyperparameters': {
                'lstm': {'hidden_units': 64, 'num_layers': 2},
                'lightgbm': {'n_estimators': 100, 'learning_rate': 0.1}
            }
        }
    
    async def _train_lstm_async(self, X_train: np.ndarray, y_train: np.ndarray):
        """Async wrapper for LSTM training."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._train_lstm_sync, X_train, y_train
        )
    
    def _train_lstm_sync(self, X_train: np.ndarray, y_train: np.ndarray):
        """Synchronous LSTM training."""
        return train_lstm(X_train, y_train)
    
    async def _train_lightgbm_async(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        lstm_model
    ):
        """Async wrapper for LightGBM training."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._train_lightgbm_sync, X_train, y_train, lstm_model
        )
    
    def _train_lightgbm_sync(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        lstm_model
    ):
        """Synchronous LightGBM training."""
        return train_lightgbm_residuals(X_train, y_train, lstm_model)
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        model.eval()
        with torch.no_grad():
            predictions = model(torch.FloatTensor(X_test))
            predictions = predictions.numpy()
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        return {'mae': mae, 'rmse': rmse}
    
    def _evaluate_lightgbm(
        self, 
        lgb_model, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        lstm_model
    ) -> Dict[str, float]:
        """Evaluate LightGBM model performance."""
        # Get LSTM predictions
        lstm_model.eval()
        with torch.no_grad():
            lstm_preds = lstm_model(torch.FloatTensor(X_test)).numpy()
        
        # Get LightGBM residuals
        lgb_preds = lgb_model.predict(X_test)
        
        # Combine predictions
        combined_preds = lstm_preds + lgb_preds
        
        mae = mean_absolute_error(y_test, combined_preds)
        rmse = np.sqrt(mean_squared_error(y_test, combined_preds))
        
        return {'mae': mae, 'rmse': rmse}
    
    async def deploy_model(self, training_results: Dict[str, Any]):
        """Deploy newly trained model."""
        async with AsyncSessionLocal() as session:
            # Register new model version
            model_version = await model_registry.register_model(
                session=session,
                version=training_results['model_version'],
                name="Weather Prediction Model",
                algorithm="LSTM+LightGBM",
                hyperparameters=training_results['hyperparameters'],
                model_path=training_results['lstm_path'],
                description=f"Automatically trained model on {datetime.now(timezone.utc)}"
            )
            
            # Activate model
            await model_registry.activate_model(
                session=session,
                version=training_results['model_version'],
                is_production=True
            )
            
            # Record metrics
            await performance_tracker.record_metrics(
                session=session,
                model_version=training_results['model_version'],
                metrics=training_results['metrics']['combined'],
                time_window="training"
            )
            
            await session.commit()
            
            logger.info("Model deployed successfully", 
                      model_version=training_results['model_version'])
    
    async def run_retraining_pipeline(self):
        """Run the complete retraining pipeline."""
        try:
            # Check if retraining should be triggered
            if not await self.check_retraining_triggers():
                return
            
            logger.info("Starting retraining pipeline...")
            
            # Collect training data
            async with AsyncSessionLocal() as session:
                df = await self.collect_training_data(session)
            
            if len(df) < self.min_data_points:
                logger.warning("Insufficient data for retraining")
                return
            
            # Train models
            training_results = await self.train_models(df)
            
            # Deploy model
            await self.deploy_model(training_results)
            
            logger.info("Retraining pipeline completed successfully")
            
        except Exception as e:
            logger.error("Retraining pipeline failed", error=str(e))
    
    def start_scheduler(self):
        """Start the retraining scheduler."""
        # Schedule daily check
        schedule.every().day.at("02:00").do(
            lambda: asyncio.run(self.run_retraining_pipeline())
        )
        
        logger.info("Retraining scheduler started")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# Global pipeline instance
retraining_pipeline = RetrainingPipeline()

if __name__ == "__main__":
    # Run retraining pipeline
    asyncio.run(retraining_pipeline.run_retraining_pipeline())