"""
Performance Prediction System for Computer Assistant

This module provides performance prediction capabilities including:
- System performance forecasting
- Resource usage prediction
- Response time estimation
- Load balancing optimization
- Capacity planning
- Anomaly detection
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class PredictionType(Enum):
    """Types of predictions"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_USAGE = "network_usage"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LOAD = "load"
    CAPACITY = "capacity"
    ANOMALY = "anomaly"


class PredictionHorizon(Enum):
    """Prediction time horizons"""
    SHORT_TERM = "short_term"  # Next few minutes
    MEDIUM_TERM = "medium_term"  # Next hour
    LONG_TERM = "long_term"  # Next day/week


class PredictionStatus(Enum):
    """Prediction status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(Enum):
    """Prediction model types"""
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


@dataclass
class MetricData:
    """Performance metric data point"""
    timestamp: datetime
    metric_type: str
    value: float
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric_type': self.metric_type,
            'value': self.value,
            'context': self.context,
            'metadata': self.metadata
        }


@dataclass
class PredictionResult:
    """Prediction result"""
    id: str
    prediction_type: PredictionType
    horizon: PredictionHorizon
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    timestamp: datetime
    target_time: datetime
    model_used: ModelType
    input_features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'prediction_type': self.prediction_type.value,
            'horizon': self.horizon.value,
            'predicted_value': self.predicted_value,
            'confidence_interval': list(self.confidence_interval),
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat(),
            'target_time': self.target_time.isoformat(),
            'model_used': self.model_used.value,
            'input_features': self.input_features,
            'metadata': self.metadata
        }


@dataclass
class PredictionModel:
    """Prediction model"""
    id: str
    name: str
    model_type: ModelType
    prediction_type: PredictionType
    parameters: Dict[str, Any]
    training_data_size: int = 0
    accuracy: float = 0.0
    last_trained: Optional[datetime] = None
    last_used: Optional[datetime] = None
    prediction_count: int = 0
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type.value,
            'prediction_type': self.prediction_type.value,
            'parameters': self.parameters,
            'training_data_size': self.training_data_size,
            'accuracy': self.accuracy,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'prediction_count': self.prediction_count,
            'error_rate': self.error_rate,
            'metadata': self.metadata
        }


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    id: str
    timestamp: datetime
    metric_type: str
    actual_value: float
    expected_value: float
    anomaly_score: float
    severity: str  # low, medium, high, critical
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'metric_type': self.metric_type,
            'actual_value': self.actual_value,
            'expected_value': self.expected_value,
            'anomaly_score': self.anomaly_score,
            'severity': self.severity,
            'description': self.description,
            'context': self.context
        }


class PerformancePredictor:
    """
    Performance prediction system for forecasting and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance predictor"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.metric_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.prediction_models: Dict[str, PredictionModel] = {}
        self.prediction_history: deque = deque(maxlen=1000)
        self.anomaly_history: deque = deque(maxlen=500)
        
        # Configuration
        self.prediction_dir = Path(self.config.get('prediction_dir', 'performance_predictions'))
        self.prediction_dir.mkdir(exist_ok=True)
        
        # Prediction settings
        self.min_data_points = self.config.get('min_data_points', 50)
        self.prediction_intervals = {
            PredictionHorizon.SHORT_TERM: timedelta(minutes=5),
            PredictionHorizon.MEDIUM_TERM: timedelta(hours=1),
            PredictionHorizon.LONG_TERM: timedelta(days=1)
        }
        
        # Model settings
        self.auto_retrain = self.config.get('auto_retrain', True)
        self.retrain_threshold = self.config.get('retrain_threshold', 0.1)  # 10% accuracy drop
        self.ensemble_models = self.config.get('ensemble_models', True)
        
        # Anomaly detection settings
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.0)  # Standard deviations
        self.anomaly_detection_enabled = self.config.get('anomaly_detection', True)
        
        # Performance tracking
        self.performance_stats = {
            'predictions_made': 0,
            'accurate_predictions': 0,
            'anomalies_detected': 0,
            'models_trained': 0,
            'average_accuracy': 0.0,
            'average_prediction_time': 0.0,
            'daily_stats': {}
        }
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Prediction lock for thread safety
        self.prediction_lock = threading.Lock()
        
        # Initialize built-in models
        self._initialize_models()
        
        # Load existing data
        self._load_prediction_data()
        
        self.logger.info("Performance Predictor initialized")
    
    def _initialize_models(self) -> None:
        """Initialize built-in prediction models"""
        try:
            # Response time prediction models
            self._create_model(
                "response_time_linear",
                ModelType.LINEAR_REGRESSION,
                PredictionType.RESPONSE_TIME,
                {'learning_rate': 0.01, 'regularization': 0.001}
            )
            
            self._create_model(
                "response_time_ma",
                ModelType.MOVING_AVERAGE,
                PredictionType.RESPONSE_TIME,
                {'window_size': 20, 'weights': 'exponential'}
            )
            
            # Memory usage prediction models
            self._create_model(
                "memory_usage_polynomial",
                ModelType.POLYNOMIAL_REGRESSION,
                PredictionType.MEMORY_USAGE,
                {'degree': 2, 'regularization': 0.01}
            )
            
            self._create_model(
                "memory_usage_smoothing",
                ModelType.EXPONENTIAL_SMOOTHING,
                PredictionType.MEMORY_USAGE,
                {'alpha': 0.3, 'beta': 0.1, 'gamma': 0.1}
            )
            
            # CPU usage prediction models
            self._create_model(
                "cpu_usage_arima",
                ModelType.ARIMA,
                PredictionType.CPU_USAGE,
                {'p': 2, 'd': 1, 'q': 2}
            )
            
            # Error rate prediction models
            self._create_model(
                "error_rate_ensemble",
                ModelType.ENSEMBLE,
                PredictionType.ERROR_RATE,
                {'models': ['linear', 'moving_average'], 'weights': [0.6, 0.4]}
            )
            
            # Throughput prediction models
            self._create_model(
                "throughput_lstm",
                ModelType.LSTM,
                PredictionType.THROUGHPUT,
                {'hidden_size': 50, 'num_layers': 2, 'sequence_length': 10}
            )
            
            self.logger.info("Built-in prediction models initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
    
    def _create_model(self, name: str, model_type: ModelType,
                     prediction_type: PredictionType, parameters: Dict[str, Any]) -> str:
        """Create a new prediction model"""
        try:
            model_id = str(uuid.uuid4())
            
            model = PredictionModel(
                id=model_id,
                name=name,
                model_type=model_type,
                prediction_type=prediction_type,
                parameters=parameters
            )
            
            self.prediction_models[model_id] = model
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            raise
    
    def _load_prediction_data(self) -> None:
        """Load existing prediction data"""
        try:
            # Load models
            models_file = self.prediction_dir / "models.json"
            if models_file.exists():
                with open(models_file, 'r') as f:
                    models_data = json.load(f)
                
                for model_data in models_data:
                    model = PredictionModel(
                        id=model_data['id'],
                        name=model_data['name'],
                        model_type=ModelType(model_data['model_type']),
                        prediction_type=PredictionType(model_data['prediction_type']),
                        parameters=model_data['parameters'],
                        training_data_size=model_data.get('training_data_size', 0),
                        accuracy=model_data.get('accuracy', 0.0),
                        last_trained=datetime.fromisoformat(model_data['last_trained']) if model_data.get('last_trained') else None,
                        last_used=datetime.fromisoformat(model_data['last_used']) if model_data.get('last_used') else None,
                        prediction_count=model_data.get('prediction_count', 0),
                        error_rate=model_data.get('error_rate', 0.0),
                        metadata=model_data.get('metadata', {})
                    )
                    self.prediction_models[model.id] = model
            
            # Load performance stats
            stats_file = self.prediction_dir / "performance_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.performance_stats.update(json.load(f))
            
            # Load recent metric data
            data_file = self.prediction_dir / "recent_metrics.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    metrics_data = json.load(f)
                
                for metric_type, data_points in metrics_data.items():
                    for point_data in data_points[-1000:]:  # Load last 1000 points
                        metric = MetricData(
                            timestamp=datetime.fromisoformat(point_data['timestamp']),
                            metric_type=point_data['metric_type'],
                            value=point_data['value'],
                            context=point_data.get('context', {}),
                            metadata=point_data.get('metadata', {})
                        )
                        self.metric_data[metric_type].append(metric)
            
            self.logger.info(f"Loaded {len(self.prediction_models)} models and metric data")
            
        except Exception as e:
            self.logger.error(f"Error loading prediction data: {e}")
    
    async def save_prediction_data(self) -> None:
        """Save prediction data to disk"""
        try:
            # Save models
            models_file = self.prediction_dir / "models.json"
            with open(models_file, 'w') as f:
                models_data = [model.to_dict() for model in self.prediction_models.values()]
                json.dump(models_data, f, indent=2)
            
            # Save performance stats
            stats_file = self.prediction_dir / "performance_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.performance_stats, f, indent=2)
            
            # Save recent metric data
            data_file = self.prediction_dir / "recent_metrics.json"
            with open(data_file, 'w') as f:
                metrics_data = {}
                for metric_type, data_points in self.metric_data.items():
                    metrics_data[metric_type] = [point.to_dict() for point in list(data_points)[-500:]]  # Save last 500 points
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info("Prediction data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving prediction data: {e}")
    
    async def add_metric_data(self, metric_type: str, value: float,
                             context: Optional[Dict[str, Any]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add new metric data point"""
        try:
            metric = MetricData(
                timestamp=datetime.now(),
                metric_type=metric_type,
                value=value,
                context=context or {},
                metadata=metadata or {}
            )
            
            self.metric_data[metric_type].append(metric)
            
            # Update daily stats
            today = datetime.now().date().isoformat()
            if today not in self.performance_stats['daily_stats']:
                self.performance_stats['daily_stats'][today] = {'metrics_added': 0}
            self.performance_stats['daily_stats'][today]['metrics_added'] += 1
            
            # Anomaly detection
            if self.anomaly_detection_enabled:
                await self._detect_anomaly(metric)
            
            # Auto-retrain models if needed
            if self.auto_retrain:
                await self._check_retrain_models(metric_type)
            
            self.logger.debug(f"Added metric data: {metric_type} = {value}")
            
        except Exception as e:
            self.logger.error(f"Error adding metric data: {e}")
            raise
    
    async def predict(self, prediction_type: PredictionType,
                     horizon: PredictionHorizon,
                     context: Optional[Dict[str, Any]] = None,
                     model_id: Optional[str] = None) -> PredictionResult:
        """Make a prediction"""
        try:
            start_time = time.time()
            
            with self.prediction_lock:
                # Get appropriate model
                if model_id:
                    if model_id not in self.prediction_models:
                        raise ValueError(f"Model {model_id} not found")
                    model = self.prediction_models[model_id]
                else:
                    model = self._select_best_model(prediction_type)
                
                if not model:
                    raise ValueError(f"No suitable model found for {prediction_type}")
                
                # Get historical data
                metric_type = prediction_type.value
                if metric_type not in self.metric_data or len(self.metric_data[metric_type]) < self.min_data_points:
                    raise ValueError(f"Insufficient data for prediction: {len(self.metric_data.get(metric_type, []))} points")
                
                historical_data = list(self.metric_data[metric_type])
                
                # Make prediction
                prediction_result = await self._make_prediction(
                    model, historical_data, horizon, context or {}
                )
                
                # Update model stats
                model.last_used = datetime.now()
                model.prediction_count += 1
                
                # Update performance stats
                self.performance_stats['predictions_made'] += 1
                
                prediction_time = time.time() - start_time
                current_avg = self.performance_stats['average_prediction_time']
                total_predictions = self.performance_stats['predictions_made']
                self.performance_stats['average_prediction_time'] = (
                    (current_avg * (total_predictions - 1) + prediction_time) / total_predictions
                )
                
                # Add to history
                self.prediction_history.append(prediction_result)
                
                self.logger.info(f"Made prediction: {prediction_type.value} = {prediction_result.predicted_value:.2f}")
                return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise
    
    def _select_best_model(self, prediction_type: PredictionType) -> Optional[PredictionModel]:
        """Select the best model for prediction type"""
        try:
            # Find models for this prediction type
            suitable_models = [
                model for model in self.prediction_models.values()
                if model.prediction_type == prediction_type
            ]
            
            if not suitable_models:
                return None
            
            # Select model with highest accuracy
            best_model = max(suitable_models, key=lambda m: m.accuracy)
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error selecting model: {e}")
            return None
    
    async def _make_prediction(self, model: PredictionModel, historical_data: List[MetricData],
                              horizon: PredictionHorizon, context: Dict[str, Any]) -> PredictionResult:
        """Make prediction using specified model"""
        try:
            prediction_id = str(uuid.uuid4())
            current_time = datetime.now()
            target_time = current_time + self.prediction_intervals[horizon]
            
            # Extract values and timestamps
            values = [point.value for point in historical_data]
            timestamps = [point.timestamp for point in historical_data]
            
            # Make prediction based on model type
            if model.model_type == ModelType.LINEAR_REGRESSION:
                predicted_value, confidence = await self._linear_regression_predict(values, model.parameters)
            
            elif model.model_type == ModelType.POLYNOMIAL_REGRESSION:
                predicted_value, confidence = await self._polynomial_regression_predict(values, model.parameters)
            
            elif model.model_type == ModelType.MOVING_AVERAGE:
                predicted_value, confidence = await self._moving_average_predict(values, model.parameters)
            
            elif model.model_type == ModelType.EXPONENTIAL_SMOOTHING:
                predicted_value, confidence = await self._exponential_smoothing_predict(values, model.parameters)
            
            elif model.model_type == ModelType.ARIMA:
                predicted_value, confidence = await self._arima_predict(values, model.parameters)
            
            elif model.model_type == ModelType.LSTM:
                predicted_value, confidence = await self._lstm_predict(values, model.parameters)
            
            elif model.model_type == ModelType.ENSEMBLE:
                predicted_value, confidence = await self._ensemble_predict(values, model.parameters, model.prediction_type)
            
            else:
                raise ValueError(f"Unknown model type: {model.model_type}")
            
            # Calculate confidence interval
            margin = confidence * 1.96  # 95% confidence interval
            confidence_interval = (predicted_value - margin, predicted_value + margin)
            
            # Create prediction result
            result = PredictionResult(
                id=prediction_id,
                prediction_type=model.prediction_type,
                horizon=horizon,
                predicted_value=predicted_value,
                confidence_interval=confidence_interval,
                confidence_score=confidence,
                timestamp=current_time,
                target_time=target_time,
                model_used=model.model_type,
                input_features=context,
                metadata={
                    'model_id': model.id,
                    'data_points_used': len(values),
                    'historical_mean': np.mean(values),
                    'historical_std': np.std(values)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction with model {model.name}: {e}")
            raise
    
    async def _linear_regression_predict(self, values: List[float], parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Linear regression prediction"""
        try:
            if len(values) < 2:
                return values[-1] if values else 0.0, 0.5
            
            # Simple linear regression
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope and intercept
            n = len(values)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict next value
            next_x = len(values)
            predicted_value = slope * next_x + intercept
            
            # Calculate confidence (based on R-squared)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            confidence = max(0.1, r_squared)
            
            return float(predicted_value), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error in linear regression prediction: {e}")
            return values[-1] if values else 0.0, 0.1
    
    async def _polynomial_regression_predict(self, values: List[float], parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Polynomial regression prediction"""
        try:
            if len(values) < 3:
                return values[-1] if values else 0.0, 0.5
            
            degree = parameters.get('degree', 2)
            
            # Fit polynomial
            x = np.arange(len(values))
            y = np.array(values)
            
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            
            # Predict next value
            next_x = len(values)
            predicted_value = poly(next_x)
            
            # Calculate confidence (based on fit quality)
            y_pred = poly(x)
            mse = np.mean((y - y_pred) ** 2)
            confidence = max(0.1, 1.0 / (1.0 + mse))
            
            return float(predicted_value), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error in polynomial regression prediction: {e}")
            return values[-1] if values else 0.0, 0.1
    
    async def _moving_average_predict(self, values: List[float], parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Moving average prediction"""
        try:
            window_size = parameters.get('window_size', 10)
            weights_type = parameters.get('weights', 'equal')
            
            if len(values) < window_size:
                window_size = len(values)
            
            recent_values = values[-window_size:]
            
            if weights_type == 'exponential':
                # Exponential weights (more recent values have higher weight)
                weights = np.exp(np.arange(window_size))
                weights = weights / np.sum(weights)
                predicted_value = np.sum(recent_values * weights)
            else:
                # Equal weights
                predicted_value = np.mean(recent_values)
            
            # Confidence based on variance
            variance = np.var(recent_values)
            confidence = max(0.1, 1.0 / (1.0 + variance))
            
            return float(predicted_value), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error in moving average prediction: {e}")
            return values[-1] if values else 0.0, 0.1
    
    async def _exponential_smoothing_predict(self, values: List[float], parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Exponential smoothing prediction"""
        try:
            alpha = parameters.get('alpha', 0.3)
            
            if not values:
                return 0.0, 0.1
            
            # Simple exponential smoothing
            smoothed = values[0]
            for value in values[1:]:
                smoothed = alpha * value + (1 - alpha) * smoothed
            
            predicted_value = smoothed
            
            # Confidence based on recent trend stability
            if len(values) >= 5:
                recent_trend = np.std(values[-5:])
                confidence = max(0.1, 1.0 / (1.0 + recent_trend))
            else:
                confidence = 0.5
            
            return float(predicted_value), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error in exponential smoothing prediction: {e}")
            return values[-1] if values else 0.0, 0.1
    
    async def _arima_predict(self, values: List[float], parameters: Dict[str, Any]) -> Tuple[float, float]:
        """ARIMA prediction (simplified implementation)"""
        try:
            # Simplified ARIMA - using autoregressive component only
            p = parameters.get('p', 2)
            
            if len(values) < p + 1:
                return values[-1] if values else 0.0, 0.5
            
            # Fit AR model
            y = np.array(values)
            
            # Simple AR(p) model
            X = []
            Y = []
            for i in range(p, len(y)):
                X.append(y[i-p:i])
                Y.append(y[i])
            
            if not X:
                return values[-1], 0.5
            
            X = np.array(X)
            Y = np.array(Y)
            
            # Solve normal equations (simplified)
            try:
                coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
            except:
                return values[-1], 0.1
            
            # Predict next value
            last_values = y[-p:]
            predicted_value = np.dot(coeffs, last_values)
            
            # Calculate confidence
            if len(Y) > 0:
                predictions = X @ coeffs
                mse = np.mean((Y - predictions) ** 2)
                confidence = max(0.1, 1.0 / (1.0 + mse))
            else:
                confidence = 0.5
            
            return float(predicted_value), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA prediction: {e}")
            return values[-1] if values else 0.0, 0.1
    
    async def _lstm_predict(self, values: List[float], parameters: Dict[str, Any]) -> Tuple[float, float]:
        """LSTM prediction (simplified implementation)"""
        try:
            # Simplified LSTM - using weighted recent values
            sequence_length = parameters.get('sequence_length', 10)
            
            if len(values) < sequence_length:
                sequence_length = len(values)
            
            recent_values = values[-sequence_length:]
            
            # Simple weighted prediction (mimicking LSTM behavior)
            weights = np.exp(np.arange(sequence_length) * 0.1)  # Exponential weights
            weights = weights / np.sum(weights)
            
            predicted_value = np.sum(recent_values * weights)
            
            # Confidence based on sequence stability
            if len(recent_values) >= 3:
                trend_stability = 1.0 / (1.0 + np.std(np.diff(recent_values)))
                confidence = max(0.1, trend_stability)
            else:
                confidence = 0.5
            
            return float(predicted_value), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            return values[-1] if values else 0.0, 0.1
    
    async def _ensemble_predict(self, values: List[float], parameters: Dict[str, Any],
                               prediction_type: PredictionType) -> Tuple[float, float]:
        """Ensemble prediction"""
        try:
            model_types = parameters.get('models', ['linear', 'moving_average'])
            weights = parameters.get('weights', [0.5, 0.5])
            
            predictions = []
            confidences = []
            
            # Get predictions from different models
            for model_type in model_types:
                if model_type == 'linear':
                    pred, conf = await self._linear_regression_predict(values, {})
                elif model_type == 'moving_average':
                    pred, conf = await self._moving_average_predict(values, {'window_size': 10})
                elif model_type == 'exponential_smoothing':
                    pred, conf = await self._exponential_smoothing_predict(values, {'alpha': 0.3})
                else:
                    pred, conf = values[-1] if values else 0.0, 0.1
                
                predictions.append(pred)
                confidences.append(conf)
            
            # Weighted ensemble
            if len(weights) != len(predictions):
                weights = [1.0 / len(predictions)] * len(predictions)
            
            ensemble_prediction = sum(p * w for p, w in zip(predictions, weights))
            ensemble_confidence = sum(c * w for c, w in zip(confidences, weights))
            
            return float(ensemble_prediction), float(ensemble_confidence)
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return values[-1] if values else 0.0, 0.1
    
    async def _detect_anomaly(self, metric: MetricData) -> Optional[AnomalyDetection]:
        """Detect anomalies in metric data"""
        try:
            metric_type = metric.metric_type
            
            if len(self.metric_data[metric_type]) < 10:
                return None  # Need more data for anomaly detection
            
            # Get recent historical data
            recent_data = list(self.metric_data[metric_type])[-50:]  # Last 50 points
            values = [point.value for point in recent_data[:-1]]  # Exclude current point
            
            # Calculate statistics
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            if std_value == 0:
                return None  # No variation in data
            
            # Calculate anomaly score (z-score)
            anomaly_score = abs(metric.value - mean_value) / std_value
            
            if anomaly_score > self.anomaly_threshold:
                # Determine severity
                if anomaly_score > 4.0:
                    severity = "critical"
                elif anomaly_score > 3.0:
                    severity = "high"
                elif anomaly_score > 2.5:
                    severity = "medium"
                else:
                    severity = "low"
                
                anomaly = AnomalyDetection(
                    id=str(uuid.uuid4()),
                    timestamp=metric.timestamp,
                    metric_type=metric_type,
                    actual_value=metric.value,
                    expected_value=mean_value,
                    anomaly_score=anomaly_score,
                    severity=severity,
                    description=f"Anomalous {metric_type}: {metric.value:.2f} (expected ~{mean_value:.2f})",
                    context=metric.context
                )
                
                self.anomaly_history.append(anomaly)
                self.performance_stats['anomalies_detected'] += 1
                
                self.logger.warning(f"Anomaly detected: {anomaly.description}")
                return anomaly
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting anomaly: {e}")
            return None
    
    async def _check_retrain_models(self, metric_type: str) -> None:
        """Check if models need retraining"""
        try:
            # Find models for this metric type
            prediction_type = PredictionType(metric_type)
            models_to_retrain = []
            
            for model in self.prediction_models.values():
                if model.prediction_type == prediction_type:
                    # Check if accuracy has dropped
                    if model.accuracy > 0 and model.error_rate > self.retrain_threshold:
                        models_to_retrain.append(model)
                    
                    # Check if enough time has passed since last training
                    if model.last_trained:
                        time_since_training = datetime.now() - model.last_trained
                        if time_since_training > timedelta(days=7):  # Retrain weekly
                            models_to_retrain.append(model)
            
            # Retrain models
            for model in models_to_retrain:
                await self._retrain_model(model)
            
        except Exception as e:
            self.logger.error(f"Error checking model retraining: {e}")
    
    async def _retrain_model(self, model: PredictionModel) -> None:
        """Retrain a prediction model"""
        try:
            metric_type = model.prediction_type.value
            
            if metric_type not in self.metric_data:
                return
            
            # Get training data
            training_data = list(self.metric_data[metric_type])
            
            if len(training_data) < self.min_data_points:
                return
            
            # Update model training data size
            model.training_data_size = len(training_data)
            model.last_trained = datetime.now()
            
            # Calculate new accuracy (simplified)
            values = [point.value for point in training_data]
            
            # Use cross-validation approach
            if len(values) >= 20:
                # Split data for validation
                split_point = int(len(values) * 0.8)
                train_values = values[:split_point]
                test_values = values[split_point:]
                
                # Make predictions on test set
                predictions = []
                for i in range(len(test_values)):
                    if model.model_type == ModelType.LINEAR_REGRESSION:
                        pred, _ = await self._linear_regression_predict(train_values + test_values[:i], model.parameters)
                    elif model.model_type == ModelType.MOVING_AVERAGE:
                        pred, _ = await self._moving_average_predict(train_values + test_values[:i], model.parameters)
                    else:
                        pred = test_values[i-1] if i > 0 else train_values[-1]
                    
                    predictions.append(pred)
                
                # Calculate accuracy
                if predictions and test_values:
                    mse = np.mean([(p - a) ** 2 for p, a in zip(predictions, test_values)])
                    model.accuracy = max(0.0, 1.0 / (1.0 + mse))
                    model.error_rate = mse / (np.mean(test_values) ** 2) if np.mean(test_values) != 0 else 1.0
            
            self.performance_stats['models_trained'] += 1
            
            self.logger.info(f"Retrained model {model.name}, accuracy: {model.accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model {model.name}: {e}")
    
    async def predict_multiple(self, prediction_types: List[PredictionType],
                              horizon: PredictionHorizon,
                              context: Optional[Dict[str, Any]] = None) -> List[PredictionResult]:
        """Make multiple predictions"""
        try:
            results = []
            
            for prediction_type in prediction_types:
                try:
                    result = await self.predict(prediction_type, horizon, context)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error predicting {prediction_type}: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error making multiple predictions: {e}")
            return []
    
    async def validate_prediction(self, prediction_id: str, actual_value: float) -> None:
        """Validate a prediction with actual value"""
        try:
            # Find prediction in history
            prediction = None
            for pred in self.prediction_history:
                if pred.id == prediction_id:
                    prediction = pred
                    break
            
            if not prediction:
                self.logger.warning(f"Prediction {prediction_id} not found for validation")
                return
            
            # Calculate accuracy
            error = abs(prediction.predicted_value - actual_value)
            relative_error = error / actual_value if actual_value != 0 else error
            
            # Update model accuracy
            model = None
            for m in self.prediction_models.values():
                if m.id == prediction.metadata.get('model_id'):
                    model = m
                    break
            
            if model:
                # Update running accuracy
                total_predictions = model.prediction_count
                current_accuracy = model.accuracy
                
                # New accuracy based on this validation
                new_accuracy = max(0.0, 1.0 - relative_error)
                
                # Update running average
                if total_predictions > 1:
                    model.accuracy = ((current_accuracy * (total_predictions - 1)) + new_accuracy) / total_predictions
                else:
                    model.accuracy = new_accuracy
                
                # Update error rate
                model.error_rate = ((model.error_rate * (total_predictions - 1)) + relative_error) / total_predictions
            
            # Update global stats
            if relative_error < 0.1:  # Consider accurate if within 10%
                self.performance_stats['accurate_predictions'] += 1
            
            total_predictions = self.performance_stats['predictions_made']
            if total_predictions > 0:
                self.performance_stats['average_accuracy'] = (
                    self.performance_stats['accurate_predictions'] / total_predictions
                )
            
            self.logger.info(f"Validated prediction {prediction_id}: error = {relative_error:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error validating prediction: {e}")
    
    def get_predictions(self, prediction_type: Optional[PredictionType] = None,
                       limit: int = 100) -> List[PredictionResult]:
        """Get prediction history"""
        predictions = list(self.prediction_history)
        
        if prediction_type:
            predictions = [p for p in predictions if p.prediction_type == prediction_type]
        
        return sorted(predictions, key=lambda p: p.timestamp, reverse=True)[:limit]
    
    def get_anomalies(self, metric_type: Optional[str] = None,
                     severity: Optional[str] = None,
                     limit: int = 100) -> List[AnomalyDetection]:
        """Get anomaly history"""
        anomalies = list(self.anomaly_history)
        
        if metric_type:
            anomalies = [a for a in anomalies if a.metric_type == metric_type]
        
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        
        return sorted(anomalies, key=lambda a: a.timestamp, reverse=True)[:limit]
    
    def get_models(self, prediction_type: Optional[PredictionType] = None) -> List[PredictionModel]:
        """Get prediction models"""
        models = list(self.prediction_models.values())
        
        if prediction_type:
            models = [m for m in models if m.prediction_type == prediction_type]
        
        return sorted(models, key=lambda m: m.accuracy, reverse=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    def get_metric_summary(self, metric_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get metric summary for specified time period"""
        try:
            if metric_type not in self.metric_data:
                return {}
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [
                point for point in self.metric_data[metric_type]
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_data:
                return {}
            
            values = [point.value for point in recent_data]
            
            return {
                'metric_type': metric_type,
                'time_period_hours': hours,
                'data_points': len(values),
                'min_value': min(values),
                'max_value': max(values),
                'mean_value': np.mean(values),
                'median_value': np.median(values),
                'std_deviation': np.std(values),
                'latest_value': values[-1],
                'trend': 'increasing' if len(values) >= 2 and values[-1] > values[0] else 'decreasing' if len(values) >= 2 else 'stable'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metric summary: {e}")
            return {}
    
    async def create_model(self, name: str, model_type: ModelType,
                          prediction_type: PredictionType,
                          parameters: Dict[str, Any]) -> str:
        """Create a new prediction model"""
        return self._create_model(name, model_type, prediction_type, parameters)
    
    async def delete_model(self, model_id: str) -> None:
        """Delete a prediction model"""
        if model_id in self.prediction_models:
            del self.prediction_models[model_id]
            self.logger.info(f"Deleted model: {model_id}")
        else:
            raise ValueError(f"Model {model_id} not found")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        
        # Update specific settings
        if 'min_data_points' in new_config:
            self.min_data_points = new_config['min_data_points']
        
        if 'auto_retrain' in new_config:
            self.auto_retrain = new_config['auto_retrain']
        
        if 'retrain_threshold' in new_config:
            self.retrain_threshold = new_config['retrain_threshold']
        
        if 'anomaly_threshold' in new_config:
            self.anomaly_threshold = new_config['anomaly_threshold']
        
        if 'anomaly_detection' in new_config:
            self.anomaly_detection_enabled = new_config['anomaly_detection']
        
        self.logger.info("Performance Predictor configuration updated")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Save prediction data
            await self.save_prediction_data()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear old data
            cutoff_time = datetime.now() - timedelta(days=30)
            
            # Clear old metric data
            for metric_type in self.metric_data:
                self.metric_data[metric_type] = deque(
                    [point for point in self.metric_data[metric_type] if point.timestamp >= cutoff_time],
                    maxlen=10000
                )
            
            # Clear old daily stats
            if 'daily_stats' in self.performance_stats:
                cutoff_date = (datetime.now() - timedelta(days=30)).date()
                self.performance_stats['daily_stats'] = {
                    date: stats for date, stats in self.performance_stats['daily_stats'].items()
                    if datetime.fromisoformat(date).date() >= cutoff_date
                }
            
            self.logger.info("Performance Predictor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")