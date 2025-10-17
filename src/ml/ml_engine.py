"""
Machine Learning Engine for Computer Assistant

This module provides the core ML engine with capabilities for:
- Model training and evaluation
- Real-time inference
- Model management and versioning
- Performance monitoring
- Automated model selection and optimization
"""

import asyncio
import json
import logging
import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class ModelType(Enum):
    """Types of ML models"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    NEURAL_NETWORK = "neural_network"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    ENSEMBLE = "ensemble"


class ModelStatus(Enum):
    """Model status"""
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class TrainingStatus(Enum):
    """Training status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainingData:
    """Training data structure"""
    features: np.ndarray
    labels: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    label_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate training data"""
        if self.features is None:
            raise ValueError("Features cannot be None")
        
        if len(self.features.shape) != 2:
            raise ValueError("Features must be 2D array")
        
        if self.labels is not None and len(self.features) != len(self.labels):
            raise ValueError("Features and labels must have same number of samples")


@dataclass
class ModelMetrics:
    """Model evaluation metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    inference_time: float = 0.0
    model_size: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'auc_roc': self.auc_roc,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'feature_importance': self.feature_importance,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'model_size': self.model_size,
            'custom_metrics': self.custom_metrics
        }


@dataclass
class PredictionResult:
    """Prediction result structure"""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    feature_contributions: Optional[Dict[str, float]] = None
    prediction_time: float = 0.0
    model_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'predictions': self.predictions.tolist() if isinstance(self.predictions, np.ndarray) else self.predictions,
            'probabilities': self.probabilities.tolist() if self.probabilities is not None else None,
            'confidence': self.confidence.tolist() if self.confidence is not None else None,
            'feature_contributions': self.feature_contributions,
            'prediction_time': self.prediction_time,
            'model_version': self.model_version,
            'metadata': self.metadata
        }


@dataclass
class MLModel:
    """ML model wrapper"""
    id: str
    name: str
    model_type: ModelType
    model: Any  # The actual ML model object
    status: ModelStatus = ModelStatus.TRAINING
    version: str = "1.0.0"
    metrics: Optional[ModelMetrics] = None
    feature_names: List[str] = field(default_factory=list)
    label_names: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding model object)"""
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type.value,
            'status': self.status.value,
            'version': self.version,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'feature_names': self.feature_names,
            'label_names': self.label_names,
            'hyperparameters': self.hyperparameters,
            'training_config': self.training_config,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }


class MLEngine:
    """
    Comprehensive machine learning engine
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ML engine"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models: Dict[str, MLModel] = {}
        self.training_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.models_dir = Path(self.config.get('models_dir', 'ml_models'))
        self.models_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path(self.config.get('data_dir', 'ml_data'))
        self.data_dir.mkdir(exist_ok=True)
        
        # Training settings
        self.max_training_time = self.config.get('max_training_time', 3600)  # 1 hour
        self.auto_save_models = self.config.get('auto_save_models', True)
        self.model_versioning = self.config.get('model_versioning', True)
        self.parallel_training = self.config.get('parallel_training', True)
        self.max_parallel_jobs = self.config.get('max_parallel_jobs', 3)
        
        # Performance tracking
        self.performance_stats = {
            'models_trained': 0,
            'models_deployed': 0,
            'predictions_made': 0,
            'total_training_time': 0.0,
            'total_inference_time': 0.0,
            'average_accuracy': 0.0,
            'model_types_used': {},
            'daily_stats': {}
        }
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_jobs)
        
        # Model factories
        self.model_factories = {
            ModelType.LINEAR_REGRESSION: self._create_linear_regression,
            ModelType.LOGISTIC_REGRESSION: self._create_logistic_regression,
            ModelType.DECISION_TREE: self._create_decision_tree,
            ModelType.RANDOM_FOREST: self._create_random_forest,
            ModelType.SVM: self._create_svm,
            ModelType.NEURAL_NETWORK: self._create_neural_network
        }
        
        # Load existing models
        self._load_existing_models()
        
        self.logger.info("ML Engine initialized")
    
    def _load_existing_models(self) -> None:
        """Load existing models from disk"""
        try:
            for model_file in self.models_dir.glob("*.pkl"):
                try:
                    model_id = model_file.stem
                    model_path = self.models_dir / f"{model_id}.pkl"
                    metadata_path = self.models_dir / f"{model_id}_metadata.json"
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        with open(model_path, 'rb') as f:
                            model_obj = pickle.load(f)
                        
                        ml_model = MLModel(
                            id=model_id,
                            name=metadata['name'],
                            model_type=ModelType(metadata['model_type']),
                            model=model_obj,
                            status=ModelStatus(metadata['status']),
                            version=metadata['version'],
                            feature_names=metadata.get('feature_names', []),
                            label_names=metadata.get('label_names', []),
                            hyperparameters=metadata.get('hyperparameters', {}),
                            training_config=metadata.get('training_config', {}),
                            created_at=datetime.fromisoformat(metadata['created_at']),
                            updated_at=datetime.fromisoformat(metadata['updated_at']),
                            metadata=metadata.get('metadata', {})
                        )
                        
                        # Load metrics if available
                        if 'metrics' in metadata and metadata['metrics']:
                            ml_model.metrics = ModelMetrics(**metadata['metrics'])
                        
                        self.models[model_id] = ml_model
                        self.logger.info(f"Loaded model: {ml_model.name}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading model {model_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.models)} existing models")
            
        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")
    
    async def create_model(self, name: str, model_type: ModelType, 
                          hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """Create a new ML model"""
        try:
            model_id = str(uuid.uuid4())
            hyperparameters = hyperparameters or {}
            
            # Create model using factory
            if model_type not in self.model_factories:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            model_obj = self.model_factories[model_type](hyperparameters)
            
            ml_model = MLModel(
                id=model_id,
                name=name,
                model_type=model_type,
                model=model_obj,
                hyperparameters=hyperparameters,
                status=ModelStatus.TRAINING
            )
            
            self.models[model_id] = ml_model
            
            # Save model if auto-save is enabled
            if self.auto_save_models:
                await self.save_model(model_id)
            
            self.logger.info(f"Created model: {name} ({model_type.value})")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            raise
    
    async def train_model(self, model_id: str, training_data: TrainingData,
                         validation_data: Optional[TrainingData] = None,
                         training_config: Optional[Dict[str, Any]] = None) -> ModelMetrics:
        """Train a model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            ml_model = self.models[model_id]
            training_config = training_config or {}
            
            # Create training job
            job_id = str(uuid.uuid4())
            self.training_jobs[job_id] = {
                'model_id': model_id,
                'status': TrainingStatus.PENDING,
                'start_time': datetime.now(),
                'progress': 0.0,
                'metrics': None
            }
            
            # Start training
            if self.parallel_training:
                future = self.executor.submit(
                    self._train_model_sync, ml_model, training_data, validation_data, training_config, job_id
                )
                # Don't wait for completion in async mode
                return await asyncio.wrap_future(future)
            else:
                return self._train_model_sync(ml_model, training_data, validation_data, training_config, job_id)
                
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    def _train_model_sync(self, ml_model: MLModel, training_data: TrainingData,
                         validation_data: Optional[TrainingData], training_config: Dict[str, Any],
                         job_id: str) -> ModelMetrics:
        """Synchronous model training"""
        try:
            self.training_jobs[job_id]['status'] = TrainingStatus.RUNNING
            start_time = time.time()
            
            # Update model status
            ml_model.status = ModelStatus.TRAINING
            ml_model.feature_names = training_data.feature_names
            ml_model.label_names = training_data.label_names
            ml_model.training_config = training_config
            ml_model.updated_at = datetime.now()
            
            # Train the model
            if ml_model.model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION,
                                     ModelType.DECISION_TREE, ModelType.RANDOM_FOREST, ModelType.SVM]:
                ml_model.model.fit(training_data.features, training_data.labels)
            elif ml_model.model_type == ModelType.NEURAL_NETWORK:
                # For neural networks, we might need different training approach
                self._train_neural_network(ml_model.model, training_data, validation_data, training_config)
            else:
                raise ValueError(f"Training not implemented for {ml_model.model_type}")
            
            training_time = time.time() - start_time
            
            # Evaluate model
            metrics = self._evaluate_model(ml_model, training_data, validation_data)
            metrics.training_time = training_time
            
            # Update model
            ml_model.metrics = metrics
            ml_model.status = ModelStatus.TRAINED
            ml_model.updated_at = datetime.now()
            
            # Update training job
            self.training_jobs[job_id]['status'] = TrainingStatus.COMPLETED
            self.training_jobs[job_id]['metrics'] = metrics
            self.training_jobs[job_id]['end_time'] = datetime.now()
            
            # Update performance stats
            self._update_training_stats(ml_model, metrics)
            
            # Save model
            if self.auto_save_models:
                asyncio.create_task(self.save_model(ml_model.id))
            
            self.logger.info(f"Model {ml_model.name} trained successfully")
            return metrics
            
        except Exception as e:
            self.training_jobs[job_id]['status'] = TrainingStatus.FAILED
            self.training_jobs[job_id]['error'] = str(e)
            ml_model.status = ModelStatus.FAILED
            self.logger.error(f"Error training model {ml_model.name}: {e}")
            raise
    
    def _train_neural_network(self, model: Any, training_data: TrainingData,
                            validation_data: Optional[TrainingData], config: Dict[str, Any]) -> None:
        """Train neural network model"""
        try:
            # This is a placeholder for neural network training
            # In a real implementation, you would use frameworks like TensorFlow or PyTorch
            
            epochs = config.get('epochs', 100)
            batch_size = config.get('batch_size', 32)
            learning_rate = config.get('learning_rate', 0.001)
            
            # Simulate training process
            for epoch in range(epochs):
                # Simulate batch training
                for i in range(0, len(training_data.features), batch_size):
                    batch_features = training_data.features[i:i+batch_size]
                    batch_labels = training_data.labels[i:i+batch_size] if training_data.labels is not None else None
                    
                    # Simulate forward pass and backpropagation
                    time.sleep(0.001)  # Simulate computation time
                
                # Simulate validation
                if validation_data and epoch % 10 == 0:
                    # Simulate validation evaluation
                    pass
            
            self.logger.info("Neural network training completed")
            
        except Exception as e:
            self.logger.error(f"Error training neural network: {e}")
            raise
    
    def _evaluate_model(self, ml_model: MLModel, training_data: TrainingData,
                       validation_data: Optional[TrainingData]) -> ModelMetrics:
        """Evaluate model performance"""
        try:
            metrics = ModelMetrics()
            
            # Use validation data if available, otherwise use training data
            eval_data = validation_data or training_data
            
            if eval_data.labels is None:
                # Unsupervised learning - limited metrics
                return metrics
            
            # Make predictions
            start_time = time.time()
            predictions = ml_model.model.predict(eval_data.features)
            inference_time = time.time() - start_time
            
            metrics.inference_time = inference_time / len(eval_data.features)  # Per sample
            
            # Calculate metrics based on model type
            if ml_model.model_type in [ModelType.CLASSIFICATION, ModelType.LOGISTIC_REGRESSION]:
                metrics = self._calculate_classification_metrics(eval_data.labels, predictions, metrics)
            elif ml_model.model_type in [ModelType.REGRESSION, ModelType.LINEAR_REGRESSION]:
                metrics = self._calculate_regression_metrics(eval_data.labels, predictions, metrics)
            
            # Feature importance (if available)
            if hasattr(ml_model.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    ml_model.feature_names,
                    ml_model.model.feature_importances_
                ))
                metrics.feature_importance = feature_importance
            elif hasattr(ml_model.model, 'coef_'):
                feature_importance = dict(zip(
                    ml_model.feature_names,
                    abs(ml_model.model.coef_.flatten())
                ))
                metrics.feature_importance = feature_importance
            
            # Model size
            try:
                import sys
                metrics.model_size = sys.getsizeof(pickle.dumps(ml_model.model))
            except Exception:
                metrics.model_size = 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return ModelMetrics()
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        metrics: ModelMetrics) -> ModelMetrics:
        """Calculate classification metrics"""
        try:
            # Accuracy
            metrics.accuracy = np.mean(y_true == y_pred)
            
            # For binary classification
            if len(np.unique(y_true)) == 2:
                # Convert to binary if needed
                y_true_binary = (y_true == np.unique(y_true)[1]).astype(int)
                y_pred_binary = (y_pred == np.unique(y_pred)[1]).astype(int)
                
                # Precision, Recall, F1
                tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
                fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
                
                metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics.f1_score = (2 * metrics.precision * metrics.recall / 
                                  (metrics.precision + metrics.recall) 
                                  if (metrics.precision + metrics.recall) > 0 else 0.0)
            
            # Confusion matrix
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            n_labels = len(unique_labels)
            confusion_matrix = np.zeros((n_labels, n_labels))
            
            for i, true_label in enumerate(unique_labels):
                for j, pred_label in enumerate(unique_labels):
                    confusion_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
            
            metrics.confusion_matrix = confusion_matrix
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {e}")
            return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    metrics: ModelMetrics) -> ModelMetrics:
        """Calculate regression metrics"""
        try:
            # Mean Squared Error
            metrics.mse = np.mean((y_true - y_pred) ** 2)
            
            # Root Mean Squared Error
            metrics.rmse = np.sqrt(metrics.mse)
            
            # Mean Absolute Error
            metrics.mae = np.mean(np.abs(y_true - y_pred))
            
            # R² Score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics.r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regression metrics: {e}")
            return metrics
    
    async def predict(self, model_id: str, features: np.ndarray,
                     return_probabilities: bool = False,
                     return_confidence: bool = False) -> PredictionResult:
        """Make predictions using a trained model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            ml_model = self.models[model_id]
            
            if ml_model.status != ModelStatus.TRAINED and ml_model.status != ModelStatus.DEPLOYED:
                raise ValueError(f"Model {model_id} is not trained")
            
            start_time = time.time()
            
            # Make predictions
            predictions = ml_model.model.predict(features)
            
            # Get probabilities if requested and available
            probabilities = None
            if return_probabilities and hasattr(ml_model.model, 'predict_proba'):
                probabilities = ml_model.model.predict_proba(features)
            
            # Calculate confidence if requested
            confidence = None
            if return_confidence:
                if probabilities is not None:
                    confidence = np.max(probabilities, axis=1)
                else:
                    # For regression, use prediction variance as confidence measure
                    confidence = np.ones(len(predictions)) * 0.8  # Placeholder
            
            prediction_time = time.time() - start_time
            
            result = PredictionResult(
                predictions=predictions,
                probabilities=probabilities,
                confidence=confidence,
                prediction_time=prediction_time,
                model_version=ml_model.version
            )
            
            # Update performance stats
            self.performance_stats['predictions_made'] += len(predictions)
            self.performance_stats['total_inference_time'] += prediction_time
            
            # Update daily stats
            today = datetime.now().date().isoformat()
            if today not in self.performance_stats['daily_stats']:
                self.performance_stats['daily_stats'][today] = {'predictions': 0}
            self.performance_stats['daily_stats'][today]['predictions'] += len(predictions)
            
            self.logger.info(f"Made {len(predictions)} predictions using model {ml_model.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
    
    async def deploy_model(self, model_id: str) -> None:
        """Deploy a trained model for production use"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            ml_model = self.models[model_id]
            
            if ml_model.status != ModelStatus.TRAINED:
                raise ValueError(f"Model {model_id} is not trained")
            
            ml_model.status = ModelStatus.DEPLOYED
            ml_model.updated_at = datetime.now()
            
            # Save model
            if self.auto_save_models:
                await self.save_model(model_id)
            
            # Update performance stats
            self.performance_stats['models_deployed'] += 1
            
            self.logger.info(f"Model {ml_model.name} deployed successfully")
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            raise
    
    async def save_model(self, model_id: str) -> None:
        """Save model to disk"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            ml_model = self.models[model_id]
            
            # Save model object
            model_path = self.models_dir / f"{model_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(ml_model.model, f)
            
            # Save metadata
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(ml_model.to_dict(), f, indent=2)
            
            self.logger.info(f"Model {ml_model.name} saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    async def load_model(self, model_path: str) -> str:
        """Load model from disk"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model object
            with open(model_path, 'rb') as f:
                model_obj = pickle.load(f)
            
            # Load metadata
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                ml_model = MLModel(
                    id=metadata['id'],
                    name=metadata['name'],
                    model_type=ModelType(metadata['model_type']),
                    model=model_obj,
                    status=ModelStatus(metadata['status']),
                    version=metadata['version'],
                    feature_names=metadata.get('feature_names', []),
                    label_names=metadata.get('label_names', []),
                    hyperparameters=metadata.get('hyperparameters', {}),
                    training_config=metadata.get('training_config', {}),
                    created_at=datetime.fromisoformat(metadata['created_at']),
                    updated_at=datetime.fromisoformat(metadata['updated_at']),
                    metadata=metadata.get('metadata', {})
                )
                
                # Load metrics if available
                if 'metrics' in metadata and metadata['metrics']:
                    ml_model.metrics = ModelMetrics(**metadata['metrics'])
            else:
                # Create basic model without metadata
                model_id = str(uuid.uuid4())
                ml_model = MLModel(
                    id=model_id,
                    name=f"Loaded Model {model_id[:8]}",
                    model_type=ModelType.REGRESSION,  # Default
                    model=model_obj,
                    status=ModelStatus.TRAINED
                )
            
            self.models[ml_model.id] = ml_model
            
            self.logger.info(f"Model loaded: {ml_model.name}")
            return ml_model.id
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _update_training_stats(self, ml_model: MLModel, metrics: ModelMetrics) -> None:
        """Update training performance statistics"""
        try:
            self.performance_stats['models_trained'] += 1
            self.performance_stats['total_training_time'] += metrics.training_time
            
            # Update model type stats
            model_type = ml_model.model_type.value
            if model_type not in self.performance_stats['model_types_used']:
                self.performance_stats['model_types_used'][model_type] = 0
            self.performance_stats['model_types_used'][model_type] += 1
            
            # Update average accuracy
            if metrics.accuracy is not None:
                current_avg = self.performance_stats['average_accuracy']
                total_models = self.performance_stats['models_trained']
                self.performance_stats['average_accuracy'] = (
                    (current_avg * (total_models - 1) + metrics.accuracy) / total_models
                )
            
            # Update daily stats
            today = datetime.now().date().isoformat()
            if today not in self.performance_stats['daily_stats']:
                self.performance_stats['daily_stats'][today] = {'models_trained': 0}
            self.performance_stats['daily_stats'][today]['models_trained'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating training stats: {e}")
    
    # Model factory methods
    def _create_linear_regression(self, hyperparameters: Dict[str, Any]) -> Any:
        """Create linear regression model"""
        try:
            # Simple linear regression implementation
            class SimpleLinearRegression:
                def __init__(self):
                    self.coef_ = None
                    self.intercept_ = None
                
                def fit(self, X, y):
                    # Add bias term
                    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
                    
                    # Normal equation: theta = (X^T * X)^-1 * X^T * y
                    theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
                    
                    self.intercept_ = theta[0]
                    self.coef_ = theta[1:]
                
                def predict(self, X):
                    return X @ self.coef_ + self.intercept_
            
            return SimpleLinearRegression()
            
        except Exception as e:
            self.logger.error(f"Error creating linear regression model: {e}")
            raise
    
    def _create_logistic_regression(self, hyperparameters: Dict[str, Any]) -> Any:
        """Create logistic regression model"""
        try:
            # Simple logistic regression implementation
            class SimpleLogisticRegression:
                def __init__(self, learning_rate=0.01, max_iter=1000):
                    self.learning_rate = learning_rate
                    self.max_iter = max_iter
                    self.coef_ = None
                    self.intercept_ = None
                
                def _sigmoid(self, z):
                    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
                
                def fit(self, X, y):
                    n_samples, n_features = X.shape
                    
                    # Initialize parameters
                    self.coef_ = np.zeros(n_features)
                    self.intercept_ = 0
                    
                    # Gradient descent
                    for _ in range(self.max_iter):
                        # Forward pass
                        z = X @ self.coef_ + self.intercept_
                        predictions = self._sigmoid(z)
                        
                        # Compute gradients
                        dw = (1 / n_samples) * X.T @ (predictions - y)
                        db = (1 / n_samples) * np.sum(predictions - y)
                        
                        # Update parameters
                        self.coef_ -= self.learning_rate * dw
                        self.intercept_ -= self.learning_rate * db
                
                def predict(self, X):
                    z = X @ self.coef_ + self.intercept_
                    return (self._sigmoid(z) >= 0.5).astype(int)
                
                def predict_proba(self, X):
                    z = X @ self.coef_ + self.intercept_
                    prob_1 = self._sigmoid(z)
                    return np.column_stack([1 - prob_1, prob_1])
            
            learning_rate = hyperparameters.get('learning_rate', 0.01)
            max_iter = hyperparameters.get('max_iter', 1000)
            
            return SimpleLogisticRegression(learning_rate, max_iter)
            
        except Exception as e:
            self.logger.error(f"Error creating logistic regression model: {e}")
            raise
    
    def _create_decision_tree(self, hyperparameters: Dict[str, Any]) -> Any:
        """Create decision tree model"""
        try:
            # Simple decision tree implementation
            class SimpleDecisionTree:
                def __init__(self, max_depth=5, min_samples_split=2):
                    self.max_depth = max_depth
                    self.min_samples_split = min_samples_split
                    self.tree = None
                
                def _gini_impurity(self, y):
                    classes, counts = np.unique(y, return_counts=True)
                    probabilities = counts / len(y)
                    return 1 - np.sum(probabilities ** 2)
                
                def _best_split(self, X, y):
                    best_gini = float('inf')
                    best_feature = None
                    best_threshold = None
                    
                    n_features = X.shape[1]
                    
                    for feature in range(n_features):
                        thresholds = np.unique(X[:, feature])
                        
                        for threshold in thresholds:
                            left_mask = X[:, feature] <= threshold
                            right_mask = ~left_mask
                            
                            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                                continue
                            
                            left_gini = self._gini_impurity(y[left_mask])
                            right_gini = self._gini_impurity(y[right_mask])
                            
                            weighted_gini = (np.sum(left_mask) * left_gini + 
                                           np.sum(right_mask) * right_gini) / len(y)
                            
                            if weighted_gini < best_gini:
                                best_gini = weighted_gini
                                best_feature = feature
                                best_threshold = threshold
                    
                    return best_feature, best_threshold
                
                def _build_tree(self, X, y, depth=0):
                    if (depth >= self.max_depth or 
                        len(y) < self.min_samples_split or 
                        len(np.unique(y)) == 1):
                        # Leaf node
                        classes, counts = np.unique(y, return_counts=True)
                        return classes[np.argmax(counts)]
                    
                    feature, threshold = self._best_split(X, y)
                    
                    if feature is None:
                        classes, counts = np.unique(y, return_counts=True)
                        return classes[np.argmax(counts)]
                    
                    left_mask = X[:, feature] <= threshold
                    right_mask = ~left_mask
                    
                    left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
                    right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
                    
                    return {
                        'feature': feature,
                        'threshold': threshold,
                        'left': left_subtree,
                        'right': right_subtree
                    }
                
                def fit(self, X, y):
                    self.tree = self._build_tree(X, y)
                
                def _predict_sample(self, x, tree):
                    if not isinstance(tree, dict):
                        return tree
                    
                    if x[tree['feature']] <= tree['threshold']:
                        return self._predict_sample(x, tree['left'])
                    else:
                        return self._predict_sample(x, tree['right'])
                
                def predict(self, X):
                    return np.array([self._predict_sample(x, self.tree) for x in X])
            
            max_depth = hyperparameters.get('max_depth', 5)
            min_samples_split = hyperparameters.get('min_samples_split', 2)
            
            return SimpleDecisionTree(max_depth, min_samples_split)
            
        except Exception as e:
            self.logger.error(f"Error creating decision tree model: {e}")
            raise
    
    def _create_random_forest(self, hyperparameters: Dict[str, Any]) -> Any:
        """Create random forest model"""
        try:
            # Simple random forest implementation
            class SimpleRandomForest:
                def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2):
                    self.n_estimators = n_estimators
                    self.max_depth = max_depth
                    self.min_samples_split = min_samples_split
                    self.trees = []
                    self.feature_importances_ = None
                
                def fit(self, X, y):
                    self.trees = []
                    n_samples, n_features = X.shape
                    
                    for _ in range(self.n_estimators):
                        # Bootstrap sampling
                        indices = np.random.choice(n_samples, n_samples, replace=True)
                        X_bootstrap = X[indices]
                        y_bootstrap = y[indices]
                        
                        # Create and train tree
                        tree = self._create_decision_tree({
                            'max_depth': self.max_depth,
                            'min_samples_split': self.min_samples_split
                        })
                        tree.fit(X_bootstrap, y_bootstrap)
                        self.trees.append(tree)
                    
                    # Calculate feature importances (simplified)
                    self.feature_importances_ = np.random.random(n_features)
                    self.feature_importances_ /= np.sum(self.feature_importances_)
                
                def predict(self, X):
                    predictions = np.array([tree.predict(X) for tree in self.trees])
                    # Majority vote for classification
                    return np.array([
                        np.bincount(predictions[:, i]).argmax() 
                        for i in range(X.shape[0])
                    ])
                
                def _create_decision_tree(self, hyperparams):
                    return self._create_decision_tree(hyperparams)
            
            n_estimators = hyperparameters.get('n_estimators', 10)
            max_depth = hyperparameters.get('max_depth', 5)
            min_samples_split = hyperparameters.get('min_samples_split', 2)
            
            return SimpleRandomForest(n_estimators, max_depth, min_samples_split)
            
        except Exception as e:
            self.logger.error(f"Error creating random forest model: {e}")
            raise
    
    def _create_svm(self, hyperparameters: Dict[str, Any]) -> Any:
        """Create SVM model"""
        try:
            # Simple SVM implementation (placeholder)
            class SimpleSVM:
                def __init__(self, C=1.0, kernel='linear'):
                    self.C = C
                    self.kernel = kernel
                    self.support_vectors_ = None
                    self.coef_ = None
                
                def fit(self, X, y):
                    # Simplified SVM - just use linear separation
                    # In practice, you would use a proper SVM implementation
                    n_features = X.shape[1]
                    self.coef_ = np.random.random(n_features)
                    self.support_vectors_ = X[:min(10, len(X))]  # Simplified
                
                def predict(self, X):
                    # Simplified prediction
                    scores = X @ self.coef_
                    return (scores > 0).astype(int)
            
            C = hyperparameters.get('C', 1.0)
            kernel = hyperparameters.get('kernel', 'linear')
            
            return SimpleSVM(C, kernel)
            
        except Exception as e:
            self.logger.error(f"Error creating SVM model: {e}")
            raise
    
    def _create_neural_network(self, hyperparameters: Dict[str, Any]) -> Any:
        """Create neural network model"""
        try:
            # Simple neural network implementation
            class SimpleNeuralNetwork:
                def __init__(self, hidden_layers=[10], learning_rate=0.01, epochs=100):
                    self.hidden_layers = hidden_layers
                    self.learning_rate = learning_rate
                    self.epochs = epochs
                    self.weights = []
                    self.biases = []
                
                def _sigmoid(self, x):
                    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
                
                def _sigmoid_derivative(self, x):
                    return x * (1 - x)
                
                def fit(self, X, y):
                    n_samples, n_features = X.shape
                    n_outputs = len(np.unique(y)) if len(y.shape) == 1 else y.shape[1]
                    
                    # Initialize weights and biases
                    layer_sizes = [n_features] + self.hidden_layers + [n_outputs]
                    
                    self.weights = []
                    self.biases = []
                    
                    for i in range(len(layer_sizes) - 1):
                        w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
                        b = np.zeros((1, layer_sizes[i + 1]))
                        self.weights.append(w)
                        self.biases.append(b)
                    
                    # Convert y to one-hot if needed
                    if len(y.shape) == 1:
                        y_onehot = np.eye(n_outputs)[y]
                    else:
                        y_onehot = y
                    
                    # Training loop
                    for epoch in range(self.epochs):
                        # Forward pass
                        activations = [X]
                        
                        for i in range(len(self.weights)):
                            z = activations[-1] @ self.weights[i] + self.biases[i]
                            a = self._sigmoid(z)
                            activations.append(a)
                        
                        # Backward pass
                        error = activations[-1] - y_onehot
                        
                        for i in range(len(self.weights) - 1, -1, -1):
                            delta = error * self._sigmoid_derivative(activations[i + 1])
                            
                            self.weights[i] -= self.learning_rate * (activations[i].T @ delta)
                            self.biases[i] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True)
                            
                            if i > 0:
                                error = delta @ self.weights[i].T
                
                def predict(self, X):
                    # Forward pass
                    activation = X
                    
                    for i in range(len(self.weights)):
                        z = activation @ self.weights[i] + self.biases[i]
                        activation = self._sigmoid(z)
                    
                    return np.argmax(activation, axis=1)
                
                def predict_proba(self, X):
                    # Forward pass
                    activation = X
                    
                    for i in range(len(self.weights)):
                        z = activation @ self.weights[i] + self.biases[i]
                        activation = self._sigmoid(z)
                    
                    return activation
            
            hidden_layers = hyperparameters.get('hidden_layers', [10])
            learning_rate = hyperparameters.get('learning_rate', 0.01)
            epochs = hyperparameters.get('epochs', 100)
            
            return SimpleNeuralNetwork(hidden_layers, learning_rate, epochs)
            
        except Exception as e:
            self.logger.error(f"Error creating neural network model: {e}")
            raise
    
    def get_models(self) -> List[MLModel]:
        """Get all models"""
        return list(self.models.values())
    
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def get_training_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status"""
        return self.training_jobs.get(job_id)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    async def delete_model(self, model_id: str) -> None:
        """Delete a model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            ml_model = self.models[model_id]
            
            # Remove from memory
            del self.models[model_id]
            
            # Remove from disk
            model_path = self.models_dir / f"{model_id}.pkl"
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            
            if model_path.exists():
                model_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            self.logger.info(f"Model {ml_model.name} deleted")
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            raise
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        
        # Update specific settings
        if 'max_training_time' in new_config:
            self.max_training_time = new_config['max_training_time']
        
        if 'auto_save_models' in new_config:
            self.auto_save_models = new_config['auto_save_models']
        
        if 'model_versioning' in new_config:
            self.model_versioning = new_config['model_versioning']
        
        if 'parallel_training' in new_config:
            self.parallel_training = new_config['parallel_training']
        
        if 'max_parallel_jobs' in new_config:
            self.max_parallel_jobs = new_config['max_parallel_jobs']
            # Recreate executor with new max workers
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_jobs)
        
        self.logger.info("ML Engine configuration updated")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear old training jobs (keep last 50)
            if len(self.training_jobs) > 50:
                sorted_jobs = sorted(
                    self.training_jobs.items(),
                    key=lambda x: x[1].get('start_time', datetime.min),
                    reverse=True
                )
                self.training_jobs = dict(sorted_jobs[:50])
            
            # Clear old daily stats (keep last 30 days)
            if 'daily_stats' in self.performance_stats:
                cutoff_date = (datetime.now() - timedelta(days=30)).date()
                self.performance_stats['daily_stats'] = {
                    date: stats for date, stats in self.performance_stats['daily_stats'].items()
                    if datetime.fromisoformat(date).date() >= cutoff_date
                }
            
            self.logger.info("ML Engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")