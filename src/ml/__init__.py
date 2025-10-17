"""
Machine Learning Framework for Computer Assistant

This package provides comprehensive machine learning capabilities including:
- Performance prediction and optimization
- Adaptive learning algorithms
- Model training and evaluation
- Real-time inference and decision making
- Continuous improvement through feedback loops
"""

from .ml_engine import (
    MLEngine,
    ModelType,
    TrainingData,
    ModelMetrics,
    PredictionResult,
    MLModel
)

from .adaptive_learner import (
    AdaptiveLearner,
    LearningStrategy,
    AdaptationMetrics,
    LearningResult
)

from .performance_predictor import (
    PerformancePredictor,
    PredictionType,
    PerformanceData,
    PredictionMetrics
)

__version__ = "1.0.0"
__author__ = "Computer Assistant ML Team"
__description__ = "Advanced machine learning framework for intelligent automation"

__all__ = [
    'MLEngine',
    'ModelType',
    'TrainingData',
    'ModelMetrics',
    'PredictionResult',
    'MLModel',
    'AdaptiveLearner',
    'LearningStrategy',
    'AdaptationMetrics',
    'LearningResult',
    'PerformancePredictor',
    'PredictionType',
    'PerformanceData',
    'PredictionMetrics'
]