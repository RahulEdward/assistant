"""
Utilities Package for Computer Assistant

This package contains utility modules for logging, configuration, performance monitoring,
testing, and other helper functions.
"""

__version__ = "1.0.0"
__author__ = "Computer Assistant Team"

# Import main utility classes
from .logger import Logger
from .performance_monitor import PerformanceMonitor, PerformanceMetric, SystemMetrics
from .test_runner import TestRunner, TestResult, TestSuite, TestReport