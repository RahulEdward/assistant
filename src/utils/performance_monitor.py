"""
Performance Monitor for Computer Assistant

This module provides comprehensive performance monitoring, profiling, and optimization
capabilities to ensure sub-second response times and high accuracy.
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
import os
from contextlib import contextmanager
import functools
import traceback

# Third-party imports
import numpy as np


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FunctionProfile:
    """Profiling data for a function."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_called: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 100.0


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    start_time: datetime
    end_time: datetime
    duration: timedelta
    system_metrics: List[SystemMetrics]
    function_profiles: Dict[str, FunctionProfile]
    custom_metrics: List[PerformanceMetric]
    bottlenecks: List[str]
    recommendations: List[str]
    overall_score: float


class PerformanceMonitor:
    """
    Advanced performance monitoring and optimization system.
    
    Features:
    - Real-time system resource monitoring
    - Function execution profiling
    - Response time tracking
    - Accuracy measurement
    - Bottleneck detection
    - Performance optimization recommendations
    - Automated performance testing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance monitor."""
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'monitoring_interval': 1.0,  # seconds
            'max_history_size': 1000,
            'performance_threshold': 1.0,  # seconds
            'accuracy_threshold': 0.99,  # 99%
            'cpu_threshold': 80.0,  # percent
            'memory_threshold': 80.0,  # percent
            'enable_profiling': True,
            'enable_system_monitoring': True,
            'report_interval': 300,  # 5 minutes
            'auto_optimization': True
        }
        
        if config:
            self.config.update(config)
        
        # Monitoring data
        self.metrics_history: deque = deque(maxlen=self.config['max_history_size'])
        self.system_metrics_history: deque = deque(maxlen=self.config['max_history_size'])
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.start_time = datetime.now()
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.accuracy_scores: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Optimization state
        self.optimization_suggestions: List[str] = []
        self.performance_alerts: List[str] = []
        
        # Initialize system monitoring
        self._init_system_monitoring()
        
        self.logger.info("PerformanceMonitor initialized successfully")
    
    def _init_system_monitoring(self):
        """Initialize system monitoring components."""
        try:
            # Test system monitoring capabilities
            psutil.cpu_percent()
            psutil.virtual_memory()
            psutil.disk_io_counters()
            psutil.net_io_counters()
            
            self.logger.info("System monitoring capabilities verified")
        except Exception as e:
            self.logger.warning(f"Some system monitoring features may not be available: {e}")
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                if self.config['enable_system_monitoring']:
                    system_metrics = self._collect_system_metrics()
                    self.system_metrics_history.append(system_metrics)
                    
                    # Check for performance issues
                    self._check_performance_thresholds(system_metrics)
                
                # Sleep until next monitoring interval
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            net_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0
            
            # GPU metrics (if available)
            gpu_percent = None
            gpu_memory_mb = None
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory_mb = gpu.memoryUsed
            except ImportError:
                pass  # GPU monitoring not available
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                gpu_percent=gpu_percent,
                gpu_memory_mb=gpu_memory_mb
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def _check_performance_thresholds(self, metrics: SystemMetrics):
        """Check if system metrics exceed performance thresholds."""
        alerts = []
        
        if metrics.cpu_percent > self.config['cpu_threshold']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.config['memory_threshold']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_percent and metrics.gpu_percent > 90:
            alerts.append(f"High GPU usage: {metrics.gpu_percent:.1f}%")
        
        # Add alerts to performance alerts list
        for alert in alerts:
            if alert not in self.performance_alerts:
                self.performance_alerts.append(alert)
                self.logger.warning(f"Performance alert: {alert}")
    
    @contextmanager
    def measure_execution_time(self, operation_name: str):
        """Context manager to measure execution time of operations."""
        start_time = time.perf_counter()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            self.error_counts[operation_name] += 1
            raise
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Record response time
            self.response_times.append(execution_time)
            
            # Record metric
            metric = PerformanceMetric(
                name=f"{operation_name}_execution_time",
                value=execution_time,
                unit="seconds",
                timestamp=datetime.now(),
                category="performance",
                metadata={"error_occurred": error_occurred}
            )
            self.metrics_history.append(metric)
            
            # Check performance threshold
            if execution_time > self.config['performance_threshold']:
                self.logger.warning(
                    f"Slow operation detected: {operation_name} took {execution_time:.3f}s"
                )
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution."""
        if not self.config['enable_profiling']:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            start_time = time.perf_counter()
            error_occurred = False
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                raise
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                # Update function profile
                self._update_function_profile(func_name, execution_time, error_occurred)
        
        return wrapper
    
    def _update_function_profile(self, func_name: str, execution_time: float, error_occurred: bool):
        """Update profiling data for a function."""
        if func_name not in self.function_profiles:
            self.function_profiles[func_name] = FunctionProfile(function_name=func_name)
        
        profile = self.function_profiles[func_name]
        profile.call_count += 1
        profile.total_time += execution_time
        profile.min_time = min(profile.min_time, execution_time)
        profile.max_time = max(profile.max_time, execution_time)
        profile.avg_time = profile.total_time / profile.call_count
        profile.last_called = datetime.now()
        
        if error_occurred:
            profile.error_count += 1
        
        profile.success_rate = ((profile.call_count - profile.error_count) / profile.call_count) * 100
    
    def record_accuracy(self, operation_name: str, accuracy_score: float):
        """Record accuracy measurement for an operation."""
        self.accuracy_scores.append(accuracy_score)
        
        metric = PerformanceMetric(
            name=f"{operation_name}_accuracy",
            value=accuracy_score,
            unit="percentage",
            timestamp=datetime.now(),
            category="accuracy"
        )
        self.metrics_history.append(metric)
        
        # Check accuracy threshold
        if accuracy_score < self.config['accuracy_threshold']:
            self.logger.warning(
                f"Low accuracy detected: {operation_name} accuracy is {accuracy_score:.2%}"
            )
    
    def record_custom_metric(self, name: str, value: float, unit: str = "", 
                           category: str = "custom", metadata: Optional[Dict[str, Any]] = None):
        """Record a custom performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            metadata=metadata or {}
        )
        
        self.metrics_history.append(metric)
        self.custom_metrics[name].append(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance metrics."""
        summary = {
            'monitoring_duration': datetime.now() - self.start_time,
            'total_operations': len(self.response_times),
            'response_times': {},
            'accuracy': {},
            'system_resources': {},
            'function_profiles': {},
            'alerts': self.performance_alerts[-10:],  # Last 10 alerts
        }
        
        # Response time statistics
        if self.response_times:
            summary['response_times'] = {
                'average': statistics.mean(self.response_times),
                'median': statistics.median(self.response_times),
                'min': min(self.response_times),
                'max': max(self.response_times),
                'p95': np.percentile(list(self.response_times), 95),
                'p99': np.percentile(list(self.response_times), 99),
                'under_threshold': sum(1 for t in self.response_times if t < self.config['performance_threshold']) / len(self.response_times)
            }
        
        # Accuracy statistics
        if self.accuracy_scores:
            summary['accuracy'] = {
                'average': statistics.mean(self.accuracy_scores),
                'min': min(self.accuracy_scores),
                'max': max(self.accuracy_scores),
                'above_threshold': sum(1 for a in self.accuracy_scores if a >= self.config['accuracy_threshold']) / len(self.accuracy_scores)
            }
        
        # System resource statistics
        if self.system_metrics_history:
            recent_metrics = list(self.system_metrics_history)[-10:]  # Last 10 measurements
            summary['system_resources'] = {
                'cpu_avg': statistics.mean([m.cpu_percent for m in recent_metrics]),
                'memory_avg': statistics.mean([m.memory_percent for m in recent_metrics]),
                'cpu_max': max([m.cpu_percent for m in recent_metrics]),
                'memory_max': max([m.memory_percent for m in recent_metrics])
            }
        
        # Function profile summary
        for func_name, profile in self.function_profiles.items():
            summary['function_profiles'][func_name] = {
                'call_count': profile.call_count,
                'avg_time': profile.avg_time,
                'success_rate': profile.success_rate,
                'total_time': profile.total_time
            }
        
        return summary
    
    def generate_performance_report(self) -> PerformanceReport:
        """Generate a comprehensive performance report."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Calculate overall performance score
        overall_score = self._calculate_performance_score()
        
        report = PerformanceReport(
            report_id=f"perf_report_{int(time.time())}",
            start_time=self.start_time,
            end_time=end_time,
            duration=duration,
            system_metrics=list(self.system_metrics_history),
            function_profiles=self.function_profiles.copy(),
            custom_metrics=list(self.metrics_history),
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            overall_score=overall_score
        )
        
        return report
    
    def _detect_bottlenecks(self) -> List[str]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        # Check slow functions
        for func_name, profile in self.function_profiles.items():
            if profile.avg_time > self.config['performance_threshold']:
                bottlenecks.append(f"Slow function: {func_name} (avg: {profile.avg_time:.3f}s)")
        
        # Check system resources
        if self.system_metrics_history:
            recent_metrics = list(self.system_metrics_history)[-10:]
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
            
            if avg_cpu > self.config['cpu_threshold']:
                bottlenecks.append(f"High CPU usage: {avg_cpu:.1f}%")
            
            if avg_memory > self.config['memory_threshold']:
                bottlenecks.append(f"High memory usage: {avg_memory:.1f}%")
        
        # Check response times
        if self.response_times:
            slow_operations = sum(1 for t in self.response_times if t > self.config['performance_threshold'])
            if slow_operations > len(self.response_times) * 0.1:  # More than 10% slow
                bottlenecks.append(f"Frequent slow operations: {slow_operations}/{len(self.response_times)}")
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Function-specific recommendations
        for func_name, profile in self.function_profiles.items():
            if profile.avg_time > self.config['performance_threshold']:
                recommendations.append(f"Optimize {func_name}: consider caching, algorithm improvements, or async execution")
            
            if profile.success_rate < 95:
                recommendations.append(f"Improve error handling in {func_name}: success rate is {profile.success_rate:.1f}%")
        
        # System-level recommendations
        if self.system_metrics_history:
            recent_metrics = list(self.system_metrics_history)[-10:]
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
            
            if avg_cpu > 70:
                recommendations.append("Consider CPU optimization: use multiprocessing, optimize algorithms, or upgrade hardware")
            
            if avg_memory > 70:
                recommendations.append("Consider memory optimization: implement caching strategies, reduce memory footprint, or add more RAM")
        
        # Response time recommendations
        if self.response_times:
            p95_time = np.percentile(list(self.response_times), 95)
            if p95_time > self.config['performance_threshold']:
                recommendations.append("Improve response times: implement caching, optimize database queries, or use async processing")
        
        return recommendations
    
    def _calculate_performance_score(self) -> float:
        """Calculate an overall performance score (0-100)."""
        score = 100.0
        
        # Response time score (40% weight)
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            response_score = max(0, 100 - (avg_response_time / self.config['performance_threshold']) * 40)
            score = score * 0.6 + response_score * 0.4
        
        # Accuracy score (30% weight)
        if self.accuracy_scores:
            avg_accuracy = statistics.mean(self.accuracy_scores)
            accuracy_score = avg_accuracy * 100
            score = score * 0.7 + accuracy_score * 0.3
        
        # System resource score (20% weight)
        if self.system_metrics_history:
            recent_metrics = list(self.system_metrics_history)[-10:]
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
            
            resource_score = max(0, 100 - max(avg_cpu, avg_memory))
            score = score * 0.8 + resource_score * 0.2
        
        # Error rate score (10% weight)
        total_calls = sum(profile.call_count for profile in self.function_profiles.values())
        total_errors = sum(profile.error_count for profile in self.function_profiles.values())
        
        if total_calls > 0:
            error_rate = total_errors / total_calls
            error_score = max(0, 100 - error_rate * 100)
            score = score * 0.9 + error_score * 0.1
        
        return min(100.0, max(0.0, score))
    
    def optimize_performance(self):
        """Automatically apply performance optimizations."""
        if not self.config['auto_optimization']:
            return
        
        self.logger.info("Running automatic performance optimization...")
        
        # Clear old data to free memory
        if len(self.metrics_history) > self.config['max_history_size'] * 0.8:
            # Keep only recent 50% of data
            keep_count = len(self.metrics_history) // 2
            self.metrics_history = deque(list(self.metrics_history)[-keep_count:], 
                                       maxlen=self.config['max_history_size'])
        
        # Clear old system metrics
        if len(self.system_metrics_history) > self.config['max_history_size'] * 0.8:
            keep_count = len(self.system_metrics_history) // 2
            self.system_metrics_history = deque(list(self.system_metrics_history)[-keep_count:], 
                                              maxlen=self.config['max_history_size'])
        
        # Clear old response times
        if len(self.response_times) > 500:
            self.response_times = deque(list(self.response_times)[-250:], maxlen=1000)
        
        # Clear old accuracy scores
        if len(self.accuracy_scores) > 500:
            self.accuracy_scores = deque(list(self.accuracy_scores)[-250:], maxlen=1000)
        
        self.logger.info("Performance optimization completed")
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to a file."""
        try:
            data = {
                'export_time': datetime.now().isoformat(),
                'config': self.config,
                'summary': self.get_performance_summary(),
                'function_profiles': {name: {
                    'function_name': profile.function_name,
                    'call_count': profile.call_count,
                    'total_time': profile.total_time,
                    'avg_time': profile.avg_time,
                    'min_time': profile.min_time,
                    'max_time': profile.max_time,
                    'error_count': profile.error_count,
                    'success_rate': profile.success_rate,
                    'last_called': profile.last_called.isoformat() if profile.last_called else None
                } for name, profile in self.function_profiles.items()},
                'recent_metrics': [
                    {
                        'name': metric.name,
                        'value': metric.value,
                        'unit': metric.unit,
                        'timestamp': metric.timestamp.isoformat(),
                        'category': metric.category,
                        'metadata': metric.metadata
                    }
                    for metric in list(self.metrics_history)[-100:]  # Last 100 metrics
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Performance metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update configuration settings."""
        self.config.update(config_updates)
        self.logger.info(f"Performance monitor configuration updated: {config_updates}")
    
    def cleanup(self):
        """Clean up resources and stop monitoring."""
        self.stop_monitoring()
        
        # Clear data structures
        self.metrics_history.clear()
        self.system_metrics_history.clear()
        self.function_profiles.clear()
        self.custom_metrics.clear()
        self.response_times.clear()
        self.accuracy_scores.clear()
        
        self.logger.info("PerformanceMonitor cleanup completed")