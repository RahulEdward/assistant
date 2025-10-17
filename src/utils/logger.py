"""
Logging utility for Desktop Assistant
Provides comprehensive logging with file rotation, different log levels, and structured output.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup a comprehensive logger with file and console output
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to 'logs')
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory
    if log_dir is None:
        log_dir = "logs"
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation
    log_file = log_path / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    
    # Error file handler (separate file for errors)
    error_file = log_path / f"{name}_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)
    
    return logger


class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self, name: str = "Performance"):
        self.logger = setup_logger(f"{name}_Performance")
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and log the duration"""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        start_time = self.start_times.pop(operation)
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Operation '{operation}' completed in {duration:.3f} seconds")
        return duration
    
    def log_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log a performance metric"""
        self.logger.info(f"Metric: {metric_name} = {value} {unit}")


class StructuredLogger:
    """Logger for structured data and events"""
    
    def __init__(self, name: str = "Structured"):
        self.logger = setup_logger(f"{name}_Structured")
    
    def log_event(self, event_type: str, data: dict):
        """Log a structured event"""
        timestamp = datetime.now().isoformat()
        log_data = {
            'timestamp': timestamp,
            'event_type': event_type,
            'data': data
        }
        self.logger.info(f"EVENT: {log_data}")
    
    def log_command(self, command: str, parameters: dict, result: dict):
        """Log command execution"""
        self.log_event('command_execution', {
            'command': command,
            'parameters': parameters,
            'result': result
        })
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log structured error information"""
        self.log_event('error', {
            'error_type': error_type,
            'message': error_message,
            'context': context or {}
        })


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return setup_logger(name)


def set_global_log_level(level: str):
    """Set the log level for all existing loggers"""
    log_level = getattr(logging, level.upper())
    
    # Set root logger level
    logging.getLogger().setLevel(log_level)
    
    # Set level for all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Update console handlers
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)