"""
Logging Configuration
====================

Centralized logging setup for the Bank Churn Prediction system.
Provides consistent logging across all modules with file and console handlers.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import functools
import time


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
            
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_'):
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Store original levelname
        original_levelname = record.levelname
        
        # Add color to level name
        if original_levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[original_levelname]}{original_levelname:<8}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False,
    colored_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        json_format: Use JSON format for file logs
        colored_console: Use colored output for console logs
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    
    # Create logs directory
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if colored_console and sys.stdout.isatty():
            console_format = ColoredFormatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_format = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if log_to_file:
        from logging.handlers import RotatingFileHandler
        
        timestamp = datetime.now().strftime('%Y%m%d')
        
        # Main log file with rotation
        main_log_file = Path(log_dir) / f"bank_churn_{timestamp}.log"
        file_handler = RotatingFileHandler(
            main_log_file, 
            mode='a', 
            encoding='utf-8',
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | [%(filename)s:%(lineno)d] | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file (errors only)
        error_log_file = Path(log_dir) / f"errors_{timestamp}.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            mode='a',
            encoding='utf-8',
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    
    # Log initial message
    root_logger.info(f"Logging initialized at {log_level} level")
    if log_to_file:
        root_logger.info(f"Logs directory: {log_dir}")


def get_logger(name: str, extra_data: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__ of the module)
        extra_data: Extra data to include in all log messages
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Add extra data if provided using log record factory
    if extra_data:
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.extra_data = extra_data
            return record
        
        logging.setLogRecordFactory(record_factory)
    
    return logger


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: Logger instance
        level: Logging level for the function call messages
    
    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log function call
            logger.log(level, f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.log(level, f"{func.__name__} completed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


def setup_audit_logging(audit_log_dir: str = "logs") -> logging.Logger:
    """
    Setup audit logging for tracking all predictions and model decisions.
    
    Args:
        audit_log_dir: Directory for audit logs
    
    Returns:
        Audit logger instance
    """
    from logging.handlers import RotatingFileHandler
    
    audit_log_path = Path(audit_log_dir)
    audit_log_path.mkdir(parents=True, exist_ok=True)
    
    audit_logger = logging.getLogger('audit')
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Don't propagate to root logger
    
    # Remove existing handlers to avoid duplicates
    for handler in audit_logger.handlers[:]:
        handler.close()
        audit_logger.removeHandler(handler)
    
    # Audit log file with CSV-like format
    audit_file = audit_log_path / "audit_trail.csv"
    
    # Create file with header if it doesn't exist
    if not audit_file.exists():
        with open(audit_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,user,action,customer_id,prediction,probability,model_version,features_used\n")
    
    audit_handler = RotatingFileHandler(
        audit_file, 
        mode='a', 
        encoding='utf-8',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    audit_handler.setLevel(logging.INFO)
    
    # Simple formatter for CSV-like format
    audit_formatter = logging.Formatter('%(message)s')
    audit_handler.setFormatter(audit_formatter)
    
    audit_logger.addHandler(audit_handler)
    
    return audit_logger


class PerformanceLogger:
    """Helper class for performance monitoring and logging"""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {execution_time:.4f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {execution_time:.4f}s: {exc_val}")
            
    @classmethod
    def time_operation(cls, logger: logging.Logger, operation_name: str):
        """Context manager for timing operations"""
        return cls(logger, operation_name)


# Don't auto-initialize logging by default to allow explicit configuration
# Users should call setup_logging() explicitly in their main script