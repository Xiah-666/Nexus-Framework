"""
Centralized logging configuration for NEMESIS-NEXUS framework.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# Global logger registry
_loggers = {}


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup centralized logging for the application."""
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified name."""
    if name not in _loggers:
        _loggers[name] = logging.getLogger(f"nemesis.{name}")
    return _loggers[name]
