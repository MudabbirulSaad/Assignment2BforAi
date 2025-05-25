#!/usr/bin/env python3
"""
TBRGS Logging Module

This module provides centralized logging functionality for the TBRGS project.
It configures logging with both console and file handlers.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

class TBRGSLogger:
    """
    Centralized logging class for the TBRGS project.
    
    This class configures logging with both console and file handlers,
    allowing for consistent logging across the entire project.
    """
    
    # Class variable to track if logger has been initialized
    _initialized = False
    
    @classmethod
    def setup_logging(cls, log_level=logging.INFO, log_file=None):
        """
        Set up logging configuration for the entire application.
        
        Args:
            log_level (int): Logging level (default: logging.INFO)
            log_file (str): Path to log file (default: None, will use default path)
        
        Returns:
            logging.Logger: Configured logger instance
        """
        if cls._initialized:
            return logging.getLogger('tbrgs')
        
        # Create logs directory if it doesn't exist
        if log_file is None:
            # Get project root directory (3 levels up from this file)
            project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(
                               os.path.dirname(os.path.abspath(__file__))))))
            logs_dir = project_root / 'app' / 'core' / 'logging' / 'logs'
            logs_dir.mkdir(exist_ok=True, parents=True)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = logs_dir / f'tbrgs_{timestamp}.log'
        
        # Configure root logger
        logger = logging.getLogger('tbrgs')
        logger.setLevel(log_level)
        
        # Remove existing handlers if any
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)  # File gets everything
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Mark as initialized
        cls._initialized = True
        
        logger.info("Logging system initialized")
        return logger
    
    @classmethod
    def get_logger(cls, name=None):
        """
        Get a logger instance for the specified module.
        
        Args:
            name (str): Name of the module requesting the logger
                        (default: None, returns root logger)
        
        Returns:
            logging.Logger: Logger instance
        """
        if not cls._initialized:
            cls.setup_logging()
        
        if name:
            return logging.getLogger(f'tbrgs.{name}')
        return logging.getLogger('tbrgs')

# Initialize the default logger
logger = TBRGSLogger.setup_logging()
