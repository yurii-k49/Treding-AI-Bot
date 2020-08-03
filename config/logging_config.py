# config/logging_config.py
import logging
import os
from datetime import datetime
import sys

def setup_logging():
    """Setup detailed logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Create file handler for all logs
    full_log_file = f"{log_dir}/trading_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(full_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Create file handler for errors only
    error_log_file = f"{log_dir}/errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_file_handler = logging.FileHandler(error_log_file)
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Setup trading logger
    trading_logger = logging.getLogger('trading')
    trading_logger.setLevel(logging.DEBUG)
    trading_logger.addHandler(file_handler)
    trading_logger.addHandler(error_file_handler)
    trading_logger.addHandler(console_handler)

    # Setup analysis logger
    analysis_logger = logging.getLogger('analysis')
    analysis_logger.setLevel(logging.DEBUG)
    analysis_logger.addHandler(file_handler)
    analysis_logger.addHandler(error_file_handler)
    analysis_logger.addHandler(console_handler)

    # Setup model logger
    model_logger = logging.getLogger('model')
    model_logger.setLevel(logging.DEBUG)
    model_logger.addHandler(file_handler)
    model_logger.addHandler(error_file_handler)
    model_logger.addHandler(console_handler)

    # Setup utility logger
    util_logger = logging.getLogger('util')
    util_logger.setLevel(logging.DEBUG)
    util_logger.addHandler(file_handler)
    util_logger.addHandler(error_file_handler)
    util_logger.addHandler(console_handler)

    return trading_logger, analysis_logger

def get_logger(name):
    """Get logger by name"""
    return logging.getLogger(name)