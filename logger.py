import os
import sys
import logging
from datetime import datetime
from pathlib import Path

class SynapseLogger:
    """
    A class to handle logging for the SynapseClusterEM project.
    Encapsulates both console and file logging functionality.
    """
    
    def __init__(self, name='SynapseClusterEM', level=logging.INFO):
        """
        Initialize the logger with console output.
        
        Args:
            name (str): Name of the logger
            level (int): Logging level (default: logging.INFO)
        """
        # Set up basic configuration for console logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
            ]
        )
        
        # Get the logger instance
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
    
    def setup_file_logger(self, output_dir):
        """
        Set up file logging in addition to console logging.
        
        Args:
            output_dir (str or Path): Directory where log file will be saved
            
        Returns:
            str: Path to the created log file
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        log_file = output_dir / f"synapse_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging to {log_file}")
        return str(log_file)
    
    # Convenience methods that delegate to the logger
    def info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)
    
    def critical(self, message):
        """Log a critical message"""
        self.logger.critical(message) 