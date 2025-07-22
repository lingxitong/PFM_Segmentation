import os
import logging
import sys

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )