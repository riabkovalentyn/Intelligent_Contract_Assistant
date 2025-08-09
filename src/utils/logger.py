import logging
import os 
from logging import StreamHandler, FileHandler, Formatter, Logger
    
def get_logger(name: str='app') -> Logger:
    level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger