import logging
import os


def init_logger(log_path: str):
    """
    log_path: str, path to save log
    level: str, level of logging [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    the_logger = logging.getLogger('logger')

    formatter = logging.Formatter('[%(asctime)s][%(filename)s][%(levelname)s]%(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    the_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    the_logger.addHandler(stream_handler)

    return the_logger
