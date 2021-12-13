"""
Some things that can be used in any file
Contains:
    - Timing decorator
    - Very simple Logger class
"""

import time
import logging


def easy_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)-10s- %(levelname)-7s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = easy_logger(__name__)


def timing(func):
    def wrapper(*args, **kwargs):
        separator = "-"
        separate = 79 * separator
        print(separate)
        logger.info(f"Starting timer for function '{func.__name__}'\n")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print()
        logger.info(f"'{func.__name__}' took {end_time - start_time:6.2f} "
                    f"seconds to execute")
        print(separate)
        return result
    return wrapper
