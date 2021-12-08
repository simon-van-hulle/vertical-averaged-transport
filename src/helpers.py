"""
Some things that can be used in any file
"""

import time


def timing(func):
    def wrapper(*args, **kwargs):
        separator = "-"
        separate = 79 * separator
        print(separate)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\n{func.__name__} took {end_time - start_time:6.2f} seconds "
              f"to execute")
        print(separate)
        return result
    return wrapper


CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0


class Logger:
    def __init__(self, level=NOTSET):
        self.level = level

    def debug(self, *args, **kwargs):
        if self.level <= DEBUG:
            print('[  DEBUG ]', *args, **kwargs)

    def info(self, *args, **kwargs):
        if self.level <= INFO:
            print('[  INFO  ]', *args, **kwargs)

    def warn(self, *args, **kwargs):
        if self.level <= WARNING:
            print('[  WARN  ]', *args, **kwargs)

    def error(self, *args, **kwargs):
        if self.level <= WARNING:
            print('[  ERROR ]', *args, **kwargs)

    def critical(self, *args, **kwargs):
        if self.level <= CRITICAL:
            print('[CRITICAL]', *args, **kwargs)

    def warning(self, *args, **kwargs):
        self.warn(*args, **kwargs)

    def fatal(self, *args, **kwargs):
        self.critical(*args, **kwargs)
