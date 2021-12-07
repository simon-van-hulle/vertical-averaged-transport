"""
Very simple Logger class without fancy features. Just logging
"""

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
