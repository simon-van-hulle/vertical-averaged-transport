"""
Some things that can be used in any file
Contains:
    - Timing decorator
    - Very simple Logger class
"""

import time
import logging
import numpy as np

import scipy.optimize as spopt

def easy_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(name)-10s- %(levelname)-7s: %(message)s')
    formatter = logging.Formatter('%(message)s')
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



def strong_convg_f(xy_best, xy_approx):
    return np.linalg.norm(xy_best - xy_approx, axis=0).mean()


def weak_convg_f(xy_best, xy_approx, h):
    return np.linalg.norm((h(xy_best) - h(xy_approx)).mean(axis=1))


def error_f(x, j, k):
        return k * x ** j

def order_convg(dts, errors):
    popt, pcov = spopt.curve_fit(error_f, dts, errors)
    stdevs = np.sqrt(np.diag(pcov))
    res = errors - error_f(dts, *popt)
    return popt, stdevs