#!/usr/bin/env python3


import logging
import os
import sys
from ast import Call
from cgitb import strong
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

import scipy.optimize as spopt

import particle_model.helpers as h
from particle_model.particle_model_oo import *

logger = h.easy_logger(__name__, logging.INFO)



CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")

CONVG_PADDING = 20

REFINEMENTS = 20
REFINE_METHOD = 'linear'


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


def plus_minus_str(popt, stdevs, decimals=3):
    strs = []
    for p, stdev in zip(popt, stdevs):
        strs.append(f"{p:.{decimals}f} +- {stdev:.{decimals}f}")

    return strs


@dataclass
class Convg:
    name: str = field(default="Convg")
    convg_type: str = 'strong'
    errors: List[float] = field(default_factory=list)
    last_error: int = None
    weak_f: Callable[[np.ndarray], np.ndarray] = None
    j: float = None
    k: float = None
    j_stdev: float = None
    k_stdev: float = None
    j_str: str = None
    k_str: str = None
    scheme: str = None

    def convg_f(self, *args, **kwargs):
        if self.convg_type == 'weak':
            return weak_convg_f(*args, **kwargs, h=self.weak_f)
        elif self.convg_type == 'strong':
            return strong_convg_f(*args, **kwargs)

    def calc_error(self, *args):
        self.last_error = self.convg_f(*args)
        self.errors.append(self.last_error)

    def order_convg(self, dts):
        popt, stdevs = order_convg(dts, self.errors)
        self.j, self.k = popt
        self.j_stdev, self.k_stdev = stdevs
        self.j_str, self.k_str = plus_minus_str(popt, stdevs)


def convg_f_name(simul):
    return out_file("convg_" + simul.standard_title() + ".txt")


def write_header(filename, convg_list, simul):
    with open(filename, "w") as convg_f:
        convg_f.write(f"# Convergence Analysis with parameters:\n"
                      f"#   scheme              : {simul._scheme:s}\n"
                      f"#   Final time T        : {simul._end_time:f}\n"
                      f"#   Number of particles : {simul._num_particles}\n"
                      f"#\n"
                      f"# NOTE: The most recent j and K data is on the bottom\n"
                      f"#\n")

        convg_f.write(f"# {'dt':<{CONVG_PADDING}s}")

        for c in convg_list:
            convg_f.write(f"{c.name:<{CONVG_PADDING}s}")

        convg_f.write("\n")


def write_errors(filename, dts, convg_list):
    with open(filename, "a") as convg_f:
        for i, dt in enumerate(dts):
            convg_f.write(f"  {dt:<{CONVG_PADDING}f}")

            for c in convg_list:
                convg_f.write(f"{c.errors[i]:<{CONVG_PADDING}f}")

            convg_f.write("\n")
        convg_f.write("\n")


def write_j_k(filename, convg_list):
    with open(filename, "a") as convg_f:
        convg_f.write(f"# {'Order j':<{CONVG_PADDING}s}")

        for c in convg_list:
            convg_f.write(f"{c.j_str:<{CONVG_PADDING}s}")

        convg_f.write("\n")
        convg_f.write(f"# {'Factor K':<{CONVG_PADDING}s}")

        for c in convg_list:
            convg_f.write(f"{c.k_str:<{CONVG_PADDING}s}")

        convg_f.write("\n\n")


def write_to_file(simul, dts, convg_list, filename):
    if os.path.isfile(filename) == 0:
        write_header(filename, convg_list, simul)

    write_errors(filename, dts, convg_list)
    write_j_k(filename, convg_list)

    logger.info(f"\n"
                f"Finished convergence analysis\n"
                f"  Saved convergence report to {h.file_link(filename)}.\n\n"
                f"  Visualise with the `plot_convergence` script.")


def refine_func(i, method='linear'):
    if method == 'linear':
        return 2 * i + 1
    elif method == 'exp':
        return 2 ** i


def summarise(name, value):
    logger.info(f"{name:<{2*CONVG_PADDING}s}: {value:<{CONVG_PADDING}f}")


def convergence_tests(config, refinements='5', refine_method='exp'):

    logger.info(f"Starting convergence analysis with\n"
                f"  {refinements} dt refinements.\n"
                f"  {refine_method} refinement is used.\n")

    simul = ParticleSimulation(config)
    filename = convg_f_name(simul)

    dts = []
    xy_best = None

    convg_list = [
        Convg("Strong", 'strong'),
        Convg("Weak-($h(X)=X$)", 'weak', weak_f=lambda x: x),
        Convg("Weak-($h(X)=X^2$)", 'weak', weak_f=lambda x: x * x)
    ]

    for i in range(refinements + 1):
        simul.set_steps_per_it(refine_func(i, refine_method))
        dt = simul.dt()

        try:
            xy_final = simul.run(convergenc_tests=True)
            if type(xy_final) == type(0) or type(xy_final) == type(None):
                raise TypeError
        except:
            logger.warning("At least some particles left the domain.")
            break

        if i == 0:
            xy_best = xy_final

        elif np.all(xy_final):

            summarise("dt", dt)
            for c in convg_list:
                c.calc_error(xy_best, xy_final)
                summarise(c.name, c.last_error)

            dts.append(dt)

    for c in convg_list:
        c.order_convg(dts)

    write_to_file(simul, dts, convg_list, filename)


if __name__ == "__main__":
    np.seterr(all='raise')

    config = parse_configuration()
    config.store_plot = False
    config.make_animation = False
    config.show_end = False

    convergence_tests(config, REFINEMENTS, REFINE_METHOD)

    if config.show_end:
        plt.show()
