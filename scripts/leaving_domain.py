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
from sympy import interactive_traversal

import particle_model.helpers as h
from particle_model.particle_model_oo import *

logger = h.easy_logger(__name__, logging.INFO)


CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")

DOMAIN_PADDING = 20


def domain_f_name(config):
    return out_file(f"domain_{config.scheme}_S{config.total_steps}.txt")


def write_header(config, filename):
    with open(filename, "a") as domain_f:
        domain_f.write(f"# Domain Analysis with parameters:\n"
                       f"#   Scheme              : {config.scheme:s}\n"
                       f"#   Total Steps         : {config.total_steps:d}\n"
                       f"#"
                       f"#\n")

        domain_f.write(f"# {'dt':<{DOMAIN_PADDING}s}"
                       f"{'% left domain':<{DOMAIN_PADDING}}")

        domain_f.write("\n")


def write_results(dts, perc_left, filename):
    with open(filename, "a") as domain_f:
        for dt, percentage in zip(dts, perc_left):
            domain_f.write(f"  "
                           f"{dt:<{DOMAIN_PADDING}f}"
                           f"{percentage:<{DOMAIN_PADDING}f}"
                           f"\n")


def write_to_file(config, dts, perc_left, filename):
    if os.path.isfile(filename) == 0:
        write_header(config, filename)

    write_results(dts, perc_left, filename)

    logger.info(f"\n"
                f"Finished domain analysis\n"
                f"  Saved report to {h.file_link(filename)}.\n\n"
                f"  Visualise with the `plot_leaving_domain` script.")


def domain_test(config, repetitions=1, intervals=1, interval_size=1):

    logger.info(f"Starting domain analysis with\n"
                f"  {repetitions:4d} repetitions per dt value.\n"
                f"  {intervals:4d} intervals in end-time.\n"
                f"  {interval_size:4.1f} interval size")

    filename = domain_f_name(config)

    dts = []
    perc_left_domain = []

    n_particles = config.num_particles

    for interval in range(intervals):
        config.end_time += interval_size

        for repetition in range(repetitions):
            simul = ParticleSimulation(config)
            logger.info(f"\nTIME STEP: {simul.dt():f}")

            try:
                last_left_domain = simul.run()
                if type(last_left_domain) == type(None):
                    raise ValueError
            except:
                continue

            dts.append(simul.dt())
            percentage = 100 * last_left_domain / n_particles
            perc_left_domain.append(percentage)

            logger.info(f"{percentage} % of the particles left the domain.")

    write_to_file(config, dts, perc_left_domain, filename)


if __name__ == "__main__":
    np.seterr(all='raise')

    config = parse_configuration()

    config.make_animation = False
    config.store_plot = False
    config.show_end = False

    REPETITIONS = 6
    TIME_INTERVALS = 15
    TIME_INTERVAL_SIZE = 0.5

    domain_test(config, repetitions=REPETITIONS, intervals=TIME_INTERVALS,
                interval_size=TIME_INTERVAL_SIZE)
