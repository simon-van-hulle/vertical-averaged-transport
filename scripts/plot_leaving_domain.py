#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import logging
import sys
import particle_model.helpers as h

logger = h.easy_logger(__name__, logging.INFO)

def plot_file(filename, label):

    data= np.genfromtxt(filename).T
    dts = data[0]
    percentages = data[1]

    plt.scatter(dts, percentages, label=label)
    plt.axvline(np.max(dts), color='k', linestyle='--')
    plt.xlabel(f"Time step $\Delta t$")
    plt.ylabel(f"Percentage of particles that left the domain.")
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        logger.error("Expected file name")

    for filename, label in zip(sys.argv[1:], ["Euler", "Milstein"]):
        plot_file(filename, label)
    plt.axhline(0, color='k')
    plt.axvline(0, color='k')

    
    plt.show()
