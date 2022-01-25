#!/usr/bin/env python3


import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import particle_model.helpers as h

logger = h.easy_logger(__name__, logging.INFO)


import convergence as convg


def main():

    if len(sys.argv) < 2:
        sys.exit("Specify a file name")


    filename = sys.argv[1]
    data = np.genfromtxt(filename, comments='#').T

    dts = data[0]
    convg_list = [convg.Convg(errors = data[i]) for i in range(1, len(data))]

    for c in convg_list:
        c.order_convg(dts)
    
    convg.write_j_k(filename, convg_list)

    logger.info(f"\nConvergence analysis appended to {h.file_link(filename)}")

    return 0


if __name__ == "__main__":
    main()
