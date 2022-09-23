#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt

from particle_model_oo import *


def main():

    # nSteps = 61
    nSteps = 31
    nPart = 1
    dt = 0.05

    w = WienerProcess(nSteps, nPart, dt)

    title = "Wiener Process"
    plt.figure(title)
    plt.clf()

    for i, ls in zip([1, 2, 3], ['-', ':', '--']):
        process = w.get_process()[::i]
        steps = np.arange(0, nSteps, i)
        plt.plot(steps * dt, process, linestyle=ls,
                 label=fr"$\Delta t = {dt * i:.2f}$")

    plt.suptitle(title, fontsize=16, weight='bold')
    # plt.title(rf"{self._n_particles} particles $-$ $\Delta t$"
    #             rf" = {self._dt}", fontsize=11)
    plt.ylabel(rf"Wiener process $W_t$")
    plt.xlabel(rf"Time $t$")
    plt.legend()

    return 0


if __name__ == '__main__':
    main()
    plt.show()
