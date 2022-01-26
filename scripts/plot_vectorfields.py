#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt


def H(x, y):
    return 15 + 5 * x


def u(x, y):
    return -y * (1 - x * x) / H(x, y)


def v(x, y):
    return x * (1 - y * y) / H(x, y)


def velPlot(lim, n):
    x, y = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))

    uVec = u(x,y)
    vVec = v(x,y)

    plt.quiver(x, y, uVec, vVec, linewidths=np.linspace(0, 1, x.size))
    plt.show()

if __name__ == "__main__":

    plt.hlines(1, -1, 1)
    plt.hlines(-1, -1, 1)
    plt.vlines(-1, -1, 1)
    plt.vlines(1, -1, 1)

    N = 20
    LIM = 1.5

    velPlot(LIM, N)

