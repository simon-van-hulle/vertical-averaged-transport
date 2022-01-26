#!/usr/bin/env python3


from particle_model.particle_model_oo import *


def main():

    Nx = 20
    Ny = 20
    N = Nx * Ny

    dt = 0.005
    nSteps = 10

    x, y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))

    w = WienerProcess(nSteps, N, dt)

    particles = Particles(N, w, w, x=x.flatten(), y=y.flatten())

    steps = [1]

    for step in steps:
        wStep = w.get_step(step, 1)
        particles.calc_dispersion()
        dxE, dyE = particles.euler_step(wStep, wStep, dt)
        dxM, dyM = particles.milstein_step(wStep, wStep, dt)

        dxDiff = dxM - dxE
        dyDiff = dyM - dyE
        plt.figure()
        # plt.quiver(x, y, dxE, dyE, color='g', label="Euler")
        plt.quiver(x, y, dxM, dyM, color='r', label="Milstein")
        # plt.quiver(x, y, dxDiff, dyDiff, color='k', label="Difference")

        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()

