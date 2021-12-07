#!/usr/bin/env python3

"""
A first, intuitive implementation.
I realise this object-oriented approach is completely unnecessary and rather
inefficient, so let's make a better version using numpy arrays.

The code spits out results, but haven't checked anything (not reliable...)
"""

import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import log

logger = log.Logger()


class Domain:
    def __init__(self, xmin=-1, xmax=1, ymin=-1, ymax=1):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.minDx = 0.01 * (self.xmax - self.xmin)
        self.minDy = 0.01 * (self.ymax - self.ymin)


def depth_func(x, y=None):
    return 15 + 5 * x


def dispersion_coeffs(x, y):
    return [1 + m.cos(m.pi * x), 1 + m.cos(m.pi * y)]


def depth_avgd_disp_der(x, y):
    # TODO: Refactor
    depth = depth_func(x, y)
    x_comp = 5 * (1 + m.cos(m.pi * x) - (3 + x) * m.sin(m.pi * x)) / depth
    y_comp = -5 * m.pi * (3 + x) * m.sin(m.pi * y) / depth
    return [x_comp, y_comp]


def velocities(x, y):
    depth = depth_func(x, y)
    u = - y * (1 - x * x) / depth
    v = x * (1 - y * y) / depth
    return u, v


def wiener(dt):
    return np.random.normal(0, m.sqrt(dt))


class Particle:
    def __init__(self, domain, x=0.5, y=0.5):
        self.domain = domain
        self.x = x
        self.y = y
        self.xTrack = [x]
        self.yTrack = [y]
        self.dispersion = None
        self.depth_avgd_disp = None

    def calc_dispersion(self):
        """
        Calculate the dispersion coefficient in x- and y-directions
        """
        self.dispersion = dispersion_coeffs(self.x, self.y)
        self.depth_avgd_disp = depth_avgd_disp_der(self.x, self.y)

    def in_domain(self):
        test = (self.x > self.domain.xmin) * (self.x < self.domain.xmax)
        test *= (self.y > self.domain.ymin) * (self.y < self.domain.ymax)
        return test

    def correct_coords(self):
        if self.x < self.domain.xmin:
            self.x = self.domain.xmin + self.domain.minDx
        elif self.x > self.domain.xmax:
            self.x = self.domain.xmax - self.domain.minDx
        if self.y < self.domain.ymin:
            self.y = self.domain.ymin + self.domain.minDy
        elif self.y > self.domain.ymax:
            self.y = self.domain.ymax - self.domain.minDy

    def euler_step(self, dt):
        u, v = velocities(self.x, self.y)
        self.calc_dispersion()

        dy = (v + self.depth_avgd_disp[1]) * dt + \
            m.sqrt(2 * self.dispersion[0]) * wiener(dt)
        dx = (u + self.depth_avgd_disp[0]) * dt + \
            m.sqrt(2 * self.dispersion[1]) * wiener(dt)

        self.x += dx
        self.y += dy

        self.xTrack.append(self.x)
        self.yTrack.append(self.y)

        if not self.in_domain():
            self.correct_coords()
            logger.debug(self.x, self.y)


class ParticleSimulation():
    def __init__(self, n_particles, n_steps, end_time=100):
        # Protected variables
        self._num_particles = n_particles
        self._num_steps = n_steps
        self._end_time = end_time
        self._dt = self.calc_dt()

        # Public variables
        self.domain = Domain()
        self.particles = None

    # Calculating dependent variables.
    def calc_dt(self):
        return self._end_time / self._num_steps

    # Setters for protected variables
    def set_end_time(self, end_time):
        self._end_time = end_time
        self._dt = self.calc_dt()

    def set_num_steps(self, n_steps):
        self._num_steps = n_steps
        self._dt = self.calc_dt()

    def set_num_particles(self, n_particles):
        self._num_particles = n_particles

    # Member functions
    def euler_step(self):
        [part.euler_step(self._dt) for part in self.particles]

    def plot_current(self):
        plt.figure("Particle Distribution")
        plt.xlim([self.domain.xmin, self.domain.xmax])
        plt.ylim([self.domain.ymin, self.domain.ymax])
        [plt.scatter(part.x, part.y, color='gray') for part in self.particles]

    def run(self):
        self.particles = [Particle(self.domain)
                          for i in range(self._num_particles)]

        self.plot_current()

        for i in range(self._num_steps):
            [part.euler_step(self._dt) for part in self.particles]

        self.plot_current()

        # particleAnimation(particles)


def particleAnimation(particles):
    logger.info("Starting Animation")

    xData = np.ones(len(particles)) * 0.5
    yData = np.ones(len(particles)) * 0.5

    fig, ax = plt.subplots()
    ln = ax.scatter(xData, yData, color='gray')

    def update(frame):
        xData = [particle.xTrack[frame] for particle in particles]
        yData = [particle.yTrack[frame] for particle in particles]

        # ax.clear()
        # ax.scatter(xData, yData)
        ln.set_offsets(np.vstack((xData, yData)).T)
        return ln

    fig.tight_layout()
    animation = anim.FuncAnimation(fig, update,
                                   frames=len(particles[0].xTrack),
                                   interval=200)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    animation.save("particle_simul.mp4")
    return animation


if __name__ == "__main__":

    simul = ParticleSimulation(n_particles=1000, n_steps=10000, end_time=100)
    simul.run()

    plt.show()
