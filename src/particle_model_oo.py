#!/usr/bin/env python3

"""
A first, intuitive implementation.
The code spits out results, but haven't checked anything (not reliable...)
"""

import math as m
import os

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

import helpers as h

# Setup logger and absolute paths
logger = h.Logger()
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")


class Domain:
    """
    We might want to do some magic with keeping everything inside the domain
    That's why I thought it might be nice to have a separate class.

    We might remove this later.
    """

    def __init__(self, xmin=-1, xmax=1, ymin=-1, ymax=1):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.minDx = 0.01 * (self.xmax - self.xmin)
        self.minDy = 0.01 * (self.ymax - self.ymin)


def depth_func(x, y=None):
    """
    Calculates the depth of the water at any location.

    :param x: x position (array)
    :param y: y position (array), defaults to None
    :return: depth
    """
    return 15 + 5 * x


def dispersion_coeffs(x, y):
    """
    Returns a list of dispersion coefficients (arrays) for all input locations

    :param x: x position (array)
    :param y: y position (array)
    :return: list of dispersion coefficients [Dx, Dy]
    """
    return [1 + np.cos(np.pi * x), 1 + np.cos(np.pi * y)]


def depth_avgd_disp_der(x, y):
    """
    Equation term with derivative and division by H
    $\dfrac{1}{H}\dfrac{\partial(HDx)}{\partial x}$ (and equivalent for y)  

    :param x: x position (array)
    :param y: y position (array)
    :return: list of the derivative terms [d(HDx)/dx/H, d(HDy)/dy/H]
    """
    # TODO: Refactor
    depth = depth_func(x, y)
    x_comp = 5 * (1 + np.cos(np.pi * x) - (3 + x) * np.sin(np.pi * x)) / depth
    y_comp = -5 * np.pi * (3 + x) * np.sin(np.pi * y) / depth
    return [x_comp, y_comp]


def velocities(x, y):
    """
    Calculate water velocity at specified location(s)

    :param x: x position
    :param y: y position
    :return: x and y velocity [u, v]
    """
    depth = depth_func(x, y)
    u = - y * (1 - x * x) / depth
    v = x * (1 - y * y) / depth
    return u, v


def wiener_steps(dt, n):
    """
    n realisations of a wiener process step with $\Delta t = 1$
    $W_{t+\Delta t} - W_{t} = N(0, \Delta t)$

    :param dt: Time step
    :param n: number of steps to generate (e.g. number of particles)
    :return: np array with n steps of the Wiener process
    """
    return np.random.normal(0, np.sqrt(dt), n)


class Particles:
    def __init__(self, N, domain, x=0.5, y=0.5):
        self.size = N
        self.domain = domain
        self.pos_x = x * np.ones(self.size)
        # self.pos_x = x * np.random.uniform(size=self.size)
        self.pos_y = y * np.ones(self.size)
        # self.pos_y = y * np.random.uniform(size=self.size)
        self.history_x = [self.pos_x.copy()]
        self.history_y = [self.pos_y.copy()]
        self.dispersion = [np.zeros(self.size), np.zeros(self.size)]
        self.depth_avgd_disp = [np.zeros(self.size), np.zeros(self.size)]

    def calc_dispersion(self):
        """
        Calculate the dispersion coefficient in x- and y-directions
        """
        self.dispersion = dispersion_coeffs(self.pos_x, self.pos_y)
        self.depth_avgd_disp = depth_avgd_disp_der(self.pos_x, self.pos_y)

    def correct_coords(self):
        """
        Adjust the positions to make sure they are in the predfined domain
        TODO: Make this better! This is very very preliminary.
        """
        for i in range(self.size):
            if self.pos_x[i] < self.domain.xmin:
                self.pos_x[i] = self.domain.xmin + self.domain.minDx
            elif self.pos_x[i] > self.domain.xmax:
                self.pos_x[i] = self.domain.xmax - self.domain.minDx

            if self.pos_y[i] < self.domain.ymin:
                self.pos_y[i] = self.domain.ymin + self.domain.minDy
            elif self.pos_y[i] > self.domain.ymax:
                self.pos_y[i] = self.domain.ymax - self.domain.minDy

    def euler_step(self, dt):
        """
        Perform one solver step with an Euler scheme implementation

        :param dt: Time step for the numerical solver
        """
        u, v = velocities(self.pos_x, self.pos_y)
        self.calc_dispersion()

        dy = (v + self.depth_avgd_disp[1]) * dt + \
            np.sqrt(2 * self.dispersion[0]) * wiener_steps(dt, self.size)
        dx = (u + self.depth_avgd_disp[0]) * dt + \
            np.sqrt(2 * self.dispersion[1]) * wiener_steps(dt, self.size)

        self.pos_x += dx
        self.pos_y += dy

        self.history_x.append(self.pos_x.copy())
        self.history_y.append(self.pos_y.copy())

        # TODO: How to do this with numpy arrays?
        self.correct_coords()

    def scatter(self, color='r'):
        """
        Scatter plot all of the particles at their current position

        :param color: particle color, defaults to 'r'
        """
        plt.scatter(self.pos_x, self.pos_y, color=color, s=3)

    def scatter_plot(self, title=None):
        title = title or f"Particles - P{self.size}"
        plt.figure(title)
        self.scatter()
        plt.title(title)
        plt.xlim([self.domain.xmin, self.domain.xmax])
        plt.ylim([self.domain.ymin, self.domain.ymax])


class ParticleSimulation():
    def __init__(self, n_particles, n_steps, end_time=100):
        # Protected variables
        self._num_particles = n_particles
        self._num_steps = n_steps
        self._end_time = end_time
        self._dt = self.calc_dt()
        self._time = 0

        # Public variables
        self.domain = Domain()
        self.particles = Particles(n_particles, self.domain)

    # Calculating dependent variables.
    def calc_dt(self):
        """
        Get the new value of _dt if anything updated
        """
        return self._end_time / self._num_steps

    # Setters for protected variables
    def set_end_time(self, end_time):
        """
        Set a new end time for the simulation and update _dt
        """
        self._end_time = end_time
        self._dt = self.calc_dt()

    def set_num_steps(self, n_steps):
        """
        Set new number of steps for the simulation and update _dt
        """
        self._num_steps = n_steps
        self._dt = self.calc_dt()

    def set_num_particles(self, n_particles):
        """
        Set number of pollutant particles present in the simulation
        """
        self._num_particles = n_particles

    # Member functions
    def euler_step(self):
        """
        Perform one Euler scheme step for all the particles
        """
        self.particles.euler_step()

    def plot_current(self, show=False, file_name=False):
        """
        Plot the current configuration of particles and potentially store

        :param show: show the plot during run, defaults to False
        :param file_name: file name to store the file, defaults to False. 
                            Use None to prevent storing the file.
        """
        plt.figure("Particle Distribution")
        plt.xlim([self.domain.xmin, self.domain.xmax])
        plt.ylim([self.domain.ymin, self.domain.ymax])
        self.particles.scatter()

        if file_name:
            if file_name == True:
                file_name = f"particles-P{self._num_particles}-S{self._num_steps}-T{self._time:.0f}"

            file_name = os.path.join(OUTPUT_DIR, file_name)
            plt.savefig(file_name)
            logger.info(
                f"Saved state plot at time {self._time:.2f} as {file_name}")

        if show:
            plt.show()

    @h.timing
    def run(self, show_plot=False, plot_name=False, animation=False):
        logger.info("Starting particle model run")

        self.plot_current()

        for i in range(self._num_steps):
            status = (i + 1) / self._num_steps * 100
            logger.debug(f"Simulation status: {status:4.1f} % done", end='\r')
            self._time += self._dt
            self.particles.euler_step(self._dt)

        logger.info("Finished particle model run")

        self.plot_current(show=show_plot, file_name=plot_name)

        if animation:
            file_name = f"animation-P{self._num_particles}-S{self._num_steps}-T{self._time:.0f}.mp4"
            file_name = os.path.join(OUTPUT_DIR, file_name)
            self.particleAnimation(file_name)

    def particleAnimation(self, file_name=False):
        logger.info("Starting Animation")

        particles = self.particles

        xData = particles.history_x[0]
        yData = particles.history_y[0]

        fig, ax = plt.subplots()
        ln = ax.scatter(xData, yData, color='r', s=3)

        def update(frame):
            xData = particles.history_x[frame]
            yData = particles.history_y[frame]

            ln.set_offsets(np.vstack((xData, yData)).T)
            logger.debug(f"Rendering frame {frame}", end='\r')
            return ln

        fig.tight_layout()
        animation = anim.FuncAnimation(fig, update, frames=self._num_steps,
                                       interval=200)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

        if file_name:
            animation.save(file_name)
            logger.info(f"Saved animation as {file_name}")

        return animation


if __name__ == "__main__":
    simul = ParticleSimulation(n_particles=10000, n_steps=1000, end_time=1000)
    simul.run(show_plot=False, plot_name=True, animation=False)

    plt.show()
