#!/usr/bin/env python3

"""
A first, intuitive implementation.
The code spits out results, but haven't checked anything (not reliable...)
"""
import math as m
import os
import sys

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
    def __init__(self, xmin=-1, xmax=1, ymin=-1, ymax=1, min_factor=0.01):
        self.xmin : float = xmin
        self.xmax : float = xmax
        self.ymin : float = ymin
        self.ymax = ymax
        self.minDx = min_factor * (self.xmax - self.xmin)
        self.minDy = min_factor * (self.ymax - self.ymin)

def depth_func(x, y=None):
    return 15 + 5 * x

def dispersion_coeffs(x, y):
    """
    Returns a list of dispersion coefficients (arrays) for all input locations
    """
    Dx = 1 + np.cos(np.pi * x)
    Dy = 1 + np.cos(np.pi * y)
    return [Dx, Dy]

def dispersion_der(x, y):
    """
    Returns the dispersion coefficient derivatives 
    :return: list of dispersion coefficient derivatives [dDx/dx, dDy/dy]
    """
    dDxdx = - np.pi * np.sin(np.pi * x)
    dDydy = - np.pi * np.cos(np.pi * y)
    return [dDxdx, dDydy]

def depth_avgd_disp_der(x, y):
    """
    Equation term with derivative and division by H
    $\dfrac{1}{H}\dfrac{\partial(HDx)}{\partial x}$ (and equivalent for y)  

    :return: list of the derivative terms [d(HDx)/dx/H, d(HDy)/dy/H]
    """
    depth = depth_func(x, y)
    x_comp = 5 * (1 + np.cos(np.pi * x) - (3 + x) * np.sin(np.pi * x)) / depth
    y_comp = -5 * np.pi * (3 + x) * np.sin(np.pi * y) / depth
    return [x_comp, y_comp]

def velocities(x, y):
    """
    Calculate water velocity at specified location(s)
    """
    depth = depth_func(x, y)
    u = - y * (1 - x * x) / depth
    v = x * (1 - y * y) / depth
    return u, v

def velocities_der(x, y):
    """
    Calculate water velocity at specified location(s)
    """
    depth = depth_func(x, y)
    dudx = 2 * x * y / depth
    dvdx = -2 * x * y / depth
    return dudx, dvdx

def wiener_steps(dt, n):
    """
    $W_{t+\Delta t} - W_{t} = N(0, \Delta t)$
    """
    return np.random.normal(0, np.sqrt(dt), n)

class Particles:
    def __init__(self, N, domain=None, x=0.5, y=0.5):
        self.size = N
        self.domain = domain or Domain()
        self.pos_x = x * np.ones(self.size)
        self.pos_y = y * np.ones(self.size)
        self.history_x = [self.pos_x.copy()]
        self.history_y = [self.pos_y.copy()]
        self.dispersion = [np.zeros(self.size), np.zeros(self.size)]
        self.dispersion_der = [np.zeros(self.size), np.zeros(self.size)]
        self.depth_avgd_disp = [np.zeros(self.size), np.zeros(self.size)]

    def calc_dispersion(self):
        """
        Calculate the dispersion coefficient in x- and y-directions
        """
        self.dispersion = dispersion_coeffs(self.pos_x, self.pos_y)
        self.dispersion_der = dispersion_der(self.pos_x, self.pos_y)
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
        """
        u, v = velocities(self.pos_x, self.pos_y)
        self.calc_dispersion()

        dx = (u + self.depth_avgd_disp[0]) * dt + np.sqrt(2 * self.dispersion[0]) * wiener_steps(dt, self.size)
        dy = (v + self.depth_avgd_disp[1]) * dt + np.sqrt(2 * self.dispersion[1]) * wiener_steps(dt, self.size)

        return dx, dy

    def milstein_step(self, dt):
        """
        Perform one solver step with a Milstein scheme implementation
        """
        u, v = velocities(self.pos_x, self.pos_y)
        dudx, dvdx = velocities_der(self.pos_x, self.pos_y)
        self.calc_dispersion()

        euler_dx, euler_dy = self.euler_step(dt)

        gx_der = 1 / np.sqrt(2 * self.dispersion[0]) * self.dispersion_der[0]
        gy_der = 1 / np.sqrt(2 * self.dispersion[1]) * self.dispersion_der[1]

        dx = euler_dx + np.sqrt(self.dispersion[0] / 2) * gx_der * (wiener_steps(dt, self.size) ** 2 - dt)
        dy = euler_dy + np.sqrt(self.dispersion[1] / 2) * gy_der * (wiener_steps(dt, self.size) ** 2 - dt)

        return dx, dy

    def perform_step(self, dt, scheme="euler"):
        """
        Perform one numerical step in the scheme of choice
        """
        if scheme == "euler":
            dx, dy = self.euler_step(dt)
        elif scheme == "milstein":
            dx, dy = self.milstein_step(dt)
        else:
            print("\n")
            logger.critical(f"The {scheme} scheme is not implemented.")
            sys.exit(f"\tExiting...")

        self.pos_x += dx
        self.pos_y += dy

        self.history_x.append(self.pos_x.copy())
        self.history_y.append(self.pos_y.copy())

        # TODO: How to do this with numpy arrays?
        self.correct_coords()

    def scatter(self, color='r'):
        """
        Scatter plot all of the particles at their current position
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
    def __init__(self, n_particles, n_steps, end_time=100, scheme="euler"):
        # Protected variables
        self._num_particles = n_particles
        self._num_steps = n_steps
        self._end_time = end_time
        self._dt = self.calc_dt()
        self._time = 0
        self._scheme = scheme

        # Public variables
        self.domain = Domain()
        self.particles = Particles(n_particles, self.domain)

    def standard_title(self):
        title = f"{self._scheme}-P{self._num_particles}"
        title += f"-S{self._num_steps}-T{self._time:.0f}"
        return title

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
    def step(self):
        """
        Perform one step for all the particles with the numerical scheme of 
        choice.
        """
        self.particles.perform_step(self._dt, self._scheme)

    def plot_current(self, show=False, file_name=False):
        plt.figure("Particle Distribution")
        plt.xlim([self.domain.xmin, self.domain.xmax])
        plt.ylim([self.domain.ymin, self.domain.ymax])
        self.particles.scatter()

        if file_name:
            if file_name == True:
                file_name = f"particles-{self.standard_title()}"

            file_name = os.path.join(OUTPUT_DIR, file_name)
            plt.savefig(file_name)
            logger.info(
                f"Saved state plot at time {self._time:.2f} as {file_name}")

        if show:
            plt.show()

    @h.timing
    def run(self, show_plot=False, plot_name=False, animation=False):
        logger.info("Starting particle model run")

        for i in range(self._num_steps):
            status = (i + 1) / self._num_steps * 100
            logger.debug(f"Simulation status: {status:4.1f} % done", end='\r')
            self._time += self._dt
            self.step()

        logger.info("Finished particle model run")

        self.plot_current(show=show_plot, file_name=plot_name)

        if animation:
            file_name = f"animation-{self.standard_title()}.mp4"
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
    simul = ParticleSimulation(n_particles=10000, n_steps=100000, end_time=1000, scheme="euler")
    simul.run(show_plot=False, plot_name=True, animation=False)

    plt.show()
