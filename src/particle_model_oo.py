#!/usr/bin/env python3

"""
A first, intuitive implementation.
The code spits out results, but haven't checked anything (not reliable...)
"""
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


def wiener_steps(dt, n_particles):
    """
    $W_{t+\Delta t} - W_{t} = N(0, \Delta t)$
    """
    return np.random.normal(0, np.sqrt(dt), n_particles)


class Domain:
    def __init__(self, xmin=-1, xmax=1, ymin=-1, ymax=1, min_factor=0.01):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.minDx = min_factor * (self.xmax - self.xmin)
        self.minDy = min_factor * (self.ymax - self.ymin)


class WienerProcess:
    def __init__(self, n_steps, n_particles, dt):
        self.dt = dt
        self.n_steps = n_steps
        self.n_particles = n_particles
        self.wiener_steps = self.generate_steps()
        self.process = self.get_process()

    def generate_steps(self):
        logger.info("Generating Wiener Proces steps")
        std_deviation = np.sqrt(self.dt)
        shape = (self.n_steps, self.n_particles)
        steps = np.random.normal(0, std_deviation, shape)
        return steps

    def get_process(self):
        wiener = np.zeros((self.n_steps, self.n_particles))
        wiener[0] = self.wiener_steps[0]
        for i, step in enumerate(self.wiener_steps[1:]):
            wiener[i] = wiener[i - 1] + step
        return wiener

    def get_step(self, step_num):
        return self.wiener_steps[step_num]

    def plot(self):
        title = "Wiener Process"
        plt.figure(title)
        plt.title(title)
        plt.plot(self.process)


class Particles:
    def __init__(self, N, wiener_x, wiener_y, domain=None, x=0.5, y=0.5):
        self.size = N
        self.domain = domain or Domain()
        self.wiener_x = wiener_x
        self.wiener_y = wiener_y

        self.pos_x = x * np.ones(self.size)
        self.pos_y = y * np.ones(self.size)
        self.history_x = [self.pos_x.copy()]
        self.history_y = [self.pos_y.copy()]
        self.dispersion = None
        self.dispersion_der = None
        self.depth_avgd_disp = None

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

    def euler_step(self, wiener_step_x, wiener_step_y, dt):
        """
        Perform one solver step with an Euler scheme implementation
        """
        u, v = velocities(self.pos_x, self.pos_y)

        fx = u + self.depth_avgd_disp[0]
        fy = v + self.depth_avgd_disp[1]
        gx = np.sqrt(2 * self.dispersion[0])
        gy = np.sqrt(2 * self.dispersion[1])

        dx = fx * dt + gx * wiener_step_x
        dy = fy * dt + gy * wiener_step_y

        return dx, dy

    def milstein_step(self, wiener_step_x, wiener_step_y, dt):
        """
        Perform one solver step with a Milstein scheme implementation
        """

        euler_dx, euler_dy = self.euler_step(wiener_step_x, wiener_step_y, dt)

        dx = euler_dx + self.dispersion_der[0] / 2 * (wiener_step_x ** 2 - dt)
        dy = euler_dy + self.dispersion_der[1] / 2 * (wiener_step_y ** 2 - dt)

        return dx, dy

    def perform_step(self, current_step, dt, scheme="euler"):
        """
        Perform one numerical step in the scheme of choice
        """
        self.calc_dispersion()
        wiener_step_x = self.wiener_x.get_step(current_step)
        wiener_step_y = self.wiener_y.get_step(current_step)

        if scheme == "euler":
            dx, dy = self.euler_step(wiener_step_x, wiener_step_y, dt)
        elif scheme == "milstein":
            dx, dy = self.milstein_step(wiener_step_x, wiener_step_y, dt)
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
        plt.scatter(self.pos_x, self.pos_y, color=color, s=5)

    def scatter_plot(self, title=None):
        title = title or f"Particles - P{self.size}"
        plt.figure(title)
        plt.title(title)
        self.scatter()
        plt.xlim([self.domain.xmin, self.domain.xmax])
        plt.ylim([self.domain.ymin, self.domain.ymax])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

class ParticleSimulation():
    def __init__(self, n_particles, n_steps, end_time=100, scheme="euler"):
        self._num_particles = n_particles
        self._num_steps = n_steps
        self._end_time = end_time
        self._dt = self.calc_dt()
        self._time = 0
        self._current_step = 0
        self._scheme = scheme

        self.domain = Domain()
        self.wiener_x = WienerProcess(n_steps, n_particles, self._dt)
        self.wiener_y = WienerProcess(n_steps, n_particles, self._dt)
        self.particles = Particles(
            n_particles, self.wiener_x, self.wiener_y, self.domain)

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
        status = (self._current_step + 1) / self._num_steps * 100
        logger.debug(f"Simulation status: {status:4.1f} % done", end='\r')
        self._time += self._dt
        self.particles.perform_step(self._current_step, self._dt, self._scheme)

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
            logger.info(f"Saved state plot at time {self._time:.2f} as {file_name}")

        if show:
            plt.show()

    @h.timing
    def run(self, show_plot=False, plot_name=False, animation=False):
        logger.info("Starting particle model run")
        
        for self._current_step in range(self._num_steps):
            self.step()

        logger.info("Finished particle model run")

        self.plot_current(show=show_plot, file_name=plot_name)

        if animation:
            self.particleAnimation()

    def particleAnimation(self):
        logger.info("Starting Animation")

        file_name = f"animation-{self.standard_title()}.mp4"
        file_name = os.path.join(OUTPUT_DIR, file_name)

        particles = self.particles

        xData = particles.history_x[0]
        yData = particles.history_y[0]

        fig, ax = plt.subplots()
        ln = ax.scatter(xData, yData, color='r', s=5)

        def update(frame):
            xData = particles.history_x[frame]
            yData = particles.history_y[frame]

            ln.set_offsets(np.vstack((xData, yData)).T)
            logger.debug(f"Rendering frame {frame}", end='\r')
            return ln

        fig.tight_layout()
        animation = anim.FuncAnimation(fig, update, frames=self._num_steps,
                                       interval=200)
        ax.set_xlim([self.domain.xmin, self.domain.xmax])
        ax.set_ylim([self.domain.ymin, self.domain.ymax])

        if file_name:
            animation.save(file_name)
            logger.info(f"Saved animation as {file_name}")

        return animation


if __name__ == "__main__":
    simul = ParticleSimulation(n_particles=10, n_steps=100000,
                               end_time=1000, scheme="milstein")
    simul.run(show_plot=False, plot_name=True, animation=False)

    plt.show()
