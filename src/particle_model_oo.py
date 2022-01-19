#!/usr/bin/env python3

"""
A first, intuitive implementation.
The code spits out results, but haven't checked anything (not reliable...)
"""
from cProfile import run
import os
import sys

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as spstats

import logging
import enlighten
import configargparse

import helpers as h

logger = h.easy_logger(__name__, logging.INFO)

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")


def out_file(name):
    return os.path.join(OUTPUT_DIR, name)


def depth_func(x, y):
    return 15 + 5 * x


def dispersion_coeffs(x, y):
    Dx = 1 + np.cos(np.pi * x)
    Dy = 1 + np.cos(np.pi * y)
    return [Dx, Dy]


def dispersion_der(x, y):
    dDxdx = - np.pi * np.sin(np.pi * x)
    dDydy = - np.pi * np.sin(np.pi * y)
    return [dDxdx, dDydy]


def depth_avgd_disp_der(x, y):
    """
    Equation term with derivative and division by H
    $\dfrac{1}{H}\dfrac{\partial(HDx)}{\partial x}$ (and equivalent for y)
    """
    depth = depth_func(x, y)
    x_comp = 5 * (1 + np.cos(np.pi * x) - np.pi * (3 + x) * np.sin(np.pi * x))
    y_comp = -5 * np.pi * (3 + x) * np.sin(np.pi * y)
    return [x_comp / depth, y_comp / depth]


def velocities(x, y):
    depth = depth_func(x, y)
    u = - y * (1 - x * x) / depth
    v = x * (1 - y * y) / depth
    return u, v


class Domain:
    def __init__(self, xmin=-1, xmax=1, ymin=-1, ymax=1, min_factor=0.01):
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax


class WienerProcess:
    def __init__(self, n_steps, n_particles, dt):
        self._dt = dt
        self._n_steps = n_steps
        self._n_particles = n_particles
        self._wiener_steps = self.generate_steps()
        self._process = self.get_process()

    def generate_steps(self):
        logger.info(f"Generating Wiener Proces with {self._n_steps} steps")
        std_deviation = np.sqrt(self._dt)
        shape = (self._n_steps, self._n_particles)
        steps = np.random.normal(0, std_deviation, shape)
        return steps

    def get_process(self):
        wiener = np.zeros((self._n_steps, self._n_particles))
        wiener[0] = self._wiener_steps[0]
        for i, step in enumerate(self._wiener_steps[1:]):
            wiener[i] = wiener[i - 1] + step
        return wiener

    def get_step(self, step_num, steps_per_it):
        return self._wiener_steps[step_num: (step_num + steps_per_it)].sum(axis=0)

    def plot(self):
        title = "Wiener Process"
        plt.figure(title)
        plt.clf()
        plt.title(title)
        plt.plot(self._process)


class Particles:
    def __init__(self, N, wiener_x, wiener_y, domain=None, x=0.5, y=0.5):
        self._xy_init = [x, y]
        self._size = N
        self._domain = domain or Domain()
        self._wiener_x = wiener_x
        self._wiener_y = wiener_y

        self._pos_x = x * np.ones(self._size)
        self._pos_y = y * np.ones(self._size)
        self._history_x = [self._pos_x.copy()]
        self._history_y = [self._pos_y.copy()]
        self._dispersion = None
        self._dispersion_der = None
        self._depth_avgd_disp = None

    def calc_dispersion(self):
        self._dispersion = dispersion_coeffs(self._pos_x, self._pos_y)
        self._dispersion_der = dispersion_der(self._pos_x, self._pos_y)
        self._depth_avgd_disp = depth_avgd_disp_der(self._pos_x, self._pos_y)

    def euler_step(self, wiener_step_x, wiener_step_y, dt):
        u, v = velocities(self._pos_x, self._pos_y)

        fx = u + self._depth_avgd_disp[0]
        fy = v + self._depth_avgd_disp[1]
        gx = np.sqrt(2 * self._dispersion[0])
        gy = np.sqrt(2 * self._dispersion[1])

        dx = fx * dt + gx * wiener_step_x
        dy = fy * dt + gy * wiener_step_y

        return dx, dy

    def milstein_step(self, wiener_step_x, wiener_step_y, dt):
        euler_dx, euler_dy = self.euler_step(wiener_step_x, wiener_step_y, dt)

        dx = euler_dx + self._dispersion_der[0] / 2 * (wiener_step_x ** 2 - dt)
        dy = euler_dy + self._dispersion_der[1] / 2 * (wiener_step_y ** 2 - dt)

        return dx, dy

    def perform_step(self, current_step, steps_per_it, dt, scheme="euler",
                     store_history=False):
        """Perform one numerical step in the scheme of choice
        """
        self.calc_dispersion()
        wiener_step_x = self._wiener_x.get_step(current_step, steps_per_it)
        wiener_step_y = self._wiener_y.get_step(current_step, steps_per_it)

        if scheme == "euler":
            dx, dy = self.euler_step(wiener_step_x, wiener_step_y, dt)

        elif scheme == "milstein":
            dx, dy = self.milstein_step(wiener_step_x, wiener_step_y, dt)

        else:
            print("\n")
            logger.critical(f"The {scheme} scheme is not implemented.")
            sys.exit(f"\tExiting...")

        self._pos_x += dx
        self._pos_y += dy

        if store_history:
            self._history_x.append(self._pos_x.copy())
            self._history_y.append(self._pos_y.copy())

    def reset(self):
        self._pos_x = 0 * self._pos_x + self._xy_init[0]
        self._pos_y = 0 * self._pos_y + self._xy_init[1]
        self._history_x = [self._pos_x.copy()]
        self._history_y = [self._pos_y.copy()]
        return 0

    def scatter(self, color='r'):
        """Scatter plot all of the particles at their current position
        """
        xy = np.vstack([self._pos_x, self._pos_y])
        z = spstats.gaussian_kde(xy)(xy)

        plt.scatter(self._pos_x, self._pos_y, c=z, s=20)
        plt.colorbar(orientation='vertical', label="Probability Density")


class ParticleSimulation():
    def __init__(self, config):
        c = config
        self.config = config
        self._num_particles = c.num_particles
        self._total_steps = c.total_steps
        self._end_time = c.end_time
        self._steps_per_it = 1
        self._time = 0
        self._current_step = 0
        self._scheme = c.scheme

        self._domain = Domain()
        self._wiener_x = WienerProcess(c.total_steps, c.num_particles,
                                       self.dt())
        self._wiener_y = WienerProcess(c.total_steps, c.num_particles,
                                       self.dt())
        self._particles = Particles(c.num_particles, self._wiener_x,
                                    self._wiener_y, self._domain)

    def standard_title(self):
        title = f"{self._scheme}-P{self._num_particles}"
        title += f"-S{self.num_steps()}-T{self._time:.0f}"
        return title

    def num_steps(self):
        return self._total_steps // self._steps_per_it

    def dt(self):
        return self._end_time / self.num_steps()

    # Setters for protected variables
    def set_end_time(self, end_time):
        self._end_time = end_time
        return 0

    def set_total_steps(self, n_steps):
        self._total_steps = n_steps
        return 0

    def set_steps_per_it(self, steps_per_it):
        self._steps_per_it = steps_per_it
        return 0

    def set_num_particles(self, n_particles):
        self._num_particles = n_particles
        return 0

    # Member functions
    def step(self):
        """One step for all particles with the numerical scheme of choice.
        """
        self._time += self.dt()
        self._particles.perform_step(self._current_step, self._steps_per_it,
                                     self.dt(), self._scheme,
                                     self.config.make_animation)
        return 0

    def plot_current(self, show_plot=False, store_plot=False, img_name=False):
        plt.figure("Particle Distribution")
        plt.clf()

        self._particles.scatter()
        plt.suptitle(f"Particle Distribution - {self._scheme.capitalize()}",
                     weight='bold', fontsize=16)
        plt.title(rf"{self._num_particles} particles $-$ $\Delta t$"
                  rf" = {self.dt()} $-$ T = {self._time:.1f}", fontsize=11)
        plt.xlim([self._domain._xmin, self._domain._xmax])
        plt.ylim([self._domain._ymin, self._domain._ymax])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        if not img_name:
            img_name = f"particles-{self.standard_title()}.png"

        if store_plot:
            img_name = out_file(img_name)
            plt.savefig(img_name)
            logger.info(f"Saved state plot at time {self._time:.2f} as "
                        f"{h.file_link(img_name)}")

        if show_plot:
            plt.show()

        return 0

    def run(self):
        self.reset()
        c = self.config

        logger.info(f"\nStarting particle model run with {self._scheme} "
                    f"scheme and the Wiener process in {self.num_steps()} "
                    f"steps.")
        run_progress_bar = enlighten.Counter(total=self.num_steps(),
                                             desc='Particle Model Run',
                                             min_delta=1,
                                             unit='ticks')

        for self._current_step in range(0, self._total_steps, self._steps_per_it):
            self.step()
            run_progress_bar.update()

        logger.info("Finished particle model run")

        if c.show_all or c.store_plot:
            self.plot_current(show_plot=c.show_all, store_plot=c.store_plot)

        if c.make_animation:
            self.particleAnimation()

        return np.array([self._particles._pos_x, self._particles._pos_y])

    def particleAnimation(self):
        logger.info("Starting Animation")
        animation_progress_bar = enlighten.Counter(total=self.num_steps(),
                                                   desc='Generating Animation',
                                                   min_delta=1,
                                                   unit='ticks')

        file_name = f"animation-{self.standard_title()}.mp4"
        file_name = out_file(file_name)

        particles = self._particles

        xData = particles._history_x[0]
        yData = particles._history_y[0]

        fig, ax = plt.subplots()
        ln = ax.scatter(xData, yData, color='r', s=5)

        def update(frame):
            xData = particles._history_x[frame]
            yData = particles._history_y[frame]

            ln.set_offsets(np.vstack((xData, yData)).T)
            animation_progress_bar.update()
            return ln

        fig.tight_layout()
        animation = anim.FuncAnimation(fig, update, frames=self.num_steps(),
                                       interval=200)
        ax.set_xlim([self._domain._xmin, self._domain._xmax])
        ax.set_ylim([self._domain._ymin, self._domain._ymax])

        if file_name:
            animation.save(file_name)
            print("\n")
            logger.info(f"Saved animation as {h.file_link(file_name)}")

        return animation

    def reset(self):
        self._time = 0
        self._current_step = 0
        self._particles.reset()
        return 0


def parse_configuration():
    p = configargparse.ArgumentParser()
    p = configargparse.ArgParser(
        default_config_files=[f'{CURRENT_DIR}/config.yaml'])
    p.add('-c', '--my-config', is_config_file=True, metavar='',
          help='config file path')
    p.add('-p', '--num-particles', type=int, default=1000, metavar='',
          help='number of particles in the simulation')
    p.add('-st', '--total-steps', type=int, default=1000, metavar='',
          help='number of numerical steps in the simulation')
    p.add('-t', '--end-time', type=int, default=1000, metavar='',
          help='time (seconds) to stop the simulation')
    p.add('-sc', '--scheme', type=str, default="euler", metavar='',
          help='numerical scheme for the simulation')
    p.add('-a', '--make-animation', action='store_true',
          help='make animation or not')
    p.add('-psa', '--show-all', action='store_true',
          help='show all plots and block excecution')
    p.add('-ps', '--store-plot', action='store_true',
          help='store plot in results/ directory')
    p.add('-es', '--show-end', action='store_true',
          help='show plot of end state')

    args = p.parse_args()
    logger.info("Configuration parameters:\n")
    print(p.format_values())

    return args


if __name__ == "__main__":
    config = parse_configuration()

    simul = ParticleSimulation(config)
    simul.run()

    if config.show_end:
        plt.show()
