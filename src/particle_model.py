import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import util

grid_res = 25
n_particles = 5000

dt = 0.001
t_final = 10

def H(particles):
    return 15 + 5 * particles[:,0]

def D(particles):
    return 1 + np.cos(np.pi * particles)

def V(particles):
    return np.array([-1, 1]) * particles[:,::-1] * (1 - particles * particles) / H(particles)[:, None]

def diff_HD(particles):
    intermediate = np.empty((n_particles, 2))
    intermediate[:,0] = 5 * ( (1 + np.cos(np.pi * particles[:,0])) - np.pi * (particles[:,0] + 3) * np.sin(np.pi * particles[:,0]) )
    intermediate[:,1] = -5 * np.pi * (particles[:,0] + 3) * np.sin(np.pi * particles[:,1])
    return intermediate

def diff_sqrt2D(particles):
    return - np.pi * np.sin(np.pi * particles) / np.sqrt(2 * (1 + np.cos(np.pi * particles)))

def dW():
    return np.random.normal(0, np.sqrt(dt), (n_particles, 2))

def advance_euler(particles):
    particles += V(particles) * dt + diff_HD(particles) / H(particles)[:, None] * dt + np.sqrt(2 * D(particles)) * dW()

def advance_milstein(particles):
    advance_euler(particles)
    particles += 0.5 * np.sqrt(2 * D(particles)) * diff_sqrt2D(particles) * (dW() * dW() - dt)

def animate(i):
    advance_euler(particles)
    scatterplt.set_offsets(particles)
    axs[1].clear()
    axs[1].contour(X, Y, util.particle_density(X, Y, particles, 0.1))

particles = 0.5 * np.ones((n_particles, 2)) # Each row is a particle, column 0 is x pos, column 1 is y pos

# Grid for contour plot
x = np.linspace(-1, 1, grid_res)
y = np.linspace(-1, 1, grid_res)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(1,2,figsize=(14,7))
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(-1, 1)
axs[0].set_box_aspect(1)
axs[1].set_xlim(-1, 1)
axs[1].set_ylim(-1, 1)
axs[1].set_box_aspect(1)
scatterplt = axs[0].scatter(particles[:,0], particles[:,1])
countourplt = axs[1].contour(X, Y, util.particle_density(X, Y, particles, 0.1))


animation = anim.FuncAnimation(fig, animate, interval=5, frames=int(t_final/dt))
plt.show()