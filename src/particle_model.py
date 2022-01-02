import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

n_particles = 100

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
    return np.random.normal(0, dt, (n_particles, 2))

def advance_euler(particles):
    advection = V(particles) * dt
    diffusion1 = diff_HD(particles) / H(particles)[:, None] * dt
    diffusion2 = np.sqrt(2 * D(particles)) * dW()
    particles += advection + diffusion1 + diffusion2

def advance_milstein(particles):
    advance_euler(particles)
    particles += 0.5 * np.sqrt(2 * D(particles)) * diff_sqrt2D(particles) * (dW() * dW() - dt)

def animate(i):
    advance_euler(particles)
    sc.set_offsets(particles)

particles = 0.5 * np.ones((n_particles, 2)) # Each row is a particle, column 0 is x pos, column 1 is y pos

fig, ax = plt.subplots()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
sc = ax.scatter(particles[:,0], particles[:,1])

animation = anim.FuncAnimation(fig, animate, interval=5, frames=int(t_final/dt))
plt.show()