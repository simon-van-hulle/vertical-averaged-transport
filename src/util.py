import numpy as np
    

def particle_density(grid_x, grid_y, particles, var):
    coords = np.stack((grid_x, grid_y), axis=2)
    coords = coords.reshape((-1,2))
    coords = coords[:,np.newaxis,:]
    coords = np.repeat(coords, particles.shape[0], axis=1)
    diff =  coords - particles
    diff2 = diff * diff
    r2 = np.sqrt(np.sum(diff2, axis=2))

    gaussian = np.exp(-r2 / (2 * var)) / np.sqrt(2 * np.pi * var) / particles.shape[0]
    gaussian_superpos = gaussian.sum(axis=1).reshape(grid_x.shape)

    return gaussian_superpos

