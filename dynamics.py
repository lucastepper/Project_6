import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pot_force as lj
from langevin import langevin


### define stuff ##
n_steps = int(1E3)
n_particles = 2
dimensions = 2
mass = np.ones(n_particles)
time_step = 0.01
damping = 0
beta = 0.01
k = int(1)

# position particels on unit circle, v = 0
angle = 0
x_init = np.zeros((n_particles, dimensions))
for i in range(n_particles):
    x_init[i, :] = [np.cos(angle), np.sin(angle)]
    angle += 2 * np.pi / n_particles
v_init = np.zeros((n_particles, dimensions))
t = np.linspace(0, time_step * n_steps, n_steps + 1)

def gradient(x):
    return lj.gradient(x, k)


### generate solution ###
x_sol, v_sol = langevin(gradient, n_steps, x_init, v_init,
                        mass, time_step, damping, beta)


### calculate stuff ###
r_sol = np.linalg.norm(x_sol, axis=2)
r_sol = r_sol.reshape(-1)
r_sol_mean = np.mean(r_sol)
r_sol_max = np.ndarray.max(r_sol)

print('mean distance from centre:', r_sol_mean)
print('maximum distance from centre:', r_sol_max)


### plot suff ###
for i in range(n_particles):
    plt.plot(x_sol[:,i,0], x_sol[:,i,1])
# for i in range(n_particles):
#     plt.plot(t, lj.potential(x_sol, k))
plt.show()

def chi(x, xmin, xmax):
    return np.logical_and(xmin <= x, x < xmax)


positions = np.random.uniform(low=-5, high=5, size=100000)
edges = np.linspace(-5, 5, 31)
centers = (edges[:-1] + edges[1:]) / 2

histogram = [np.sum(chi(positions, x, y)) / positions.size
             for x, y in zip(edges[:-1], edges[1:])]

fig, ax = plt.subplots()
ax.bar(centers, [h for h in histogram], (edges[1] - edges[0]) * 0.9)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\frac{1}{N}\sum_{n=1}^N \chi_i(x_n)$')
fig.tight_layout()
