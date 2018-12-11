import numpy as np
import matplotlib.pyplot as plt
import lennard_jones_1 as lj
from langevin import langevin


n_steps = int(1E4)
n_particles = 2
dimensions = 2
x_init = np.arange(n_particles * dimensions).reshape(n_particles, dimensions)
v_init = np.arange(n_particles * dimensions).reshape(n_particles, dimensions)
mass = np.ones(n_particles)
time_step = 0.001
damping = 0.1
beta = 0.01
k = int(5E-3)

def gradient(x):
    return lj.gradient(x, k)

x_sol, v_sol = langevin(gradient, n_steps, x_init, v_init,
                        mass, time_step, damping, beta)

r_sol = np.linalg.norm(x_sol, axis=2)
r_sol = r_sol.reshape(-1)
r_sol_mean = np.mean(r_sol)
r_sol_max = np.ndarray.max(r_sol)

print('mean distance from centre:', r_sol_mean)
print('maximum distance from centre:', r_sol_max)


plt.plot(x_sol[:,0,0], x_sol[:,0,1])
plt.show()
