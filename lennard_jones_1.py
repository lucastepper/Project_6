import numpy as np


x1 = np.array([[1,1],[0,0],[3,3],[1,0]])


def potential_harmonic(x, k):
    """ Computes potential of n particles in harmonic potenital
        arround the origin

        Arguments:
            x (numpy.ndarray(n, d)): configuration
            k (float): spring constant

        Returns:
            phi (float): total harmonic potential """

    return np.sum(k * (np.sum(x ** 2, axis=1)))


def gradient_harmonic(x, k):
    """ Computes gradient of n particles in harmonic potenital
        arround the origin at position of particles

        Arguments:
            x (np.ndarray(n, d)): configuration
            k (float): spring constant

        Returns:
            phi (np.ndarray(n)): gradient harmonic potential """

    return -2 * x


def potential_lennard_jones(x):
    """ Computes Lennard Jones potential of n particles

        Arguments:
            x (np.ndarray(n, d)): configuration

        Returns:
            phi (float): total Lennard Jones potetnial """

    n_particles = x.shape[0]
    potential_list = []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            distance = np.linalg.norm(x[i,:] - x[j,:])
            potential = 4 / distance ** 12 - 4 / distance ** 6
            potential_list.append(potential)

    return np.sum(potential_list)


def gradient_lennard_jones(x):
    """ Computes gradient Lennard Jones potential of n particles
        at position of particles

        Arguments:
            x (np.ndarray(n, d)): configuration

        Returns:
            phi (np.ndarray(n)): gradient of Lennard Jones potential """

    n_particles = x.shape[0]
    gradient_list = []
    for i in range(1):
        gradient0_list = []
        for j in range(n_particles):
            distance = np.linalg.norm(x[i,:] - x[j,:])
            if distance != 0:
                gradient0 = ((x[i,:] - x[j,:])
                           * (6 * 4 / distance ** 8 - 12 * 4 / distance ** 14))
                gradient0_list.append(gradient0)
        gradient = np.sum(gradient0_list, axis=0)
        gradient_list.append(gradient)

    return np.array(gradient_list)


def potential(x, k):
    return potential_harmonic(x, k) +  potential_lennard_jones(x)


def gradient(x, k):
    return gradient_harmonic(x, k) + potential_lennard_jones(x)
