import numpy as np

import sys
sys.path.append("../")
from src.synthetic_catalog_creation import random_vector
from src.synthetic_catalog_creation import stress_to_instability


def random_normals(N, seed=1000):
    rng = np.random.default_rng(seed)
    nn, ne, nd = random_vector(N, rng)
    # upward normal: nd<0
    flip = nd > 0
    nn[flip] *= -1
    ne[flip] *= -1
    nd[flip] *= -1

    return np.column_stack([nn, ne, nd])

def instability_PDF(R, m, n):
    sigma = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -(1 - 2 * R)]
    ])
    # sort the principal stresses    * ps1: most compressional stress (compression positive)
    eigvals = np.linalg.eigvalsh(sigma)
    ps1, _, ps3 = -np.sort(eigvals)

    traction = n @ sigma.T

    sigma_n = np.sum(traction * n, axis=1)
    sigma_n_scaled = (-sigma_n - (ps1 + ps3)/2) * 2 / (ps1 - ps3)

    tau_vec = traction - sigma_n[:, None] * n
    tau = np.linalg.norm(tau_vec, axis=1)
    tau_scaled = tau * 2 / (ps1 - ps3)

    instability = stress_to_instability(sigma_n_scaled, tau_scaled, m)

    bins = np.arange(-1.005, 1.01, 0.01)
    hist, edges = np.histogram(instability, bins=bins)

    center = (edges[1:] + edges[:-1]) / 2
    PDF = hist / (0.01 * hist.sum())
    return center, PDF