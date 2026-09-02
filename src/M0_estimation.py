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

def prepare_instability_pdf(instability, R, m, n, theob_calc):

    data = []
    for i in range(len(instability)):
        centers, PDF = instability_PDF(R[i], m, n)
        idx = np.argmin(np.abs(instability[i] - centers))
        b = theob_calc(centers)
        dI = centers[1] - centers[0]

        data.append({
            "I_obs": instability[i],
            "idx": idx,
            "centers": centers,
            "PDF": PDF,
            "b": b,
            "dI": dI,
        })

    return data


def compute_log_likelihood(M0, data, Mc):
    log_likelihood = 0.0

    for x in data:
        PDF = x["PDF"]
        b = x["b"]
        dI = x["dI"]
        idx = x["idx"]

        if PDF[idx]==0:
            return -np.inf   # because P=0

        weight = 10**(-b * (Mc - M0))
        Z = np.sum(PDF * weight) * dI

        P = PDF[idx] * weight[idx] / Z
        log_likelihood += np.log(P)
    return log_likelihood


def M0_confidence_interval(log_likelihood_history):
    log_likelihood_history = np.asarray(log_likelihood_history, dtype=float)
    M0 = log_likelihood_history[:,0]
    log_likelihood = log_likelihood_history[:,1]

    # sort M0
    sort_idx = np.argsort(M0)
    M0 = M0[sort_idx]
    log_likelihood = log_likelihood[sort_idx]

    M0min_sample_idx = np.argwhere(log_likelihood.max()-log_likelihood<1.92).min()
    M0max_sample_idx = np.argwhere(log_likelihood.max()-log_likelihood<1.92).max()

    quad_coeff = np.polyfit(
                M0[M0min_sample_idx:M0max_sample_idx+1],
                log_likelihood[M0min_sample_idx:M0max_sample_idx+1],
                2
            )
    a, b, c = quad_coeff

    M0_MLE_quad = - b / (2 * a)
    ll_hat = c - b**2 / (4 * a)     #maximum value of log_likelihood (quadratic approximation)
    D = b**2 - 4*a*(c - ll_hat + 1.92)
    if D<0:
        print("No intersection with likelihood threshold.")
    M0_1 = (-b - np.sqrt(D)) / (2 * a)
    M0_2 = (-b + np.sqrt(D)) / (2 * a)

    M0_min_quad, M0_max_quad = sorted([M0_1, M0_2])

    return M0_MLE_quad, M0_min_quad, M0_max_quad, quad_coeff


def M0_estimation(data, M0_min, M0_max, tolerance, Mc):
    log_likelihood_history = []

    while M0_max - M0_min > tolerance:
        M0_1 = M0_min + (M0_max - M0_min) / 3
        M0_2 = M0_max - (M0_max - M0_min) / 3

        log_likelihood_M0_1 = compute_log_likelihood(M0_1, data, Mc)
        log_likelihood_M0_2 = compute_log_likelihood(M0_2, data, Mc)

        log_likelihood_history.append((M0_1, log_likelihood_M0_1))
        log_likelihood_history.append((M0_2, log_likelihood_M0_2))

        if log_likelihood_M0_1 < log_likelihood_M0_2:
            M0_min = M0_1
        else:
            M0_max = M0_2

    ## M0_MLE ##
    M0_MLE_quad, M0_min_quad, M0_max_quad, quad_coeff = M0_confidence_interval(log_likelihood_history)

    log_likelihood_MLE = compute_log_likelihood(M0_MLE_quad, data, Mc)
    log_likelihood_history.append((M0_MLE_quad, log_likelihood_MLE))

    return M0_MLE_quad, M0_min_quad, M0_max_quad, quad_coeff, np.asarray(log_likelihood_history, dtype=float)