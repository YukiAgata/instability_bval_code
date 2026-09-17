# N-E-D coordinates
# stress tensors: tension positive
# the order of the events are the same as the input

from numba import njit
import emcee
import numpy as np
from obspy.imaging.beachball import aux_plane
import pandas as pd

### instability calculation ###

# normal / shear stress, friction coefficients -> instability
def stress_to_instability(sigma_n, tau_e, friction_coefficient):
    sigma_n = np.array(sigma_n)
    tau_e = np.array(tau_e)
    instability = (tau_e - friction_coefficient * sigma_n) / np.sqrt(friction_coefficient**2 + 1) #圧縮正なので
    return instability

# focal mechanisms -> normal / shear (slip direction) stress
def scaled_stress_on_plane(strike, dip, rake, sigma):
    # strike, dip, rake: degrees
    N = len(strike)
    strike = np.radians(strike)
    dip = np.radians(dip)
    rake = np.radians(rake)

    # sort the principal stresses    * ps1: most compressional stress (compression positive)
    eigvals = np.linalg.eigvalsh(sigma)  # (N, 3)
    eigvals_sorted = np.sort(eigvals, axis=1)
    ps1 = - eigvals_sorted[:, 0]    # maximum compression
    ps3 = - eigvals_sorted[:, 2]    # minimum compression

    # fault normal vector
    n = np.column_stack([
        - np.sin(dip) * np.sin(strike),
        np.sin(dip) * np.cos(strike),
        - np.cos(dip)
        ])
    # slip vector
    s = np.column_stack([
        np.cos(rake)*np.cos(strike) + np.cos(dip)*np.sin(rake)*np.sin(strike),
        np.cos(rake)*np.sin(strike) - np.cos(dip)*np.sin(rake)*np.cos(strike),
        -np.sin(dip)*np.sin(rake)
        ])

    # traction: T = σn
    Traction = np.einsum('nij,nj->ni', sigma, n)
    # normal stress
    sigma_normal = np.sum(Traction * n, axis=1)
    sigma_normal_scaled = (- sigma_normal-(ps1+ps3)/2)*2/(ps1-ps3)    # transformation into compression positive + normalization
    # shear stress (slip direction)
    tau_vec = Traction - sigma_normal[:, None] * n
    tau_slip1 = np.sum(tau_vec * s, axis=1)
    tau_slip_scaled = (tau_slip1)*2/(ps1-ps3)    # noramlization
    return sigma_normal_scaled, tau_slip_scaled

def catalog_to_stress_and_instability(sigma, friction_coefficient, strike, dip, rake):
    # strike, dip, rake: degrees
    # sigma = np.array([
    #             [Snn, Sne, Sdn],
    #             [Sne, See, Sed],
    #             [Sdn, Sed, Sdd]
    #         ])

    # adujsting sigma   * broadcast (3,3) → (N,3,3)
    if sigma.ndim == 2:
        sigma = np.repeat(sigma[None, :, :], len(strike), axis=0)

    normal_stress, shear_stress = scaled_stress_on_plane(strike, dip, rake, sigma)
    instability = stress_to_instability(normal_stress, shear_stress, friction_coefficient)
    return normal_stress, shear_stress, instability


def process_catalog(df, m_analysis):
    magnitude = df["magnitude"].to_numpy()
    strike = df["strike"].to_numpy()
    dip = df["dip"].to_numpy()
    rake = df["rake"].to_numpy()

    sigma = df[
        ["Snn", "Sne", "Snd",
         "Sne", "See", "Sed",
         "Snd", "Sed", "Sdd"]
    ].to_numpy().reshape(-1, 3, 3)

    n_events = len(df)
    strike_aux, dip_aux, rake_aux = np.zeros(n_events), np.zeros(n_events), np.zeros(n_events)
    for k in range(n_events):
        strike_aux[k], dip_aux[k], rake_aux[k] = aux_plane(strike[k], dip[k], rake[k])

    magnitude_instability_list = []
    _, _, instability1 = catalog_to_stress_and_instability(
        sigma, m_analysis, strike, dip, rake
    )
    _, _, instability2 = catalog_to_stress_and_instability(
        sigma, m_analysis, strike_aux, dip_aux, rake_aux
    )

    magnitude_instability_list.append(pd.DataFrame({
        "m_analysis": m_analysis,
        "magnitude": magnitude,
        "instability1": instability1,
        "instability2": instability2
    }))

    return pd.concat(magnitude_instability_list, ignore_index=True)

def bval_estimation(magnitude, Mc):
    magnitude = magnitude[magnitude >= Mc]
    bval_est = np.log10(np.e) / (magnitude.mean() - Mc)
    bval_err = bval_est/np.sqrt(len(magnitude))
    return bval_est, bval_err, magnitude

def instability_bval_relationship(instability, magnitude, bin_number, Mc):
    instability_bin_min, instability_bin_ave, instability_bin_max, bval_bin, berror_bin = [
        np.zeros(bin_number) for _ in range(5)
    ]

    idx = np.argsort(instability)[::-1]
    instability = instability[idx]
    magnitude = magnitude[idx]
    split_instability = np.array_split(instability, bin_number)
    split_magnitude = np.array_split(magnitude, bin_number)

    for k in range(bin_number):
        instability_bin_min[k] = split_instability[k].min()
        instability_bin_ave[k] = split_instability[k].mean()
        instability_bin_max[k] = split_instability[k].max()
        bval_bin[k], berror_bin[k], _ = bval_estimation(split_magnitude[k], Mc)

    return instability_bin_min, instability_bin_ave, instability_bin_max, bval_bin, berror_bin


## parameter estimation ##
@njit
def compute_theob_exp(b_max, b_min, delta_I, instability):
    return b_max - (b_max - b_min) * np.exp((instability - 1) / delta_I)

@njit
def compute_theob_linear(b_max, b_min, delta_I, instability):
    return np.where(
        instability >= delta_I,
        b_max - (b_max - b_min) / (1 - delta_I) * (instability - delta_I),
        b_max
    )

@njit
def compute_log_likelihood(params, theob_calc, instability, magnitude, Mc):
    # theob_calc: compute_theob_exp / compute_theob_linear
    Nevent = len(magnitude)
    if Nevent != len(instability):
        print("Magnitude data and instability data have different lengths.")

    b_max, b_min, delta_I = params
    ## restriction of parameters ##
    if b_max > 1.5:
        ll = -np.inf
        return ll
    if b_max < b_min:
        ll = -np.inf
        return ll
    if b_min < 0:
        ll = -np.inf
        return ll
    if delta_I <= 0:
        ll = -np.inf
        return ll

    ## likelihood calculation ##
    theob = theob_calc(b_max, b_min, delta_I, instability)
    if np.any(theob <= 0) or np.any(~np.isfinite(theob)):
        ll = -np.inf
        return ll
    else:
        ll = np.sum(np.log(theob)) + Nevent * np.log(np.log(10)) - np.sum(theob * np.log(10) * (magnitude - Mc))
        return ll

def MCMC_on_given_instability(instability, magnitude, theob_calc, Mc, pos):
    nwalkers = 32
    ndim = 3

    pos = np.asarray(pos, dtype=float)
    pos = pos + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                compute_log_likelihood,
                args=(theob_calc, instability, magnitude, Mc)
                )
    sampler.run_mcmc(pos, 5000, progress=True)
    samples = sampler.get_chain(
                discard=1000,
                thin=10,
                flat=True
                )

    log_probs = sampler.get_log_prob(
                    discard=1000,
                    thin=10,
                    flat=True
                )

    imax = np.argmax(log_probs)
    params_MLE = samples[imax]
    ll_MLE = log_probs[imax]

    return samples, params_MLE, ll_MLE

# quadratic approximation, determining MLE and 95% confidence interval
def friction_quad_approx(log_likelihood_history):
    log_likelihood_history = np.asarray(log_likelihood_history, dtype=float)

    m = log_likelihood_history[:,0]
    log_likelihood = log_likelihood_history[:,1]

    # sort m
    sort_idx = np.argsort(m)
    m = m[sort_idx]
    log_likelihood = log_likelihood[sort_idx]

    mmin_sample_idx = np.argwhere(log_likelihood.max()-log_likelihood<1.92).min()
    mmax_sample_idx = np.argwhere(log_likelihood.max()-log_likelihood<1.92).max()

    quad_coeff = np.polyfit(
                m[mmin_sample_idx:mmax_sample_idx+1],
                log_likelihood[mmin_sample_idx:mmax_sample_idx+1],
                2
            )
    a, b, c = quad_coeff

    m_MLE_quad = - b / (2 * a)
    ll_hat = c - b**2 / (4 * a)     # maximum value of log_likelihood (quadratic approximation)
    D = b**2 - 4*a*(c - ll_hat + 1.92)
    if D<0:
        print("No intersection with likelihood threshold.")
    m1 = (-b - np.sqrt(D)) / (2 * a)
    m2 = (-b + np.sqrt(D)) / (2 * a)

    m_min_quad, m_max_quad = sorted([m1, m2])
    m_min_quad = max(0, m_min_quad)
    m_max_quad = min(1, m_max_quad)

    return m_MLE_quad, m_min_quad, m_max_quad, quad_coeff


def function_estimation(name, df, m_min, m_max, tolerance, Mc, seed, pos=[1, 0.8, 0.3]):
    if "exp" in name:
        theob_calc = compute_theob_exp
    elif "linear" in name:
        theob_calc = compute_theob_linear

    # fix the seed
    np.random.seed(seed)

    magnitude = df["magnitude"].to_numpy()

    log_likelihood_history = []

    ## log_likelihood of m_min, m_max ##
    m1 = m_min
    m2 = m_max

    magnitude_instability_list1 = process_catalog(df, m1)
    magnitude_instability_list2 = process_catalog(df, m2)

    if "no_error" in name:
        instability_m1 = magnitude_instability_list1["instability1"].to_numpy()
        instability_m2 = magnitude_instability_list2["instability1"].to_numpy()
    else:
        instability_m1 = np.maximum(
            magnitude_instability_list1["instability1"].to_numpy(),
            magnitude_instability_list1["instability2"].to_numpy()
        )
        instability_m2 = np.maximum(
            magnitude_instability_list2["instability1"].to_numpy(),
            magnitude_instability_list2["instability2"].to_numpy()
        )

    # m1
    _, _, log_likelihood_m1 = MCMC_on_given_instability(
                                instability_m1,
                                magnitude,
                                theob_calc,
                                Mc,
                                pos
                                )
    log_likelihood_history.append((m1, log_likelihood_m1))

    # m2
    _, _, log_likelihood_m2 = MCMC_on_given_instability(
                                instability_m2,
                                magnitude,
                                theob_calc,
                                Mc,
                                pos
                                )
    log_likelihood_history.append((m2, log_likelihood_m2))

    while m_max - m_min > tolerance:
        if m_max < m_min:
            print(f"m_max({m_max}) is less than m_min({m_min})")
        m1 = m_min + (m_max - m_min) / 3
        m2 = m_max - (m_max - m_min) / 3
        
        magnitude_instability_list1 = process_catalog(df, m1)
        magnitude_instability_list2 = process_catalog(df, m2)

        if "no_error" in name:
            instability_m1 = magnitude_instability_list1["instability1"].to_numpy()
            instability_m2 = magnitude_instability_list2["instability1"].to_numpy()
        else:
            instability_m1 = np.maximum(
                magnitude_instability_list1["instability1"].to_numpy(),
                magnitude_instability_list1["instability2"].to_numpy()
            )
            instability_m2 = np.maximum(
                magnitude_instability_list2["instability1"].to_numpy(),
                magnitude_instability_list2["instability2"].to_numpy()
            )

        # m1
        _, _, log_likelihood_m1 = MCMC_on_given_instability(
                                    instability_m1,
                                    magnitude,
                                    theob_calc,
                                    Mc,
                                    pos
                                    )
        log_likelihood_history.append((m1, log_likelihood_m1))

        # m2
        _, _, log_likelihood_m2 = MCMC_on_given_instability(
                                    instability_m2,
                                    magnitude,
                                    theob_calc,
                                    Mc,
                                    pos
                                    )
        log_likelihood_history.append((m2, log_likelihood_m2))

        if log_likelihood_m1 < log_likelihood_m2:
            m_min = m1
        else:
            m_max = m2

    ## m_MLE_quad ##
    m_MLE_quad, m_min_quad, m_max_quad, quad_coeff = friction_quad_approx(log_likelihood_history)

    magnitude_instability_MLE_quad = process_catalog(df, m_MLE_quad)

    if "no_error" in name:
        instability_MLE_quad = magnitude_instability_MLE_quad["instability1"].to_numpy()
    else:
        instability_MLE_quad = np.maximum(
            magnitude_instability_MLE_quad["instability1"].to_numpy(),
            magnitude_instability_MLE_quad["instability2"].to_numpy()
        )

    samples_MLE_quad, params_MLE_quad, log_likelihood_MLE_quad = MCMC_on_given_instability(
                                                                    instability_MLE_quad,
                                                                    magnitude,
                                                                    theob_calc,
                                                                    Mc,
                                                                    pos
                                                                    )
    log_likelihood_history.append((m_MLE_quad, log_likelihood_MLE_quad))

    return (
        m_MLE_quad, m_min_quad, m_max_quad, quad_coeff,
        samples_MLE_quad,
        np.asarray(log_likelihood_history, dtype=float),
        params_MLE_quad,
        log_likelihood_MLE_quad
    )