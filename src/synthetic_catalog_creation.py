# N-E-D coordinate
# stress: tension positive

import numpy as np
import os
from pathlib import Path
import runpy

# create N vectors with random orientations
def random_vector(N, rng):
    # theta, phi: random angles (uniform distribution on unit sphere)
    u = rng.random(N)
    v = rng.random(N)
    theta = np.arccos(2 * u - 1)
    phi = 2 * np.pi * v
    nn = np.sin(theta)*np.cos(phi)
    ne = np.sin(theta)*np.sin(phi)
    nd = np.cos(theta)
    return nn, ne, nd

# create N fault planes with random directions
def generate_random_planes(N, rng):
    nn, ne, nd = random_vector(N, rng)
    # fault normal vector（upward: nd < 0）
    flip = nd > 0
    nn[flip] *= -1
    ne[flip] *= -1
    nd[flip] *= -1
    # calculate strike and dip of the fault plane
    strike = np.arctan2(- nn, ne)   # radian
    dip    = np.arccos(- nd)    # radina
    return np.degrees(strike), np.degrees(dip)  # degree

# normal/shear stress, friction coefficient → instability
def stress_to_instability(sigma_n, tau_e, friction_coefficient):
    sigma_n = np.array(sigma_n)
    tau_e = np.array(tau_e)
    instability = 1 - np.abs(friction_coefficient * sigma_n - tau_e + np.sqrt(friction_coefficient**2 + 1)) / np.sqrt(friction_coefficient**2 + 1)
    return instability

# magnitude sampling
def magnitude_sampling(b, M0, rng):
    b = np.asarray(b)
    u = rng.random(len(b))    # random upper cummulative probability (0-1)
    magnitude = - np.log10(u) / b + M0
        # sample the matnitude whose distribution follows the GR law 
    return magnitude

### create synthetic catalog ###
def random_focalmech_and_magnitude(sigma,
                                   model_i_to_b,
                                   model_type,
                                   N_earthquake,
                                   M0,
                                   friction_coefficient,
                                   rng):
    # sigma = np.array([
    #           [Snn, Sne, Sdn],
    #           [Sne, See, Sed],
    #           [Sdn, Sed, Sdd]
    #           ])
    # instability_to_theobval (exsample)
    #   def instability_to_theobval(instability):
    #       return 1.1 - 0.05 * np.exp(2.1 * instability)

    # create N_earthquake random fault planes
    strike, dip = generate_random_planes(N_earthquake, rng)    # degree
    strike = np.radians(strike) # radian
    dip = np.radians(dip)       # radian
    N = len(strike)

    # sort the principal stresses    * ps1: most compressional stress (compression positive)
    eigvals = np.linalg.eigvalsh(sigma)
    ps1, _, ps3 = - np.sort(eigvals)

    # fault normal vector
    n1 = np.column_stack([
        - np.sin(dip) * np.sin(strike),
        np.sin(dip) * np.cos(strike),
        - np.cos(dip)])
    # traction: T = σn
    Traction1 = n1 @ sigma.T
    # normal stress
    sigma_n1 = np.sum(Traction1 * n1, axis=1)
    sigma_n1_scaled = (- sigma_n1-(ps1+ps3)/2)*2/(ps1-ps3)    # transformation into compression positive + normalization
    # shear stress (slip direction)
    tau_vec1 = Traction1 - sigma_n1[:, None] * n1
    tau_slip1 = np.linalg.norm(tau_vec1, axis=1)
    tau_slip1_scaled = (tau_slip1)*2/(ps1-ps3)  # normalization
    tau_hat = tau_vec1 / np.linalg.norm(tau_vec1, axis=1)[:, None]  #unit vector

    # strike vector
    t = np.column_stack([np.cos(strike), np.sin(strike), np.zeros(N)])
    # calculating rake (direction of shear traction compared from strike)
    s1 = np.sum(tau_hat * t, axis=1)
    s2 = np.sum(tau_hat * np.cross(n1, t), axis=1)
    rake = np.arctan2(s2, s1)  # radian
    instability = stress_to_instability(sigma_n1_scaled, tau_slip1_scaled, friction_coefficient)

    # instability -> model b-value -> magnitude
    theobval = model_i_to_b(instability, model_type)
    magnitude = magnitude_sampling(theobval, M0, rng)

    return np.degrees(strike), np.degrees(dip), np.degrees(rake), magnitude
    # strike, dip, rake: degree

def synthetic_catalog_creation(sigma,
                               model_i_to_b,
                               model_type,
                               M0,
                               Mc,
                               friction_coefficient,
                               N_earthquake,
                               N_subgroup,
                               seed=10):
    # fix the seed
    rng = np.random.default_rng(seed)

    strike_list = []
    dip_list = []
    rake_list = []
    magnitude_list = []

    n_selected = 0

    while n_selected < N_earthquake:
        strike_sub, dip_sub, rake_sub, magnitude_sub = random_focalmech_and_magnitude(
                sigma,
                model_i_to_b,
                model_type,
                N_subgroup,
                M0,
                friction_coefficient,
                rng
            )
        mask = magnitude_sub >= Mc
        strike_list.append(strike_sub[mask])
        dip_list.append(dip_sub[mask])
        rake_list.append(rake_sub[mask])
        magnitude_list.append(magnitude_sub[mask])

        n_selected += np.sum(mask)

    strike = np.concatenate(strike_list)[:N_earthquake]
    dip = np.concatenate(dip_list)[:N_earthquake]
    rake = np.concatenate(rake_list)[:N_earthquake]
    magnitude = np.concatenate(magnitude_list)[:N_earthquake]

    return strike, dip, rake, magnitude

### perturbation of focal mechanisms ###
# rotate the vector: n around the vector: u by angle: theta (Rodrigues's formula)
def rotate(n, u, theta):
    return (n*np.cos(theta)[:, None] +
            np.cross(u, n)*np.sin(theta)[:, None] +
            u*np.sum(u*n, axis=1, keepdims=True)*(1 - np.cos(theta)[:, None])
            )

def focmech_perturbation(strike, dip, rake, focmech_uncertainty, seed):
    #input angles: degree

    # fix the seed
    rng = np.random.default_rng(seed)

    strike = np.radians(strike) # radian
    dip = np.radians(dip)       # radian
    rake = np.radians(rake)     # radian

    N = len(strike)

    # normal vector (original)
    n = np.column_stack([
        - np.sin(dip) * np.sin(strike),
        np.sin(dip) * np.cos(strike),
        - np.cos(dip)
        ])
    # slip vector (original)
    s = np.column_stack([
        np.cos(rake)*np.cos(strike) + np.cos(dip)*np.sin(rake)*np.sin(strike),
        np.cos(rake)*np.sin(strike) - np.cos(dip)*np.sin(rake)*np.cos(strike),
        -np.sin(dip)*np.sin(rake)
        ])

    # P/T axis (original)
    T = s + n
    T /= np.linalg.norm(T, axis=1, keepdims=True)   # normalization
    P = s - n
    P /= np.linalg.norm(P, axis=1, keepdims=True)   # normalization

    # u: rotation axis (random orientation)
    un, ue, ud = random_vector(N, rng)
    u = np.column_stack([un, ue, ud])
    # theta (radian): rotation angle which is randomly sampled (1 sigma: fault_plane_uncertainty)
    theta = rng.normal(
        loc=0.0,
        scale=np.radians(focmech_uncertainty),
        size=N
    )

    T_new = rotate(T, u, theta)
    P_new = rotate(P, u, theta)

    n_new = (T_new - P_new)/np.sqrt(2)
    s_new = (T_new + P_new)/np.sqrt(2)
    flip = n_new[:,2] > 0   # flip the vectors whose n_new are downwards
    n_new[flip] *= -1
    s_new[flip] *= -1

    strike_perturbed1 = np.mod(
                            np.degrees(np.arctan2(-n_new[:,0], n_new[:,1])),
                            360.0
                        )
    dip_perturbed1 = np.degrees(np.arccos(-n_new[:,2]))

    t_new = np.column_stack([
                np.cos(np.radians(strike_perturbed1)),
                       np.sin(np.radians(strike_perturbed1)),
                       np.zeros(N)
            ])

    s_new_on_plane = s_new - np.sum(s_new * n_new, axis=1, keepdims=True) * n_new
    s1 = np.sum(s_new_on_plane * t_new, axis=1)
    s2 = np.sum(s_new_on_plane * (np.cross(n_new, t_new)/np.linalg.norm(np.cross(n_new, t_new), axis=1, keepdims=True)), axis=1)    # 数値誤差による規定崩壊回避
    rake_perturbed1 = np.degrees(np.arctan2(s2, s1))

    return strike_perturbed1, dip_perturbed1, rake_perturbed1
        # degree


### perturbation of stress field ###
def stress_inverse(strike, dip, rake, stressinverse_dir):
    # save the current directory
    original_dir = Path.cwd()
    stressinverse_dir = Path(stressinverse_dir).resolve()
    data_dir = stressinverse_dir / "Data"
    program_dir = stressinverse_dir / "Programs_PYTHON"
    
    try:
        # move to the Data directory
        os.chdir(data_dir)
        # save "optrake_mech.txt"
        data = np.column_stack([strike, dip, rake])
        np.savetxt(
            "optrake_mech.txt",
            data,
            delimiter="\t",
            header="% strike\tdip\trake",
            comments=""
        )

        # move to the Programs_PYTHON directory
        os.chdir(program_dir)
        # conduct StressInverse.py
        result = runpy.run_path("StressInverse.py")
        sigma_inv = result["tau_optimum"]
        R_inv = result["shape_ratio"]

    finally:
        # always return to the original directory
        os.chdir(original_dir)

    return sigma_inv, R_inv