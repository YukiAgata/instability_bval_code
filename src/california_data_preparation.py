### dataset creation ###
# integrate earthquake and stress catalog
# exclude earthquakes whose hypocenters are farther than max_dist_km = 10 [km]

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def latlon_to_km(lat, lon, lat0, lon0):
    # transform latitude/longitude into distance [km]
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 111.32
    return x, y


def merge_earthquake_and_stress(df_earthquake, df_stress, max_dist_km = 10.0):
    # merge earthquake and stress catalogs
    # *exclude earthquakes whose hypocenters are farther than max_dist_km = 10 [km]
    lat0 = df_stress["latitude_stress"].mean()
    lon0 = df_stress["longitude_stress"].mean()

    # stress catalog (Cartesian coordinate)
    sx, sy = latlon_to_km(
        df_stress["latitude_stress"].to_numpy(),
        df_stress["longitude_stress"].to_numpy(),
        lat0, lon0
    )
    # earthquake catalog (Cartesian coordinate)
    ex, ey = latlon_to_km(
        df_earthquake["latitude_event"].to_numpy(),
        df_earthquake["longitude_event"].to_numpy(),
        lat0, lon0
    )

    # nearest neighbor search
    tree = cKDTree(np.column_stack([sx, sy]))
    distances, indices = tree.query(np.column_stack([ex, ey]), k=1)

    # distance filtering
    mask = distances <= max_dist_km
    df_earthquake_filtered = df_earthquake[mask].reset_index(drop=True)
    df_stress_nearest = df_stress.iloc[indices[mask]].reset_index(drop=True)

    df = pd.concat([df_earthquake_filtered, df_stress_nearest], axis=1)
    return df


def exclude_geotherm(df, r_km = 0):
    # dist_mask: threshold distance from geothermal field [km]

    # coso
    dist_coso_ew = (df["longitude_stress"].to_numpy()-(-117.79))*111.32
    dist_coso_ns = (df["latitude_stress"].to_numpy()-36.02)*111.32
    dist_coso_sq = dist_coso_ew**2 + dist_coso_ns**2

    # desert hotsprings
    dist_dh_ew = (df["longitude_stress"].to_numpy()-(-116.6))*111.32
    dist_dh_ns = (df["latitude_stress"].to_numpy()-34)*111.32
    dist_dh_sq = dist_dh_ew**2 + dist_dh_ns**2

    # salton sea
    dist_ss_ew = (df["longitude_stress"].to_numpy()-(-115.6))*111.32
    dist_ss_ns = (df["latitude_stress"].to_numpy()-33.16)*111.32
    dist_ss_sq = dist_ss_ew**2 + dist_ss_ns**2

    # North Brawley
    dist_nb_ew = (df["longitude_stress"].to_numpy()-(-115.52))*111.32
    dist_nb_ns = (df["latitude_stress"].to_numpy()-33.02)*111.32
    dist_nb_sq = dist_nb_ew**2 + dist_nb_ns**2

    # Haber
    dist_haber_ew = (df["longitude_stress"].to_numpy()-(-115.54))*111.32
    dist_haber_ns = (df["latitude_stress"].to_numpy()-32.71)*111.32
    dist_haber_sq = dist_haber_ew**2 + dist_haber_ns**2

    mask_coso = dist_coso_sq < r_km**2
    mask_ss = dist_ss_sq < r_km**2
    mask_dh = dist_dh_sq < r_km**2
    mask_nb = dist_nb_sq < r_km**2
    mask_haber = dist_haber_sq < r_km**2
    mask_geotherm = mask_coso|mask_dh|mask_ss|mask_nb|mask_haber

    return df[mask_geotherm], df[~mask_geotherm]


# Mc estimation (Goodness of Fit)
def bval_estimation(magnitude, Mc):
    magnitude = magnitude[magnitude >= Mc]
    bval_est = np.log10(np.e) / (magnitude.mean() - Mc)
    bval_err = bval_est/np.sqrt(len(magnitude))
    return bval_est, bval_err, magnitude

def CFMD(magnitude, magnitude_binedges, Mc_candidate):
    counts, bin_edges = np.histogram(magnitude, bins=magnitude_binedges)
    magnitude_centers = (bin_edges[:-1] + bin_edges[1:])/2
    Mc_candidate_idx = np.argmin(np.abs(magnitude_centers - Mc_candidate))
    CFMD_obs = np.cumsum(counts[::-1])[::-1]
    CFMD_pred = CFMD_obs.copy().astype(float)
    b, _, _ = bval_estimation(magnitude, Mc_candidate)
    for i in range(len(magnitude_centers)):
        CFMD_pred[i] = CFMD_obs[Mc_candidate_idx] * 10 ** (-b * (magnitude_centers[i] - magnitude_centers[Mc_candidate_idx]))
    return CFMD_obs, CFMD_pred, b, magnitude_centers

def Mc_estimation(magnitude, GoF_threshold):
    Mc_candidate = np.arange(0, 5, 0.1)
    GoF = np.zeros(len(Mc_candidate))
    b = np.zeros(len(Mc_candidate))
    CFMD_plotsite = np.arange(-1.5, 8.1, 0.1)
    CFMD_obs = np.zeros((len(Mc_candidate), len(CFMD_plotsite)-1))
    CFMD_pred = np.zeros((len(Mc_candidate), len(CFMD_plotsite)-1))
    for i, Mc_i in enumerate(Mc_candidate):
        CFMD_obs[i], CFMD_pred[i], b[i], magnitude_centers = CFMD(magnitude, CFMD_plotsite-0.05, Mc_i)
        Mc_i_idx = np.argmin(np.abs(magnitude_centers - Mc_i))
        GoF[i] = 100 - 100 * np.abs(CFMD_obs[i, Mc_i_idx:] - CFMD_pred[i, Mc_i_idx:]).sum()/CFMD_obs[i, Mc_i_idx:].sum()
    Mc_idx = np.where(GoF >= GoF_threshold)[0][0]
    return Mc_idx, Mc_candidate, GoF, b, CFMD_obs, CFMD_pred, magnitude_centers