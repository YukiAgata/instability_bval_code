from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append("../")
from src.catalog_analysis import process_catalog
from src.M0_estimation import random_normals
from src.M0_estimation import prepare_instability_pdf
from src.M0_estimation import M0_estimation
from src.M0_estimation import M0_confidence_interval
from src.plot import plot_M0_likelihood

def run_analysis(name, path, Mc, friction_coefficient, params_MCMC):
    print(f"Processing: {name}...")

    bmax, bmin, dI = params_MCMC

    df = pd.read_csv(path)

    # R input
    R = df["R"].to_numpy()

    # instability input
    magnitude_instability = process_catalog(df, friction_coefficient)
    if "no_error" in name:
        instability = magnitude_instability["instability1"].to_numpy()
    else:
        instability = np.maximum(
            magnitude_instability["instability1"].to_numpy(),
            magnitude_instability["instability2"].to_numpy()
        )

    # definition of function: "theob_calc"
    if name.endswith("exp"):
        def theob_calc(instability):
            return bmax - (bmax - bmin) * np.exp((instability - 1) / dI)
    elif name.endswith("linear"):
        def theob_calc(instability):
            return np.where(
                instability >= dI,
                (bmax-bmin)/(1-dI) * (1 - instability) + bmin,
                bmax
            )

    # base catalog and idx corresponding to each instability value
    N = 10**6
    n = random_normals(N, seed=1000)
    data = prepare_instability_pdf(
        instability,
        R,
        friction_coefficient,
        n,
        theob_calc,
    )

    invalid = [
        i for i, x in enumerate(data)
        if x["PDF"][x["idx"]] == 0
    ]

    if invalid:
        print(
            f"{name}: {len(invalid)} observations are "
            "outside the physical support."
        )
        for i in invalid:
            print(
                f"  index={i}, I_obs={data[i]['I_obs']:.6f}, "
                f"nearest_center={data[i]['centers'][data[i]['idx']]:.6f}"
            )
        return name, np.nan

    (
        M0_MLE_quad, M0_min_quad, M0_max_quad,
        quad_coeff,
        log_likelihood_history
    ) = M0_estimation(
        data,
        M0_min = -5,
        M0_max = 5,
        tolerance = 0.01,
        Mc = Mc
    )

    ## plot
    fig = plot_M0_likelihood(
        log_likelihood_history,
        M0_MLE_quad,
        M0_min_quad,
        M0_max_quad,
        quad_coeff
    )

    ## save figures
    output_dir = Path(f"../figures/result/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir/"M0_estimation.png", dpi=300)

    return name, M0_MLE_quad


jobs = [
    (
        "no_error_exp",
        "../data/synthetic/catalog_no_error_exp.csv",
        2.0,
        0.3,
        np.array([1.1, 0.7, 0.4]),
        # 0.33488571638243947,
        # np.array([1.20690819, 0.71769527, 0.50666853])
    ),
    # (
    #     "moderate_error_exp",
    #     "../data/synthetic/catalog_moderate_error_exp.csv",
    #     2.0,
    #     0.22600439183277168,
    #     np.array([1.00400233, 0.72069507, 0.3384601 ])
    # ),
    # (
    #     "large_error_exp",
    #     "../data/synthetic/catalog_large_error_exp.csv",
    #     2.0,
    #     0.551773385945006,
    #     np.array([0.91287519, 0.74430288, 0.28887689])
    # ),
    # (
    #     "dataset1_exp",
    #     "../data/california/processed/dataset1.csv",
    #     2.2,
    #     0.3247698239315371,
    #     np.array([1.03612091, 0.86910025, 0.23268873])
    # ),
    # (
    #     "dataset2_exp",
    #     "../data/california/processed/dataset2.csv",
    #     2.2,
    #     0.3512901714395389,
    #     np.array([1.06930792, 0.85532783, 0.18792537])
    # ),
    (
        "no_error_linear",
        "../data/synthetic/catalog_no_error_linear.csv",
        2.0,
        0.3,
        np.array([1.1, 0.7, 0.9]),
        # 0.31013790806505337,
        # np.array([1.09242589, 0.68960332, 0.91178735])
    ),
    # (
    #     "moderate_error_linear",
    #     "../data/synthetic/catalog_moderate_error_linear.csv",
    #     2.0,
    #     0.42512715428429504,
    #     np.array([1.07575645, 0.75650728, 0.70323366])
    # ),
    # (
    #     "large_error_linear",
    #     "../data/synthetic/catalog_large_error_linear.csv",
    #     2.0,
    #     0.4453363590690218,
    #     np.array([0.99975709, 0.78583168, 0.58406466])
    # ),
    # (
    #     "dataset1_linear",
    #     "../data/california/processed/dataset1.csv",
    #     2.2,
    #     0.3182667502130999,
    #     np.array([1.0323004 , 0.89899286, 0.49994493])
    # ),
    # (
    #     "dataset2_linear",
    #     "../data/california/processed/dataset2.csv",
    #     2.2,
    #     0.36903814346277386,
    #     np.array([1.06562332, 0.88327444, 0.63615478])
    # ),
]

def run_job(job):
    return run_analysis(*job)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_job, jobs))

    print("All jobs finished.")
    print(results)