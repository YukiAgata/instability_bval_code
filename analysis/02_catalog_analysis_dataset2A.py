from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append("../")
from src.catalog_analysis import process_catalog
from src.catalog_analysis import instability_bval_relationship
from src.catalog_analysis import parameter_estimation
from src.plot import plot_instability_bval_relationship
from src.plot import plot_m_likelihood

bin_number = 10
m_analysis = np.arange(0, 1.05, 0.05)
Mc_dict = {
    "dataset2_A_exp": 2.2,
    "dataset2_A_linear": 2.2,
}

def theob_calc(params, model_type, instability):
    bmax, bmin, dI = params
    if model_type == "exp":
        return bmax - (bmax - bmin) * np.exp((instability - 1) / dI)
    if model_type == "linear":
        return np.where(
            instability >= dI,
            (bmax-bmin)/(1-dI) * (1 - instability) + bmin,
            bmax
        )

def run_analysis(name, path, model_type, seed):
    print(f'Processins: {name}...')

    df = pd.read_csv(path)

    ## parameter estimation
    (
        m_MLE_quad, m_min_quad, m_max_quad, quad_coeff,
        samples_MLE_quad,
        log_likelihood_history,
        params_MLE_quad
    ) = parameter_estimation(
        name,
        df,
        m_min=0,
        m_max=1,
        tolerance=0.01,
        Mc=Mc_dict[name],
        seed=seed
    )

    ## analysis at m_MCMC (preparation for plot)
    magnitude_instability = process_catalog(df, m_MLE_quad)
    magnitude = magnitude_instability["magnitude"].to_numpy()
    if "no_error" in name:
        instability = magnitude_instability["instability1"].to_numpy()
    else:
        instability = np.maximum(
            magnitude_instability["instability1"].to_numpy(),
            magnitude_instability["instability2"].to_numpy()
        )

    (
        instability_bin_min,
        instability_bin_ave,
        instability_bin_max,
        bval_bin,
        berror_bin
    ) = instability_bval_relationship(
        instability,
        magnitude,
        bin_number,
        Mc_dict[name]
    )

    ## theoretical models
    def theob_calc_MCMC(instability):
        return theob_calc(params_MLE_quad, model_type, instability)

    if name.startswith("dataset1") or name.startswith("dataset2"):
        model_i_to_b = None
    else:
        if model_type == "exp":
            true_params = [1.1, 0.7, 0.4]
        elif model_type == "linear":
            true_params = [1.1, 0.7, 0.9]

        def model_i_to_b(instability):
            return theob_calc(
                true_params,
                model_type,
                instability
            )

    ## plot
    fig, _, fig_corner = plot_instability_bval_relationship(
        instability_bin_min,
        instability_bin_ave,
        instability_bin_max,
        bval_bin,
        berror_bin,
        theob_calc_MCMC,
        model_i_to_b,
        params_MLE_quad,
        samples_MLE_quad
    )

    fig_m = plot_m_likelihood(log_likelihood_history, m_MLE_quad, m_min_quad, m_max_quad, quad_coeff)

    ## save figures
    output_dir = Path(f"../figures/result/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir/"instability_bval.png", bbox_inches="tight", dpi=300)
    fig_corner.savefig(output_dir/"corner_plot.png", bbox_inches="tight", dpi=300)
    fig_m.savefig(output_dir/"m_estimation.png", bbox_inches="tight", dpi=300)

    plt.close(fig)
    plt.close(fig_corner)
    plt.close(fig_m)

    return name, m_MLE_quad, params_MLE_quad


jobs = [
    (
        "dataset2_A_exp",
        "../data/california/processed/dataset2_A.csv",
        "exp",
        112
    ),
    (
        "dataset2_A_linear",
        "../data/california/processed/dataset2_A.csv",
        "linear",
        113
    )
]

def run_job(job):
    return run_analysis(*job)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_job, jobs))

    # results を DataFrame に変換
    results_df = pd.DataFrame(
        results,
        columns=["name", "m_MLE_quad", "params_MLE_quad"]
    )

    # decompose params_MCMC
    results_df[["bmax", "bmin", "dI"]] = pd.DataFrame(
        results_df["params_MLE_quad"].tolist(),
        index=results_df.index
    )

    # delete "params_MCMC"
    results_df = results_df.drop(columns=["params_MLE_quad"])

    # save results as csv file
    output_path = Path("../data/analysis_results_dataset2_A.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
