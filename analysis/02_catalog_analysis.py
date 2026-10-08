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
from src.catalog_analysis import function_estimation
from src.plot import plot_binning
from src.plot import plot_parameter_estimation
from src.plot import plot_corner
from src.plot import plot_m_likelihood

bin_number = 10
m_analysis = np.arange(0, 1.05, 0.05)
Mc_dict = {
    "no_error_exp": 2.0,
    "moderate_error_exp": 2.0,
    "large_error_exp": 2.0,
    "dataset1_exp": 2.2,
    "dataset2_exp": 2.2,
    "no_error_linear": 2.0,
    "moderate_error_linear": 2.0,
    "large_error_linear": 2.0,    
    "dataset1_linear": 2.2,
    "dataset2_linear": 2.2,
    "dataset1_GTF10in_exp": 2.2,
    "dataset1_GTF10out_exp": 2.2,
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

    ## binning approach (Section 2.2.1)
    magnitude_instability_03 = process_catalog(df, 0.3)
    magnitude_instability_06 = process_catalog(df, 0.6)
    magnitude_instability_09 = process_catalog(df, 0.9)

    magnitude_03 = magnitude_instability_03["magnitude"].to_numpy()
    magnitude_06 = magnitude_instability_06["magnitude"].to_numpy()
    magnitude_09 = magnitude_instability_09["magnitude"].to_numpy()

    if "no_error" in name:
        instability_03 = magnitude_instability_03["instability1"].to_numpy()
        instability_06 = magnitude_instability_06["instability1"].to_numpy()
        instability_09 = magnitude_instability_09["instability1"].to_numpy()
    else:
        instability_03 = np.maximum(
            magnitude_instability_03["instability1"].to_numpy(),
            magnitude_instability_03["instability2"].to_numpy()
        )
        instability_06 = np.maximum(
            magnitude_instability_06["instability1"].to_numpy(),
            magnitude_instability_06["instability2"].to_numpy()
        )
        instability_09 = np.maximum(
            magnitude_instability_09["instability1"].to_numpy(),
            magnitude_instability_09["instability2"].to_numpy()
        )

    (
        instability_bin_min_03,
        instability_bin_ave_03,
        instability_bin_max_03,
        bval_bin_03,
        berror_bin_03
    ) = instability_bval_relationship(
        instability_03,
        magnitude_03,
        bin_number,
        Mc_dict[name]
    )
    (
        instability_bin_min_06,
        instability_bin_ave_06,
        instability_bin_max_06,
        bval_bin_06,
        berror_bin_06
    ) = instability_bval_relationship(
        instability_06,
        magnitude_06,
        bin_number,
        Mc_dict[name]
    )
    (
        instability_bin_min_09,
        instability_bin_ave_09,
        instability_bin_max_09,
        bval_bin_09,
        berror_bin_09
    ) = instability_bval_relationship(
        instability_09,
        magnitude_09,
        bin_number,
        Mc_dict[name]
    )

    ## defining model_i_to_b
    if name.startswith("dataset1") or name.startswith("dataset2"):
        model_i_to_b = None
    else:
        if model_type == "exp":
            true_params = [1.1, 0.7, 0.4]
        elif model_type == "linear":
            true_params = [1.1, 0.7, 0.7]

        def model_i_to_b(instability):
            return theob_calc(
                true_params,
                model_type,
                instability
            )

    fig_bin_03 = plot_binning(
        instability_bin_min_03,
        instability_bin_ave_03,
        instability_bin_max_03,
        bval_bin_03,
        berror_bin_03,
        model_i_to_b
    )
    fig_bin_06 = plot_binning(
        instability_bin_min_06,
        instability_bin_ave_06,
        instability_bin_max_06,
        bval_bin_06,
        berror_bin_06,
        model_i_to_b
    )
    fig_bin_09 = plot_binning(
        instability_bin_min_09,
        instability_bin_ave_09,
        instability_bin_max_09,
        bval_bin_09,
        berror_bin_09,
        model_i_to_b
    )

    ## function estimation (Section 2.2.2)
    (
        m_MLE_quad, m_min_quad, m_max_quad, quad_coeff,
        samples_MLE_quad,
        log_likelihood_history,
        params_MLE_quad,
        log_likelihood_MLE_quad
    ) = function_estimation(
        name,
        df,
        m_min=0,
        m_max=1,
        tolerance=0.01,
        Mc=Mc_dict[name],
        model_type=model_type,
        seed=seed
    )

    # binning approach for μ_MLE
    magnitude_instability_MLE = process_catalog(df, m_MLE_quad)
    magnitude_MLE = magnitude_instability_MLE["magnitude"].to_numpy()
    if "no_error" in name:
        instability_MLE = magnitude_instability_MLE["instability1"].to_numpy()
    else:
        instability_MLE = np.maximum(
            magnitude_instability_MLE["instability1"].to_numpy(),
            magnitude_instability_MLE["instability2"].to_numpy()
        )
    (
        instability_bin_min_MLE,
        instability_bin_ave_MLE,
        instability_bin_max_MLE,
        bval_bin_MLE,
        berror_bin_MLE
    ) = instability_bval_relationship(
        instability_MLE,
        magnitude_MLE,
        bin_number,
        Mc_dict[name]
    )

    ## defining MLE_i_to_b
    def MLE_i_to_b(instability):
        return theob_calc(params_MLE_quad, model_type, instability)

    fig_func_est = plot_parameter_estimation(
        model_i_to_b,
        MLE_i_to_b,
        params_MLE_quad,
        instability_bin_min_MLE,
        instability_bin_ave_MLE,
        instability_bin_max_MLE,
        bval_bin_MLE,
        berror_bin_MLE
    )

    fig_corner = plot_corner(samples_MLE_quad, params_MLE_quad)

    fig_m = plot_m_likelihood(log_likelihood_history, m_MLE_quad, m_min_quad, m_max_quad, quad_coeff)

    ## save figures
    output_dir = Path(f"../figures/result/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_bin_03.savefig(output_dir/"binning_03.svg", bbox_inches="tight", dpi=300)
    fig_bin_06.savefig(output_dir/"binning_06.svg", bbox_inches="tight", dpi=300)
    fig_bin_09.savefig(output_dir/"binning_09.svg", bbox_inches="tight", dpi=300)
    fig_func_est.savefig(output_dir/"function_estimation.svg", bbox_inches="tight", dpi=300)
    fig_corner.savefig(output_dir/"corner_plot.svg", bbox_inches="tight", dpi=300)
    fig_m.savefig(output_dir/"m_estimation.svg", bbox_inches="tight", dpi=300)

    plt.close(fig_bin_03)
    plt.close(fig_bin_06)
    plt.close(fig_bin_09)
    plt.close(fig_func_est)
    plt.close(fig_corner)
    plt.close(fig_m)

    return name, m_MLE_quad, params_MLE_quad, log_likelihood_MLE_quad


jobs = [
    (
        "no_error_exp",
        "../data/synthetic/catalog_no_error_exp.csv",
        "exp",
        100
    ),
    (
        "moderate_error_exp",
        "../data/synthetic/catalog_moderate_error_exp.csv",
        "exp",
        101
    ),
    (
        "large_error_exp",
        "../data/synthetic/catalog_large_error_exp.csv",
        "exp",
        102
    ),
    # (
    #     "dataset1_exp",
    #     "../data/california/processed/dataset1.csv",
    #     "exp",
    #     103
    # ),
    # (
    #     "dataset2_exp",
    #     "../data/california/processed/dataset2.csv",
    #     "exp",
    #     104
    # ),
    (
        "no_error_linear",
        "../data/synthetic/catalog_no_error_linear.csv",
        "linear",
        105
    ),
    (
        "moderate_error_linear",
        "../data/synthetic/catalog_moderate_error_linear.csv",
        "linear",
        106
    ),
    (
        "large_error_linear",
        "../data/synthetic/catalog_large_error_linear.csv",
        "linear",
        107
    ),
    # (
    #     "dataset1_linear",
    #     "../data/california/processed/dataset1.csv",
    #     "linear",
    #     108
    # ),
    # (
    #     "dataset2_linear",
    #     "../data/california/processed/dataset2.csv",
    #     "linear",
    #     109
    # ),
    # (
    #     "dataset1_GTF10in_exp",
    #     "../data/california/processed/dataset1_GTF10in.csv",
    #     "exp",
    #     110
    # ),
    # (
    #     "dataset1_GTF10out_exp",
    #     "../data/california/processed/dataset1_GTF10out.csv",
    #     "exp",
    #     111
    # )
]

def run_job(job):
    return run_analysis(*job)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4, max_tasks_per_child=1) as executor:
        results = list(executor.map(run_job, jobs))

    # results を DataFrame に変換
    results_df = pd.DataFrame(
        results,
        columns=["name", "m_MLE_quad", "params_MLE_quad", "log_likelihood_MLE_quad"]
    )

    # decompose params_MCMC
    results_df[["bmax", "bmin", "dI"]] = pd.DataFrame(
        results_df["params_MLE_quad"].tolist(),
        index=results_df.index
    )

    # delete "params_MCMC"
    results_df = results_df.drop(columns=["params_MLE_quad"])

    # save results as csv file
    output_path = Path("../data/analysis_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
