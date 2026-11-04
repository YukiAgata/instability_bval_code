# Code for "Faulting Instability Controlling b-value and Earthquake Growth"

This repository contains the code for the analysis presented in "Faulting Instability Controlling b-value and Earthquake Growth."

## Repository structure

- `analysis/`: Jupyter notebooks and analysis scripts used in the study.
- `src/`: Python source code.
- `data/`: Data used in the analysis.
- `figures/`: Figures generated during the analysis(created automatically).

## Analysis workflow

### Analysis of the Synthetic Catalogs
1. `analysis/01_synthetic_catalog_creation.ipynb`
    - Creates the synthetic earthquake and stress catalog.
    - Set `stressinverse_dir` to the directory containing "Stressinverse_1.1.3" (Vavryčuk, 2014). Update this path according to your local environment.

2. `analysis/02_catalog_analysis.py`
    - Analyzes the catalogs. Run:
    ```bash
    cd analysis
    python 02_catalog_analysis.py
    ``` 
    - By default, real datasets are also analyzed simultaneously.
    - Conducts maximum likelihood estimation of parameters ($\mu, b_{max}, b_{min}, \Delta I$).
    - The parameter estimation results are saved in `data/analysis_results.csv`.
        - Output figures are saved in the corresponding subfolders in `figures/result/`. These outputs include Figs. 5 and S5.
        - Tables 1 and S1 are generated using these output files.

### Analysis of the Real Catalogs
1. `data/README_data.md`
    - Download the required data following the instructions in `data/README_data.md`.

2. `analysis/00_california_data_preparation.ipynb`
    - Prepares DataFrames from the downloaded data. The output files are:
        - 4 CSV files in `data/processed/`
        - 13 figures in `figures/dataset/`(including Figs. 6, 9a, 9b, and S2)
    - Estimates the magnitude of completeness ($M_c$) as well.
    - Creates datasets accounting for geothermal fields (GTF). The radius can be modified by adjusting the `r_km` variable.

3. `analysis/02_catalog_analysis.py`
    - Run
    ```bash
    cd analysis
    python 02_catalog_analysis.py
    ```
    - Analyzes the catalogs. By default, synthetic catalogs are also analyzed simultaneously.
    - Maximum likelihood estimation of parameters ($\mu, b_{max}, b_{min}, \Delta I$) is conducted simultaneously.
    - The parameter estimation results are saved in `data/analysis_results.csv`.
    - Output figures are saved in `figures/result/`. These outputs include Figs. 7, 8, 9c, 9d, S1, and S3.
    - Tables 2, 3, and 4 are generated using these output files.

### Frequency Distribution by Instability
1. `analysis/03_instability_frequency.py`
    - Computes the frequency distribution of earthquakes by instability and the distribution of stress states on the Mohr circle.
    - Output files are saved in `figures/instability_frequency/`. These correspond to Figs. 10 and S6.

## Requirements

The analysis was performed using Python 3.11.15.

Required packages include:

- cartopy
- corner
- emcee
- matplotlib
- numba
- numpy
- obspy
- pandas
- scipy
- tqdm
