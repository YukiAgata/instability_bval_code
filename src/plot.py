import cartopy.crs as ccrs
import cartopy.feature as cfeature
import corner
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_GoFcurve(Mc_idx, Mc_candidate, GoF, GoF_threshold):
    LABEL_SIZE = 18
    TICK_SIZE = 14
    LEGEND_SIZE = 14

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(Mc_candidate, GoF, marker=".")
    ax.axhline(GoF_threshold, color="orange")
    ax.axvline(Mc_candidate[Mc_idx], label=f"Mc={Mc_candidate[Mc_idx]:.1f}", c="r")
    ax.set_xlabel("Mc candidates", fontsize=LABEL_SIZE)
    ax.set_ylabel("Goodness of Fit [%]", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    return fig, ax, Mc_candidate[Mc_idx]

def plot_Mc_confirmation(Mc_idx, CFMD_obs, CFMD_pred, b, Mc, magnitude_centers):
    LABEL_SIZE = 18
    TICK_SIZE = 14
    LEGEND_SIZE = 14

    fig, ax = plt.subplots(constrained_layout=True)
    mask = magnitude_centers >= 1.9

    ax.plot(magnitude_centers[mask], CFMD_obs[Mc_idx][mask], marker=".", label="B (observed)")
    ax.plot(magnitude_centers[mask], CFMD_pred[Mc_idx][mask], marker=".", label=f"S (predicted); b={b[Mc_idx]:.2f}")
    ax.axvline(Mc, label=f"Mc={Mc:.1f}", c="r")
    ax.set_xlabel("Magnitude", fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequency (upper cumulative)", fontsize=LABEL_SIZE)
    ax.set_yscale('log')
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    return fig, ax

def plot_spatial_distribution(latitude, longitude, color=None):
    LABEL_SIZE = 18

    fig, ax = plt.subplots(constrained_layout=True, subplot_kw={"projection": ccrs.PlateCarree()})
    # mapping only near California
    ax.set_extent([-126, -114, 31.7, 42.3], crs=ccrs.PlateCarree())
    # background map
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    # latitude and longitude grids
    # 緯度経度グリッド
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(2)
    gl.ylocator = mticker.MultipleLocator(2)
    gl.xlabel_style = {"size": LABEL_SIZE}
    gl.ylabel_style = {"size": LABEL_SIZE}

    # spatial distribution
    sc = ax.scatter(
        longitude, latitude,
        color=color,
        s=1,
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=2
    )
    return fig, ax

## (following two functions) mapping in case of GTF
def add_label_with_line(ax, lon, lat, text, dx, dy):
    # 線の終点
    label_lon = lon + dx
    label_lat = lat + dy

    # 点からラベル位置まで線を引く
    ax.plot(
        [lon, label_lon],
        [lat, label_lat],
        color="black",
        linewidth=1.0,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    # ラベル
    ax.text(
        label_lon,
        label_lat,
        text,
        fontsize=14,
        ha="left" if dx > 0 else "right",
        va="center",
        transform=ccrs.PlateCarree(),
        zorder=4,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.7,
            pad=1.5,
        ),
    )

def plot_spatial_distribution_GTF(latitude, longitude, color=None):
    LABEL_SIZE = 18

    fig, ax = plt.subplots(constrained_layout=True, subplot_kw={"projection": ccrs.PlateCarree()})
    # mapping only near California
    ax.set_extent([-126, -114, 31.7, 42.3], crs=ccrs.PlateCarree())
    # background map
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    # latitude and longitude grids
    # 緯度経度グリッド
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(2)
    gl.ylocator = mticker.MultipleLocator(2)
    gl.xlabel_style = {"size": LABEL_SIZE}
    gl.ylabel_style = {"size": LABEL_SIZE}

    # spatial distribution
    sc = ax.scatter(
        longitude, latitude,
        color=color,
        s=1,
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=2
    )

    # Labels with leader lines
    add_label_with_line(ax, -122.8, 38.8, "The Geysers", 0.6, 0.3)
    add_label_with_line(ax, -117.79, 36.02, "Coso", 0.6, 0.25)
    add_label_with_line(ax, -116.6, 34.0, "Desert Hotsprings", 0.6, 0.25)
    add_label_with_line(ax, -115.6, 33.16, "Salton Sea", 0.6, 0.25)
    add_label_with_line(ax, -115.52, 33.02, "North Brawley", 0.6, -0.25)
    add_label_with_line(ax, -115.54, 32.71, "Haber", 0.6, -0.5)
    return fig, ax

def plot_CMFD_dataset1GTF(magnitude_in, magnitude_out, bval_in, bval_out, Mc):
    FONT_SIZE = 18
    TICK_SIZE = 14
    LEGEND_SIZE = 14

    bins = np.arange(Mc, 7.5, 0.1)

    fig, ax = plt.subplots(constrained_layout=True)

    ## geothermal group
    N = len(magnitude_in)

    cumulative_N = np.array([np.sum(magnitude_in >= m) for m in bins])
    mask = cumulative_N > 0
    ax.plot(bins[mask], cumulative_N[mask], marker="o", color="C0", label=rf"geothermal: $b={bval_in:.2f}\pm{bval_in/np.sqrt(N):.2f}$")

    cumulative_N_theo = N * 10 ** (- bval_in * (bins - Mc))
    ax.plot(bins[mask], cumulative_N_theo[mask], "--", color="C0", alpha=0.5)

    ##  non-geothermal group
    N = len(magnitude_out)

    cumulative_N = np.array([np.sum(magnitude_out >= m) for m in bins])
    mask = cumulative_N > 0
    ax.plot(bins[mask], cumulative_N[mask], marker="o", color="C1", label=rf"non-geothermal: $b={bval_out:.3f}\pm{bval_out/np.sqrt(N):.3f}$")

    cumulative_N_theo = N * 10 ** (- bval_out * (bins - Mc))
    ax.plot(bins[mask], cumulative_N_theo[mask], "--", color="C1", alpha=0.5)

    ax.set_xlabel("Magnitude", fontsize=FONT_SIZE)
    ax.set_ylabel("Frequency (upper cumulative)", fontsize=FONT_SIZE)
    ax.set_ylim(0.55, 350000)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    return fig, ax

def plot_binning(
        instability_bin_min,
        instability_bin_ave,
        instability_bin_max,
        bval_bin,
        berror_bin,
        model_i_to_b
):
    # Font size settings
    LABEL_SIZE = 18
    TICK_SIZE = 14
    LEGEND_SIZE = 14

    fig, ax = plt.subplots(constrained_layout=True)
    i_vals = np.arange(-1, 1, 0.01)

    ax.errorbar(instability_bin_ave, bval_bin, berror_bin,
                xerr=[instability_bin_ave-instability_bin_min,
                      instability_bin_max-instability_bin_ave],
                color="#0072B2",
                fmt="none",
                label="binning"
                )
    if model_i_to_b is not None:
        ax.plot(i_vals,
                model_i_to_b(i_vals),
                color="#E69F00",
                linewidth=2,
                label="model")
    ax.set_xlabel("instability", fontsize=LABEL_SIZE)
    ax.set_ylabel("b-value", fontsize=LABEL_SIZE)
    ax.set_ylim(0.6, 1.3)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    if model_i_to_b is not None:
        ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, linestyle=":", alpha=0.6)
    return fig

def plot_parameter_estimation(
        model_i_to_b,
        MLE_i_to_b,
        params_MLE,
        instability_bin_min,
        instability_bin_ave,
        instability_bin_max,
        bval_bin,
        berror_bin
):
    LABEL_SIZE = 18
    TICK_SIZE = 14
    LEGEND_SIZE = 14

    fig, ax = plt.subplots(constrained_layout=True)
    i_vals = np.arange(-1, 1, 0.01)

    ax.errorbar(instability_bin_ave, bval_bin, berror_bin,
                xerr=[instability_bin_ave-instability_bin_min,
                      instability_bin_max-instability_bin_ave],
                color="#0072B2",
                fmt="none",
                label="binning"
                )
    ax.plot(i_vals,
            MLE_i_to_b(i_vals),
            color="#CC79A7",
            linewidth=2,
            label=(
                f"MLE:"
                f"$b_{{\\max}}={params_MLE[0]:.2f}, "
                f"b_{{\\min}}={params_MLE[1]:.2f}, "
                f"\\Delta I={params_MLE[2]:.2f}$"
            ))
    if model_i_to_b is not None:
        ax.plot(i_vals,
                model_i_to_b(i_vals),
                color="#E69F00",
                linewidth=2,
                label="model")
    ax.set_xlabel("instability", fontsize=LABEL_SIZE)
    ax.set_ylabel("b-value", fontsize=LABEL_SIZE)
    ax.set_ylim(0.6, 1.3)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, linestyle=":", alpha=0.6)
    return fig

def plot_corner(samples_MCMC, params_MCMC):
    LABEL_SIZE = 18
    TICK_SIZE = 14
    TITLE_SIZE = 16

    fig = corner.corner(
        samples_MCMC,
        labels=[r"$b_{\rm max}$", r"$b_{\rm min}$", r"$\Delta I$"],
        show_titles=True,
        title_fmt=".3f",
        title_quantiles=[0.025, 0.5, 0.975],
        label_kwargs={"fontsize": LABEL_SIZE},
        title_kwargs={"fontsize": TITLE_SIZE},
    )
    ndim = len(params_MCMC)
    # show each param_MCMC value in the figures
    for i in range(ndim):
        for j in range(ndim):
            ax = fig.axes[i * ndim + j]
            # diagonal：1D-histogram
            if i == j:
                ax.axvline(params_MCMC[i], color="red", linestyle="--", linewidth=1.5)
            # non-diagonal
            elif i > j:
                # x-axis：j-th parameter
                ax.axvline(params_MCMC[j], color="red", linestyle="--", linewidth=1.2)
                # y-axis：i-th parameter
                ax.axhline(params_MCMC[i], color="red", linestyle="--", linewidth=1.2)
    # tick labels
    for ax in fig.get_axes():
        ax.tick_params(labelsize=TICK_SIZE)
    fig.tight_layout()
    return fig


def plot_unitmohr_with_coulomb(normal_stress, shear_stress, friction_coefficient):
    LABEL_SIZE = 18
    TICK_SIZE = 14

    fig, ax = plt.subplots(constrained_layout=True)
    theta = np.linspace(0, 2*np.pi, 400)

    R = 1
    x_circle = R * np.cos(theta)
    y_circle = R * np.sin(theta)
    ax.plot(x_circle, y_circle, color="black", linewidth=1)

    x_line = np.linspace(-1.5, 1.5, 500)
    y_line = friction_coefficient * x_line + np.sqrt(1 + friction_coefficient**2)
    ax.plot(
        x_line, y_line,
        color="red",
        linestyle="--",
        linewidth=1,
    )

    sc = ax.scatter(
        normal_stress, shear_stress,
        s=0.5,
        color="#0072B2",
        edgecolors="none",
    )

    ax.set_xlabel(r"$\sigma_n$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\tau_e$", fontsize=LABEL_SIZE)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, linestyle=":", alpha=0.6)
    return fig, ax


def plot_instability_PDF(ax, center, PDF, label=None, color=None):
    LABEL_SIZE = 18
    TICK_SIZE = 14

    ax.plot(center, PDF, label=label, color=color)
    ax.set_xlabel("instability", fontsize=LABEL_SIZE)
    ax.set_ylabel("probability density", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


# supplementary figures
def plot_m_likelihood(log_likelihood_history, m_MLE_quad, m_min_quad, m_max_quad, quad_coeff):
    fig, ax = plt.subplots(constrained_layout=True)

    m = log_likelihood_history[:,0]
    log_likelihood = log_likelihood_history[:,1]

    # sort m
    sort_idx = np.argsort(m)
    m_sorted = m[sort_idx]
    log_likelihood_sorted = log_likelihood[sort_idx]

    ax.plot(
        m_sorted,
        log_likelihood_sorted,
        marker=".",
        label="profile lnL"
        )
    ax.axvline(
        m_MLE_quad,
        c="r",
        label=f"$\\mu_{{MLE}}={m_MLE_quad:.2f}$"
    )
    ax.axvspan(
        m_min_quad,
        m_max_quad,
        color="red",
        alpha=0.3,
        label=f"95% confidence interval:{m_min_quad:.2f}-{m_max_quad:.2f}"
        )

    def quad_approx(m, quad_coeff):
        return quad_coeff[0]*m**2 + quad_coeff[1]*m + quad_coeff[2]

    ax.plot(
        np.arange(0, 1.01, 0.01),
        quad_approx(np.arange(0, 1.01, 0.01), quad_coeff),
        color="black",
        linestyle=':',
        label="quadratic approximation"
        )

    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("lnL")
    ax.legend()
    return fig