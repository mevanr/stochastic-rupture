# -*- coding: utf-8 -*-
"""
FIGURE 3
Directly generates a 2x2 multipanel Figure 3 from N_fiber_sensitivity_map.csv.

Panels
------
(a) Heat map of mean causal avalanche size <S>
(b) Mean avalanche size vs normalized machine stiffness kappa_B
(c) System-spanning probability vs kappa_B
(d) Mean avalanche size vs one-step hazard amplification A_seed

Input
-----
N_fiber_sensitivity_map.csv

Output
------
Figure_3_combined_direct.png
Figure_3_combined_direct.pdf
Figure_3_combined_direct.eps
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
CSV_FILE = HERE / "N_fiber_sensitivity_map.csv"

OUT_PNG = HERE / "Figure_3_combined_direct.png"
OUT_PDF = HERE / "Figure_3_combined_direct.pdf"
OUT_EPS = HERE / "Figure_3_combined_direct.eps"


# =============================================================================
# USER SETTINGS
# =============================================================================

ETA_LINES_MEAN = [5, 10, 20, 25, 30]
ETA_LINES_SPAN = [20, 25, 30]

SAVE_DPI = 600


# =============================================================================
# PUBLICATION STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


# =============================================================================
# LOAD DATA
# =============================================================================

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{CSV_FILE}\n\n"
        "Place N_fiber_sensitivity_map.csv in the same folder as this script."
    )

df = pd.read_csv(CSV_FILE)

required = {
    "kappa_bundle_Km_over_Nk",
    "eta_alphaT0",
    "mean_size",
}

missing = required.difference(df.columns)

if missing:
    raise ValueError(
        "Missing required CSV columns:\n"
        + "\n".join(sorted(missing))
    )

numeric_cols = [
    "kappa_bundle_Km_over_Nk",
    "eta_alphaT0",
    "mean_size",
    "P_spanning",
    "n_avalanches",
    "N",
    "q0_Nh0W",
    "A_seed_one_step",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

df = df.sort_values(
    ["eta_alphaT0", "kappa_bundle_Km_over_Nk"]
).reset_index(drop=True)


# =============================================================================
# HELPERS
# =============================================================================

def nearest_eta(dataframe, target_eta):
    values = np.sort(dataframe["eta_alphaT0"].unique())
    return values[np.argmin(np.abs(values - target_eta))]


def binomial_se(p, n):
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    return np.sqrt(np.clip(p * (1.0 - p) / n, 0, None))


def panel_label(ax, label):
    ax.text(
        -0.14,
        1.06,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


# =============================================================================
# PREPARE HEAT MAP DATA
# =============================================================================

pivot = df.pivot(
    index="eta_alphaT0",
    columns="kappa_bundle_Km_over_Nk",
    values="mean_size",
)

eta_values = pivot.index.to_numpy(dtype=float)
kappa_values = pivot.columns.to_numpy(dtype=float)
mean_size_grid = pivot.to_numpy(dtype=float)

log_kappa = np.log10(kappa_values)


# =============================================================================
# BUILD 2 x 2 FIGURE
# =============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9.2, 7.5),
)

ax_a, ax_b, ax_c, ax_d = axes.ravel()


# =============================================================================
# PANEL (a): HEAT MAP
# =============================================================================

mesh = ax_a.pcolormesh(
    log_kappa,
    eta_values,
    mean_size_grid,
    shading="auto",
)

cbar = fig.colorbar(
    mesh,
    ax=ax_a,
    pad=0.02,
    fraction=0.050,
)

cbar.set_label(
    r"Mean causal avalanche size $\langle S\rangle$"
)

zmin = float(np.nanmin(mean_size_grid))
zmax = float(np.nanmax(mean_size_grid))

candidate_levels = [2, 5, 10, 15]
levels = [
    level for level in candidate_levels
    if zmin < level < zmax
]

if levels:
    contours = ax_a.contour(
        log_kappa,
        eta_values,
        mean_size_grid,
        levels=levels,
        linewidths=0.7,
    )

    ax_a.clabel(
        contours,
        fontsize=6.5,
        inline=True,
        fmt=lambda x: f"{x:g}",
    )

decades = np.arange(
    np.floor(log_kappa.min()),
    np.ceil(log_kappa.max()) + 1,
)

ax_a.set_xticks(decades)
ax_a.set_xticklabels(
    [rf"$10^{{{int(d)}}}$" for d in decades]
)

ax_a.set_xlabel(
    r"Normalized machine stiffness $\kappa_B=K_m/(Nk)$"
)

ax_a.set_ylabel(
    r"Load sensitivity $\eta=\alpha T_0$"
)

ax_a.set_title(
    "Mean avalanche-size landscape"
)

panel_label(ax_a, "a")


# =============================================================================
# PANEL (b): MEAN SIZE VS STIFFNESS
# =============================================================================

used_eta = []

for eta_target in ETA_LINES_MEAN:

    eta_use = nearest_eta(
        df,
        eta_target
    )

    if eta_use in used_eta:
        continue

    used_eta.append(
        eta_use
    )

    sub = df[
        df["eta_alphaT0"] == eta_use
    ].sort_values(
        "kappa_bundle_Km_over_Nk"
    )

    x = sub[
        "kappa_bundle_Km_over_Nk"
    ].to_numpy()

    y = sub[
        "mean_size"
    ].to_numpy()

    ax_b.plot(
        x,
        y,
        marker="o",
        markersize=3.5,
        label=rf"$\eta={eta_use:g}$",
    )

ax_b.set_xscale(
    "log"
)

ax_b.set_xlabel(
    r"Normalized machine stiffness $\kappa_B$"
)

ax_b.set_ylabel(
    r"Mean avalanche size $\langle S\rangle$"
)

ax_b.set_title(
    "Machine-stiffness dependence"
)

ax_b.legend(
    frameon=False,
)

panel_label(ax_b, "b")


# =============================================================================
# PANEL (c): SYSTEM-SPANNING PROBABILITY
# =============================================================================

if "P_spanning" in df.columns:

    used_eta = []

    for eta_target in ETA_LINES_SPAN:

        eta_use = nearest_eta(
            df,
            eta_target
        )

        if eta_use in used_eta:
            continue

        used_eta.append(
            eta_use
        )

        sub = df[
            df["eta_alphaT0"] == eta_use
        ].sort_values(
            "kappa_bundle_Km_over_Nk"
        )

        x = sub[
            "kappa_bundle_Km_over_Nk"
        ].to_numpy()

        y = sub[
            "P_spanning"
        ].to_numpy()

        if "n_avalanches" in sub.columns:
            n = sub[
                "n_avalanches"
            ].to_numpy()

            yerr = 1.96 * binomial_se(
                y,
                n,
            )

            ax_c.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                markersize=3.5,
                capsize=2,
                label=rf"$\eta={eta_use:g}$",
            )

        else:
            ax_c.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                label=rf"$\eta={eta_use:g}$",
            )

    ax_c.set_xscale(
        "log"
    )

    ax_c.set_xlabel(
        r"Normalized machine stiffness $\kappa_B$"
    )

    ax_c.set_ylabel(
        r"System-spanning probability $P_{\rm span}$"
    )

    ax_c.set_ylim(
        bottom=0
    )

    ax_c.set_title(
        "Probability of large collective failure"
    )

    ax_c.legend(
        frameon=False,
    )

else:

    ax_c.text(
        0.5,
        0.5,
        "P_spanning column\nnot found in CSV",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )

    ax_c.set_axis_off()

panel_label(ax_c, "c")


# =============================================================================
# PANEL (d): LOCAL HAZARD AMPLIFICATION VS COLLECTIVE SIZE
# =============================================================================

if "A_seed_one_step" in df.columns:

    used_eta = []

    for eta_target in ETA_LINES_MEAN:

        eta_use = nearest_eta(
            df,
            eta_target
        )

        if eta_use in used_eta:
            continue

        used_eta.append(
            eta_use
        )

        sub = df[
            df["eta_alphaT0"] == eta_use
        ].sort_values(
            "A_seed_one_step"
        )

        x = sub[
            "A_seed_one_step"
        ].to_numpy()

        y = sub[
            "mean_size"
        ].to_numpy()

        ax_d.plot(
            x,
            y,
            marker="o",
            markersize=3.5,
            label=rf"$\eta={eta_use:g}$",
        )

    ax_d.set_xlabel(
        r"One-step hazard amplification $\mathcal{A}_{\rm seed}$"
    )

    ax_d.set_ylabel(
        r"Mean avalanche size $\langle S\rangle$"
    )

    ax_d.set_title(
        "Local amplification and collective response"
    )

    ax_d.legend(
        frameon=False,
    )

else:

    ax_d.text(
        0.5,
        0.5,
        "A_seed_one_step column\nnot found in CSV",
        transform=ax_d.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )

    ax_d.set_axis_off()

panel_label(ax_d, "d")


# =============================================================================
# COMMON FORMATTING
# =============================================================================

for ax in [ax_a, ax_b, ax_c, ax_d]:
    if ax.axison:
        ax.grid(False)


# Optional parameter annotation on panel (a)
info = []

if "N" in df.columns and df["N"].nunique() == 1:
    info.append(
        rf"$N={int(df['N'].iloc[0])}$"
    )

if (
    "q0_Nh0W" in df.columns
    and df["q0_Nh0W"].nunique() == 1
):
    info.append(
        rf"$q_0={df['q0_Nh0W'].iloc[0]:g}$"
    )

if (
    "n_avalanches" in df.columns
    and df["n_avalanches"].nunique() == 1
):
    info.append(
        rf"$n_{{\rm sim}}={int(df['n_avalanches'].iloc[0])}$ per cell"
    )

if info:
    ax_a.text(
        0.02,
        0.98,
        ",  ".join(info),
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="0.7",
            alpha=0.85,
        ),
    )


# =============================================================================
# LAYOUT AND SAVE
# =============================================================================

fig.tight_layout(
    pad=1.2,
    w_pad=1.4,
    h_pad=1.6,
)

fig.savefig(
    OUT_PNG,
    dpi=SAVE_DPI,
)

fig.savefig(
    OUT_PDF,
)

fig.savefig(
    OUT_EPS,
    format="eps",
)

plt.show()

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_EPS)
