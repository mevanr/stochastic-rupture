# -*- coding: utf-8 -*-
"""
Revised Figure 2
Analytical and Monte Carlo validation of the causal amplification framework.

Panels:
(a) A_c^(D) vs machine stiffness K_m/k.
(b) A_c^(F) vs unloading fraction beta for several feedback delays.
(c) Force-control regime map with A_c=1 contour.
(d) P_sync and P_tr vs normalized feedback delay.

Outputs:
    Figure_2_revised.png
    Figure_2_revised.pdf
    Figure_2_revised.eps
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi

SEED = 20260912
rng = np.random.default_rng(SEED)

# -------------------------------------------------------------------------
# Common dimensionless parameters
# -------------------------------------------------------------------------
k = 1.0
alpha = 1.0
r0 = 0.02
D0 = 0.5
v = 0.50
tp = 0.20
W = 1.0

# MC size for figure validation
NMC = 100_000

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def cumhaz_from_prob(p):
    eps = 0.5/NMC
    p = np.clip(p, eps, 1-eps)
    return -np.log1p(-p)

def mc_A_from_H(H, H0, n=NMC):
    """Paired cumulative-hazard-threshold Monte Carlo."""
    z = rng.exponential(size=n)
    p = np.mean(z <= H)
    p0 = np.mean(z <= H0)
    return cumhaz_from_prob(p)/cumhaz_from_prob(p0)

# -------------------------------------------------------------------------
# Displacement-driven analytical model
# -------------------------------------------------------------------------
def disp_coeff(kappa):
    Km = kappa*k
    c2 = k*Km/(Km+2*k)
    c1 = k*Km/(Km+k)
    A2 = r0*np.exp(alpha*c2*D0)
    b2 = alpha*c2*v
    A1 = r0*np.exp(alpha*c1*D0)
    b1 = alpha*c1*v
    return A1, b1, A2, b2

def median_t1(kappa):
    A1, b1, A2, b2 = disp_coeff(kappa)
    return np.log1p((b2/(2*A2))*np.log(2.0))/b2

def disp_H(kappa, t1):
    A1, b1, A2, b2 = disp_coeff(kappa)
    H = (A1/b1)*np.exp(b1*(t1+tp))*np.expm1(b1*W)
    H0 = (A2/b2)*np.exp(b2*(t1+tp))*np.expm1(b2*W)
    return H, H0

# -------------------------------------------------------------------------
# Force-control analytical model
# -------------------------------------------------------------------------
def force_H(chi, beta, delta_hat, rho):
    """
    Correct Ei sign:
      int_0^U exp[-a exp(-s/tau)] ds
      = tau [Ei(-a) - Ei(-a exp(-U/tau))].
    """
    delta = delta_hat*W
    tau_rec = rho*W
    H0 = r0*np.exp(chi/2)*W

    if beta == 0 or W <= delta:
        H = r0*np.exp(chi)*W
        return H, H0

    a = chi*beta
    U = W-delta
    if a == 0:
        integral = U
    else:
        integral = tau_rec*(
            expi(-a) - expi(-a*np.exp(-U/tau_rec))
        )

    H = r0*np.exp(chi)*(delta + integral)
    return H, H0

def force_A(chi, beta, delta_hat, rho):
    H, H0 = force_H(chi, beta, delta_hat, rho)
    return H/H0

def force_Psync(chi):
    return 1 - np.exp(-r0*np.exp(chi/2)*tp)

# -------------------------------------------------------------------------
# Figure layout
# -------------------------------------------------------------------------
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
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
})

fig = plt.figure(figsize=(9.4, 7.3))
gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.42)

# -------------------------------------------------------------------------
# Panel a: displacement A_c vs K_m/k
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])

kappa_line = np.logspace(-1.2, 3.0, 300)
A_line = []
for kap in kappa_line:
    t1 = median_t1(kap)
    H, H0 = disp_H(kap, t1)
    A_line.append(H/H0)
A_line = np.asarray(A_line)

kappa_mc = np.array([0.1, 0.3, 1, 3, 10, 30, 100])
A_mc = []
for kap in kappa_mc:
    t1 = median_t1(kap)
    H, H0 = disp_H(kap, t1)
    A_mc.append(mc_A_from_H(H, H0))
A_mc = np.asarray(A_mc)

ax.plot(kappa_line, A_line, label="Analytical")
ax.plot(kappa_mc, A_mc, "o", label="Monte Carlo")
ax.axhline(1, color="0.45", lw=1.0, linestyle="--")
ax.set_xscale("log")
ax.set_xlabel(r"Machine stiffness ratio $\kappa=K_m/k$")
ax.set_ylabel(r"$\mathcal{A}_c^{(D)}$")
ax.set_title("a  Displacement drive: stiffness crossover",
             loc="left", fontweight="bold", fontsize=9.5)
ax.legend(frameon=False)
ax.grid(False)

# -------------------------------------------------------------------------
# Panel b: force A_c vs beta
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])

chi = 2.0
rho = 1.0
beta_line = np.linspace(0, 1, 300)
delta_curves = [0.0, 0.10, 0.25, 0.50, 1.00]

for dh in delta_curves:
    vals = [force_A(chi, b, dh, rho) for b in beta_line]
    ax.plot(beta_line, vals, label=rf"$\widehat{{\delta}}={dh:g}$")

# MC points for two representative delays
for dh, marker in [(0.0, "o"), (0.50, "s")]:
    beta_pts = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    ypts = []
    for b in beta_pts:
        H, H0 = force_H(chi, b, dh, rho)
        ypts.append(mc_A_from_H(H, H0))
    ax.plot(beta_pts, ypts, marker, linestyle="none",
            markersize=4.5)

ax.axhline(1, color="0.45", lw=1.0, linestyle="--")
ax.set_xlabel(r"Transient unloading fraction $\beta$")
ax.set_ylabel(r"$\mathcal{A}_c^{(F)}$")
ax.set_title("b  Total-force loading: delayed unloading",
             loc="left", fontweight="bold", fontsize=9.5)
ax.legend(frameon=False, ncol=2)
ax.grid(False)

# -------------------------------------------------------------------------
# Panel c: regime map
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])

beta_grid = np.linspace(0, 1, 181)
delta_grid = np.linspace(0, 1.20, 181)
Z = np.empty((len(delta_grid), len(beta_grid)))

for i, dh in enumerate(delta_grid):
    for j, b in enumerate(beta_grid):
        Z[i, j] = np.log10(force_A(chi, b, dh, rho))

mesh = ax.pcolormesh(beta_grid, delta_grid, Z, shading="auto")
cb = fig.colorbar(mesh, ax=ax, pad=0.02)
cb.set_label(r"$\log_{10}\mathcal{A}_c^{(F)}$")
cs = ax.contour(beta_grid, delta_grid, Z, levels=[0.0],
                linewidths=1.5)
ax.clabel(cs, fmt={0.0: r"$\mathcal{A}_c=1$"}, fontsize=8)
ax.set_xlabel(r"Transient unloading fraction $\beta$")
ax.set_ylabel(r"Normalized feedback delay $\widehat{\delta}$")
ax.set_title("c  Enhancement--suppression boundary",
             loc="left", fontweight="bold", fontsize=9.5)
ax.grid(False)

# -------------------------------------------------------------------------
# Panel d: P_sync vs P_tr
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])

delta_vals = np.linspace(0, 1.20, 250)
beta_demo = 1.0
Psync = np.full_like(delta_vals, force_Psync(chi))
Ptr = np.empty_like(delta_vals)
Ptr0 = np.empty_like(delta_vals)

for i, dh in enumerate(delta_vals):
    H, H0 = force_H(chi, beta_demo, dh, rho)
    Ptr[i] = 1 - np.exp(-H)
    Ptr0[i] = 1 - np.exp(-H0)

ax.plot(delta_vals, Psync, label=r"$P_{\rm sync}$")
ax.plot(delta_vals, Ptr, label=r"$P_{\rm tr}$")
ax.plot(delta_vals, Ptr0, linestyle="--",
        label=r"$P_{\rm tr}^{(0)}$ (no-transfer baseline)")
ax.set_xlabel(r"Normalized feedback delay $\widehat{\delta}$")
ax.set_ylabel("Probability")
ax.set_title("d  Synchrony vs post-arrival triggering",
             loc="left", fontweight="bold", fontsize=9.5)
ax.legend(frameon=False)
ax.grid(False)

plt.savefig("Figure_2_revised.png", dpi=600)
plt.savefig("Figure_2_revised.pdf")
plt.savefig("Figure_2_revised.eps", format="eps")
plt.show()

print("Saved Figure_2_revised.png/.pdf/.eps")
