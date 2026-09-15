# -*- coding: utf-8 -*-
"""
Revised Figure 1
Mechanically consistent parallel two-fiber bundle, causal timing, and
protocol-dependent post-rupture load histories.

Outputs:
    Figure_1_revised.png
    Figure_1_revised.pdf
    Figure_1_revised.eps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import matplotlib.lines as mlines

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
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
})

fig = plt.figure(figsize=(8.2, 8.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.18, 0.92, 1.0],
                      hspace=0.55, wspace=0.34)

# -------------------------------------------------------------------------
# Panel a: Parallel two-fiber bundle + finite-stiffness machine
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.2)
ax.axis("off")

# Fixed wall
ax.add_patch(Rectangle((0.55, 1.0), 0.28, 3.0, facecolor="0.85",
                       edgecolor="black", linewidth=1.3))
for yy in np.linspace(1.05, 3.95, 8):
    ax.plot([0.35, 0.55], [yy-0.15, yy+0.15], color="black", lw=0.8)

# Left and right rigid bundle plates
ax.plot([2.1, 2.1], [1.35, 3.65], color="black", lw=2.0)
ax.plot([6.2, 6.2], [1.35, 3.65], color="black", lw=2.0)

# Link wall to left bundle plate
ax.plot([0.83, 2.1], [2.5, 2.5], color="black", lw=2.0)

def spring(ax, x0, x1, y, amp=0.14, turns=9, lw=1.8):
    xs = np.linspace(x0, x1, 250)
    ys = y + amp*np.sin(turns*2*np.pi*(xs-x0)/(x1-x0))
    ax.plot(xs, ys, color="black", lw=lw)

# Parallel fibers
spring(ax, 2.2, 6.1, 3.15)
spring(ax, 2.2, 6.1, 1.85)
ax.text(4.15, 3.46, r"fiber 1: $k$", ha="center")
ax.text(4.15, 1.38, r"fiber 2: $k$", ha="center")

# Separation L
ax.annotate("", xy=(3.25, 2.52), xytext=(5.05, 2.52),
            arrowprops=dict(arrowstyle="<->", linewidth=1.2))
ax.text(4.15, 2.70, r"$L$", ha="center")

# Machine spring K_m
spring(ax, 6.35, 9.1, 2.5, amp=0.18, turns=8, lw=2.0)
ax.text(7.72, 2.92, r"loading train $K_m$", ha="center")

# Actuator
ax.add_patch(Rectangle((9.15, 1.65), 0.8, 1.7, facecolor="0.90",
                       edgecolor="black", linewidth=1.3))
ax.annotate("", xy=(10.65, 2.5), xytext=(9.95, 2.5),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.8))
ax.text(10.28, 2.83, r"$D(t)$", ha="center")

ax.text(5.5, 4.72,
        "Parallel two-fiber bundle coupled to a finite-stiffness machine",
        ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(0.01, 0.99, "a", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top")

# -------------------------------------------------------------------------
# Panel b: Causal timeline
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, :])
ax.set_xlim(-0.05, 1.15)
ax.set_ylim(-0.8, 1.6)
ax.axis("off")

t1 = 0.12
tp = 0.42
tfb = 0.76
Wend = 1.06

ax.annotate("", xy=(1.10, 0.15), xytext=(0.0, 0.15),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.5))

for x, lab in [(t1, r"$t_1$"), (tp, r"$t_1+t_p$"),
               (tfb, r"$t_1+t_{\rm fb}$"), (Wend, r"$t_1+t_p+W$")]:
    ax.plot([x, x], [-0.02, 0.32], color="black", lw=1.0)
    ax.text(x, -0.18, lab, ha="center", va="top")

ax.annotate("", xy=(tp, 0.68), xytext=(t1, 0.68),
            arrowprops=dict(arrowstyle="<->", linewidth=1.4))
ax.text((t1+tp)/2, 0.86, r"material propagation $t_p=L/c_s$",
        ha="center")

ax.annotate("", xy=(tfb, 1.16), xytext=(t1, 1.16),
            arrowprops=dict(arrowstyle="<->", linewidth=1.4))
ax.text((t1+tfb)/2, 1.34, r"apparatus feedback time $t_{\rm fb}$",
        ha="center")

ax.add_patch(Rectangle((t1, 0.00), tp-t1, 0.30,
                       facecolor="0.90", edgecolor="none"))
ax.text((t1+tp)/2, 0.46, r"$P_{\rm sync}$: pre-influence",
        ha="center", fontsize=8)

ax.add_patch(Rectangle((tp, 0.00), Wend-tp, 0.30,
                       facecolor="0.82", edgecolor="none"))
ax.text((tp+Wend)/2, 0.46, r"$P_{\rm tr}$: post-arrival window $W$",
        ha="center", fontsize=8)

ax.text(0.01, 0.99, "b", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top")

# -------------------------------------------------------------------------
# Panel c: Displacement-driven load history
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[2, 0])

s = np.linspace(-0.25, 1.25, 600)
tp_rel = 0.20
D = 1.0
k = 1.0

def load_per_fiber(kappa, n):
    Km = kappa*k
    return k*Km*D/(Km+n*k)

for kappa, ls, label in [
    (1e6, "--", r"hard device, $K_m/k\to\infty$"),
    (3.0, "-", r"$K_m/k=3$"),
    (0.3, ":", r"$K_m/k=0.3$")
]:
    T2 = load_per_fiber(kappa, 2)
    T1 = load_per_fiber(kappa, 1)
    T = np.where(s < tp_rel, T2, T1)
    ax.plot(s, T/T2, linestyle=ls, label=label)

ax.axvline(0, color="0.4", lw=1.0)
ax.axvline(tp_rel, color="0.4", lw=1.0, linestyle="--")
ax.text(0, 1.78, r"$t_1$", ha="center", va="top")
ax.text(tp_rel, 1.78, r"$t_1+t_p$", ha="center", va="top")
ax.axhline(1.0, color="0.6", lw=0.8)
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(0.92, 1.82)
ax.set_xlabel(r"Time relative to first rupture")
ax.set_ylabel(r"Survivor load / pre-rupture load")
ax.set_title("c  Finite-stiffness displacement drive",
             loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper right")
ax.grid(False)

# -------------------------------------------------------------------------
# Panel d: Total-force load history with delayed unloading
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[2, 1])

s = np.linspace(-0.25, 1.6, 800)
tp_rel = 0.20
tfb_rel = 0.58
F = 1.0
beta = 0.75
tau_rec = 0.55

T = np.empty_like(s)
T[s < tp_rel] = F/2
mask1 = (s >= tp_rel) & (s < tfb_rel)
T[mask1] = F
mask2 = s >= tfb_rel
T[mask2] = F*(1-beta*np.exp(-(s[mask2]-tfb_rel)/tau_rec))

ax.plot(s, T/(F/2), linewidth=2.0)
ax.axhline(1.0, color="0.6", lw=0.8, label="pre-rupture per-fiber load")
ax.axvline(0, color="0.4", lw=1.0)
ax.axvline(tp_rel, color="0.4", lw=1.0, linestyle="--")
ax.axvline(tfb_rel, color="0.4", lw=1.0, linestyle=":")
ax.text(0, 2.10, r"$t_1$", ha="center", va="top")
ax.text(tp_rel, 2.10, r"$t_1+t_p$", ha="center", va="top")
ax.text(tfb_rel, 2.10, r"$t_1+t_{\rm fb}$", ha="center", va="top")
ax.annotate("load transfer", xy=(tp_rel+0.03, 1.92), xytext=(0.40, 1.75),
            arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=8)
ax.annotate("delayed unloading", xy=(tfb_rel+0.03, 0.55), xytext=(0.85, 0.78),
            arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=8)
ax.set_xlim(-0.25, 1.6)
ax.set_ylim(0.38, 2.12)
ax.set_xlabel(r"Time relative to first rupture")
ax.set_ylabel(r"Survivor load / $(F/2)$")
ax.set_title("d  Total-force loading with delayed unloading",
             loc="left", fontweight="bold")
ax.grid(False)

plt.savefig("Figure_1_revised.png", dpi=600)
plt.savefig("Figure_1_revised.pdf")
plt.savefig("Figure_1_revised.eps", format="eps")
plt.show()

print("Saved Figure_1_revised.png/.pdf/.eps")
