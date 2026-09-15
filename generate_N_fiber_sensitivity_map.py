# -*- coding: utf-8 -*-
"""
Generate N_fiber_sensitivity_map.csv
====================================

Seeded N-fiber global-load-sharing simulation for the finite-stiffness
parallel-bundle rupture model used in Figure 3.

Model
-----
For n intact fibers at fixed actuator displacement D,

    T_n(D) = k K_m D / (K_m + n k)

with

    kappa_B = K_m / (N k)
    eta     = alpha T0
    q0      = N h(T0) W

The pre-seed displacement is chosen so that T_N = T0.  After an imposed
seed rupture, D is held fixed.  When n fibers remain,

    lambda_n = n h(T_n)

and the waiting time to the next rupture is exponentially distributed.
The avalanche continues if Delta t_n <= W.

The CSV is written in the format expected by Figure_3_four_figures.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

N = 100
Q0 = 0.5
W = 1.0
N_AVALANCHES = 2000

# Fixed seed for complete reproducibility.
RNG_SEED = 20260912

# Smooth grid for the Figure 3 heat map and line cuts.
# Includes kappa_B = 0.01 and 10 exactly.
KAPPA_VALUES = np.logspace(-2, 1, 25)

# Includes eta = 5, 10, 20, 25, 30 exactly.
ETA_VALUES = np.arange(5.0, 31.0, 1.0)

SPANNING_FRACTION = 0.80


# =============================================================================
# OUTPUT PATH
# =============================================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_CSV = HERE / "N_fiber_sensitivity_map.csv"


# =============================================================================
# MODEL HELPERS
# =============================================================================

def tension_ratio(n, N, kappa_B):
    """
    Return T_n / T0 after the seed, under fixed actuator displacement.

    From:
        K_m = kappa_B * N * k
        D chosen so T_N(D) = T0

    one obtains:
        T_n/T0 = (kappa_B + 1) / (kappa_B + n/N)
    """
    return (kappa_B + 1.0) / (kappa_B + n / N)


def log_lambda_times_W(n, N, q0, eta, kappa_B):
    """
    Compute log(lambda_n * W) without overflow.

    Because
        q0 = N h(T0) W
    and
        h(T_n)/h(T0) = exp[eta * (T_n/T0 - 1)],

    we have
        lambda_n W
        = (n/N) q0 exp[eta * (T_n/T0 - 1)].
    """
    ratio = tension_ratio(n, N, kappa_B)

    return (
        np.log(q0)
        + np.log(n / N)
        + eta * (ratio - 1.0)
    )


def one_step_seed_amplification(N, eta, kappa_B):
    """
    Per-fiber hazard amplification immediately after the imposed seed:
        A_seed = h(T_{N-1}) / h(T0).
    """
    ratio = tension_ratio(N - 1, N, kappa_B)
    log_A = eta * (ratio - 1.0)

    # This is modest over the present parameter grid, but clipping
    # keeps the function numerically safe if the grid is later expanded.
    return float(np.exp(np.clip(log_A, -700.0, 700.0)))


def simulate_seeded_avalanches(
    rng,
    N,
    q0,
    eta,
    kappa_B,
    n_avalanches,
):
    """
    Simulate independent seeded avalanches.

    S starts at 1 because the initial seed rupture is imposed.

    If E ~ Exp(1), then Delta t_n = E/lambda_n.
    The continuation criterion Delta t_n <= W is equivalent to

        log(E) <= log(lambda_n W),

    which avoids numerical overflow at large hazard.
    """
    sizes = np.ones(n_avalanches, dtype=np.int16)
    active = np.ones(n_avalanches, dtype=bool)

    # Immediately after the imposed seed there are N-1 intact fibers.
    for n in range(N - 1, 0, -1):

        if not np.any(active):
            break

        log_lamW = log_lambda_times_W(
            n=n,
            N=N,
            q0=q0,
            eta=eta,
            kappa_B=kappa_B,
        )

        active_idx = np.flatnonzero(active)

        # Exponential cumulative-hazard thresholds.
        E = rng.exponential(scale=1.0, size=active_idx.size)

        continue_event = np.log(E) <= log_lamW

        accepted = active_idx[continue_event]
        stopped = active_idx[~continue_event]

        # An accepted next event adds one failed fiber to the avalanche.
        sizes[accepted] += 1

        # Avalanches with Delta t_n > W terminate.
        active[stopped] = False

    return sizes


def wilson_interval(successes, trials, z=1.959963984540054):
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return np.nan, np.nan

    p = successes / trials
    z2 = z * z

    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    half = (
        z
        * np.sqrt(
            p * (1.0 - p) / trials
            + z2 / (4.0 * trials * trials)
        )
        / denom
    )

    return max(0.0, center - half), min(1.0, center + half)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():

    rng = np.random.default_rng(RNG_SEED)

    spanning_threshold = int(np.ceil(SPANNING_FRACTION * N))

    rows = []

    total_cells = len(ETA_VALUES) * len(KAPPA_VALUES)
    cell = 0

    print("=" * 78)
    print("N-FIBER FINITE-STIFFNESS SENSITIVITY MAP")
    print("=" * 78)
    print(f"N                   = {N}")
    print(f"q0 = N h(T0) W      = {Q0}")
    print(f"W                   = {W}")
    print(f"avalanches / cell   = {N_AVALANCHES}")
    print(f"RNG seed            = {RNG_SEED}")
    print(f"spanning threshold  = S >= {spanning_threshold}")
    print(f"parameter cells     = {total_cells}")
    print("=" * 78)

    for eta in ETA_VALUES:

        for kappa_B in KAPPA_VALUES:

            cell += 1

            sizes = simulate_seeded_avalanches(
                rng=rng,
                N=N,
                q0=Q0,
                eta=float(eta),
                kappa_B=float(kappa_B),
                n_avalanches=N_AVALANCHES,
            )

            spanning = sizes >= spanning_threshold
            n_spanning = int(np.sum(spanning))
            p_spanning = float(np.mean(spanning))

            # Normal-approximation SE retained because the existing Figure 3
            # plotting script uses this quantity conceptually.
            span_se = float(
                np.sqrt(
                    p_spanning * (1.0 - p_spanning)
                    / N_AVALANCHES
                )
            )

            wilson_low, wilson_high = wilson_interval(
                n_spanning,
                N_AVALANCHES,
            )

            rows.append({
                "kappa_bundle_Km_over_Nk": float(kappa_B),
                "eta_alphaT0": float(eta),
                "mean_size": float(np.mean(sizes)),
                "median_size": float(np.median(sizes)),
                "sd_size": float(np.std(sizes, ddof=1)),
                "p90_size": float(np.quantile(sizes, 0.90)),
                "p99_size": float(np.quantile(sizes, 0.99)),
                "max_size": int(np.max(sizes)),
                "P_spanning": p_spanning,
                "n_spanning": n_spanning,
                "span_SE": span_se,
                "span_Wilson95_low": float(wilson_low),
                "span_Wilson95_high": float(wilson_high),
                "A_seed_one_step": one_step_seed_amplification(
                    N=N,
                    eta=float(eta),
                    kappa_B=float(kappa_B),
                ),
                "n_avalanches": N_AVALANCHES,
                "N": N,
                "q0_Nh0W": Q0,
                "W": W,
                "spanning_fraction": SPANNING_FRACTION,
                "rng_seed": RNG_SEED,
            })

            if cell % 25 == 0 or cell == total_cells:
                print(
                    f"Completed {cell:4d}/{total_cells} cells "
                    f"({100.0 * cell / total_cells:5.1f}%)"
                )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["eta_alphaT0", "kappa_bundle_Km_over_Nk"]
    ).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)

    print("\nSaved:")
    print(OUT_CSV)

    # -------------------------------------------------------------------------
    # Numerical anchors used in the manuscript discussion
    # -------------------------------------------------------------------------
    print("\nSelected numerical anchors from THIS reproducible run:")

    anchors = [
        (5, 0.01),
        (5, 10.0),
        (25, 0.01),
        (25, 10.0),
        (30, 0.01),
        (30, 10.0),
    ]

    for eta_target, kappa_target in anchors:

        sub = df[
            np.isclose(df["eta_alphaT0"], eta_target)
        ].copy()

        idx = (
            sub["kappa_bundle_Km_over_Nk"] - kappa_target
        ).abs().idxmin()

        row = sub.loc[idx]

        print(
            f"eta={row['eta_alphaT0']:g}, "
            f"kappa_B={row['kappa_bundle_Km_over_Nk']:.5g}: "
            f"<S>={row['mean_size']:.3f}, "
            f"P_span={row['P_spanning']:.4f}, "
            f"P90={row['p90_size']:.1f}, "
            f"P99={row['p99_size']:.1f}, "
            f"A_seed={row['A_seed_one_step']:.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
