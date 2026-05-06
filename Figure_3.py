# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:49:09 2026
@author: mrajakaruna

Figure 3: Minimal coarse-grained spreading simulation on a random interaction graph

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ===============================
# PARAMETERS
# ===============================

N = 2000                 # number of nodes
z_mean = 4               # average degree
trials = 1000            # number of avalanche trials per parameter
lambda_vals = np.linspace(0.1, 3.0, 15)  # effective loading parameter

# Derived network
p_edge = z_mean / (N - 1)
G = nx.erdos_renyi_graph(N, p_edge)

# ===============================
# Avalanche simulation
# ===============================

def run_avalanche(G, p_trigger):
    """Simulate one avalanche starting from a single seed."""
    failed = set()
    active = set([np.random.randint(0, N)])
    failed.update(active)
    
    while active:
        new_active = set()
        for node in active:
            for neighbor in G.neighbors(node):
                if neighbor not in failed:
                    if np.random.rand() < p_trigger:
                        new_active.add(neighbor)
                        failed.add(neighbor)
        active = new_active
    
    return len(failed)

# ===============================
# Displacement control
# p ≈ lambda_eff / 2
# ===============================

mean_sizes_disp = []
mean_sizes_force = []

beta = 0.3     # unloading fraction
alphaF = 3.0   # load scale (for force control prefactor)
r0 = 0.01
tmat = 0.05

for lam in lambda_vals:
    
    # Displacement control
    p_disp = lam / 2.0
    
    sizes = []
    for _ in range(trials):
        sizes.append(run_avalanche(G, p_disp))
    mean_sizes_disp.append(np.mean(sizes))
    
    # Force control (suppressed)
    p_force = r0 * np.exp(alphaF) * np.exp(-alphaF * beta) * tmat
    sizes_force = []
    for _ in range(trials):
        sizes_force.append(run_avalanche(G, p_force))
    mean_sizes_force.append(np.mean(sizes_force))

# ===============================
# Plot
# ===============================

plt.figure(figsize=(7,5))
plt.plot(lambda_vals, mean_sizes_disp, 'o-', label="Displacement control")
plt.plot(lambda_vals, mean_sizes_force, 's--', label="Force control (β=0.3)")
plt.axvline(2.0/z_mean, color='gray', linestyle=':', label="R=1 threshold")
plt.xlabel(r'$\lambda_{\mathrm{eff}}$')
plt.ylabel(r'$\langle S \rangle$')
plt.legend(frameon=False, fancybox=False)
plt.tight_layout()
plt.show()