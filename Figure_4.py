# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:49:09 2026
@author: mrajakaruna

Figure 4: Cross-disciplinary analogies - earthquakes, neuronal avalanches, non-reciprocal phases

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.lines as mlines

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Create figure (slightly wider to prevent "shrunken" top row)
fig = plt.figure(figsize=(9.2, 6))

# Less whitespace between columns -> wider top panels
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.28, wspace=0.18)

# =============================================================================
# Panel a: Earthquake physics
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0])

# Tighter x-range so drawing fills panel width
ax_a.set_xlim(0.5, 7.2)
ax_a.set_ylim(0, 4)
ax_a.axis('off')

# Draw fault line
ax_a.plot([1, 7], [2, 2], 'k-', linewidth=3)

# Draw tectonic plates
plate1 = Rectangle((0.8, 1.5), 2.6, 1, facecolor='lightgray', edgecolor='black', linewidth=1.5)
plate2 = Rectangle((4.6, 1.5), 2.6, 1, facecolor='lightgray', edgecolor='black', linewidth=1.5)
ax_a.add_patch(plate1)
ax_a.add_patch(plate2)

# Add plate motion arrows
ax_a.annotate('', xy=(2.0, 2.5), xytext=(3.1, 2.5),
              arrowprops=dict(arrowstyle='->', color='blue', linewidth=2))
ax_a.annotate('', xy=(6.3, 1.5), xytext=(5.1, 1.5),
              arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

ax_a.text(2.55, 2.78, 'Plate motion', fontsize=7, color='blue', clip_on=False)
ax_a.text(5.35, 1.18, 'Plate motion', fontsize=7, color='red', clip_on=False)

# Mark rupture initiation
rupture_point = (3.2, 2)
circle = Circle(rupture_point, 0.2, facecolor='red', edgecolor='darkred', linewidth=2, zorder=10)
ax_a.add_patch(circle)
ax_a.text(2.75, 2.3,
          r'$t_1$',
          fontsize=9,
          fontweight='bold',
          clip_on=False)

# Draw seismic wave fronts
for i in range(3):
    radius = i * 0.8 + 0.4
    circle_wave = Circle(rupture_point, radius, facecolor='none',
                         edgecolor='orange', linewidth=1.5, alpha=0.65, linestyle='--')
    ax_a.add_patch(circle_wave)

# Mark second rupture site
second_rupture = (5.7, 2)
circle2 = Circle(second_rupture, 0.2, facecolor='yellow', edgecolor='orange', linewidth=2, zorder=10)
ax_a.add_patch(circle2)

ax_a.text(5.85, 2.28,
          r'$t_2 = t_1 + \tau$',
          fontsize=8,
          clip_on=False)

# Add distance annotation
ax_a.annotate('', xy=(3.4, 1.5), xytext=(5.5, 1.5),
              arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.5))
ax_a.text(4.45, 1.28, 'L', ha='center', fontsize=8, clip_on=False)

# Add causal condition
ax_a.text(4.45, 0.52, r'Cascading if $\tau < L/c_s$',
          ha='center', fontsize=7,
          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'),
          clip_on=False)

# Title + panel label
ax_a.text(4.0, 3.7, 'Earthquake physics', ha='center', fontsize=9, fontweight='bold', clip_on=False)
ax_a.text(0.02, 0.98, 'a', fontsize=12, fontweight='bold', va='top', transform=ax_a.transAxes)

# =============================================================================
# Panel b: Neuronal avalanches
# =============================================================================
ax_b = fig.add_subplot(gs[0, 1])

# Tighter x-range to fill panel
ax_b.set_xlim(1.0, 7.0)
ax_b.set_ylim(0, 4)
ax_b.axis('off')

neuron_positions = [(2, 2), (4, 1.5), (6, 2.5), (3, 3), (5, 1)]
for i, (x, y) in enumerate(neuron_positions):
    circ = Circle((x, y), 0.3, facecolor='lightblue', edgecolor='blue', linewidth=1.5)
    ax_b.add_patch(circ)
    ax_b.text(x, y, f'N{i+1}', ha='center', va='center', fontsize=6)

# Excitatory connections (positive feedback)
connections = [(2, 2, 4, 1.5), (2, 2, 3, 3), (4, 1.5, 6, 2.5), (3, 3, 5, 1)]
for x1, y1, x2, y2 in connections:
    ax_b.annotate('', xy=(x2, y2), xytext=(x1, y1),
                  arrowprops=dict(arrowstyle='->', color='green', linewidth=1.5, alpha=0.7))

# Inhibitory connection (negative feedback)
ax_b.annotate('', xy=(5, 1), xytext=(6, 2.5),
              arrowprops=dict(arrowstyle='->', color='red', linewidth=2, alpha=0.7))

ax_b.text(2.05, 2.55, 'Excitatory\n(positive fb)', fontsize=6, color='green', ha='center', clip_on=False)
ax_b.text(6.10, 1.5, 'Inhibitory\n(negative fb)', fontsize=6, color='red', ha='center', clip_on=False)

# Firing event
firing_neuron = (2, 2)
circle_fire = Circle(firing_neuron, 0.4, facecolor='none', edgecolor='orange', linewidth=2, linestyle='--')
ax_b.add_patch(circle_fire)
ax_b.text(2, 1.2, 'Initial spike', fontsize=6, ha='center', clip_on=False)

# Avalanche region
avalanche = FancyBboxPatch((3.55, 1.0), 2.95, 2.0, boxstyle='round,pad=0.1',
                           facecolor='lightcoral', alpha=0.18, edgecolor='red', linewidth=1)
ax_b.add_patch(avalanche)
ax_b.text(5.0, 2.05, r'Avalanche'+'\n'+r'if $\tau < t_{\mathrm{mat}}$',
          ha='center', fontsize=7, clip_on=False)

ax_b.text(4.0, 3.7, 'Neuronal avalanches', ha='center', fontsize=9, fontweight='bold', clip_on=False)
ax_b.text(0.02, 0.98, 'b', fontsize=12, fontweight='bold', va='top', transform=ax_b.transAxes)

# =============================================================================
# Panel c: Non-reciprocal phase transitions
# =============================================================================
ax_c = fig.add_subplot(gs[0, 2])

# Tighter x-range (and extend a bit right so "Collective phase" doesn't clip)
ax_c.set_xlim(1.3, 7.9)
ax_c.set_ylim(0, 4)
ax_c.axis('off')

bond1 = Circle((3, 2), 0.5, facecolor='lightblue', edgecolor='blue', linewidth=2)
bond2 = Circle((5, 2), 0.5, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax_c.add_patch(bond1)
ax_c.add_patch(bond2)

ax_c.text(3, 2, 'Bond 1', ha='center', va='center', fontsize=6)
ax_c.text(5, 2, 'Bond 2', ha='center', va='center', fontsize=6)

ax_c.annotate('', xy=(4.5, 2.2), xytext=(3.5, 2.2),
              arrowprops=dict(arrowstyle='->', color='red', linewidth=3))
ax_c.annotate('', xy=(3.5, 1.8), xytext=(4.5, 1.8),
              arrowprops=dict(arrowstyle='->', color='blue', linewidth=1, alpha=0.5))

ax_c.text(4, 2.72, 'Strong coupling\n(positive feedback)', ha='center', fontsize=6, color='red', clip_on=False)
ax_c.text(4, 1.13, 'Weak/no coupling\n(no feedback)', ha='center', fontsize=6, color='blue', clip_on=False)

ax_c.text(3, 0.90,
          r'Failed at $t_1$',
          ha='center',
          fontsize=7,        # slightly larger helps clarity
          fontweight='bold',
          clip_on=False)
check = Circle((3, 0.7), 0.1, facecolor='red', edgecolor='darkred')
ax_c.add_patch(check)

# Phase transition indication (move slightly left + ensure visible)
ax_c.annotate('', xy=(6.4, 2), xytext=(7.2, 2),
              arrowprops=dict(arrowstyle='->', color='purple', linewidth=2))
ax_c.text(6.8, 2.3, 'Collective\nphase', fontsize=6, ha='center', color='purple', clip_on=False)

ax_c.text(4.0, 3.7, 'Non-reciprocal phases', ha='center', fontsize=9, fontweight='bold', clip_on=False)
ax_c.text(0.02, 0.98, 'c', fontsize=12, fontweight='bold', va='top', transform=ax_c.transAxes)

# =============================================================================
# Panel d: Summary schematic
# =============================================================================
ax_d = fig.add_subplot(gs[1, :])
ax_d.set_xlim(0, 10)
ax_d.set_ylim(0, 4)
ax_d.axis('off')

column_positions = [2, 5, 8]
column_labels = ['Earthquake\nphysics', 'Neuronal\navalanches', 'Non-reciprocal\nphase transitions']

for i, (x0, label) in enumerate(zip(column_positions, column_labels)):
    rect = Rectangle((x0-1.5, 0.5), 3, 3, facecolor='lightgray', alpha=0.2,
                     edgecolor='black', linewidth=0.8)
    ax_d.add_patch(rect)
    ax_d.text(x0, 3.5, label, ha='center', va='center', fontsize=8, fontweight='bold')

    if i == 0:
        ax_d.text(x0, 2.5, 'Positive\nfeedback', ha='center', fontsize=8, color='red')
        ax_d.text(x0, 1.5, 'Cascading\nrupture', ha='center', fontsize=8)
        ax_d.annotate('', xy=(x0+0.5, 2.2), xytext=(x0-0.5, 2.2),
                      arrowprops=dict(arrowstyle='->', color='red', linewidth=2))
    elif i == 1:
        ax_d.text(x0, 2.7, 'Excitatory: +', ha='center', fontsize=7, color='green')
        ax_d.text(x0, 2.2, 'Inhibitory: -', ha='center', fontsize=7, color='red')
        ax_d.text(x0, 1.5, 'Avalanches/\nSuppression', ha='center', fontsize=7)
        ax_d.annotate('', xy=(x0+0.5, 2.9), xytext=(x0-0.5, 2.9),
                      arrowprops=dict(arrowstyle='->', color='green', linewidth=1.5))
        ax_d.annotate('', xy=(x0-0.5, 2.4), xytext=(x0+0.5, 2.4),
                      arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5))
    else:
        ax_d.text(x0 +0.35, 2.5, 'Asymmetric\nfeedback', ha='center', fontsize=8, color='purple')
        ax_d.text(x0, 1.5, 'Collective\nphase', ha='center', fontsize=8)
        ax_d.annotate('', xy=(x0+0.3, 2.2), xytext=(x0-0.3, 2.8),
                      arrowprops=dict(arrowstyle='->', color='purple', linewidth=1.5))

ax_d.text(5, 0.2, '↓ Feedback polarity determines universality class ↓',
          ha='center', fontsize=9, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

ax_d.text(0.02, 0.98, 'd', fontsize=12, fontweight='bold', va='top', transform=ax_d.transAxes)

plt.tight_layout()
plt.savefig('fig3_analogies.eps', format='eps')
plt.savefig('fig3_analogies.png', format='png', dpi=600)
plt.savefig('fig3_analogies.pdf', format='pdf')
plt.show()

print("Figure 3 saved as fig3_analogies.eps, .png, .pdf")