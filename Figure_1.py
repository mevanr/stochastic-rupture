# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:49:09 2026
@author: mrajakaruna

Figure 1: Minimal stochastic rupture model and causal synchronization

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
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
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Create figure
fig = plt.figure(figsize=(7.5, 8))

# Grid:
# - Use row 0 for panel a
# - Use row 1 for panels b/c
# - Use row 2 for panel d
# - Use row 3 as an empty spacer row to keep panel d clear
gs = fig.add_gridspec(
    4, 2,
    height_ratios=[1.00, 1.00, 1.20, 0.25],  # last row is spacer
    hspace=0.45,                              # smaller -> b/c move upward
    wspace=0.30
)

# ------------------------------------------------------------
# Panel a: Three-particle system (protocol-neutral schematic)
# ------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, :])
ax_a.set_xlim(-1, 8)
ax_a.set_ylim(-1, 3)
ax_a.set_aspect('equal')
ax_a.axis('off')

particle_positions = [1, 3, 5]
particle_radius = 0.4

for i, x in enumerate(particle_positions):
    circle = Circle((x, 1), particle_radius,
                    facecolor='lightblue',
                    edgecolor='black',
                    linewidth=2,
                    zorder=5)
    ax_a.add_patch(circle)
    ax_a.text(x, 1, f'{i+1}',
              ha='center', va='center',
              fontsize=10, fontweight='bold')

bond1 = mlines.Line2D([particle_positions[0] + particle_radius,
                      particle_positions[1] - particle_radius],
                     [1, 1],
                     color='black', linewidth=2.5, solid_capstyle='round')
bond2 = mlines.Line2D([particle_positions[1] + particle_radius,
                      particle_positions[2] - particle_radius],
                     [1, 1],
                     color='black', linewidth=2.5, solid_capstyle='round')
ax_a.add_line(bond1)
ax_a.add_line(bond2)

spring_x1 = np.linspace(particle_positions[0] + particle_radius + 0.2,
                        particle_positions[1] - particle_radius - 0.2, 100)
spring_y1 = 1 + 0.10 * np.sin(
    6 * np.pi * (spring_x1 - spring_x1[0]) / (spring_x1[-1] - spring_x1[0])
)
ax_a.plot(spring_x1, spring_y1, color='black', linewidth=1.8)

spring_x2 = np.linspace(particle_positions[1] + particle_radius + 0.2,
                        particle_positions[2] - particle_radius - 0.2, 100)
spring_y2 = 1 + 0.10 * np.sin(
    6 * np.pi * (spring_x2 - spring_x2[0]) / (spring_x2[-1] - spring_x2[0])
)
ax_a.plot(spring_x2, spring_y2, color='black', linewidth=1.8)

ax_a.text(2, 1.6, r'$k,\, \ell_0$', ha='center', fontsize=10)
ax_a.text(4, 1.6, r'$k,\, \ell_0$', ha='center', fontsize=10)

x_left = particle_positions[1] + particle_radius
x_right = particle_positions[2] - particle_radius
y_arrow = 0.6

ax_a.annotate('',
              xy=(x_right, y_arrow),
              xytext=(x_left, y_arrow),
              arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.5))

ax_a.text((x_left + x_right) / 2,
          y_arrow - 0.18,
          r'$L$',
          ha='center', va='top', fontsize=11)

# Panel label a
ax_a.text(-0.08, 1.02, 'a',
          fontsize=13, fontweight='bold', va='top',
          transform=ax_a.transAxes)

# ------------------------------------------------------------
# Panel b: Displacement control tension evolution
# ------------------------------------------------------------
ax_b = fig.add_subplot(gs[1, 0])
time = np.linspace(0, 2, 1000)
t1 = 0.8

T_pre = time
T_post = np.where(time > t1, 2*(time - t1) + t1, np.nan)

ax_b.plot(time, T_pre, 'b-', linewidth=2, label='Pre-rupture (each bond)')
ax_b.plot(time, T_post, 'r-', linewidth=2, label='Post-rupture (survivor)')
ax_b.axvline(x=t1, color='gray', linestyle='--', linewidth=1, label=r'First rupture at $t_1$')
ax_b.axhline(y=t1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

x_arrow = t1 + 0.3
y_pre = x_arrow
y_post = 2*(x_arrow - t1) + t1
y_bottom = y_pre + 0.02
y_top = y_post - 0.02

ax_b.annotate('',
              xy=(x_arrow, y_top),
              xytext=(x_arrow, y_bottom),
              arrowprops=dict(arrowstyle='->', color='red', linewidth=1))

ax_b.text(x_arrow + 0.06,
          y_bottom - 0.02,
          'Loading rate doubles',
          fontsize=7, ha='left', va='top', color='red')

ax_b.set_xlabel('Time (arb. units)')
ax_b.set_ylabel('Tension T')
ax_b.set_title('Displacement control: positive feedback', fontsize=9)

legb = ax_b.legend(loc='upper left', frameon=False, fancybox=False)
legb.get_frame().set_alpha(0.0)

ax_b.grid(False)
ax_b.set_xlim(0, 2)
ax_b.set_ylim(0, 2.5)

# Panel label b (added)
ax_b.text(-0.18, 1.02, 'b',
          fontsize=13, fontweight='bold', va='top',
          transform=ax_b.transAxes)

# ------------------------------------------------------------
# Panel c: Force control tension evolution
# ------------------------------------------------------------
ax_c = fig.add_subplot(gs[1, 1])
time = np.linspace(0, 4, 1000)
t1 = 1.0
F = 1.0
beta = 0.7
tau_rec = 1.5

T_pre_force = F * np.ones_like(time)
T_post_force = np.where(time > t1, F * (1 - beta * np.exp(-(time - t1)/tau_rec)), np.nan)

ax_c.plot(time, T_pre_force, 'b-', linewidth=2, label='Pre-rupture')
ax_c.plot(time, T_post_force, 'r-', linewidth=2, label='Post-rupture')
ax_c.axvline(x=t1, color='gray', linestyle='--', linewidth=1, label=r'First rupture at $t_1$')
ax_c.axhline(y=F*(1-beta), color='red', linestyle=':', linewidth=1, alpha=0.7)

x_arrow = t1 + 0.6
y_curve = F * (1 - beta * np.exp(-(x_arrow - t1)/tau_rec))
y_top = y_curve + 0.20
y_bottom = y_curve + 0.05

ax_c.annotate('',
              xy=(x_arrow, y_bottom),
              xytext=(x_arrow, y_top),
              arrowprops=dict(arrowstyle='->', color='red', linewidth=1))

ax_c.text(x_arrow, y_top + 0.03,
          r'Unloading' + '\n' + rf'$\beta = {beta}$',
          fontsize=7, ha='center', va='bottom', color='red')

ax_c.set_xlabel('Time (arb. units)')
ax_c.set_ylabel('Tension T')
ax_c.set_title('Force control: negative feedback', fontsize=9)

legc = ax_c.legend(loc='lower right', frameon=False, fancybox=False)
legc.get_frame().set_alpha(0.0)

ax_c.grid(False)
ax_c.set_xlim(0, 4)
ax_c.set_ylim(0, 1.2)

# Panel label c (added)
ax_c.text(-0.18, 1.02, 'c',
          fontsize=13, fontweight='bold', va='top',
          transform=ax_c.transAxes)

# ------------------------------------------------------------
# Panel d: Causal synchronization schematic
# ------------------------------------------------------------
ax_d = fig.add_subplot(gs[2, :])
ax_d.set_xlim(-1, 8)
ax_d.set_ylim(-1, 4)
ax_d.set_aspect('equal')
ax_d.axis('off')

circle1 = Circle((2, 2), 0.5, facecolor='lightcoral', edgecolor='red', linewidth=2, zorder=10)
ax_d.add_patch(circle1)
circle2 = Circle((6, 2), 0.5, facecolor='lightblue', edgecolor='blue', linewidth=2, zorder=10)
ax_d.add_patch(circle2)

ax_d.text(2, 2, r'$t_1$', ha='center', va='center', fontsize=11, fontweight='bold', zorder=20)
ax_d.text(6, 2.5, r'$t_2 = t_1 + \tau$', ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=20)

for r in [1.0, 2.2, 3.4]:
    circle_wave = Circle((2, 2), r, facecolor='none', edgecolor='orange',
                         linewidth=1.2, alpha=0.5, linestyle='--', zorder=1)
    ax_d.add_patch(circle_wave)

arrow = FancyArrowPatch((2.5, 2.7), (4.8, 2.7),
                        arrowstyle='-|>', mutation_scale=18,
                        linewidth=2.5, color='orange', zorder=30)
ax_d.add_patch(arrow)

ax_d.text(3.65, 3.25, r'Stress wave speed $c_s$', ha='center', fontsize=9)

ax_d.annotate('',
              xy=(2, 1),
              xytext=(6, 1),
              arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.5))

ax_d.text(4, 0.5, r'$L$', ha='center', fontsize=10)
ax_d.text(4, 1.3, r'$t_{\mathrm{mat}} = L/c_s$', ha='center', fontsize=10)

ax_d.text(4, -0.4,
          r'If $\tau < t_{\mathrm{mat}}$: synchronized' + '\n' +
          r'If $\tau > t_{\mathrm{mat}}$: independent',
          ha='center', va='center', fontsize=9,
          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

# Panel label d
ax_d.text(-0.08, 1.02, 'd',
          fontsize=13, fontweight='bold', va='top',
          transform=ax_d.transAxes)

# ------------------------------------------------------------
# Spacer axis (empty) to preserve clearance below panel d
# ------------------------------------------------------------
ax_spacer = fig.add_subplot(gs[3, :])
ax_spacer.axis('off')

# Save figure
plt.savefig('fig1_schematic.eps', format='eps')
plt.savefig('fig1_schematic.png', format='png', dpi=600)
plt.savefig('fig1_schematic.pdf', format='pdf')
plt.show()

print("Figure 1 saved as fig1_schematic.eps, .png, .pdf")