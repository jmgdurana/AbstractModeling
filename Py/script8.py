# =====================================================================
# Script 8: Geometric Mapping of Valid Commercial Cables Suite
# Author: José González de Durana
# Article: Abstract modeling: formal reduction to praxis interpretation
# Environment: Python 3.13 / NumPy 2.0+ Compliant
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

# 1. FIXED NETWORK CONFIGURATION (M1 Operational Layer)
V_s = 12.0
R2_nominal = 50.0  # Nominal load resistance (from Script 7 final state)

# Cable baseline parameters per kilometer (typical commercial microgrid values)
R_per_km = 0.50     # Ohms/km
L_per_km = 150e-6   # Henries/km

# 2. PARAMETER SPACE RESOLUTION (Grid Definition)
lengths = np.linspace(0.1, 5.0, 400)      # Line length from 100m to 5km
capacitances = np.linspace(1e-6, 500e-6, 400) # Filter capacity from 1uF to 500uF

# Create 2D mesh grid
Length_Grid, C1_Grid = np.meshgrid(lengths, capacitances)

# Calculate localized lumped parameters across the grid
R1_Grid = Length_Grid * R_per_km
L1_Grid = Length_Grid * L_per_km

# 3. SECOND-ORDER CHARACTERISTIC EVALUATION & TIMESCALE SEPARATION
# Alpha (attenuation factor) and Omega_0 squared (undamped natural frequency)
alpha = 0.5 * (R1_Grid / L1_Grid + 1.0 / (R2_nominal * C1_Grid))
omega_0_sq = (R1_Grid + R2_nominal) / (R2_nominal * L1_Grid * C1_Grid)

# Discriminant array for overdamped regime validation
discriminant = alpha**2 - omega_0_sq

# Isolate real roots across the parametric mesh
s1 = -alpha + np.sqrt(np.maximum(0, discriminant)) # Slower dominant pole
s2 = -alpha - np.sqrt(np.maximum(0, discriminant)) # Faster pole

# Define validation criteria filters
is_overdamped = discriminant > 0
timescale_separation = np.abs(s2 / s1) >= 5.0 # Strict factor of 5 separation

# Combine masks into the definitive "Valid Cable" domain matrix
valid_cables_mask = is_overdamped & timescale_separation

# 4. SCIENTIFIC VISUALIZATION (MDPI ENERGIES GEOMETRIC LAYOUT)
plt.figure(figsize=(8.5, 6))

# Plot the parametric design domains using filled contours
cp = plt.contourf(Length_Grid, C1_Grid * 1e6, valid_cables_mask.astype(int),
                  levels=[-0.5, 0.5, 1.5],
                  colors=['#f8d7da', '#d4edda'], alpha=0.9)

# Overlay the explicit boundaries
plt.contour(Length_Grid, C1_Grid * 1e6, discriminant, levels=[0],
            colors=['red'], linewidths=2.0, linestyles='--')

# Plot labels and architectural styling
plt.title(r'Geometric Boundaries of the "Valid Cables Collection" ($M_3$ Design Space)', fontsize=12, pad=15)
plt.xlabel('Cable Length (km)', fontsize=10)
plt.ylabel(r'Filter Capacitance $C_1$ ($\mu$F)', fontsize=10)

# Visual anchors (custom annotations for the chart)
plt.text(1.0, 100, 'VALID COLLECTION\n(First-Order Abstraction\nMatches Real Transient)',
         color='green', fontsize=10, weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

plt.text(3.5, 400, 'INVALID ZONE\n(Oscillatory or Non-Dominant\nSecond-Order Dynamics)',
         color='darkred', fontsize=10, weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

# Overlay line marking the Overdamped exact border for the paper
proxy_line = plt.Line2D([0], [0], color='red', linestyle='--', lw=2, label=r'Overdamped Boundary ($\alpha^2 = \omega_0^2$)')
plt.legend(handles=[proxy_line], loc='upper right')

plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('script8_valid_cables_map.pdf', format='pdf', bbox_inches='tight')
plt.show()

