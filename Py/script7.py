# =====================================================================
# Script 7: Abstract Modeling Optimization & Systematic Synthetic Identity
# Author: José González de Durana
# Article: Abstract modeling: formal reduction to praxis interpretation
# Corrected for NumPy 2.0+ / Python 3.13 Environment
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. PHYSICAL CIRCUIT PARAMETERS (M1 Layer)
V_s = 12.0
R1 = 10.0
L1 = 0.001
C1 = 0.0001
R2_initial = 100.0
R2_final = 50.0  

# Steady-state initial conditions at t=0- (Pre-load change)
v3_0 = (R2_initial / (R1 + R2_initial)) * V_s  
iL_0 = V_s / (R1 + R2_initial)                 
x0 = [v3_0, iL_0]

# 2. NUMERICAL RESOLUTION OF THE REAL SECOND-ORDER MODEL (M2)
def circuit_dynamics_after_switch(t, x):
    v3, i_L = x[0], x[1]
    dv3_dt = (i_L - v3 / R2_final) / C1
    di_L_dt = (V_s - R1 * i_L - v3) / L1
    return [dv3_dt, di_L_dt]

t_span = (0, 0.006) # 6 milliseconds transient window
t_eval = np.linspace(t_span[0], t_span[1], 1500)
sol_m2 = solve_ivp(circuit_dynamics_after_switch, t_span, x0, t_eval=t_eval, method='Radau')

t_ms = sol_m2.t * 1000
v3_m2 = sol_m2.y[0]
p_R2_m2 = (v3_m2 ** 2) / R2_final  # Real Power Flow

# 3. ANALYTICAL SYNTHESIS OF THE ABSTRACT FIRST-ORDER MODEL (M3)
# Characteristic equation coefficients for the final state (R2_final)
alpha = 0.5 * (R1/L1 + 1/(R2_final*C1))
omega_0_sq = (R1 + R2_final) / (R2_final * L1 * C1)

# Dominant pole isolation (s1 is the slower root closest to 0)
s1 = -alpha + np.sqrt(alpha**2 - omega_0_sq)
k_abstract = abs(s1)

# Steady-state limits for gain matching (A parameter)
v3_inf = (R2_final / (R1 + R2_final)) * V_s
A_abstract = v3_inf * k_abstract

# First-order explicit voltage transient response
v3_m1 = v3_inf + (v3_0 - v3_inf) * np.exp(-k_abstract * sol_m2.t)
p_R2_m1 = (v3_m1 ** 2) / R2_final  # Abstract Power Flow

# 4. METRIC CONVERGENCE VIA INVARIANT TRAPEZOIDAL VECTOR SLICING
# Functional manual integration to replace deprecated np.trapz across versions
def modern_trapz(y, x):
    return np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0

iape_energy_error = modern_trapz(np.abs(p_R2_m2 - p_R2_m1), sol_m2.t)
relative_error_pct = (iape_energy_error / modern_trapz(p_R2_m2, sol_m2.t)) * 100

print("=========================================================")
print("GNOSIOLOGICAL CLOSURE PARAMETERS (M3 Domain):")
print(f"Dominant Pole (Inverse Time Constant, k): {k_abstract:.2f} rad/s")
print(f"Abstract Gain Factor (A):                 {A_abstract:.2f} V/s")
print(f"Integrated Absolute Power Error (IAPE):   {iape_energy_error*1000:.4f} mJ")
print(f"Relative Energy Mismatch Profile:         {relative_error_pct:.4f} %")
print("=========================================================")

# 5. HIGH-QUALITY PLOTTING BLOCK (MDPI ENERGIES STANDARD)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Top Graph: Voltage Abstraction Layer (Using Raw Strings r'...' for LaTeX strings)
ax1.plot(t_ms, v3_m2, 'b-', lw=2.5, label=r'Real $2^{\mathrm{nd}}$-Order Voltage ($M_1$)')
ax1.plot(t_ms, v3_m1, 'm--', lw=2, label=r'Abstract $1^{\mathrm{st}}$-Order Voltage ($M_3$)')
ax1.set_ylabel('Voltage $v_3(t)$ (V)')
ax1.set_title('Voltage State Space Trajectory Comparison', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right')

# Bottom Graph: Power Flow Discontinuity & Identity Closure
ax2.plot(t_ms, p_R2_m2, 'r-', lw=2.5, label='Real Load Power ($M_1$)')
ax2.plot(t_ms, p_R2_m1, 'm--', lw=2, label='Abstract Load Power ($M_3$)')
ax2.fill_between(t_ms, p_R2_m2, p_R2_m1, color='red', alpha=0.2, label='Energy Error Area')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Power $p_{R2}(t)$ (W)')
ax2.set_title('Transient Power Flow and Energy Mismatch Profile', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('script7.pdf', format='pdf', bbox_inches='tight')
plt.show()
