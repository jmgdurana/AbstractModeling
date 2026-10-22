# =====================================================================
# 1. IMPORT LABELS & LIBRARIES
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =====================================================================
# 2. CIRCUIT PARAMETERS AND INITIAL CONDITIONS (t = 0-)
# =====================================================================
V_s = 12.0
R1 = 10.0
L1 = 0.001
C1 = 0.0001
R2_initial = 100.0
R2_final = 50.0  

# Steady-state initial conditions just before switching (t=0-)
v3_0 = (R2_initial / (R1 + R2_initial)) * V_s  
iL_0 = V_s / (R1 + R2_initial)                 
x0 = [v3_0, iL_0]

# =====================================================================
# 3. CIRCUIT DYNAMICS AND NUMERICAL RESOLUTION (t >= 0)
# =====================================================================
def circuit_dynamics_after_switch(t, x):
    v3, i_L = x[0], x[1]
    dv3_dt = (i_L - v3 / R2_final) / C1
    di_L_dt = (V_s - R1 * i_L - v3) / L1
    return [dv3_dt, di_L_dt]

# Simulation time window (0 to 6 milliseconds)
t_span = (0, 0.006)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Numerical ODE resolution
sol = solve_ivp(circuit_dynamics_after_switch, t_span, x0, t_eval=t_eval, method='Radau')

# Data extraction
t_ms = sol.t * 1000
v3_sim = sol.y[0]
iL_sim = sol.y[1]

# =====================================================================
# 4. TRANSIENT POWER CALCULATIONS
# =====================================================================
p_total = V_s * iL_sim
p_R2 = (v3_sim ** 2) / R2_final

# =====================================================================
# 5. PLOTTING BLOCK (POWERS ONLY - ENGLISH LABELS)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

# Power transient curves
ax.plot(t_ms, p_total, 'g-', lw=2, label='Total Source Power ($p_{gen}$)')
ax.plot(t_ms, p_R2, 'r-', lw=2, label='Load Power ($p_{R2}$)')

# Axis layout and scientific labeling
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Power (W)')
ax.set_title('Transient Power Evaluation')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('script6_powers_publication.pdf', format='pdf', bbox_inches='tight')
plt.show()
