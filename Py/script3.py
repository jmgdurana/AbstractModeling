# Script 3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

V_s = 12.0
R1 = 10.0
L1 = 0.001
C1 = 0.0001
R2 = 100.0

def circuit_dynamics(t, x):
    v3, i_L = x[0], x[1]
    dv3_dt = (i_L - v3/R2) / C1
    di_L_dt = (V_s - R1*i_L - v3) / L1
    return [dv3_dt, di_L_dt]

x0 = [0.0, 0.0]
t_span = (0, 0.01)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(circuit_dynamics, t_span, x0, t_eval=t_eval, method='Radau')

v3 = sol.y[0]
i_L = sol.y[1]

# Power calculations
p_total = V_s * i_L
p_R2 = (v3**2) / R2

# Plotting results
plt.figure(figsize=(8, 4))
plt.plot(sol.t * 1000, p_total, 'g-', label='Total Supplied Power ($p_{gen}$)')
plt.plot(sol.t * 1000, p_R2, 'r-', label='Load Power ($p_{R2}$)')
plt.title('Case 1: Transient Power Evaluation')
plt.xlabel('Time (ms)')
plt.ylabel('Power (W)')
plt.grid(True)
plt.legend()
plt.savefig('script3.pdf', format='pdf', bbox_inches='tight')
plt.show()
