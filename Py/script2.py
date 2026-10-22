# Script 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

V_s = 12.0
R1 = 10.0
L1 = 0.001
C1 = 0.0001
R2_initial = 100.0
t_switch = 0.004  # Time step of the resistance change

def circuit_dynamics_dynamic(t, x):
    v3, i_L = x[0], x[1]
    R2 = R2_initial if t < t_switch else (R2_initial / 2.0)
    dv3_dt = (i_L - v3/R2) / C1
    di_L_dt = (V_s - R1*i_L - v3) / L1
    return [dv3_dt, di_L_dt]

x0 = [0.0, 0.0]
t_span = (0, 0.01)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(circuit_dynamics_dynamic, t_span, x0, t_eval=t_eval, method='Radau', max_step=1e-5)

# Plotting results
plt.figure(figsize=(8, 4))
plt.plot(sol.t * 1000, sol.y[0], 'b-', label='$v_3(t)$')
plt.axvline(x=t_switch*1000, color='k', linestyle='--', label='$R_2$ Change')
plt.title('Case 2: Response under Resistance Variation in $R_2$')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()
plt.savefig('script2.pdf', format='pdf', bbox_inches='tight')
plt.show()
