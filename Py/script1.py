import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Circuit parameters
V_s = 12.0
R1 = 10.0
L1 = 0.001
C1 = 0.0001
R2 = 100.0

# System of first-order differential equations (State variables: x[0] = v3, x[1] = i_L)
def circuit_dynamics(t, x):
    v3, i_L = x[0], x[1]
    dv3_dt = (i_L - v3/R2) / C1
    di_L_dt = (V_s - R1*i_L - v3) / L1
    return [dv3_dt, di_L_dt]

# Zero initial conditions at t=0
x0 = [0.0, 0.0]
t_span = (0, 0.01) # 10 ms to capture the complete transient response
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(circuit_dynamics, t_span, x0, t_eval=t_eval, method='Radau')

# Plotting results
plt.figure(figsize=(8, 4))
plt.plot(sol.t * 1000, sol.y[0], 'b-', label='$v_3(t)$')
plt.title('Case 1: Step Response of the Output Voltage $v_3(t)$')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()
plt.savefig('script1.pdf', format='pdf', bbox_inches='tight')
plt.show()
