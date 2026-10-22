# Script 4
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

V_s = 12.0
R1 = 10.0
L1 = 0.001
C1 = 0.0001
R2_initial = 100.0
t_switch = 0.004

def circuit_dynamics_dynamic(t, x):
    # Usamos desempaquetado directo para evitar los corchetes numéricos
    v3, i_L = x
    R2 = R2_initial if t < t_switch else (R2_initial / 2.0)
    dv3_dt = (i_L - v3/R2) / C1
    di_L_dt = (V_s - R1*i_L - v3) / L1
    return [dv3_dt, di_L_dt]

x0 = [0.0, 0.0]

# Extraemos los tiempos usando variables asignadas para evitar los corchetes
t_inicio = 0
t_fin = 0.01
t_span = (t_inicio, t_fin)
t_eval = np.linspace(t_inicio, t_fin, 1000)

sol = solve_ivp(circuit_dynamics_dynamic, t_span, x0, t_eval=t_eval, method='Radau', max_step=1e-5)

# Desempaquetamos las dos filas de la matriz de soluciones directamente
v3, i_L = sol.y

# Re-evaluate R2 at each time step for accurate absorbed power calculation
R2_array = np.where(sol.t < t_switch, R2_initial, R2_initial / 2.0)
p_total = V_s * i_L
p_R2 = (v3**2) / R2_array

plt.figure(figsize=(8, 4))
plt.plot(sol.t * 1000, p_total, 'g-', label='Total Supplied Power ($p_{gen}$)')
plt.plot(sol.t * 1000, p_R2, 'r-', label='Load Power ($p_{R2}$)')
plt.axvline(x=t_switch*1000, color='k', linestyle='--', label='$R_2$ Change')
plt.title('Case 2: Power Evaluation under Resistance Variation')
plt.xlabel('Time (ms)')
plt.ylabel('Power (W)')
plt.grid(True)
plt.legend()
plt.savefig('script4.pdf', format='pdf', bbox_inches='tight')
plt.show()
