# Script 5: 
# Simulación de conmutación de carga en t=0 con circuito pre-energizado
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del circuito
V_s = 12.0
R1 = 10.0
L1 = 0.001
C1 = 0.0001
R2_initial = 100.0
R2_final = 50.0  # Cambia a la mitad en t=0

# 1. Condiciones iniciales en t=0+ debido al régimen permanente en t=0-
v3_0 = (R2_initial / (R1 + R2_initial)) * V_s  # ~10.91 V
iL_0 = V_s / (R1 + R2_initial)                 # ~0.109 A
x0 = [v3_0, iL_0]

# 2. Dinámica del circuito para t >= 0 (ya con R2 modificada)
def circuit_dynamics_after_switch(t, x):
    v3, i_L = x[0], x[1]
    R2 = R2_final  # Para todo t >= 0, la resistencia ya vale la mitad
    dv3_dt = (i_L - v3 / R2) / C1
    di_L_dt = (V_s - R1 * i_L - v3) / L1
    return [dv3_dt, di_L_dt]

# Ventana de simulación: 6 ms para apreciar el transitorio de caída
t_span = (0, 0.006)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Resolver la ODE numéricamente
sol = solve_ivp(circuit_dynamics_after_switch, t_span, x0, t_eval=t_eval, method='Radau')

# Extraer resultados
t_ms = sol.t * 1000
v3_sim = sol.y[0]
iL_sim = sol.y[1]

# 3. Cálculo de potencias transitorias para t >= 0
p_total = V_s * iL_sim
p_R2 = (v3_sim ** 2) / R2_final

# --- GRAFICACIÓN ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Gráfico 1: Tensión de salida v3(t)
ax1.plot(t_ms, v3_sim, 'r-', lw=2.5, label='$v_3(t)$')
ax1.axhline(v3_0, color='k', linestyle=':', label='Estado inicial ($t=0^-$)')
ax1.axhline((R2_final / (R1 + R2_final)) * V_s, color='g', linestyle='--', label='Nuevo estado estable ($t\\to\\infty$)')
ax1.set_ylabel('Tensión (V)')
ax1.set_title('Respuesta dinámica de la Tensión $v_3(t)$ tras la conmutación en $t=0$')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Gráfico 2: Potencia Total vs Potencia en la Carga
ax2.plot(t_ms, p_total, 'g-', lw=2, label='Potencia Total del Generador ($p_{gen}$)')
ax2.plot(t_ms, p_R2, 'm-', lw=2, label='Potencia en la Carga ($p_{R2}$)')
ax2.set_xlabel('Tiempo (ms)')
ax2.set_ylabel('Potencia (W)')
ax2.set_title('Evaluación de Potencias Transitorias')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('script5.pdf', format='pdf', bbox_inches='tight')
plt.show()
