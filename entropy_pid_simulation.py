import numpy as np
from scipy.integrate import odeint

# PID parametry z vaší teorie
k_P, k_I, k_D = 1.047, 2.31e-3, 0.178  
S_6D = 3.2e19  # Entropie vakua [kB]

def cosmic_PID(H, t):
    dHdt = -k_P * H - k_I * S_6D * t + k_D * (S_6D / t)
    return dHdt

t = np.linspace(0, 1e10, 1000)  # Časová osa
H0 = 70  # Hubbleova konstanta [km/s/Mpc]
solution = odeint(cosmic_PID, H0, t)

# Uložení dat pro graf
np.savetxt('pid_evolution.txt', np.column_stack((t, solution)))
