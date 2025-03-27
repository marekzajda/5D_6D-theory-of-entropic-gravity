"""
5D Holographic Bound Calculator
Physical constants from DOI:10.5281/zenodo.15085762
"""

import numpy as np
import matplotlib.pyplot as plt

# Fundamental constants (CODATA 2018)
ħ = 1.054571817e-34     # Reduced Planck constant [J·s]
c = 299792458           # Speed of light [m/s]
G = 6.67430e-11         # Gravitational constant [m³/kg·s²]

# Derived 5D Planck units
L_5D = (ħ*G/c**3)**0.5  # Planck length [m]
M_5D = (ħ*c/G)**0.5     # Planck mass [kg]
T_5D = (ħ*G/c**5)**0.5  # Planck time [s]

def entropy_5D(r):
    """Calculate dimensionless entropy (S/k_B)"""
    # Surface area of 5D "sphere"
    A = (8/3)*np.pi**2 * r**3  
    
    # Reference area (4*Planck area)
    A0 = 4 * L_5D**2
    
    # Main term + corrections (α=0.1, β=0.01)
    ratio = A/A0
    return ratio**1.5 + 0.1*np.log(abs(ratio)) + 0.01*ratio**-0.5

# Physical range from Planck scale to 1cm
radii = np.logspace(np.log10(L_5D), -2, 500)
entropies = [entropy_5D(r) for r in radii]

# Generate plot
plt.figure(figsize=(10,6))
plt.loglog(radii/L_5D, entropies)
plt.xlabel('Radius (in Planck units)', fontsize=12)
plt.ylabel('Entropy S/k$_B$', fontsize=12)
plt.title('5D Entropic Gravity', fontsize=14)
plt.grid(True, which="both", ls="--")
plt.savefig('5D_entropy.png', dpi=300, bbox_inches='tight')

# Key results
print(f"5D Planck length = {L_5D:.3e} m")
print(f"Entropy at r=1µm: {entropy_5D(1e-6):.3e}")
print("Plot saved to 5D_entropy.png")
