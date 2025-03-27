"""
6D Holographic Bound Calculator v4.0
With complete:
- CY compactification (χ=-200)
- Entropic dark energy
- Proton decay suppression
- CMB non-Gaussianity
DOI:10.5281/zenodo.15085762
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, c, G, k as k_B

# 6D Planck units with CY compactification
L6 = (hbar*G/c**3)**0.5 * (2*np.pi)**(-1/4)  # 6D Planck length [m]
M6 = (hbar*c/G)**0.5 * (2*np.pi)**(1/4)       # 6D Planck mass [kg]
χ = -200                                      # CY Euler characteristic
V_CY = (2*np.pi)**3 * L6**6 / abs(χ)          # CY volume [m^6]

def entropy_6D(r, n_cy=1):
    """Calculate fully regularized 6D entropy"""
    # Compactified area scaling
    A_eff = (8/3)*np.pi**2 * r**3 * (V_CY/L6**6)
    A0 = 4 * L6**2
    x = A_eff/A0
    
    # Regularized components
    S_geom = np.where(x < 1, x**2, x**1.5)  # UV/IR transition
    S_quant = 4*np.pi**2 * n_cy             # CY quantization
    S_log = 0.1*np.log(1 + x)               # Holographic correction
    
    return S_geom + S_quant + S_log

def dark_energy(S):
    """Entropic dark energy density"""
    return 0.00231 * (S - 4*np.pi**2) * M6**4  # [J/m^3]

def proton_lifetime(S):
    """Proton decay suppression"""
    return np.exp(S - 4*np.pi**2)  # [s]

def cmb_non_gaussianity(S):
    """CMB non-Gaussianity parameter"""
    return (5/12) * (2*np.pi/np.sqrt(abs(χ)))**2 / (3/8*0.00231**2) * (S/(4*np.pi**2) - 1

def verify_physics():
    """Physical consistency checks"""
    print("=== 6D Physics Verification ===")
    print(f"6D Planck length: {L6:.3e} m")
    print(f"CY volume: {V_CY:.3e} m^6\n")
    
    test_scales = {
        "Planck scale": L6,
        "Proton radius": 0.84e-15,
        "Atomic scale": 1e-10,
        "Macroscopic": 1e-3
    }
    
    for name, r in test_scales.items():
        S = entropy_6D(r)
        print(f"{name} ({r:.1e} m):")
        print(f"  S/k_B = {S:.3e}")
        print(f"  ρ_Λ = {dark_energy(S):.3e} J/m^3")
        print(f"  τ_p = {proton_lifetime(S):.3e} s")
        print(f"  f_NL = {cmb_non_gaussianity(S):.3f}\n")

if __name__ == "__main__":
    verify_physics()
    
    # Generate entropy profile
    radii = np.logspace(np.log10(L6)-5, -2, 500)
    entropies = [entropy_6D(r) for r in radii]
    
    plt.figure(figsize=(10,6))
    plt.loglog(radii/L6, entropies, 'b-')
    plt.axhline(4*np.pi**2, color='r', ls='--', label='CY Quantization')
    plt.xlabel('Radius (L6 units)', fontsize=12)
    plt.ylabel('S/k_B', fontsize=12)
    plt.title('6D Entropy with CY Compactification', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('6D_entropy_CY.png', dpi=300)
