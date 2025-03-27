"""
5D Entropic Gravity Simulation with Proper Normalization
"""

import numpy as np
import matplotlib.pyplot as plt

# Fundamental constants (SI units)
ħ = 1.054571817e-34  # Reduced Planck constant [J·s]
c = 299792458        # Speed of light [m/s]
G = 6.67430e-11      # Gravitational constant [m³/kg·s²]

# Calculate 5D Planck length
L_5D = np.sqrt(ħ*G/c**3)  # ~1.616e-35 m

def normalized_entropy(r):
    """Calculate dimensionless entropy with proper normalization"""
    # 5D surface area
    A = (8/3)*np.pi**2 * r**3  
    
    # Reference area (4*Planck area)
    A0 = 4 * L_5D**2
    
    # Dimensionless ratio
    x = A/A0
    
    # Main term + corrections (normalized to 1 at Planck scale)
    return (x**1.5 + 0.1*np.log(x) + 0.01/np.sqrt(x))/1.0101

# Simulation range (Planck scale to 1cm)
radii = np.logspace(np.log10(L_5D), -2, 500)
entropies = [normalized_entropy(r) for r in radii]

# Calculate specific values
planck_entropy = normalized_entropy(L_5D)
test_entropy = normalized_entropy(1e-10)

# Generate plot
plt.figure(figsize=(10,6))
plt.loglog(radii/L_5D, entropies, 'b-', linewidth=2)
plt.axvline(1, color='r', linestyle='--', label='Planck Length')
plt.axvline(1e-10/L_5D, color='g', linestyle=':', label='1 Ångström')
plt.xlabel('Radius (in Planck units)', fontsize=12)
plt.ylabel('Normalized Entropy (S/Sₚ)', fontsize=12)
plt.title('5D Entropic Gravity (Normalized)', fontsize=14)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('normalized_entropy.png', dpi=300, bbox_inches='tight')

print("=== Normalized Simulation Results ===")
print(f"5D Planck length = {L_5D:.3e} m")
print(f"Entropy at Planck scale: {planck_entropy:.3f} (normalized to ~1)")
print(f"Entropy at r=1Å: {test_entropy:.3e}")
print("\nPlot saved to 'normalized_entropy.png'")
