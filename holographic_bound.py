import numpy as np
import matplotlib.pyplot as plt

# Physical constants (SI units)
h = 6.62607015e-34      # Planck constant [J·s]
hbar = h/(2*np.pi)      # Reduced Planck constant
c = 299792458           # Speed of light [m/s]
G = 6.67430e-11         # Gravitational constant [m³/kg·s²]

# Calculate 5D Planck length (adjusted scaling)
L_P5 = (hbar*G/c**3)**0.5  # Standard Planck length
L_P5_effective = L_P5 * 1e8  # Adjusted for 5D (temporary scaling factor)

def entropy_5D(radius):
    """Improved entropy calculation with proper scaling"""
    A = 2*np.pi**2 * radius**3  # 5D surface area
    A0 = 4 * L_P5_effective**2   # Scaled reference area
    
    # Main term + corrections (with dimensionless coefficients)
    ratio = A/A0
    return (ratio**1.5 + 0.1*np.log(abs(ratio)) + 0.01/ratio**0.5)

# Simulation parameters (physical range)
radii = np.logspace(-35, -5, 500)  # From Planck scale to 10μm
entropies = [entropy_5D(r) for r in radii]

# Plot with physical annotations
plt.figure(figsize=(12,7))
plt.loglog(radii, entropies, 'b-', linewidth=2)
plt.axvline(L_P5, color='r', linestyle='--', label='Planck Length')
plt.axvline(1e-10, color='g', linestyle=':', label='Test Radius (1Å)')

plt.xlabel('Radius (m)', fontsize=12)
plt.ylabel('S/k$_B$', fontsize=12)
plt.title('5D Black Hole Entropy (Improved Scaling)', fontsize=14)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('entropy_simulation.png', dpi=300, bbox_inches='tight')

# Physical validation
test_radius = 1e-10
test_entropy = entropy_5D(test_radius)
planck_entropy = entropy_5D(L_P5)

print("\n=== Simulation Results ===")
print(f"5D Planck length: {L_P5_effective:.3e} m (effective)")
print(f"Entropy at Planck scale: {planck_entropy:.3f} (should be ~1)")
print(f"Entropy at r={test_radius:.1e}m: {test_entropy:.3e}")
print("\nPlot saved to 'entropy_simulation.png'")
