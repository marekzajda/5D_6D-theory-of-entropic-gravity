import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, k, c

# Fundamental constants
PLANCK_LENGTH = np.sqrt(hbar*1.0545718e-34/(c*6.67430e-11))  # More precise calculation
BOLTZMANN = k
G5 = 6.708e-39  # 5D gravitational constant

def calculate_5d_entropy(radius):
    """Calculate 5D black hole entropy with proper units"""
    # Bekenstein-Hawking entropy for 5D (S ~ r^3)
    entropy = (np.pi**2 * radius**3) / (2 * PLANCK_LENGTH**3) * BOLTZMANN
    return entropy

def generate_entropy_plot():
    radii = np.logspace(-35, -10, 100)  # From Planck scale to atomic scale
    entropies = [calculate_5d_entropy(r) for r in radii]
    
    plt.figure(figsize=(10,6))
    plt.loglog(radii, entropies, 'b-', linewidth=2)
    plt.xlabel('Radius [m]', fontsize=12)
    plt.ylabel('Entropy [J/K]', fontsize=12)
    plt.title('5D Black Hole Entropy Scaling', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.savefig('5D_entropy_plot.png', dpi=300)
    plt.close()

def run_tests():
    """Verify calculations against known theoretical values"""
    test_radius = 1e-10  # Atomic scale
    calculated = calculate_5d_entropy(test_radius)
    expected = 1.38e-23 * (test_radius/PLANCK_LENGTH)**3  # Theoretical expectation
    
    print(f"\n=== 5D Black Hole Entropy Calculator ===")
    print(f"Reference value at r={test_radius:.1e} m: {calculated:.2e} J/K")
    
    # Tolerance check (1% difference)
    if np.abs(calculated - expected)/expected < 0.01:
        print("\nTest Status: Test passed successfully")
        return True
    else:
        print(f"\nTest Status: Test failed (Expected ~{expected:.2e} J/K)")
        return False

if __name__ == "__main__":
    generate_entropy_plot()
    test_result = run_tests()
    
    # Additional verification points
    verification_radii = {
        'Planck scale': PLANCK_LENGTH,
        'Quantum scale': 1e-15,
        'Atomic scale': 1e-10
    }
    
    print("\nVerification points:")
    for name, r in verification_radii.items():
        print(f"{name+':':<15} r = {r:.1e} m → S = {calculate_5d_entropy(r):.2e} J/K")
    
    if not test_result:
        print("\nDebugging info:")
        print(f"- Planck length used: {PLANCK_LENGTH:.2e} m")
        print(f"- Boltzmann constant: {BOLTZMANN:.2e} J/K")
        print("- Consider checking:")
        print("  1) Dimensional analysis of entropy formula")
        print("  2) Numerical precision in calculations")
        print("  3) Physical constants values")
