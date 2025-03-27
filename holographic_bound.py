import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, k, c, G

# ==============================================
# Corrected 5D Entropy Calculation
# ==============================================

# Fundamental constants
PLANCK_LENGTH = np.sqrt(hbar*G/c**3)  # ~1.616e-35 m
PLANCK_MASS = np.sqrt(hbar*c/G)       # ~2.176e-8 kg
BOLTZMANN = k                         # 1.381e-23 J/K

# 5D gravitational constant (adjusted for your theory)
G5 = 6.674e-11 * (PLANCK_LENGTH)      # m⁴/kg·s² in 5D

def calculate_5d_entropy(radius):
    """
    Corrected 5D entropy formula:
    S = (3π/2) (r³/l₅³) k_B
    where l₅ is the 5D Planck length
    """
    # Calculate 5D Planck length (l₅ = (ħG₅/c³)^(1/3)
    l5 = (hbar*G5/c**3)**(1/3)
    
    # 5D entropy formula
    entropy = (3*np.pi/2) * (radius**3 / l5**3) * BOLTZMANN
    return entropy

# ==============================================
# Verification and Plotting
# ==============================================

def verify_calculation():
    test_radius = 1e-10  # Atomic scale
    
    # Calculate values
    calculated = calculate_5d_entropy(test_radius)
    theoretical = 1.200e-31  # Your expected value
    
    # Compute error
    error = np.abs(calculated - theoretical)/theoretical * 100
    
    print("=== 5D Black Hole Entropy Calculator ===")
    print(f"Using G5D = {G5:.3e} m⁴/kg·s²\n")
    print(f"At r = {test_radius:.1e} m:")
    print(f"- Calculated entropy: {calculated:.3e} J/K")
    print(f"- Theoretical value:  {theoretical:.3e} J/K")
    print(f"- Relative error:     {error:.2f}%\n")
    
    # Tolerance check (10% as shown in your output)
    if error < 10:
        print("Status: Calculation matches theory within 10% tolerance")
        return True
    else:
        print("Status: Significant discrepancy detected")
        print("Possible issues:")
        print("1. Incorrect G5D value")
        print("2. Different entropy formula expected")
        print("3. Unit conversion problems")
        return False

def generate_plot():
    radii = np.logspace(-35, -10, 100)  # Planck scale to atomic scale
    entropies = [calculate_5d_entropy(r) for r in radii]
    
    plt.figure(figsize=(10,6))
    plt.loglog(radii, entropies, 'b-', linewidth=2)
    plt.xlabel('Radius [m]', fontsize=12)
    plt.ylabel('Entropy [J/K]', fontsize=12)
    plt.title('5D Black Hole Entropy Scaling', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.savefig('5D_entropy_plot.png', dpi=300)
    plt.close()
    print("\nGenerating entropy scaling plot...")
    print("Saved to 5D_entropy_plot.png")

# ==============================================
# Main Execution
# ==============================================

if __name__ == "__main__":
    verify_calculation()
    generate_plot()
    
    # Additional diagnostic info
    l5 = (hbar*G5/c**3)**(1/3)
    print("\nDiagnostic Info:")
    print(f"5D Planck length (l₅): {l5:.3e} m")
    print(f"5D Planck mass (m₅): {(hbar*c/G5)**(1/4):.3e} kg")
    print(f"Test ratio (r/l₅): {1e-10/l5:.3e}")
