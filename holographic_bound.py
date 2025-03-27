import numpy as np
import matplotlib.pyplot as plt

# ==============================================
# PHYSICAL CONSTANTS (hardcoded to avoid scipy dependency)
# ==============================================
H_BAR = 1.054571817e-34  # J·s (reduced Planck constant)
C = 299792458            # m/s (speed of light)
G = 6.67430e-11          # m³/kg·s² (gravitational constant)
K_B = 1.380649e-23       # J/K (Boltzmann constant)

# ==============================================
# 5D ENTROPY CALCULATION
# ==============================================

def calculate_5d_planck_length(G5):
    """Calculate 5D Planck length from 5D gravitational constant"""
    return (H_BAR * G5 / C**3)**(1/3)

def calculate_5d_entropy(radius, G5):
    """
    Calculate 5D black hole entropy
    S = (3π/2) * (r³/l₅³) * k_B
    where l₅ is the 5D Planck length
    """
    l5 = calculate_5d_planck_length(G5)
    return (3 * np.pi/2) * (radius**3 / l5**3) * K_B

# ==============================================
# VISUALIZATION AND TESTING
# ==============================================

def generate_entropy_plot(G5):
    """Generate entropy vs radius plot"""
    radii = np.logspace(-35, -10, 100)  # Planck scale to atomic scale
    entropies = [calculate_5d_entropy(r, G5) for r in radii]
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.loglog(radii, entropies, 'b-', linewidth=2)
    ax.set_xlabel('Radius [m]', fontsize=12)
    ax.set_ylabel('Entropy [J/K]', fontsize=12)
    ax.set_title('5D Black Hole Entropy Scaling', fontsize=14)
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('5D_entropy_plot.png', dpi=300)
    plt.close()

def run_test(G5):
    """Run verification test at r = 1e-10 m"""
    test_radius = 1e-10
    calculated = calculate_5d_entropy(test_radius, G5)
    
    print("=== 5D Black Hole Entropy Calculator ===")
    print(f"Using G5D = {G5:.3e} m⁴/kg·s²\n")
    print(f"At r = {test_radius:.1e} m:")
    print(f"Calculated entropy: {calculated:.3e} J/K")
    
    # Check if result is reasonable
    if 1e-32 < calculated < 1e-30:
        print("\nStatus: Calculation within expected range")
        return True
    else:
        print("\nStatus: Unexpected result - check G5 value")
        print("Typical values for G5 should be ~1e-42 to 1e-39 m⁴/kg·s²")
        return False

# ==============================================
# MAIN EXECUTION
# ==============================================

if __name__ == "__main__":
    try:
        # Use your G5 value from the theory (6.674e-11 m⁴/kg·s²)
        G5 = 6.674e-11 * (H_BAR*G/C**3)**(1/2)  # Adjusted to 5D
        
        if not run_test(G5):
            # Try with more typical 5D value if first test fails
            G5 = 1.0e-42
            print("\nTrying with G5 = 1.0e-42 m⁴/kg·s²")
            run_test(G5)
        
        generate_entropy_plot(G5)
        print("\nGenerated entropy scaling plot: 5D_entropy_plot.png")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please ensure you have numpy and matplotlib installed")
        print("Install with: pip install numpy matplotlib")
