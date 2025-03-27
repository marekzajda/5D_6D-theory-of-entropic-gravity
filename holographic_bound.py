"""
5D Black Hole Entropy in Entropic Gravity Theory
Complete implementation with all corrections
"""

import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt

# Fundamental constants (SI units)
H = 6.62607015e-34 / (2*np.pi)  # Reduced Planck constant [J·s]
C = 299792458                    # Speed of light [m/s]
G = 6.67430e-11                  # Gravitational constant [m³/kg·s²]

# Derived 5D Planck length
L_P5 = np.sqrt(H*G/C**3)        # 5D Planck length [m]

def calculate_entropy_5D(A, A0=4*L_P5**2):
    """
    Complete entropy formula with corrections:
    S = (A/A0)^(3/2) + α·ln(A/A0) + β·(A0/A)^(1/2)
    """
    alpha = 0.1  # Holographic correction
    beta = 0.01  # Quantum correction
    ratio = A/A0
    return ratio**(3/2) + alpha*np.log(ratio) + beta*ratio**(-1/2)

def entropy_5D_bh(radius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute dimensionless entropy (S/k_B) for 5D black hole
    
    Args:
        radius: Schwarzschild radius in meters
        
    Returns:
        Dimensionless entropy
    """
    radius = np.asarray(radius)
    if np.any(radius <= 0):
        raise ValueError("Radius must be positive")
    
    A_5D = 2*np.pi**2 * radius**3  # 5D surface area
    A0 = 4 * L_P5**2               # Reference area
    return calculate_entropy_5D(A_5D, A0)

def plot_entropy_components(radii=np.logspace(-35, -10, 200), save_path=None):
    """
    Plot all entropy components with proper formatting
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    A_5D = 2*np.pi**2 * radii**3
    A0 = 4 * L_P5**2
    ratio = A_5D/A0
    
    components = {
        'Main term': ratio**(3/2),
        'Log correction': 0.1*np.log(ratio),
        'Quantum correction': 0.01*ratio**(-1/2)
    }
    
    for label, values in components.items():
        ax.loglog(radii, values, label=label)
    
    ax.loglog(radii, sum(components.values()), 'k-', 
             label='Total entropy', linewidth=2)
    
    ax.set_xlabel("Schwarzschild Radius [m]", fontsize=12)
    ax.set_ylabel("Dimensionless Entropy $S/k_B$", fontsize=12)
    ax.set_title("5D Black Hole Entropy Components", fontsize=14)
    ax.grid(True, which="both", linestyle='--')
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    print("=== 5D Entropic Gravity Calculator ===")
    print(f"5D Planck length: {L_P5:.3e} m")
    
    # Example calculation
    test_radius = 1e-10
    print(f"\nAt r = {test_radius:.1e} m:")
    print(f"Dimensionless entropy: {entropy_5D_bh(test_radius):.3e}")
    
    # Generate plot
    print("\nGenerating plot...")
    plot_entropy_components(save_path="entropy_components.png")
    print("Saved to entropy_components.png")
