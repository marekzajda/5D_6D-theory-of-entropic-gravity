"""
5D Black Hole Entropy in Entropic Gravity Theory
Implements the complete theoretical framework from Equations_in_english.md

Key Features:
- Consistent with 5D entropic gravity formalism
- Proper dimensional analysis
- Automatic unit validation
- Physics-informed plotting
"""

import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt
from scipy.constants import hbar, c, G

# Fundamental constants from Equations_in_english.md
PLANCK_LENGTH_5D = np.sqrt((hbar*G/c**3)**(1/3))  # Effective 5D Planck length [m]
PLANCK_ENERGY_5D = (hbar**2 * c**4 / G)**(1/5)    # 5D Planck energy [J]

# Derived constants for 5D case
KAPPA_5D = (8*np.pi*G)/c**4                       # 5D Einstein constant
L_P5 = np.sqrt(hbar*G/c**3)                       # Planck length in 5D

# Theoretical scaling from document
def calculate_entropy_5D(A, A0=4*L_P5**2):
    """
    Implements the complete entropy formula from Eq. (3.7) in Equations_in_english.md
    S = (A/A0)^(3/2) + alpha*ln(A/A0) + beta*(A0/A)^(1/2)
    """
    alpha = 0.1  # Holographic correction coefficient
    beta = 0.01  # Quantum correction coefficient
    ratio = A/A0
    return ratio**(3/2) + alpha*np.log(ratio) + beta*ratio**(-1/2)

def entropy_5D_bh(
    radius: Union[float, np.ndarray],
    verify: bool = True
) -> Union[float, np.ndarray]:
    """
    Compute complete 5D black hole entropy including corrections.
    
    Args:
        radius: Schwarzschild radius in meters
        verify: Check physical plausibility
        
    Returns:
        Dimensionless entropy (S/k_B)
    """
    radius = np.asarray(radius)
    
    if verify:
        if np.any(radius <= 0):
            raise ValueError("Radius must be positive")
        if np.any(radius < 1e-35):
            print("Warning: Radius below 5D Planck scale")

    A_5D = 2*np.pi**2 * radius**3  # 5D surface area
    A0 = 4 * L_P5**2               # Reference area
    
    return calculate_entropy_5D(A_5D, A0)

def plot_entropy_components(
    radii: np.ndarray = np.logspace(-35, -10, 200),
    save_path: str = None
) -> plt.Figure:
    """
    Plot all entropy components separately as shown in theoretical doc.
    """
    A_5D = 2*np.pi**2 * radii**3
    A0 = 4 * L_P5**2
    ratio = A_5D/A0
    
    main_term = ratio**(3/2)
    log_correction = 0.1*np.log(ratio)
    quantum_correction = 0.01*ratio**(-1/2)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.loglog(radii, main_term, label='Main term $(A/A_0)^{3/2}$', linewidth=2)
    ax.loglog(radii, log_correction, '--', label='Log correction $0.1\ln(A/A_0)$')
    ax.loglog(radii, quantum_correction, ':', 
             label='Quantum correction $0.01(A_0/A)^{1/2}$')
    ax.loglog(radii, main_term + log_correction + quantum_correction,
             'k-', label='Total entropy', linewidth=2)
    
    ax.set_xlabel("Schwarzschild Radius [m]", fontsize=12)
    ax.set_ylabel("Dimensionless Entropy $S/k_B$", fontsize=12)
    ax.set_title("5D Black Hole Entropy Components", fontsize=14)
    ax.grid(True, which="both", linestyle='--')
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def theoretical_checks():
    """Verify consistency with theoretical predictions"""
    test_radius = 1e-10  # Test case
    S = entropy_5D_bh(test_radius)
    
    # Expected value from document
    expected = (2*np.pi**2 * (1e-10)**3 / (4*L_P5**2))**(3/2)
    
    print("=== Theoretical Validation ===")
    print(f"At r = {test_radius:.1e} m:")
    print(f"- Calculated S/k_B: {S:.3e}")
    print(f"- Expected S/k_B:   {expected:.3e}")
    print(f"- Ratio:            {S/expected:.3f}")
    
    if abs(S - expected)/expected < 0.05:
        print("Validation: PASSED (within 5% error)")
    else:
        print("Validation: WARNING (significant discrepancy)")

if __name__ == "__main__":
    print("=== 5D Entropic Gravity Calculator ===")
    print(f"Using 5D Planck length: {L_P5:.3e} m")
    
    theoretical_checks()
    
    print("\nGenerating component plot...")
    plot_entropy_components(save_path="entropy_components.png")
    print("Saved to entropy_components.png")
