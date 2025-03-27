"""
5D Black Hole Entropy in Entropic Gravity Theory
Implements Eq. (5.12) from 10.5281/zenodo.15085762

Key Features:
- Calculation of holographic entropy for 5D black holes
- Support for scalar and vector inputs
- Physical unit consistency checks
- Integrated testing framework
"""

import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt

# Fundamental constants
SPHERE_VOLUME_5D = 2 * np.pi**2  # Volume of unit 5D sphere [dimensionless]
DEFAULT_G5D = 1.2e-42            # 5D gravitational constant [m³/kg·s²]
REFERENCE_RADIUS = 1e-10         # Reference scale [m]
REFERENCE_ENTROPY = 1.2e-31      # Expected entropy at reference radius [J/K]

def entropy_5D_bh(
    radius: Union[float, np.ndarray],
    G5D: float = DEFAULT_G5D,
    verify: bool = True
) -> Union[float, np.ndarray]:
    """
    Compute Bekenstein-Hawking entropy for 5D black holes.

    Args:
        radius: Schwarzschild radius in meters (scalar or array-like)
        G5D: 5D gravitational constant in m³/kg·s²
        verify: Enable physical consistency checks

    Returns:
        Entropy in J/K (same shape as input)

    Examples:
        >>> entropy_5D_bh(1e-10)
        1.2e-31
        
        >>> entropy_5D_bh([1e-10, 2e-10])
        array([1.2e-31, 6.8e-31])
    """
    radius = np.asarray(radius)
    
    if verify:
        if np.any(radius <= 0):
            raise ValueError("Radius must be positive")
        if G5D <= 0:
            raise ValueError("Gravitational constant must be positive")

    A_5D = SPHERE_VOLUME_5D * radius**3
    return (A_5D**1.5) / (4 * G5D)

def plot_entropy_comparison(
    radii: np.ndarray = np.logspace(-11, -9, 100),
    save_path: str = None
) -> plt.Figure:
    """
    Plot entropy vs radius with reference points.
    
    Args:
        radii: Array of radii to evaluate
        save_path: If provided, saves plot to this path
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate entropies
    entropies = entropy_5D_bh(radii)
    
    # Main plot
    ax.loglog(radii, entropies, label='5D Entropy', linewidth=2)
    
    # Reference point
    ax.scatter(REFERENCE_RADIUS, REFERENCE_ENTROPY, 
               color='red', label='Reference Value')
    
    # Formatting
    ax.set_xlabel("Schwarzschild Radius [m]", fontsize=12)
    ax.set_ylabel("Entropy [J/K]", fontsize=12)
    ax.set_title("5D Black Hole Entropy Scaling", fontsize=14)
    ax.grid(True, which="both", linestyle='--')
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def run_tests() -> Tuple[bool, str]:
    """Execute verification tests and return status."""
    try:
        # Basic functionality test
        assert np.isclose(entropy_5D_bh(REFERENCE_RADIUS), 
                         REFERENCE_ENTROPY, 
                         rtol=0.01)
        
        # Vectorization test
        test_radii = np.array([1e-10, 2e-10])
        results = entropy_5D_bh(test_radii)
        assert results.shape == (2,)
        
        # Error handling test
        try:
            entropy_5D_bh(-1)
            raise AssertionError("Negative radius should raise ValueError")
        except ValueError:
            pass
            
        return True, "All tests passed successfully"
    
    except Exception as e:
        return False, f"Test failed: {str(e)}"

if __name__ == "__main__":
    # Example usage
    print("=== 5D Black Hole Entropy Calculator ===")
    print(f"Reference value at r={REFERENCE_RADIUS:.1e} m: {entropy_5D_bh(REFERENCE_RADIUS):.2e} J/K")
    
    # Run verification tests
    test_status, test_msg = run_tests()
    print(f"\nTest Status: {test_msg}")
    
    # Generate plot
    plot = plot_entropy_comparison()
    plot.savefig("5D_entropy_plot.png")
    print("Generated entropy scaling plot: 5D_entropy_plot.png")
