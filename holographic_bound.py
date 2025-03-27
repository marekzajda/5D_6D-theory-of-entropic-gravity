
"""
Holographic Bound in 5D Entropic Gravity
Implements Eq. (5.12) from Zenodo 10.5281/zenodo.15085762.
"""
import numpy as np
from constants import G5D, DEFAULT_RADIUS  # Shared constants

def entropy_5D_bh(radius, G5D=G5D):
    """
    Computes entropy of a 5D black hole: S = A_5D^(3/2) / (4 G5D).
    
    Args:
        radius (float): Schwarzschild radius in meters.
        G5D (float): 5D gravitational constant (default: 1e-42 m³/kg·s²).
    """
    A_5D = 2 * np.pi**2 * radius**3  # 5D surface area
    return (A_5D**1.5) / (4 * G5D)

def test_entropy_5D_bh():
    """Test against known value from Zenodo paper (Section 5.3)."""
    assert np.isclose(entropy_5D_bh(DEFAULT_RADIUS), 1.2e-31, rtol=1e-3)

if __name__ == "__main__":
    test_entropy_5D_bh()
    print(f"S({DEFAULT_RADIUS} m) = {entropy_5D_bh(DEFAULT_RADIUS):.2e}")
