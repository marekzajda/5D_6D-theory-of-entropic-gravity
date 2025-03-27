"""  
Holographic Bound in 5D Entropic Gravity  
Implements calculations from Chapter 5 of Zenodo 10.5281/zenodo.15085762.  
"""  
import numpy as np  

def entropy_5D_bh(radius, G5D=1e-42):  
    """Computes entropy of a 5D black hole [Eq. 5.12]."""  
    A_5D = 2 * np.pi**2 * radius**3  # 5D surface area  
    return (A_5D**1.5) / (4 * G5D)  

if __name__ == "__main__":  
    # Example from Section 5.3  
    print("S(1e-10 m) =", entropy_5D_bh(1e-10))  

assert np.isclose(entropy_5D_bh(1e-10), 1.2e-31, rtol=1e-3)  # Example from paper  
