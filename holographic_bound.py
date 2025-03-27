def holographic_entropy(A_5D, G5D=1e-42):  
    """Computes S_{5D} = A_5D^{3/2} / (4 G_{5D}) [Eq. 5.12]"""  
    return (A_5D**1.5) / (4 * G5D)  
