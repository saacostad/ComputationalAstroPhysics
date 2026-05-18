import numpy as np

def calculate_eps(PHI_old, PHI_new):
    """ Cálculo del error en una iteración """
    # Replace zeros in PHI_new with a small non-zero value to avoid division by zero
    PHI_new_safe = np.where(PHI_new == 0, 1e-10, PHI_new)
    
    return np.max(100.0 * np.abs((PHI_new - PHI_old) / PHI_new_safe))
