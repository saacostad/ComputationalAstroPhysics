import numpy as np

def calculate_eps(PHI_old, PHI_new):
    """ Cálculo del error en una iteración """
    return np.max(100.0 * np.abs( (PHI_new - PHI_old) / PHI_new ))
