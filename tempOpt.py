import numpy as np 
import matplotlib.pyplot as plt


"""
Santiago Acosta.


La idea de esta tarea es encontrar la temperatura óptima de emisión estelar utilizando el método de la sección aurea 
con $\Epsilon = 50K$. Esto es, encontrar el $T$ que maximiza la función 

$$
    P(T) = \sigma T^4 \exp(-T/T_0) ( 1 - \exp(-(h\nu)/(k_B T))
$$

con $\sigma = 5.67*10^-8 Wm^-2 K^-4$, $T_0 = 10000 L$ y $h\nu / k_B = 5000 K$. 

El intervalo de $T \in [3000, 50000] K$
"""

# Definición de parámetros 
sigma = 5.67e-8
T0 = 10000
hnu_kB = 5000

# Definición de la función
def P(T):
    return sigma * T**4 * np.exp(-T / T0) * (1. - np.exp(-hnu_kB / T))]

# Definición de la búsqueda por sección aurea
