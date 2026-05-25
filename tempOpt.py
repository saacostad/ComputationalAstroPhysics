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
    return sigma * T**4 * np.exp(-T / T0) * (1. - np.exp(-hnu_kB / T))

# Definición de la búsqueda por sección aurea
def golden_section_search(f, a, b, tol=50e-3):
    gr = (np.sqrt(5) + 1) / 2  # golden ratio
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (a + b) / 2, f((a + b) / 2)

# Intervalo de T
T_min = 3000
T_max = 50000

# Búsqueda por sección aurea para encontrar el máximo de P(T)
max_T, max_P = golden_section_search(P, T_min, T_max)

print(f"Temperatura óptima: {max_T} K")
print(f"P(T) en la temperatura óptima: {max_P}")

# Creación de gráfica
T_values = np.linspace(T_min, T_max, 1000)
P_values = P(T_values)

plt.plot(T_values, P_values)
plt.xlabel('Temperatura (K)')
plt.ylabel('P(T)')
plt.title('Función de emisión estelar')
plt.grid(True)
plt.show()
