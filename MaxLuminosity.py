import numpy as np 


"""

La idea de esta tarea es encontrar el punto en donde se maximiza la luminosisdad de un Disco de Acreción. En general, lo que queremos encontrar 
son los valores de r y theta que maximicen 

L(r, theta) = r^2 sin(theta) (1+cos(theta)) e^{-r}

[considerando de una vez que r_0 = 1]. Lo anterior usando descenso del gradiente. 

En este caso, como buscamos maximizar una función, podemos "subir el gradiente" con la función misma, o "bajar el gradiente" de la función * -1
"""

# Creamos las condiciones iniciales según lo que me tocó 
r0 =  1 
theta0 = np.pi / 6.0


# Creamos la función como tal
def L(r, theta):
    return r**2 * np.sin(theta) * (1.0 + np.cos(theta)) * np.exp(-r)


"""
Algo muy importante a tener en cuenta aquí es que esta función está en coordenadas polares. Uno entonces podría pensar que se debe usar el 
gradiente en las respectivas coordenadas polares, PERO NO, pues este algorirmo trata las variables como si estuvieran en un espacio euclidiano, 
lo que se puede ver desde la regla de actualización del paso, que es meramente lineal. Por ello, lo mejor es expresar esta función en cartesianas, 
para que esta regla aproveche completamente el gradiente cartesiano.
"""

# Llama la función a maximizar pero en cartesianas.
def Lcart(x, y):
    r = np.sqrt(x**2 + y **2)
    theta = np.arctan2(y, x)]
    return L(r, theta)

