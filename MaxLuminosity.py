import numpy as np 
from matplotlib import pyplot as plt

# Qué tan precisa queremos nuestra estimación
treshold = 1e-2

"""

La idea de esta tarea es encontrar el punto en donde se maximiza la luminosisdad de un Disco de Acreción. En general, lo que queremos encontrar 
son los valores de r y theta que maximicen 

L(r, theta) = r^2 sin(theta) (1+cos(theta)) e^{-r}

[considerando de una vez que r_0 = 1]. Lo anterior usando descenso del gradiente. 

En este caso, como buscamos maximizar una función, podemos "subir el gradiente" con la función misma, o "bajar el gradiente" de la función * -1
"""

# Creamos las condiciones iniciales según lo que me tocó 
r0 =  1.0 
theta0 = np.pi / 6.0


# Convertimos a cartesianas por cuestiones
x0 = r0 * np.cos(theta0)
y0 = r0 * np.sin(theta0)


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
    theta = np.arctan2(y, x)
    return L(r, theta)

# Aquí vamos a implementar el gradiente usando diferencias finitas. En este caso, 
# utilizaremos una primera derivada de 5 puntos para hacer el gradiente preciso
def find_gradient(x, y, h = 1e-5, func = Lcart):
    
    # Implementamos diferencias finitas 
    def deriv(z, f):
        return ( -f(z + (2.0 * h)) + 8.0*f(z + h) - 8.0*f(z - h) + f(z - 2.0*h)) / (12.0 * h)
    
    # Obtenemos las derivadas parciales en cada eje dejando quieto el eje independiente
    xgrad = deriv(x, lambda z: func(z, y))
    ygrad = deriv(y, lambda z: func(x, z))

    # Devolvemos el gradiente
    return np.array([xgrad, ygrad])


"""
Ahora, algo importante es cómo dar el paso. Primeramente, podemos simplemente dar pasos muy chiquitos en la dirección positiva del gradiente (recordemos queremos
maximizar la función)
"""

# Tamaño constante del paso 
step_size = 1e-1

# Inicializamos el gradiente 
grad = find_gradient(x0, y0)

# Creamos listas con los valores de x e y 
xs = [x0] 
ys = [y0]

# Contador para revisar cositas 
count = 1

# Realizamos el bucle
while np.linalg.norm(grad) > treshold:
   
    # Damos el paso
    x0 += step_size * grad[0]
    y0 += step_size * grad[1]
    
    # Agregamos estos valores al histórico
    xs.append(x0)
    ys.append(y0)

    # Calculamos el gradiente 
    grad = find_gradient(x0, y0)

    # Imprimimos información
    print(f"Paso {count}: x = {x0} | y = {y0} | f(x, y) = {Lcart(x0, y0)} | grad = {grad}")
    count += 1 

print()
print(f"Estimación final: x = {x0} | y = {y0} | f(x, y) = {Lcart(x0, y0)} | grad = {grad}")



""" 
Ahora, para ver este comportamiento, vamos a plotear los valores 
"""

def create_graph(points_x, points_y, limsx = 3.0, limsy = 3.0, n = 100):

    # Creamos una grilla
    x = np.linspace(-limsx, limsx, n)
    y = np.linspace(-limsy, limsy, n)
    X, Y = np.meshgrid(x, y)
    Z = Lcart(X, Y)

    # Desempaquetamos nuestros puntos y los evaluamos
    px, py = points_x, points_y
    pz = [Lcart(x, y) for x, y in zip(px, py)]

    # Creamos la gráfica inicial
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    # La función
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, zorder=0)

    # Hacemos el track de los puntos
    ax.plot(px, py, pz, color="red", linewidth=2, zorder=5)
    ax.scatter(px, py, pz, color="red", s=20, zorder=6)
    
    # Labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("L(x, y)")
    plt.tight_layout()
    plt.show()

create_graph(xs, ys)

