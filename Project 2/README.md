# Pre-entrega: lo que tengo hasta el sábado 13 de junio

Hay dos partes fundamentales: la simulación propia y los códigos en PLUTO. 

## PLUTO

Primero, en vez de utilizar PLUTO, me cambié a `gPLUTO`, que es el mismo proyecto pero que eprmite utilizar la GPU para hacer los cómputos, permitiendo velocidades de simulación rapidísimas.

`gPLUTO` funciona igual que `PLUTO`, lo único que realmente hay que considerar es el compilador. En mi caso, aprovechando my GPU, uso `nvc++`, el cual se puede elegir desde el mismo `setup.py` del proyecto. Sin embargo, dejo el makefile usado en el repo.

### 2 dimensiones

En el caso de 2 dimensiones, el principal efecto que quiero mostrar es cómo un campo magnético paralelo a la interface cambia el comportamiento de las __inestabilidades de Taylor-Raylegith__, donde primero, difunde enormemente la vorticidad (inestabilidades de Kelvin-Helmotz) a campos $B_x$ relativamente pequeños. A medida que se aumenta $B_x$, se pierden las inestabilidades KH y se puede ver cómo el dedo TR encuentra "mayor resistencia" al bajar. 

Una forma intuitiva de ver esto es, como la "masa" (pues en realidad es una densidad de carga) de arriba cae perpendicular a las líneas de campo magnético en la interface, el $\vec B$ las hace rotar alrededor de la interface. Entre mayor sea el $B_x$, mayor este "torque", por lo tanto, le es más difícil al "fluido pesado" caer. 

Por último, cuando el $B_x \ge B_c$, directamente no hay formación del dedo RT, esto pues el campo magnético es lo suficientemente fuerte como para estabilizar el fluido.

En esta carpeta están los códigos para correr las simulaciones con `gPLUTO`. Hay 4 carpetas:0B, 01Bc 08Bc y 1Bc, para los respectivos valores del $B_x \propto B_c$.


### 3 dimensiones

En 3 dimensiones la cosa cambia un poco, pues ahora no importa qué tan grande sea el campo magnético que nosotros pongamos, siempre se van a presentar inestabilidades. 
Esto es algo que se explica en el paper de `PLUTO`, pero que voy a explicar nuevamente el el documento/notebook que haga al respecto. 

Lo que importa aquí es que siempre se presentan estas inestabilidades RT sin importar el valor de $B_x$, pero al campo magnético en $x$ difunde el dedo RT en dirección $x$, cosa que se ve al aumentar el campo magnético.

## Simulación propia

Para hacer la simulación en 2 dimensiones, se tienen que resolver las ecuaciones de MHD. La forma más usual de hacerlo es a través de la PDE para el vector de variables conservadas, la cual representa las 4 (5 considerando $\Nabla \cdot \vec B$) del MHD de una manera linealizada [hablaré más a detalle en el notebook].

La ecuación a resolver entonces tiene la forma

$$
\frac{\partial U}{\partial t} = - \Nabla(F) + S
$$

donde $ U = (\rho, \rho v_x, \rho v_y, \rho v_z, B_x, B_y, B_z, E)$ y $F$ representa los flujos (salen directamente de las ecuaciones del MHD). El vector $S$ son fuentes, n este caso, "gravedad" (que la interpreto mejor como un campo eléctrico en dirección $-y$), y un término adicional $S_{Powell}$, que sirve para "limpiar la divergencia del campo magnético".

Para resolver esta PDE, utilizo Runge-Kutta 2. Sin embargo, debido a la naturaleza del sistema (estamos modelando un fluido que aparte es magnétioco), hay que utiiziar algún esquema upwind. En este caso, elegí moderlar $HLLD$ según el paper original de Miyoshi-Kusano (2005), pero "generalizándolo" a 2 dimensiones. 
Lo que busca este esquema HLLD es calcular los flujos $F$ en cada interface (límites de las celdas) según la velocidad de las ondas que se estén propagando en el medio. 
En este caso, este esquema cuenta con 3 tipos principales de ondas: magnetosónicas, que usualmente son las más rápidas; alfvén, que vienen del MHD; y de contacto, las usuales. 
A partir de la estimación de las velocidades de estas ondas, se puede elegir qué flujo es el que siente la interface de la celda, y con ello considerar más efectos físicos.

Debido a razones que desconozco (pensaba que usar RK2 y HLLD iba a ser lo suficientemente no-difusivo para incluso ver inestabilidades KH), la vorticidad que sale de mis simulaciones no es mucha. 
Sin embargo, ellas muestran perfectamente el resultado principal del comportamiento del dedo a diferentes valores de $B_x$.

## Consideraciones

Los resultados en 2 dimensiones se encuentran en el repo, puesto que son simulaciones livianas y los resultados son, también, livianos. 

Los resultados en 3 dimensiones aún no sé si los pueda subir como tal, pues, guardando pocos time-stamps el peso de estos datos es del orden de los gigas (7, por ejemplo, pesan 2.25Gb). 

## TO DO

Falta realizar el notebook donde explique la teoría, sobre todo, cómo funciona el HLLD y, si es posible, las inestabilidades RT como tal.
