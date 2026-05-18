import numpy as np
import matplotlib.pyplot as plt

def create_plots(F, x, y, nx = 6, ny = 5, method = "Gauss-Seidel"):

    # Para que se vea bonito, primero formateo los x e y
    formatted_x = [f"{xi:.2f}" for xi in x]
    formatted_y = [f"{yi:.2f}" for yi in y]

    # Determinar los labels a poner
    num_labels_x = min(nx, len(formatted_x))  
    x_indices = np.linspace(0, len(x) - 1, num_labels_x).astype(int)

    num_labels_y = min(nx, len(formatted_x)) 
    y_indices = np.linspace(0, len(y) - 1, num_labels_y).astype(int)

    # Creamos el heatmap
    plt.imshow(F.T, cmap='viridis')
    plt.colorbar()
    
    plt.title(f'Heatmap usando {method}')
    
    plt.xlabel('r [U.A]')
    plt.ylabel('z [U.A]')
        
    # Ponemos los ejes correctamente
    plt.xticks(ticks=x_indices, labels=[formatted_x[i] for i in x_indices])
    plt.yticks(ticks=y_indices, labels=[formatted_y[i] for i in y_indices])

    plt.show()
