import numpy as np
import matplotlib.pyplot as plt

def create_plots(F, x, y, nx=6, ny=5, method="Gauss-Seidel"):
    # Para que se vea bonito, primero formateo los x e y
    formatted_x = [f"{xi:.2f}" for xi in x]
    formatted_y = [f"{yi:.2f}" for yi in y]

    # Determinar los labels a poner
    num_labels_x = min(nx, len(formatted_x))
    x_indices = np.linspace(0, len(x) - 1, num_labels_x).astype(int)

    num_labels_y = min(ny, len(formatted_y))  # Corrected this line to use ny instead of nx
    y_indices = np.linspace(0, len(y) - 1, num_labels_y).astype(int)

    # Creamos el heatmap
    fig, ax = plt.subplots()

    im = ax.imshow(F.T, cmap='viridis')
    
    # Create a colorbar for the heatmap
    cbar = fig.colorbar(im, orientation='horizontal', label=r"$\Phi$ [U. Arbitrarias]")
    
    ax.set_title(f'Heatmap usando {method}')
    
    ax.set_xlabel('r [U.A]')
    ax.set_ylabel('z [U.A]')
        
    # Ponemos los ejes correctamente
    ax.set_xticks(x_indices)
    ax.set_yticks(y_indices)
    ax.set_xticklabels([formatted_x[i] for i in x_indices])
    ax.set_yticklabels([formatted_y[i] for i in y_indices])

    # Plot the contour lines
    contours = ax.contour(F.T, levels=10, colors='black')

    plt.show()

# Example usage:
# F = np.random.rand(10, 10)  # Replace with your actual data
# x = np.linspace(0, 1, 10)     # Replace with your actual x values
# y = np.linspace(0, 1, 10)     # Replace with your actual y values
# create_plots(F, x, y)
