import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(PHI, x, y):
    plt.imshow(PHI, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap of PHI')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.xticks(ticks=np.arange(len(x)), labels=x)
    plt.yticks(ticks=np.arange(len(y)), labels=y)
    plt.show()

# Example usage:
# PHI = np.random.rand(10, 10)  # Replace with your actual data
# x = np.linspace(0, 1, 10)     # Replace with your actual x values
# y = np.linspace(0, 1, 10)     # Replace with your actual y values
# plot_heatmap(PHI, x, y)
