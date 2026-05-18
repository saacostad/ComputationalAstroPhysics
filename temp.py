import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(PHI, x, y):
    # Format x and y labels to at most 2 decimal places
    formatted_x = [f"{xi:.2f}" for xi in x]
    formatted_y = [f"{yi:.2f}" for yi in y]

    # Determine the indices of labels to display
    num_labels = min(10, len(formatted_x))  # Display up to 10 labels
    x_indices = np.linspace(0, len(x) - 1, num_labels).astype(int)
    y_indices = np.linspace(0, len(y) - 1, num_labels).astype(int)

    # Create a figure with subplots
    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(PHI, cmap='viridis', interpolation='nearest')
    ax.colorbar(im)
    ax.set_title('Heatmap of PHI')
    ax.set_xlabel('r')
    ax.set_ylabel('z')

    # Set x and y ticks with formatted labels
    ax.set_xticks(x_indices)
    ax.set_yticks(y_indices)
    ax.set_xticklabels([formatted_x[i] for i in x_indices])
    ax.set_yticklabels([formatted_y[i] for i in y_indices])

    # Plot the contour lines
    contours = ax.contour(PHI, levels=10, colors='black')

    plt.show()

# Example usage:
# PHI = np.random.rand(10, 10)  # Replace with your actual data
# x = np.linspace(0, 1, 10)     # Replace with your actual x values
# y = np.linspace(0, 1, 10)     # Replace with your actual y values
# plot_heatmap(PHI, x, y)
