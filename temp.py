import numpy as np
import matplotlib.pyplot as plt

def draw_radial_line(image, center, radius, thickness=1):
    """
    Draws a radial line on the image from the given center with the specified radius and thickness.
    
    :param image: 2D NumPy array representing the image.
    :param center: Tuple (x, y) representing the center of the radial line.
    :param radius: Integer representing the length of the radial line.
    :param thickness: Integer representing the thickness of the radial line.
    """
    x, y = center
    for i in range(-thickness, thickness + 1):
        for j in range(-thickness, thickness + 1):
            if (i**2 + j**2) <= radius**2:
                image[x + i, y + j] = 0  # Assuming the image is grayscale and we want to draw black lines

# Example usage
if __name__ == '__main__':
    # Create a sample image
    image = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Define the center and radius of the radial line
    center = (50, 50)
    radius = 30
    
    # Draw the radial line
    draw_radial_line(image, center, radius)
    
    # Display the image
    plt.imshow(image, cmap='gray')
    plt.show()
