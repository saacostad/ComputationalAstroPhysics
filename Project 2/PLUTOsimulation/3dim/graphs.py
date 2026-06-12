import pyPLUTO as pp 
import pyvista as pv
import numpy as np

D = pp.Load(nout='all', var=["rho", "vx2", "Bx1"])

data = D.rho[7]

threshold = np.mean(data) 

# 2. Create a 3D Uniform Grid
grid = pv.ImageData()
grid.dimensions = np.array(data.shape) + 1  # Dimensions are nodes, not cells
grid.origin = (0, 0, 0)                     # Bottom left corner
grid.spacing = (1, 1, 1)                    # Distance between nodes (adjust to your scale)

# 3. Add the scalar data to the grid (VTK expects 1D arrays in Fortran order)
grid["scalars"] = data.flatten(order="F")

# 4. Apply the threshold 
# This completely removes cells below the threshold, making them effectively alpha=0
# and vastly improving rendering performance.
thresholded_grid = grid.threshold(threshold)

# 5. Plot the result interactively
# You can adjust opacity, colormap, and show edges as needed
thresholded_grid.plot(
    cmap="viridis", 
    opacity=0.8, 
    show_edges=False,
)


