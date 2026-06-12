from numpy._core.multiarray import scalar
import pyPLUTO as pp 
import pyvista as pv
import numpy as np
import time

# Esto se cambia según lo que se necesite
num_frames = 7

# Para dar un buen valor de treshold
D = pp.Load(nout='all', var=["rho"])
data = D.rho[7]
threshold_val = (np.max(data)- 1.5*np.min(data)) / 2.0


time_series_data = [D.rho[i] for i in range(num_frames)]


#===========================================
#    ME LO DIO QWEN

# 2. Setup plotter and base grid ONCE
plotter = pv.Plotter()
grid = pv.ImageData()
grid.dimensions = np.array(time_series_data[0].shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)

grid["plot_data"] = time_series_data[0].flatten(order="F")

# 3. Use a simple class to hold mutable state for the callback
class AppState:
    def __init__(self):
        self.frame_idx = 0
        self.actor = None

state = AppState()

# 4. Define the function that runs when you press the key
def next_frame():
    # A. Advance the frame index (loops back to 0 at the end)
    state.frame_idx = (state.frame_idx + 1) % num_frames
    
    # B. Update the underlying grid data
    grid["plot_data"] = time_series_data[state.frame_idx].flatten(order="F")
    
    # C. Re-apply the threshold to get new geometry
    new_mesh = grid.threshold(value=threshold_val, scalars="plot_data")
    
    # D. Save the current camera state
    cam_pos = plotter.camera.position
    cam_focal = plotter.camera.focal_point
    cam_up = plotter.camera.up
    
    # E. Remove old actor and add the new one (Guarantees colormap/opacity persist)
    plotter.remove_actor(state.actor)
    state.actor = plotter.add_mesh(
        new_mesh,
        scalars="plot_data",
        cmap="viridis",
        opacity=0.8,      # Explicitly set
        show_edges=False
    )
    
    # F. Restore the camera state instantly (prevents jumping)
    plotter.camera.position = cam_pos
    plotter.camera.focal_point = cam_focal
    plotter.camera.up = cam_up
    
    # G. Force a redraw of the window
    plotter.render()

# 5. Add the initial actor to our state
initial_mesh = grid.threshold(value=threshold_val, scalars="plot_data")
state.actor = plotter.add_mesh(
    initial_mesh,
    scalars="plot_data",
    cmap="viridis",
    opacity=0.8,
    show_edges=False
)

# 6. Bind the key presses to the next_frame function
# You can bind multiple keys to the same function!
plotter.add_key_event('n', next_frame)
plotter.add_key_event('Right', next_frame)


# 7. Show the plot. 
# This starts PyVista's native event loop, which listens for your key presses.
plotter.show()
