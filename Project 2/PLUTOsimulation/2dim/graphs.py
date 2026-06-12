import pyPLUTO as pp 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Leemos los datos de interés
D = pp.Load(nout='all', var=["rho", "vx2", "Bx1"])

data_to_plot = D.rho

# Creamos las figuras 
fig, ax = plt.subplots()

# índice de los datos
index = 0

# Creamos la primera data
data = data_to_plot[index]

im = ax.imshow(np.rot90(data), cmap = 'inferno', animated = True)
ax.set_title("Density")


# Cramos la regla de update 
def update(frame):
    global data_to_plot

    new_data = data_to_plot[frame]
    im.set_array(np.rot90(new_data))
    return [im]


# Creamos la animación 
ani = anim.FuncAnimation(fig, update, frames=100, interval=25, blit=True)

# 5. Save the animation as an MP4
print("Saving MP4...")
# Note: If this fails, ensure ffmpeg is installed and in your system's PATH
ani.save('noise_animation.mp4', writer='ffmpeg', fps=15, dpi=100)

plt.show()



