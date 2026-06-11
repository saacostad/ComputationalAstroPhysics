import numpy as np 
from solve import hlld, create_U 
import matplotlib.pyplot as plt 


# ========== PARÁMETROS DE LA GRILLA ==========

# Distancia física
Lx = 1.0
Ly = 2.0 

# Divisiones
Nx = 128
Ny = 256

# dzs 
dx = Lx / Nx 
dy = Ly / Ny

# Grilla
x = np.linspace(-Lx/2., Lx/2., Nx)
y = np.linspace(-Ly/2., Ly/2., Ny)

X, Y = np.meshgrid(x, y)

ng = 2          # Celdas fantasma

# El CFL para ver los saltos de tiempo
CFL = 0.4

# Grilla de valores 
rho = np.zeros_like(X)

u = np.zeros_like(X)
v = np.zeros_like(X)
w = np.zeros_like(X)

bx = np.zeros_like(X)
by = np.zeros_like(X)
bz = np.zeros_like(X)

po = np.zeros_like(X)       # A partir de este calculamos la energía

# ========== PARÁMETROS FÍSICOS ==========
g = -10.0           # Gravedad
gamma = 5. / 3.     # Const. adiabática 

# Magnitud del campo magnético inicial
B0 = 0.0


# Tiempo a simular
t_max = 15.0

# Cada cuántos pasos guardar
pas_save = 100

# ========== CONDICIONES INICIALES ==========

# Densidades iniciales
rho = np.where(Y < 0, 1.0, 5.0)

# Campos magnéticos iniciales: un campo grande en Bx hará que se estabilicen las inestabilidades
bx = np.full_like(X, B0) 
# by = np.full_like(X, B0) if axis == 'y' else np.zeros_like(X)

# Presión inicial: me lo robé de algún lado pero se me olvidó de dónde :c 
p_ref = 100.0   
p0 = p_ref + rho * g * Y

# Energía inicial
Btot2 = bx**2 + by**2 + bz**2   
E = (p0) / (gamma - 1.0) + 0.5 * (rho*(u**2 + v**2 + w**2) + Btot2)


# ----- INESTABILIDAD INICIAL -----
r = np.sqrt(X**2 + Y**2)
up = np.exp(-(5. * r)**2) 
down = 10. * (np.cosh(10. * Y))**2
v -= up/down




# =============================================
# ========== SIMULACIÓN DEL SISTEMA ===========
# =============================================

# Cálculo del RHS: Vectores de flujo F y G, Source gravitacional, Powell 8-waves para limpiar divergencia
def compute_RHS(UxL, UxR, UyL, UyR, rho, v, U):
    
    F = list()
    G = list()
    cmax_list = list()
    
    # Obtenemos flujos HLLD en eje x
    for i in range(Nx - 1): 
        Fx, Px, cmaxX = hlld(UxL[:, :, i].T, UxR[:, :, i].T, B0, gamma = gamma, axis = 'x')
        F.append(Fx) 

        cmax_list.append(np.max(cmaxX))

    # Ahora en eje y
    for j in range(Ny - 1):
        # HACK: para no ponerme a molestar mucho en HLLD, implementé hlld 1dim y le paso el eje y
        # rotado 90 grados pero cambiando las variables del eje x a y como corresponde
        Gy, Py, cmaxY = hlld(UyL[:, :, j].T, UyR[:, :, j].T, B0, gamma = gamma, axis = 'y')
        Gy_unrot = Gy.copy()
        Gy_unrot[:, 1] = -Gy[:, 2]   # x-moment0
        Gy_unrot[:, 2] =  Gy[:, 1]   # y-moment0
        Gy_unrot[:, 4] = -Gy[:, 5]   # Bx
        Gy_unrot[:, 5] =  Gy[:, 4]   # By
        G.append(Gy_unrot)

        cmax_list.append(np.max(cmaxY))
    
    # Por el chistesito de arriba, tengo que transponer los flujos para que tengan las mismas dimensiones
    Fy_internal = np.array(np.transpose(G, (1, 0, 2)) )
    Fx_internal = np.array(F)

    # Cogemos sólo las celdas que no son GHOST y hacemos un roll a un índice para calcular el dF/dx
    Fx_right = Fx_internal[ng : Nx - ng, ng : Ny - ng, :]
    Fx_left  = Fx_internal[ng - 1 : Nx - ng - 1, ng : Ny - ng, :]

    Fy_top   = Fy_internal[ng : Nx - ng, ng : Ny - ng, :]
    Fy_bottom= Fy_internal[ng : Nx - ng, ng - 1 : Ny - ng - 1, :]

    # Calculamos el cmax para saber el dt
    cmax = np.max(cmax_list)
    
    # Aquí obtengo los flujos en los centros de cada celda 
    dFdx = (Fx_right - Fx_left) / dx
    dFdy = (Fy_top - Fy_bottom) / dy 

    # ========== Creación del source gravitacional: m*a para las velocidades y F*vel para E ==========
    S_grav = np.zeros_like(dFdx)
    
    # Sólo tenemos gravedad en y, entonces no toca modifica mucho
    S_grav[:, :, 2] = rho.T[ng:Nx - ng, ng:Ny-ng] * g 
    S_grav[:, :, 7] = rho.T[ng:Nx - ng, ng:Ny-ng] * g * v.T[ng:Nx - ng, ng:Ny-ng] 
    
    # ========== Estabilización de divergencia: usamos Powell 8-waves para limpiar la divergencia ==========
    S_Power = calculate_powerl(U, dFdx)

    return -dFdx - dFdy + S_grav + S_Power, cmax



# Me la robé de una IA por facilidad, condiciones de frontera periódicas en x y reflectivas en y
def apply_boundary_conditions(rho, u, v, w, bx, by, bz, p, ng=2):
    """
    Applies Periodic BC in X (Left/Right) and Reflective BC in Y (Bottom/Top).
    Assumes arrays have shape (Ny, Nx) where Ny is number of rows (y), Nx is number of columns (x).
    """
    
    # =========================================================================
    # 1. X-DIRECTION: PERIODIC (Left/Right boundaries, modify columns -> axis 1)
    # =========================================================================
    # Left ghost cells (first ng columns) copy from rightmost physical cells
    rho[:, :ng]   = rho[:, -2*ng:-ng]
    u[:, :ng]     = u[:, -2*ng:-ng]
    v[:, :ng]     = v[:, -2*ng:-ng]
    w[:, :ng]     = w[:, -2*ng:-ng]
    bx[:, :ng]    = bx[:, -2*ng:-ng]
    by[:, :ng]    = by[:, -2*ng:-ng]
    bz[:, :ng]    = bz[:, -2*ng:-ng]
    p[:, :ng]     = p[:, -2*ng:-ng]

    # Right ghost cells (last ng columns) copy from leftmost physical cells
    rho[:, -ng:]  = rho[:, ng:2*ng]
    u[:, -ng:]    = u[:, ng:2*ng]
    v[:, -ng:]    = v[:, ng:2*ng]
    w[:, -ng:]    = w[:, ng:2*ng]
    bx[:, -ng:]   = bx[:, ng:2*ng]
    by[:, -ng:]   = by[:, ng:2*ng]
    bz[:, -ng:]   = bz[:, ng:2*ng]
    p[:, -ng:]    = p[:, ng:2*ng]

    # =========================================================================
    # 2. Y-DIRECTION: REFLECTIVE (Bottom/Top boundaries, modify rows -> axis 0)
    # =========================================================================
    # Bottom ghost cells (first ng rows) mirror the bottom physical cells
    rho[:ng, :]   = rho[ng:2*ng, :]
    u[:ng, :]     = u[ng:2*ng, :]       # Tangential (vx): keep sign
    v[:ng, :]     = -v[ng:2*ng, :]      # Normal (vy): FLIP SIGN
    w[:ng, :]     = w[ng:2*ng, :]       # Tangential (vz): keep sign
    bx[:ng, :]    = bx[ng:2*ng, :]      # Tangential (Bx): keep sign
    by[:ng, :]    = -by[ng:2*ng, :]     # Normal (By): FLIP SIGN
    bz[:ng, :]    = bz[ng:2*ng, :]      # Tangential (Bz): keep sign
    p[:ng, :]     = p[ng:2*ng, :]

    # Top ghost cells (last ng rows) mirror the top physical cells
    rho[-ng:, :]  = rho[-2*ng:-ng, :]
    u[-ng:, :]    = u[-2*ng:-ng, :]     # Tangential (vx): keep sign
    v[-ng:, :]    = -v[-2*ng:-ng, :]    # Normal (vy): FLIP SIGN
    w[-ng:, :]    = w[-2*ng:-ng, :]     # Tangential (vz): keep sign
    bx[-ng:, :]   = bx[-2*ng:-ng, :]    # Tangential (Bx): keep sign
    by[-ng:, :]   = -by[-2*ng:-ng, :]   # Normal (By): FLIP SIGN
    bz[-ng:, :]   = bz[-2*ng:-ng, :]    # Tangential (Bz): keep sign
    p[-ng:, :]    = p[-2*ng:-ng, :]



def calculate_powerl(U, dFdx):
    # Permite recuperar div(B) = 0 al aplicar un término de fuente que modifica los campos magnéticos 
    # para disminuir la divergencia entre ellos 

    # Obtenemos las variables
    Bx_lab = U[:,:,4].T 
    By_lab = U[:,:,5].T
    rho_lab = U[:,:,0].T

    # Calculamos divergencia con diferencias centradas
    divB = np.zeros_like(rho_lab)
    divB[1:-1, 1:-1] = (Bx_lab[1:-1, 2:] - Bx_lab[1:-1, :-2])/(2*dx) + (By_lab[2:, 1:-1] - By_lab[:-2, 1:-1])/(2*dy)

    # Recordamos el resultado en la región que nos interesa (sin ghost cells)
    divB_phys = divB[ng:Ny-ng, ng:Nx-ng].T  

    # Obtenemos las variables en las celdas físicas
    bx_phys = U[ng:Nx-ng, ng:Ny-ng, 4]      
    by_phys = U[ng:Nx-ng, ng:Ny-ng, 5]
    bz_phys = U[ng:Nx-ng, ng:Ny-ng, 6]
    ux_phys = U[ng:Nx-ng, ng:Ny-ng, 1] / U[ng:Nx-ng, ng:Ny-ng, 0]   # Para las velocidades, dividimos sobre rho
    uy_phys = U[ng:Nx-ng, ng:Ny-ng, 2] / U[ng:Nx-ng, ng:Ny-ng, 0]
    uz_phys = U[ng:Nx-ng, ng:Ny-ng, 3] / U[ng:Nx-ng, ng:Ny-ng, 0]

    # Array de Powell source: -div(B) * {B para v, v para B y \vec v * \vec B para E}
    S_powell = np.zeros_like(dFdx)   
    S_powell[:,:,1] -= divB_phys * bx_phys
    S_powell[:,:,2] -= divB_phys * by_phys
    S_powell[:,:,3] -= divB_phys * bz_phys
    S_powell[:,:,4] -= divB_phys * ux_phys
    S_powell[:,:,5] -= divB_phys * uy_phys
    S_powell[:,:,6] -= divB_phys * uz_phys
    S_powell[:,:,7] -= divB_phys * (ux_phys*bx_phys + uy_phys*by_phys + uz_phys*bz_phys)

    return S_powell



# ======================================
# ========== BUCLE PRINCIPAL ===========
# ======================================

t = 0.              # Tiempo inicial
n = -1              # Step inicial
while t < t_max:
    
    # Empezamos aplicando boundary conditions
    apply_boundary_conditions(rho, u, v, w, bx, by, bz, E, ng=2) 
    
    # Creamos los arreglos de cantidades conservadas a izquierda y derecha (para calcular los flujos con HLLD)
    UxL = create_U(rho, u, v, w, bx, by, bz, E, gamma, side = "left", axis = "x")
    UxR = create_U(rho, u, v, w, bx, by, bz, E, gamma, side = "right", axis = "x")
    UyL = create_U(rho, u, v, w, bx, by, bz, E, gamma, side = "left", axis = "y")
    UyR = create_U(rho, u, v, w, bx, by, bz, E, gamma, side = "right", axis = "y")
    
    # El arreglo de conserved variables en el centro de la celda (Para aplicar el RK2)
    U = np.transpose(create_U(rho, u, v, w, bx, by, bz, E, gamma, side = "total", axis = "total"), (2, 1, 0))
    
    # ----- Primer parso RK2 -----
    # Obtenemos el RHS: Flujos - grav - Powell
    RHS, cmax = compute_RHS(UxL, UxR, UyL, UyR, rho, v, U)
    dt = CFL * dx/cmax

    # Damos el primer paso intermedio
    U[ng : Nx - ng, ng : Ny - ng, :] += dt * RHS
    
    # Actualizamos las variables físicas
    rho_s = U[:, :, 0].T
    u_s = U[:, :, 1].T / rho_s
    v_s = U[:, :, 2].T / rho_s
    w_s = U[:, :, 3].T / rho_s
    bx_s = U[:, :, 4].T
    by_s = U[:, :, 5].T
    bz_s = U[:, :, 6].T
    E_s = U[:, :, 7].T

    # ----- Segundo paso RK2 -----

    # Boundary conditions otra vez
    apply_boundary_conditions(rho_s, u_s, v_s, w_s, bx_s, by_s, bz_s, E_s, ng=2) 

    # Obtenemos los vectores de variables conservadas
    UxL = create_U(rho_s, u_s, v_s, w_s, bx_s, by_s, bz_s, E_s, gamma, side = "left", axis = "x")
    UxR = create_U(rho_s, u_s, v_s, w_s, bx_s, by_s, bz_s, E_s, gamma, side = "right", axis = "x")
    UyL = create_U(rho_s, u_s, v_s, w_s, bx_s, by_s, bz_s, E_s, gamma, side = "left", axis = "y")
    UyR = create_U(rho_s, u_s, v_s, w_s, bx_s, by_s, bz_s, E_s, gamma, side = "right", axis = "y")
    
    U = np.transpose(create_U(rho_s, u_s, v_s, w_s, bx_s, by_s, bz_s, E_s, gamma, side = "total", axis = "total"), (2, 1, 0))

    # RHS
    RHS_s, cmax = compute_RHS(UxL, UxR, UyL, UyR, rho_s, v_s, U)

    # Actualización final de RK2
    U[ng : Nx - ng, ng : Ny - ng, :] += (dt / 2.0) * (RHS_s - RHS) 

    rho = U[:, :, 0].T
    u = U[:, :, 1].T / rho
    v = U[:, :, 2].T / rho
    w = U[:, :, 3].T / rho
    bx = U[:, :, 4].T
    by = U[:, :, 5].T
    bz = U[:, :, 6].T
    E = U[:, :, 7].T
    
    # Aumentamos el tiempo y paso
    t += dt
    n += 1

    # Guardamos la imagen
    if n % pas_save == 0:
        
        plt.figure(figsize=(6, 10))
        
        path = "frames_noB"

        im = plt.imshow(rho, origin='lower', aspect='auto', cmap='inferno', vmin=0.0, vmax=6.0)
        plt.colorbar(im, label=r'$\rho$')
        plt.title(f'Density at t = {t:.3f}')
        plt.savefig(f"{path}/rho/rho_{n:05d}.png", dpi=150, bbox_inches='tight')
        plt.close()

        im = plt.imshow(bx, origin='lower', aspect='auto', cmap='inferno', vmin=4.5, vmax=5.5)
        plt.colorbar(im, label=r'$\rho$')
        plt.title(f'bx at t = {t:.3f}')
        plt.savefig(f"{path}/bx/bx_{n:05d}.png", dpi=150, bbox_inches='tight')
        plt.close()

        im = plt.imshow(by, origin='lower', aspect='auto', cmap='inferno', vmin=-0.01, vmax=0.01)
        plt.colorbar(im, label=r'$\rho$')
        plt.title(f'by at t = {t:.3f}')
        plt.savefig(f"{path}/by/by_{n:05d}.png", dpi=150, bbox_inches='tight')
        plt.close()

        im = plt.imshow(bz, origin='lower', aspect='auto', cmap='inferno', vmin=4.0, vmax=6.0)
        plt.colorbar(im, label=r'$\rho$')
        plt.title(f'bz at t = {t:.3f}')
        plt.savefig(f"{path}/bz/bz_{n:05d}.png", dpi=150, bbox_inches='tight')
        plt.close()


        # plt.show()
        print(f"Saved frame {n} | t = {t:.4f} | dt = {dt:.2e}") 
