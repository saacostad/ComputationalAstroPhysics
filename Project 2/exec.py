import numpy as np
from solve import Sweep
import matplotlib.pyplot as plt

# ==============================================================================
# 2. PARAMETERS & GRID
# ==============================================================================
gamma = 5. / 3.
Nx, Ny = 256, 512
Lx, Ly = 1.0, 2.0
dx, dy = Lx / Nx, Ly / Ny
CFL = 0.4
t, t_end = 0.0, 5.0
g_x, g_y = 0.0, -1.0
Bx_const = 0.0001  # Try 0.0 for pure hydro, or 0.1 to see MHD tension effects!

x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# ==============================================================================
# 3. INITIAL CONDITIONS
# ==============================================================================
rho = np.ones((Nx, Ny))
rho[:, Ny//2:] = 5.0  # Heavy fluid on top
rho[:, :Ny//2] = 1.0  # Light fluid on bottom

# Discrete Hydrostatic Integration (dp/dy = rho * g_y)
p0 = 2.5
p = np.zeros((Nx, Ny))
p[:, 0] = p0
for j in range(1, Ny):
    p[:, j] = p[:, j-1] + 0.5 * (rho[:, j] + rho[:, j-1]) * g_y * dy

vx = np.zeros_like(rho)
vy = np.zeros_like(rho)
vz = np.zeros_like(rho)

# Proper RT Perturbation
# amplitude = 1.0
# sigma = 0.5
# vy -= amplitude * (1.0 + np.cos(2.0 * np.pi * X / Lx)) * np.exp(-((Y - Ly/2.0)**2) / (2.0 * sigma**2))


X0 = X - Lx/2.0
Y0 = Y - Ly/2.0

r = np.sqrt(X0**2 + Y0**2)
up = np.exp(-(5. * r)**2)
down = 10. * (np.cosh(10. * Y0))**2

vy -= up/down

plt.imshow(vy)
plt.show()

Bx = Bx_const * np.ones_like(rho)
By = np.zeros_like(rho)
Bz = np.zeros_like(rho)

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def primitive_to_conserved(rho, vx, vy, vz, p, Bx, By, Bz):
    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    magnetic = 0.5 * (Bx**2 + By**2 + Bz**2)
    E = p / (gamma - 1.0) + kinetic + magnetic
    U = np.zeros((Nx, Ny, 7))
    U[..., 0] = rho; U[..., 1] = rho * vx; U[..., 2] = rho * vy; U[..., 3] = rho * vz
    U[..., 4] = E; U[..., 5] = By; U[..., 6] = Bz
    return U

def get_pressure(U):
    rho = U[..., 0]
    vx = U[..., 1] / rho; vy = U[..., 2] / rho; vz = U[..., 3] / rho
    Bsq = Bx_const**2 + U[..., 5]**2 + U[..., 6]**2
    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    return (gamma - 1.0) * (U[..., 4] - kinetic - 0.5 * Bsq)

def apply_boundary_conditions(U):
    # Periódico en X (índices 0 y -1)
    U[0, :, :] = U[-2, :, :]
    U[-1, :, :] = U[1, :, :]

    # Fondo (y=0, índice 0)
    U[:, 0, :] = U[:, 1, :]          # copiar todas las variables
    U[:, 0, 2] *= -1.0               # v_y antisimétrico
    # No modificar B_y (U[:,0,5]) -> simétrico (copiado)

    # Techo (y=Ny-1, índice -1)
    U[:, -1, :] = U[:, -2, :]
    U[:, -1, 2] *= -1.0
    # B_y simétrico

    return U

def get_gravity_source(U):
    src = np.zeros_like(U)
    rho = U[..., 0]; vx = U[..., 1] / rho; vy = U[..., 2] / rho
    src[..., 1] = rho * g_x; src[..., 2] = rho * g_y
    src[..., 4] = rho * (vx * g_x + vy * g_y)
    return src

def enforce_floors(U):
    U[..., 0] = np.maximum(U[..., 0], 1e-5)   # densidad mínima más alta
    rho = U[..., 0]
    vx = U[..., 1]/rho
    vy = U[..., 2]/rho
    vz = U[..., 3]/rho
    Bsq = Bx_const**2 + U[..., 5]**2 + U[..., 6]**2
    p = (gamma - 1.0) * (U[..., 4] - 0.5 * rho * (vx**2 + vy**2 + vz**2) - 0.5 * Bsq)
    p = np.maximum(p, 1e-6)   # presión mínima más alta
    # Recalcular energía si es necesario (opcional, pero mejora estabilidad)
    U[..., 4] = p/(gamma-1.0) + 0.5*rho*(vx**2+vy**2+vz**2) + 0.5*Bsq
    return U

def compute_rhs(U):
    rho = U[..., 0]; vx = U[..., 1]/rho; vy = U[..., 2]/rho; vz = U[..., 3]/rho
    sweep = Sweep(rho, vx, vy, vz, get_pressure(U), Bx_const, U[..., 5], U[..., 6], gamma)
    sweep.solve()
    rhs = np.zeros_like(U)
    rhs[1:-1, 1:-1, :] -= (sweep.flux_x[1:, 1:-1, :] - sweep.flux_x[:-1, 1:-1, :]) / dx
    rhs[1:-1, 1:-1, :] -= (sweep.flux_y[1:-1, 1:, :] - sweep.flux_y[1:-1, :-1, :]) / dy


    cmax_val = max(np.max(sweep.cmax_x), np.max(sweep.cmax_y))
    if cmax_val > 1e6:
        print(f"WARNING: cmax = {cmax_val:.2e} at t = {t:.4f}")

    return rhs, cmax_val

    return rhs, max(np.max(sweep.cmax_x), np.max(sweep.cmax_y))

# ==============================================================================
# 5. MAIN RK2 LOOP
# ==============================================================================
U = primitive_to_conserved(rho, vx, vy, vz, p, Bx, By, Bz)
U = apply_boundary_conditions(U)

step = 0
while t < t_end:
    rhs_1, cmax = compute_rhs(U)
    src_1 = get_gravity_source(U)
    
    dt = CFL * min(dx, dy) / cmax
    if t + dt > t_end: dt = t_end - t

    # RK2 Stage 1
    U1 = U + dt * (rhs_1 + src_1)
    U1 = enforce_floors(U1)
    U1 = apply_boundary_conditions(U1)

    # RK2 Stage 2
    rhs_2, _ = compute_rhs(U1)
    src_2 = get_gravity_source(U1)
    
    U_new = 0.5 * U + 0.5 * U1 + 0.5 * dt * (rhs_2 + src_2)
    U_new = enforce_floors(U_new)
    U = U_new

    t += dt; step += 1

    if step % 20 == 0:
        plt.figure(figsize=(6, 10))
        im = plt.imshow(U[..., 0].T, origin='lower', extent=[0, Lx, 0, Ly], aspect='auto', cmap='inferno')
        plt.colorbar(im, label=r'$\rho$')
        plt.title(f'Density at t = {t:.3f}')
        plt.savefig(f"frames/rho_{step:05d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved frame {step} | t = {t:.4f} | dt = {dt:.2e}")
