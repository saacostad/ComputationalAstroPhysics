"""
En este script vamos a desarrollar el modelo de HLLD para poder resolver el sistema y seguir viendo los vórtices bonitos
"""

import numpy as np
import math 
from numba import njit  # Para optimizar 


def prepare_states(rho_L, vn_L, vt_L, vb_L, p_L, Bn_L, Bt_L, Bb_L,   # Variables primitivas izquierdas
                   rho_R, vn_R, vt_R, vb_R, p_R, Bn_R, Bt_R, Bb_R,   # Variables primitivas derechas
                   gamma):
    
    """ 
    Hace el cálculo de las variables necesarias para la clase Sweep [robada de PLUTO]

    rho's -> densidades
    vn's -> velocidad del fluido en la normal de las interfaces
    vt's -> velocidades transversales
    p's -> presión térmica
    Bn's, Bt's y Bb's -> campos magnéticos en las interfaces
    gamma -> índice adiabático]

    -------------------------------------------
    RETORNA:
    vectores de cantidades conservadas u's 
    flujos de estos vectores F's 
    P's -> presiones totales 
    S's -> velocidades de la onda rápida magnetoestática 
    Bn -> Un campo magnético promediado 
    """

    #=========================================================================================================
    # ESTADO IZQUIERDO

    Btot2_L = Bn_L**2 + Bt_L**2 + Bb_L**2   # Magnitud al cuadrado de B (3 componentes)
    P_L = p_L + 0.5 * Btot2_L               # Presión total: presión térmica + magnetoestática 

    # Energía total: p/(\gamma - 1) + \frac{1}{2} \rho v**2 + \frac{1}{2} * B**2
    E_L = (p_L) / (gamma - 1.0) + 0.5 * (rho_L*(vn_L**2 + vt_L**2 + vb_L**2) + Btot2_L)  # PDT: no calculamos v**2 porque no lo volvemos a usar

    
    #----------------------------------------------------------------------------------------------------------
    # Calculamos las variables conservadas y las aplanamos (para poder usar numba a gusto y copiarnos el PLUTO)

    # U = [densidad, 3 * momentos, energía, 2 * campos magnéticos] PDT: ignoramos el B_x = B_n porque su flujo se toma como nulo
    u_L = np.stack((rho_L, rho_L * vn_L, rho_L * vt_L, rho_L * vb_L, E_L, Bt_L, Bb_L), axis = -1)
    
    # F(U) = [\rho*u, \rho u^2 + P - B_x^2, \rho u v - B_x B_y, (E + P)u - B_x *(\vec v * \vec B), \rho u w - B_x B_z, u B_y - v B_x, y B_z - w B_x]
    
    F_L = np.zeros_like(u_L)
    F_L[:, 0] = rho_L * vn_L
    F_L[:, 1] = rho_L * vn_L**2 + P_L - Bn_L**2
    F_L[:, 2] = rho_L * vn_L * vt_L - Bn_L * Bt_L
    F_L[:, 3] = rho_L * vn_L * vb_L - Bn_L * Bb_L
    F_L[:, 4] = (E_L + P_L)*vn_L - Bn_L*(vn_L*Bn_L + vt_L*Bt_L + vb_L*Bb_L)
    F_L[:, 5] = Bt_L * vn_L - vt_L * Bn_L
    F_L[:, 6] = Bb_L * vn_L - vb_L * Bn_L


    #=========================================================================================================
    # ESTADO DERECHO: nos robamos lo que hicimos pero para la derecha

    Btot2_R = Bn_R**2 + Bt_R**2 + Bb_R**2   
    P_R = p_R + 0.5 * Btot2_R               

    E_R = (p_R) / (gamma - 1.0) + 0.5 * (rho_R*(vn_R**2 + vt_R**2 + vb_R**2) + Btot2_R)  

    u_R = np.stack((rho_R, rho_R * vn_R, rho_R * vt_R, rho_R * vb_R, E_R, Bt_R, Bb_R), axis = -1)
    
    F_R = np.zeros_like(u_R)
    F_R[:, 0] = rho_R * vn_R
    F_R[:, 1] = rho_R * vn_R**2 + P_R - Bn_R**2
    F_R[:, 2] = rho_R * vn_R * vt_R - Bn_R * Bt_R
    F_R[:, 3] = rho_R * vn_R * vb_R - Bn_R * Bb_R
    F_R[:, 4] = (E_R + P_R)*vn_R - Bn_R*(vn_R*Bn_R + vt_R*Bt_R + vb_R*Bb_R)
    F_R[:, 5] = Bt_R * vn_R - vt_R * Bn_R
    F_R[:, 6] = Bb_R * vn_R - vb_R * Bn_R


    #=========================================================================================================
    # CÁLCULOS DE LAS VELOCIDADES DE ONDAS
    
    #-----------------------------------------------------------------------------
    # cositas para la IZQUIERDA 

    cs_L = gamma * p_L / rho_L   # Va dentro de la velocidad de magnetoestática 
    B_tot_rho_L = Btot2_L / rho_L # También va dentro
    
    # PDT: el maximum() lo utilizamos para evitar imaginarios que nos puedan dañar la vida entera
    # TODO: esto puede estar mal pero ajá
    # vmag_L = np.sqrt(
    #             0.5 * (cs_L + B_tot_rho_L + np.sqrt( np.maximum(0.0, (cs_L + B_tot_rho_L)**2 - 4. * cs_L * B_tot_rho_L) ))
    #         )
   
    Bn2_rho_L = Bn_L**2 / rho_L
    vmag_L = np.sqrt(0.5 * (cs_L + B_tot_rho_L + np.sqrt( np.maximum(0.0, (cs_L + B_tot_rho_L)**2 - 4. * cs_L * Bn2_rho_L) )))


    #-----------------------------------------------------------------------------
    # cositas para la DERECHA 

    cs_R = gamma * p_R / rho_R   
    B_tot_rho_R = Btot2_R / rho_R 
   
   # TODO: lo mismo de arriba
    # vmag_R = np.sqrt(
    #             0.5 * (cs_R + B_tot_rho_R + np.sqrt( np.maximum(0.0, (cs_R + B_tot_rho_R)**2 - 4. * cs_R * B_tot_rho_R) ))
    #         )

    Bn2_rho_R = Bn_R**2 / rho_R
    vmag_R = np.sqrt(0.5 * (cs_R + B_tot_rho_R + np.sqrt( np.maximum(0.0, (cs_R + B_tot_rho_R)**2 - 4. * cs_R * Bn2_rho_R) )))

    #-----------------------------------------------------------------------------
    # Velocidades magnetoestáticas rápidas

    S_L = np.minimum(vn_L - vmag_L, vn_R - vmag_R)
    S_R = np.maximum(vn_L + vmag_L, vn_R + vmag_R)


    #=============================================================================================================
    # CÁLCULO DEL CAMPO MAGNÉTICO NORMAL: promedio de Roe

    denom = S_R - S_L 
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)  # Evitamos tener cosas en 0 o negativas

    Bn = np.where(np.abs(denom) > 1e-14,
              (S_R * Bn_R - S_L * Bn_L) / denom,
              0.5 * (Bn_L + Bn_R)) 

    # D E V O L V E M O S    T O D O 
    return u_L, u_R, F_L, F_R, P_L, P_R, S_L, S_R, Bn



# =====================================================================================================
#-----------------------------------------------------------------------------------------------------
#               H L L D    C Ó D I G O 

@njit 
def hlld(u_L, u_R, F_L, F_R, P_L, P_R, S_L, S_R, Bn, gamma):
    """
    La parte 'porteada' del código de PLUOT que hace en realidad el HLLD. 
    Sólo funciona en una dimensión, para generalizar a más dimensiones toca hacer el Line Loop.
    
    INPUTS:
    u's -> vector de variables conservadas
    F's -> los flujos de las variables conservadas
    P's -> las presiones totales p. térmica + p. magnética
    S's -> velocidades magnéticas rápidas 
    Bn -> Campo magnético normal (x) promediado usando Roe 
    gamma -> constante adiabática 

    --------------------------------------------------------------

    OUTPUTS:
    F -> el flujo a utilizar: el decidido por HLLD considerando las velocidades de las ondas 
    P -> la presión a utilizar 
    cmax -> velocidad de alfvén máxima, se utiliza para ajustar correctamente el \\Delta t 
    """

    N = u_L.shape[0]   # Número de celdas

    #-----------------------------------
    # Creación de variables a utilizar 
    
    F = np.zeros((N, 7))  # 7 porque ese es el número de variables 
    P = np.zeros(N)
    cmax = np.zeros(N)


    #===============================================================
    #   BUCLE PRINCIPAL 

    for i in range(N):

        # Encontramos la velocidad máxima 
        cmax[i] = max(abs(S_L[i]), abs(S_R[i]))

        #--------------------------------------------------------------------------------------------------------------------
        #   Primera corroboración: si $S_L >= 0$, entonces estamos en un estado de flujo completo izquierdo 
        #                          si $S_R <= 0$, flujo en completo derecho
        #                           ---> Devolvemos los flujos iniciales pero con las presiones que hallamos

        if S_L[i] >= 0.0:
            for nv in range(7): F[i, nv] = F_L[i, nv]
            P[i] = P_L[i]
            continue  
            
        elif S_R[i] <= 0.0:
            for nv in range(7): F[i, nv] = F_R[i, nv]
            P[i] = P_R[i]
            continue  
        
        #-----------------------------------------------------------------------------
        # Si lo anterior no funciona, entonces tenemos que revisar las otras regiones
        #-----------------------------------------------------------------------------

        # Guardamos datos del campo magnético 
        B = Bn[i]
        sB = 1.0 if B > 0.0 else -1.0   # Signo de B 
        B2 = B*B    # B^2 

        # Guardamos las variables localmente para usarlas más cómodamente 
        rho_L = u_L[i, 0]; mx_L = u_L[i, 1]; my_L = u_L[i, 2]; mz_L = u_L[i, 3];
        E_L = u_L[i, 4]; By_L = u_L[i, 5]; Bz_L = u_L[i, 6]
        
        rho_R = u_R[i, 0]; mx_R = u_R[i, 1]; my_R = u_R[i, 2]; mz_R = u_R[i, 3]; 
        E_R = u_R[i, 4]; By_R = u_R[i, 5]; Bz_R = u_R[i, 6]


        # ------------------------------------------------------------------------------------------------
        #   Hallamos las velocidades normales: sólo tenemos info del momento, hay que hallar velocidades

        vx_L = mx_L / rho_L; vy_L = my_L / rho_L; vz_L = mz_L / rho_L; 
        vx_R = mx_R / rho_R; vy_R = my_R / rho_R; vz_R = mz_R / rho_R; 
    
        #===================================================================================================================
        #   VEL. CONTACTO 
        #   SM = \frac{\rho_R u_R(S_R - u_R) - \rho_L u_L(S_L - u_L) + P_L - P_R}{\rho_R(S_R - u_R) - \rho_L (S_L - u_L)}

        par_L = S_L[i] - vx_L; par_R = S_R[i] - vx_R        # Término en paréntesis 

        denom_val = (par_R * rho_R - par_L * rho_L)        # Denominador 
        
        if abs(denom_val) < 1e-14:
            denom_val = 1e-14 if denom_val > 0. else -1e-14

        denom = 1. / denom_val


        # -- Velocidad de contacto --
        SM = (mx_R * par_R - mx_L * par_L - P_R[i] + P_L[i]) * denom


        
        #====================================================================================================================
        #                                                   R E G I Ó N   *
        
        #--------------------------------------------------------------------------------------------------------------------
        #   Arrays para contener la info de estas región
        u_s_L = np.zeros(7)
        u_s_R = np.zeros(7)

        #--------------------------------------------------------------------------------------------------------------------
        #   PRESIÓN EN LA ZONA *: robado de PLUTO, pero son equivalentes a la del paper del HLLD

        pts  = par_R * rho_R * P_L[i] - par_L * rho_L * P_R[i] + rho_L * rho_R * par_R * par_L * (vx_R - vx_L)
        pts *= denom

        #--------------------------------------------------------------------------------------------------------------------
        #   Densidades en la región *:   \rho^* = \rho \frac{S_L - u_L}{S_L - SM}
        
        # Variables de apoyo para evitar ese cálculo apestoso
        par_L_M = (S_L[i] - SM)
        par_R_M = (S_R[i] - SM)

        # Para evitar que explote
        if abs(par_L_M) < 1e-14:
            par_L_M = 1e-14 if par_L_M >= 0 else -1e-14
        if abs(par_R_M) < 1e-14:
            par_R_M = 1e-14 if par_R_M >= 0 else -1e-14

        rho_s_L = rho_L * (par_L / par_L_M)
        rho_s_R = rho_R * (par_R / par_R_M)

        rho_s_L = max(rho_s_L, 1e-8)
        rho_s_R = max(rho_s_R, 1e-8)
        
        # Guardamos la densidad en el arreglo 
        u_s_L[0] = rho_s_L
        u_s_R[0] = rho_s_R

        # Calculamos su raíz cuadrada que usaremos luego para Alfvén 
        sqrt_rho_s_L = math.sqrt(rho_s_L); sqrt_rho_s_R = math.sqrt(rho_s_R)
        
        #--------------------------------------------------------------------------------------------------------------------
        #   Velocidades de Alfvén: S^* = SM - \frac{|B_x|}{\sqrt(\rho^*)}

        S_s_L = SM - abs(B) /sqrt_rho_s_L 
        S_s_R = SM + abs(B) /sqrt_rho_s_R 

        #-----------------------------------------------------------------------------------------------
        # Test de degeneración: lo hace PLUTO en caso tal que las ondas Alfvén (magnéticas) colapsen
        # Si Alfvén colapsa con las ondas rápidas, entonces las ecuaciones explotan. SOL: cambiar a HLL 

        degenerate = 0      # Flag para saber si todo se fue al carajo 
        
        # if (S_s_L - S_L[i]) < 1e-4*(SM - S_L[i]): degenerate = 1
        # if (S_s_R - S_R[i]) < 1e-4*(SM - S_R[i]): degenerate = 1

        if (SM - S_s_L) < 1e-10*(SM - S_L[i]): degenerate = 1
        # if (S_s_R - SM) < 1e-10*(S_R[i] - SM): degenerate = 1
        if (S_R[i] - S_s_R) < 1e-10*(S_R[i] - SM): degenerate = 1
        
        # Denominadores auxiliares 
        denom_L = (rho_L * par_L * par_L_M) - B2
        denom_R = (rho_R * par_R * par_R_M) - B2

        # Comprobamos que este denominador no nos valla a dar problemas
        if abs(denom_L) < 1e-14: denom_L = 1e-14 if denom_L >= 0 else -1e-14
        if abs(denom_R) < 1e-14: denom_R = 1e-14 if denom_R >= 0 else -1e-14



        # Si colapsan las cosas, solucionamos el sistema por HLLC
        if degenerate:
            
            # Estado HLL promedio
            denom_hll = 1.0 / (S_R[i] - S_L[i])
            u_hll = np.zeros(7)
            for nv in range(7):
                u_hll[nv] = (S_R[i] * u_R[i, nv] - S_L[i] * u_L[i, nv] + F_L[i, nv] - F_R[i, nv]) * denom_hll
            # Los estados izquierdo y derecho de la estrella colapsan al mismo valor
            u_s_L[:] = u_hll
            u_s_R[:] = u_hll
            # Velocidades de Alfvén degeneradas
            S_s_L = SM
            S_s_R = SM

        else:
            #--------------------------------------------------------------------------------------------
            # Si no estamos degenerados, tenemos que computar el cambo magnético transversal en la reg
            
            # TODO: si los campos se me alocan, es por esto
            # Factores de escala de los campos magnéticos
            scale_B_L = (rho_L * par_L**2 - B2) / denom_L 
            scale_B_R = (rho_R * par_R**2 - B2) / denom_R 

            # Guardamos los B nuevos en sus respectivos lugares
            
            u_s_L[5] = By_L * scale_B_L
            u_s_L[6] = Bz_L * scale_B_L

            u_s_R[5] = By_R * scale_B_R
            u_s_R[6] = Bz_R * scale_B_R

        #-------------------------------------------------------
        # Calculamos las velocidades transversales en la reg *

        v_s_L = vy_L - (B * By_L * (SM - vx_L) / denom_L)
        w_s_L = vz_L - (B * Bz_L * (SM - vx_L) / denom_L)
        v_s_R = vy_R - (B * By_R * (SM - vx_R) / denom_R)
        w_s_R = vz_R - (B * Bz_R * (SM - vx_R) / denom_R)

        
        #===================================================================================
        #   CALCULAMOS LAS VARIABLES CONSERVADAS: momentos

        u_s_L[1] = rho_s_L * SM; u_s_R[1] = rho_s_R * SM 
        u_s_L[2] = rho_s_L * v_s_L; u_s_R[2] = rho_s_R * v_s_R
        u_s_L[3] = rho_s_L * w_s_L; u_s_R[3] = rho_s_R * w_s_R

        #------------------------------------------------------
        # Calculamos las energías 
        By_s_L = u_s_L[5]; Bz_s_L = u_s_L[6]
        By_s_R = u_s_R[5]; Bz_s_R = u_s_R[6]

        # Izquierda
        big_par_L = (SM*B + v_s_L*By_s_L + w_s_L*Bz_s_L) - (vx_L*B + vy_L*By_L + vz_L*Bz_L)
        E_s_L = ( (S_L[i] - vx_L)*E_L - P_L[i]*vx_L + pts*SM - B*big_par_L ) / (par_L_M)
        u_s_L[4] = E_s_L

        # Derecha
        big_par_R = (SM*B + v_s_R*By_s_R + w_s_R*Bz_s_R) - (vx_R*B + vy_R*By_R + vz_R*Bz_R)
        E_s_R = ( (S_R[i] - vx_R)*E_R - P_R[i]*vx_R + pts*SM - B*big_par_R ) / (par_R_M)
        u_s_R[4] = E_s_R


        #====================================================================================
        #           COMPROBAMOS CONDICIÓN DE SEGUNDA REGIÓN

        if S_s_L >= 0.:
            for nv in range(7): F[i, nv] = F_L[i, nv] + S_L[i] * (u_s_L[nv] - u_L[i, nv])
            P[i] = pts
            continue

        elif S_s_R <= 0.:
            for nv in range(7): F[i, nv] = F_R[i, nv] + S_R[i] * (u_s_R[nv] - u_R[i, nv])
            P[i] = pts
            continue


        #=====================================================================================
        #       R E G I Ó N    **
       
        # Creamos los arreglos para guardas las cositas
        u_ss_L = np.zeros(7); u_ss_R = np.zeros(7)
       
        
        # La densidad y el primer momento se conservan
        rho_ss_L = u_s_L[0]; rho_ss_R = u_s_R[0]

        u_ss_L[0] = u_s_L[0]; u_ss_R[0] = u_s_R[0]
        u_ss_L[1] = u_s_L[1]; u_ss_R[1] = u_s_R[1]


        #------------------------------------------------------------------------------------
        #   Creamos las velocidades y campos magnéticos

        # sqrts de las densidades auxiliares 
        sqrt_rho_s_L = math.sqrt(rho_s_L); sqrt_rho_s_R = math.sqrt(rho_s_R);

        # Denominadores auxiliares
        denom = sqrt_rho_s_L + sqrt_rho_s_R

        # Velocidades 
        # TODO: mirar los signos porque ajá todo mal
        v_ss = ( v_s_L*sqrt_rho_s_L + v_s_R*sqrt_rho_s_R - sB*(By_s_R - By_s_L) ) / denom
        w_ss = ( w_s_L*sqrt_rho_s_L + w_s_R*sqrt_rho_s_R - sB*(Bz_s_R - Bz_s_L) ) / denom
       
        # Campos magnéticos 
        By_ss = ( By_s_L*sqrt_rho_s_R + By_s_R*sqrt_rho_s_L + (v_s_R - v_s_L)*sqrt_rho_s_L*sqrt_rho_s_R*sB )/denom
        Bz_ss = ( Bz_s_L*sqrt_rho_s_R + Bz_s_R*sqrt_rho_s_L + (w_s_R - w_s_L)*sqrt_rho_s_L*sqrt_rho_s_R*sB )/denom

        
        #------------------------------------------------------------------------------------
        #   Guardamos esto en el vector de variables conservadas
        
        # Momentos transversales
        u_ss_L[2] = rho_ss_L * v_ss; u_ss_R[2] = rho_ss_R * v_ss 
        u_ss_L[3] = rho_ss_L * w_ss; u_ss_R[3] = rho_ss_R * w_ss 

        # Campos magnéticos
        u_ss_L[5] = By_ss; u_ss_R[5] = By_ss 
        u_ss_L[6] = Bz_ss; u_ss_R[6] = Bz_ss

        #------------------------------------------------------------------------------------
        #    Calculamos el salto de energía
        
        B_ss_mag = (B2 + By_ss**2 + Bz_ss**2)  # Magnitud del campo magnético aquí

        first_term = (pts - 0.5*B_ss_mag)/(gamma - 1.)
        second_term = 0.5 * (SM**2 + v_ss**2 + w_ss**2)
        third_term = 0.5 * B_ss_mag

        E_ss_L = first_term + third_term + rho_s_L * second_term 
        E_ss_R = first_term + third_term + rho_s_R * second_term 

        u_ss_L[4] = E_ss_L; u_ss_R[4] = E_ss_R


        #====================================================================================
        #           COMPROBAMOS CONDICIÓN ÚLTIMA REGIÓN

        if SM >= 0.:
            for nv in range(7): F[i, nv] = F_L[i, nv] + S_s_L * (u_ss_L[nv] - u_s_L[nv]) + S_L[i] * (u_s_L[nv] - u_L[i, nv])
            P[i] = pts
            continue

        else:

            for nv in range(7): F[i, nv] = F_R[i, nv] + S_s_R * (u_ss_R[nv] - u_s_R[nv]) + S_R[i] * (u_s_R[nv] - u_R[i, nv])
            P[i] = pts
            continue

    return F, P, cmax





#=============================================================
#   C L A S E    S W E E P 
#=============================================================

class Sweep:
    def __init__(self, rho, vx, vy, vz, p, Bx, By, Bz, gamma):
        """
        Initializes the Sweep structure with 2D grid arrays.
        """
        self.gamma = gamma
        
        # Ensure all inputs are numpy arrays of the same shape
        self.rho = np.asarray(rho)
        self.vx = np.asarray(vx)
        self.vy = np.asarray(vy)
        self.vz = np.asarray(vz)
        self.p = np.asarray(p)
        
        # Handle constant Bx/By/Bz by broadcasting to grid shape if necessary
        self.Bx = np.full_like(self.rho, Bx) if np.isscalar(Bx) else np.asarray(Bx)
        self.By = np.full_like(self.rho, By) if np.isscalar(By) else np.asarray(By)
        self.Bz = np.full_like(self.rho, Bz) if np.isscalar(Bz) else np.asarray(Bz)
        
        self.flux_x = None
        self.flux_y = None
        self.cmax_x = None
        self.cmax_y = None

    def solve(self):
        """Calculates fluxes for both X and Y directions."""
        self.flux_x, self.cmax_x = self._solve_axis(0)
        self.flux_y, self.cmax_y = self._solve_axis(1)

    def _solve_axis(self, axis):
        """Extracts 1D interfaces, solves the Riemann problem, and reshapes back."""
        if axis == 0:
            # X-Sweep
            rhoL, rhoR = self.rho[:-1, :], self.rho[1:, :]
            vnL, vnR = self.vx[:-1, :], self.vx[1:, :]
            vtL, vtR = self.vy[:-1, :], self.vy[1:, :]
            vbL, vbR = self.vz[:-1, :], self.vz[1:, :]
            pL, pR = self.p[:-1, :], self.p[1:, :]
            BnL, BnR = self.Bx[:-1, :], self.Bx[1:, :]
            BtL, BtR = self.By[:-1, :], self.By[1:, :]
            BbL, BbR = self.Bz[:-1, :], self.Bz[1:, :]
        else:
            # Y-Sweep (Normal direction is Y, Transverse is X)
            rhoL, rhoR = self.rho[:, :-1], self.rho[:, 1:]
            vnL, vnR = self.vy[:, :-1], self.vy[:, 1:]
            vtL, vtR = self.vx[:, :-1], self.vx[:, 1:]
            vbL, vbR = self.vz[:, :-1], self.vz[:, 1:]
            pL, pR = self.p[:, :-1], self.p[:, 1:]
            BnL, BnR = self.By[:, :-1], self.By[:, 1:]
            BtL, BtR = self.Bx[:, :-1], self.Bx[:, 1:]
            BbL, BbR = self.Bz[:, :-1], self.Bz[:, 1:]
            
        # Flatten 2D interface arrays to 1D for the Numba solver
        rhoL, vnL, vtL, vbL, pL, BnL, BtL, BbL = [x.flatten() for x in (rhoL, vnL, vtL, vbL, pL, BnL, BtL, BbL)]
        rhoR, vnR, vtR, vbR, pR, BnR, BtR, BbR = [x.flatten() for x in (rhoR, vnR, vtR, vbR, pR, BnR, BtR, BbR)]
        
        uL, uR, fL, fR, ptL, ptR, SL, SR, Bn = prepare_states(
            rhoL, vnL, vtL, vbL, pL, BnL, BtL, BbL,
            rhoR, vnR, vtR, vbR, pR, BnR, BtR, BbR, self.gamma
        )
        
        flux_flat, _, cmax_flat = hlld(uL, uR, fL, fR, ptL, ptR, SL, SR, Bn, self.gamma)
        
        # Reshape back to 2D grid interface shape
        if axis == 0:
            shape = (self.rho.shape[0] - 1, self.rho.shape[1], 7)
        else:
            shape = (self.rho.shape[0], self.rho.shape[1] - 1, 7)
            
        return flux_flat.reshape(shape), cmax_flat.reshape(shape[:2])
