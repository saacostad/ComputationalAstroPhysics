import numpy as np 
from numba import njit 



@njit 
def hlld(UL, UR, bb, gamma = 5./3., axis = 'x'):
    """ 
    Tomando los estados de las celdas derecha e izquierda a la interface (así como otros datos),
    calcula el flujo predicho por HLLD para ser resuelto en la ecuación diferencial de MHD ideal conservativo.
    
    IMPORTANTE: para trabajar en el eje x, se deja todo normal, pero en el eje y, es importante cambiar los valores
    de momento y campos magnéticos de tal modo que las componentes en y sean las nuevas en x.

    Cada vector contiene los estados de cada celda
    ----------------------------------------------------------------------------------------------------------
    INPUTS:

    UL, UR:     las variables conservadas izq. y der. [rho, mx, my, mz, bx, by, bz, E]*N
    bb:         el campo magnético background. 
    gamma:      constante adiabática
    axis:       para saber en dónde poner nuestro campo constante
    """
    N = UL.shape[0]   # Número de celdas

    #-----------------------------------
    # Creación de variables a utilizar 
    
    F = np.zeros((N, 8))  # 7 porque ese es el número de variables 
    P = np.zeros(N)
    cmax = np.zeros(N)
    
    for i in range(N):

        # ========== GUARDAMOS LAS VARIABLES DE INTERÉS  ==========
        
        # rhol, mxl, myl, mzl, bxl, byl, bzl, El = UL[i];
        # rhor, mxr, myr, mzr, bxr, byr, bzr, Er = UR[i];

        rhol = UL[i, 0]; mxl = UL[i, 1]; myl = UL[i, 2]; mzl = UL[i, 3]
        El = UL[i, 7]; bxl = UL[i, 4]; byl = UL[i, 5]; bzl = UL[i, 6]
        
        rhor = UR[i, 0]; mxr = UR[i, 1]; myr = UR[i, 2]; mzr = UR[i, 3]
        Er = UR[i, 7]; bxr = UR[i, 4]; byr = UR[i, 5]; bzr = UR[i, 6]
        
        # ----- CALCULAMOS VELOCIDADES: \vec v = (u, v, w)  -----
        
        # Por componentes
        ul = mxl / rhol; vl = myl/rhol; wl = mzl/rhol 
        ur = mxr / rhor; vr = myr/rhor; wr = mzr/rhor 
        
        # Magnitud al cuadrado
        VEL2l = (ul**2 + vl**2 + wl**2)
        VEL2r = (ur**2 + vr**2 + wr**2)
        
        # ----- CAMPOS MAGNÉTICOS    -----
        # WARN: Intento triste de agregar un background field :,(

        # Agregamos el campo de fondo a bx
        
        # # Si el eje es el x, entonces el campo de fondo está en la dirección x
        # if axis == 'x':
        # # if True:
        #     bxl += bb 
        #     bxr += bb 
        # else:
        #     # Si el eje es y, el campo de fondo está en dirección -y
        #     byl -= bb 
        #     byr -= bb

        # Obtenemos el signo del campo magnético normal 
        sbxl = 1. if bxl > 0.0 else -1.
        sbxr = 1. if bxr > 0.0 else -1.

        # Magnitud al cuadrado
        B2l = (bxl**2 + byl**2 + bzl**2)
        B2r = (bxr**2 + byr**2 + bzr**2)

        # ----- CALCULAMOS LAS PRESIONES    -----

        # Hidroestáticas
        pl = (gamma - 1.) * (El - 0.5*rhol*VEL2l - 0.5*B2l)
        pr = (gamma - 1.) * (Er - 0.5*rhor*VEL2r - 0.5*B2r)

        # Presión total 
        ptl = pl + 0.5*B2l
        ptr = pr + 0.5*B2r

        # ============================================================
        #               COMPUTAMOS LOS FLUJOS INICIALES
        # ============================================================

        FL = np.zeros(8)
        FR = np.zeros(8)
        
        # Los flujos tal cual la PDE de conservación
        FL = np.array([rhol*ul,
            rhol*ul**2 + ptl - bxl**2, 
            rhol*vl*ul - bxl*byl,
            rhol*wl*ul - bxl*bzl,
            0.,
            byl*ul - bxl*vl,
            bzl*ul - bxl*wl,
            (El + ptl)*ul - bxl*(ul*bxl + vl*byl + wl*bzl)])


        FR = np.array([rhor*ur,
            rhor*ur**2 + ptr - bxr**2,
            rhor*vr*ur - bxr*byr,
            rhor*wr*ur - bxr*bzr,
            0.,
            byr*ur - bxr*vr,
            bzr*ur - bxr*wr,
            (Er + ptr)*ur - bxr*(ur*bxr + vr*byr + wr*bzr)])


        # ========== VELOCIDAD de SEÑAL RÁPIDA y de CONTACTO  ==========
        
        # ----- Velocidad magnetosónica -----

        innerl = gamma * pl + B2l   # Auxiliares
        innerr = gamma * pr + B2r
        

        cfl = np.sqrt( ( innerl + np.sqrt( innerl**2 - (4. * gamma * pl * bxl**2) ) ) / (2. * rhol) )
        cfr = np.sqrt( ( innerr + np.sqrt( innerr**2 - (4. * gamma * pr * bxr**2) ) ) / (2. * rhor) )
        
        # ----- Velocidad rápida: de acuerdo con eq. 12 -----

        SL = min(ul, ur) - max(cfl, cfr)
        SR = max(ul, ur) + max(cfl, cfr)
        
        # ----- Cálculo del cmax -----
        cmax[i] = max(abs(SL), abs(SR))


        # ===============================================================
        #               PRIMERA COMPROBACIÓN 
        # ===============================================================

        if SL > 0.:  
            for nv in range(8): 
                F[i, nv] = FL[nv]

            P[i] = ptl
            continue

        elif SR < 0.:  
            for nv in range(8): 
                F[i, nv] = FR[nv]

            P[i] = ptr      
            continue

        # ===============================================================
        #               PRIMERA REGIÓN INTERNA 
        # ===============================================================

        #========== Velocidad de CONTACTO y PRESIÓN TOTAL ==========
        
        par_l = SL - ul; par_r = SR - ur   # Auxiliar 
        
        # ----- Velocidad de contacto -----

        SM = (par_r * rhor * ur - par_l*rhol*ul - ptr + ptl) / (par_r*rhor - par_l*rhol) 

        # ----- Presión total región s ] -----

        denom = par_r*rhor - par_l*rhol # Auxiliar y shell 
        if abs(denom) < 1e-14: denom = 1e-14 if denom >= 0 else -1e-14 

        pts = (par_r*rhor*ptl - par_l*rhol*ptr + rhol*rhor*par_r*par_l*(ur - ul)) / (denom) 

        # ========== COMPROBAR QUE EL SISTEMA NO ESTÉ DEGENERADO ==========

        deg_flag = 0

        denoml = rhol * par_l * (SL - SM) - bxl**2
        denomr = rhor * par_r * (SR - SM) - bxr**2

        if abs(denoml) < 1e-14: deg_flag = 1
        if abs(denomr) < 1e-14: deg_flag = 1
        
        if deg_flag == 1:

            # Si esto ocurre, entonces hacemos un fallback a HLL 
            UHLL = (SR*UR[i] - SL*UL[i] - FR + FL) / (SR - SL)
           
            # -- Velocidades -- 
            usl = usr = SM
            vsl = vsr = UHLL[2] / UHLL[0]
            wsl = wsr = UHLL[3] / UHLL[0]
            
            # -- Campos magnéticos -- 
            bxsl = bxl              # El campo magnético normal no cambia entre interfases
            bxsr = bxr 

            bysl = bysr = UHLL[5]
            bzsl = bzsr = UHLL[6]

            # -- Presión -- 
            par_lm = (SL - SM); par_rm = (SR - SM)
            
            # -- Las otras cosas -- 
            if par_lm < 1e-14:
                rhosl = rhol 
                Esl = El
            else:
                rhosl = rhol * (par_l / par_lm)
                Esl = ( par_l*El - ptl*ul + pts*SM + bxl * (vl*byl + wl*bzl - vsl*bysl - wsl*bzsl) ) / par_lm

            if par_rm < 1e-14:
                rhosr = rhor 
                Esr = Er
            else:
                rhosr = rhor * (par_r / par_rm)
                Esr = ( par_r*Er - ptr*ur + pts*SM + bxr * (vr*byr + wr*bzr - vsr*bysr - wsr*bzsr) ) / par_rm

        else:
            # Si no estamos degenerados, usamos HLLD usual 

            # -- Velocidades -- 
            usl = usr = SM

            vsl = vl - (bxl*byl*(SM - ul))/denoml
            vsr = vr - (bxr*byr*(SM - ur))/denomr

            wsl = wl - (bxl*bzl*(SM - ul))/denoml
            wsr = wr - (bxr*bzr*(SM - ur))/denomr
            
            # -- Campos magnéticos -- 
            
            bxsl = bxl              # El campo magnético normal no cambia entre interfases
            bxsr = bxr 
            
            sca_l = ( rhol*par_l**2 - bxl**2 )/(denoml) 
            sca_r = ( rhor*par_r**2 - bxr**2 )/(denomr) 

            bysl = byl * sca_l
            bysr = byr * sca_r

            bzsl = bzl * sca_l 
            bzsr = bzr * sca_r


            # -- Presión -- 
            par_lm = (SL - SM); par_rm = (SR - SM)

            if abs(par_lm) < 1e-14:
                par_lm = 1e-14 if par_lm >= 0 else -1e-14
            if abs(par_rm) < 1e-14:
                par_rm = 1e-14 if par_rm >= 0 else -1e-14
            
            rhosl = rhol * (par_l / par_lm)
            Esl = ( par_l*El - ptl*ul + pts*SM + bxl * (vl*byl + wl*bzl - vsl*bysl - wsl*bzsl) ) / par_lm

            rhosr = rhor * (par_r / par_rm)
            Esr = ( par_r*Er - ptr*ur + pts*SM + bxr * (vr*byr + wr*bzr - vsr*bysr - wsr*bzsr) ) / par_rm

        # ========== Creamos los vectores de CANTIDADES CONSERVADAS en REGIÓN * ===========

        UsL = np.array([rhosl, rhosl*usl, rhosl*vsl, rhosl*wsl, bxsl, bysl, bzsl, Esl])
        UsR = np.array([rhosr, rhosr*usr, rhosr*vsr, rhosr*wsr, bxsr, bysr, bzsr, Esr])
        
        # ----- Velocidades de Alfvén -----
        SsL = SM - abs(bxl)/np.sqrt(rhosl)
        SsR = SM + abs(bxr)/np.sqrt(rhosr)


        # ========== Creamos los vectores de FLUJOS en REGIÓN * ===========

        # FsL = FL + SL*(UsL - UL[:,i])
        # FsR = FR + SR*(UsR - UR[:,i])
        
        # TODO: si se joden las bichas es culpa de esto
        FsL = FL + SL*(UsL - UL[i,:])
        FsR = FR + SR*(UsR - UR[i,:])

        # ===============================================================
        #               COMPROBACIÓN SEGUNDA REGIÓN INTERNA 
        # =============================================================== 


        if SsL >= 0.:  
            for nv in range(8): 
                F[i, nv] = FsL[nv]

            P[i] = pts
            continue

        elif SsR <= 0.:  
            for nv in range(8): 
                F[i, nv] = FsR[nv]

            P[i] = pts    
            continue

        # ===============================================================
        #               SEGUNDA REGIÓN INTERNA 
        # ===============================================================   
        
        # ----- Creamos las variables que no cambian (las creo sólo para no perderme) -----
        # Las densidades se mantienen iguales, al igual que las velocidades normales, la presión total y el B normal
        rhossl = rhosl; rhossr = rhosr 
        ussl = usl; ussr = usr 
        ptss = pts
        bxssl = bxsl; bxssr = bxsr 
        
        # Definimos variables auxiliares 
        ql = np.sqrt(rhosl); qr = np.sqrt(rhosr)

        denom = ql + qr

        # ----- Velocidades transversales: solo hay un componente ahora -----
        vss = ( ql*vsl + qr*vsr + sbxl*(bysr - bysl) ) / denom
        wss = ( ql*wsl + qr*wsr + sbxl*(bzsr - bzsl) ) / denom

        # ----- B transversales -----
        byss = ( ql*bysl + qr*bysr + sbxl*ql*qr*(vsr - vsl) ) / denom
        bzss = ( ql*bzsl + qr*bzsr + sbxl*ql*qr*(wsr - wsl) ) / denom

        # ----- Energía -----
        Essl = Esl - sbxl*ql*(vsl*bysl + wsl*bzsl - vss*byss - wss*bzss) / denom
        Essr = Esr + sbxr*qr*(vsr*bysr + wsr*bzsr - vss*byss - wss*bzss) / denom


        # =========== Creamos los vectores de CANTIDADES CONSERVADAS ==========
        UssL = np.array([rhossl, rhossl*ussl, rhossl*vss, rhossl*wss, bxl, byss, bzss, Essl])
        UssR = np.array([rhossr, rhossr*ussr, rhossr*vss, rhossr*wss, bxr, byss, bzss, Essr])

        # ========== Creamos los vectores de FLUJOS en REGIÓN ** ===========

        FssL = FsL + SsL*(UssL - UsL)
        FssR = FsR + SsR*(UssR - UsR)


        if SM >= 0.:  
            for nv in range(8): 
                F[i, nv] = FssL[nv]

            P[i] = pts
            continue

        else:  
            for nv in range(8): 
                F[i, nv] = FssR[nv]

            P[i] = pts   
            continue


    return F, P, cmax




def create_U(rho, u, v, w, bx, by, bz, E, gamma, side = "left", axis = "x", ng = 2, Nx = 128, Ny = 256):
    """ Esta fución crea el vector de cantidades conservadas a partir de los parámetros físicos del sistema 
    y el eje que estemos manejando."""

    if axis == "x":
        
        if side == "left":
            def cut(A):
                return A[:, :-1]
        else:
            def cut(A):
                return A[:, 1:]

        U = np.array([cut(rho), cut(rho*u), cut(rho*v), cut(rho*w), cut(bx), cut(by), cut(bz), cut(E)])
        return U 

    elif axis == 'y':

        if side == "left":
            def cut(A):
                # return np.rot90(A[:-1, :])
                return A[:-1, :].T
        else:
            def cut(A):
                # return np.rot90(A[1:, :])
                return A[1:, :].T

        U = np.array([cut(rho), cut(rho*v), -cut(rho*u), cut(rho*w), cut(by), -cut(bx), cut(bz), cut(E)])
        # U = np.array([cut(rho), cut(rho*v), cut(rho*u), cut(rho*w), cut(by), cut(bx), cut(bz), cut(E)])
        return U

    else:
        def cut(A):
            # return A.T[ng:Nx - ng, ng:Ny-ng]
            return A
        U = np.array([cut(rho), cut(rho*u), cut(rho*v), cut(rho*w), cut(bx),cut(by), cut(bz), cut(E)])
        return U
