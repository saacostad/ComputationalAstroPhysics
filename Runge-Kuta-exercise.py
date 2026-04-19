import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import least_squares

h = 0.01

q0 = 0.0
t0 = 0.0 

V = 10.0
R = 1000 
C = 1.0e-3 

N = int(5 * R*C / h)
Qmax = C * V
Qtarget = Qmax / 2.0


# No of iterations for the secand method
Nsec = 100

"""
------------------
EDO step solvers
------------------
"""

def ralston_step(xi, yi, f, h):

    a2 = 2.0 / 3.0
    a1 = 1.0 / 3.0
    p1 = 3.0 / 4.0 
    q11 = 3.0 / 4.0


    k1 = f(xi, yi) 
    k2 = f(xi + p1 * h, yi + q11 * k1 * h)


    return xi + h, yi + (a1 * k1 + a2 * k2) * h


def Runge_step(xi, yi, f, h):
    
    k1 = f(xi, yi)
    k2 = f(xi + 0.5*h, yi + 0.5*k1*h)
    k3 = f(xi + 0.5*h, yi + 0.5*k2*h)
    k4 = f(xi + 0.5*h, yi + k3*h)

    return xi + h, yi + (1.0 / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4) * h 


def Kutta_step(xi, yi, f, h):
    d13 = 1.0 / 3.0

    k1 = f(xi, yi)
    k2 = f(xi + d13*h, yi + d13*k1*h)
    k3 = f(xi + 2.0 * d13*h, yi - d13*k1*h + h*k2)
    k4 = f(xi + h, yi + k1*h - k2*h + k3*h)

    return xi + h, yi + (1.0 / 8.0) * (k1 + 3.0*k2 + 3.0*k3 + k4) * h 


def Euler_step(xi, yi, f, h):

    return xi + h, yi + f(xi, yi) * h 



"""
-------------------
General integrator
-------------------
"""
def Integrate(x0, y0, step_func, func):

    xs = [x0]
    ys = [y0]
    
    x_last = x0 
    y_last = y0

    for t in range(N):

        xi, yi = step_func(x_last, y_last, func, h = h) 
        xs.append(xi)
        ys.append(yi)

        x_last = xi 
        y_last = yi
    
    return np.array(xs), np.array(ys)




"""
---------------------------------------------
Secant method I stole from previous exercises
---------------------------------------------
"""
def secant(xn, xnm1, f):

    # Evaluate the function at the two previous points
    fxn = f(xn)
    fxnm1 = f(xnm1)

    # Secant update
    xn1 = xn - fxn * (xn - xnm1) / (fxn - fxnm1)

    return xn1



"""
----------------------------
The ODE perse we'll be using 
----------------------------
"""
def function(x, y):
    return (V - y / C) / R 



"""
------------------------------------------------
The shape of the analytical solution of this ODE
so we can make a regression on Vs and tau 
------------------------------------------------
"""
def RC_circuit(params, t):
    Vs, tau = params
    return Vs * (1.0 - np.exp(-t / tau))






"""
--------------------------------------------------------------
We first numerically solve the ODE using the given integrators
--------------------------------------------------------------
"""

ts_Ralston, qs_Ralston = Integrate(t0, q0, ralston_step, function)
ts_Kutta, qs_Kutta = Integrate(t0, q0, Kutta_step, function)
ts_Runge, qs_Runge = Integrate(t0, q0, Runge_step, function)
ts_Euler, qs_Euler = Integrate(t0, q0, Euler_step, function)






"""
------------------------------------------------------------------------------------
Now, using the Kutta data, we'll make the regression to find an analytical shape for 
our function and later use that on the secand method 
------------------------------------------------------------------------------------
"""

# This implementation is done with scipy's regression models
# def residuals(params, t, y_observed):
#     ''' The residuals for the least squares '''
#     return RC_circuit(params, t) - y_observed
#
#
# initial_guess = [V, 0.1] # Initial guesses for the Vs value and the tau 
#
# results = least_squares(residuals, initial_guess, args=(ts_Kutta, qs_Kutta))
#
# Vs_fit, tau_fit = results.x
#
# print(f"Fitted model with Vs = {Vs_fit} and tau = {tau_fit}")





# Now, the RC equation is not lineraizable vor Vs and tau, but if we know that Vs = V*C (the maximum charge inside of the capacitor), then we can linearize 
# it as ln(V*C - q) = ln(V*C) - 1/tau * t [furthermore, we also know that tau = R*C, nice test case]. With this, we have a linear equation of the shape
# y = a*x + b, where a = 1/tau and y = ln(q - V*C),and we already know the value of b = ln(V*C). And we can find a = (N \sum xy - \sum x \sum y) / (N (\sum x^2 - (\sum x)^2))


y = np.log(-qs_Kutta[1::] + V*C)
x = ts_Kutta[1::]

sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x_squared = np.sum(x**2)

a = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x**2)

# Finally, we recover tau

tau = -1/a

results = [V*C, tau]


"""
---------------------------------------------------------------------------------------
Lastly, we'll apply the secand method to find the time t where the q = Qmax/2.0 
---------------------------------------------------------------------------------------
"""


# First, we'll create a single-input function 
def sec_func(t):
    return RC_circuit(results, t) - Qtarget    # We substract the value we want to find


# And now, we'll call the secant method multiple times to find the thing 
t0_guess = 3.0  
t1_guess = 3.1


for i in range(Nsec):
    sec_res = secant(t0_guess, t1_guess, sec_func)

    t0_guess = t1_guess
    t1_guess = sec_res
    
    if abs(t0_guess - t1_guess) < 1.0e-6:
        break


print(f"Estimated time fow q = Qmax / 2 is {t1_guess}")


plt.plot(np.array(ts_Ralston), np.array(qs_Ralston), lw = 1.0, c = "black", label = "Ralston")
plt.plot(np.array(ts_Kutta), np.array(qs_Kutta), lw = 1.0, c = "red", label = "Kutta")
plt.plot(np.array(ts_Runge), np.array(qs_Runge), lw = 1.0, c = "blue", label = "Runge")
plt.plot(np.array(ts_Euler), np.array(qs_Euler), lw = 1.0, c = "green", label = "Euler")
plt.plot(ts_Kutta, RC_circuit(results, np.array(ts_Kutta)), lw = 1.0, c = "purple", label = "Regression")
plt.axvline(x=t1_guess, color='r', linestyle='--', label = "Half charge time")
plt.axhline(y=RC_circuit(results, t1_guess), color='r', linestyle='--')
plt.title(f"Charge evolution with h={h} and N = {N}")
plt.xlabel("time [s]")
plt.ylabel("charge [C]")
plt.legend()
plt.grid()
plt.show()
