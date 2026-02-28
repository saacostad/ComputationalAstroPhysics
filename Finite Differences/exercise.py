from math import log
import numpy as np

# #
# t0 = 8 
# t1 = 30 
#
#
# u = 2000
# m = 1400090
# q = 2100
# g = 9.8
#
# def fun(t):
#     return u * log(m) * t + (1/q) * (m - q*t)*log(m - q*t) - (m / q) + t - (g / 2) * t**2 
#
# print(fun(t1) - fun(t0))


t0 = 0.1
t1 = 1.3 
t = t1 - t0 

h = (t1 - t0) / 2.0


def f(x):
    return 5*x*np.exp(-2.0 * x)

print( (h/3) * (f(t0) + 4*f((t0+t1)/2) + f(t1))) 
