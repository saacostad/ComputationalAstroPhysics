import numpy as np 
import matplotlib.pyplot as plt 


x0 = 1.0
fun = np.sin 
SOD_fun = lambda x: -1.0 * np.sin(x) 



def abs_err(Fex, Fnum, dx, x=x0):
    return np.abs( Fex(x) - Fnum(dx, x) )

def rel_err(Fex, Fnum, dx, x=x0):
    return np.abs( ( Fex(x) - Fnum(dx, x) ) / Fex(x))

def second_OD(F, x, dx):
    return ( F(x+dx) - 2.0*F(x) + F(x-dx) ) / (dx**2)

def second_OD_F(F, x, dx):
    return ( F(x) - 2.0*F(x + 1.0*dx) + F(x + 2.0*dx) ) / (dx**2)

def second_OD_B(F, x, dx):
    return ( F(x) - 2.0*F(x - 1.0*dx) + F(x - 2.0*dx) ) / (dx**2)



dxs = np.linspace(0.001, 0.1, 15)




abs_errors = abs_err(SOD_fun, lambda dx, x: second_OD(fun, x, dx), dxs)
rel_errors = rel_err(SOD_fun, lambda dx, x: second_OD(fun, x, dx), dxs)

F_abs_errors = abs_err(SOD_fun, lambda dx, x: second_OD_F(fun, x, dx), dxs)
F_rel_errors = rel_err(SOD_fun, lambda dx, x: second_OD_F(fun, x, dx), dxs)

B_abs_errors = abs_err(SOD_fun, lambda dx, x: second_OD_B(fun, x, dx), dxs)
B_rel_errors = rel_err(SOD_fun, lambda dx, x: second_OD_B(fun, x, dx), dxs)



plt.loglog(dxs, abs_errors, '-^', label = "C abs err")
plt.loglog(dxs, rel_errors, '-^', label = 'C rel err')

plt.loglog(dxs, F_abs_errors, '-^', label = "F abs err")
plt.loglog(dxs, F_rel_errors, '-^', label = 'F rel err')

plt.loglog(dxs, B_abs_errors, '-^', label = "B abs err")
plt.loglog(dxs, B_rel_errors, '-^', label = 'B rel err')



plt.legend()
plt.grid()

plt.savefig("test.png")



