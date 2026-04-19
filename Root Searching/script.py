"""
The idea of this script is to ease the LaTeX writing process by automatically printing
the needed LaTeX commands I need for the homework




In general, we need to show the process of root-finding by Bisection, Newton-Rhapson and Secant methods of the function

x^3 - 0.165 x^2 + 3.993 \times 10^-4 = 0 
"""
from math import log10
def F(x):
    s = f"{x:.3g}"
    if "e" in s:
        base, exp = s.split("e")
        return f"{base} \\times 10^{{{int(exp)}}}"
    return s


# We first define the function itself
def f(x):
    return x**3 - 0.165*x**2 + 3.993e-4

# And now, for Newton-Rhapson, we'll also need the derivative
def fp(x): 
    return 3*x**2 - 2.0*0.165*x 



# We will define now the bisection method
def bisection(xl, xu):

    # Given the first interval, it will divide it in 2 
    xm = xl + (xu - xl) / 2.0

    print(f"Intervalo: $(x_l = {F(xl)}, x_u = {F(xu)})$", f"donde $x_m = {F(xl)} + \\frac{{ {F(xu)} - {F(xl)} }}{{ 2 }} = {F(xm)}$ \\\\")
    print(f"Intervalos a examinar: $({F(xl)}, {F(xm)})$ y $({F(xm)}, {F(xu)})$")
    
    # Then it will evaluate the 3 points
    fl = f(xl)
    fm = f(xm)
    fu = f(xu)
    
    sl = "<" if fl < 0.0 else ">"
    sm = "<" if fm < 0.0 else ">"
    su = "<" if fu < 0.0 else ">"

    pl = "-" if fl < 0.0 else "+"
    pm = "-" if fm < 0.0 else "+"
    pu = "-" if fu < 0.0 else "+"

    # Then, it will check the sign of these evaluations on each interval 
    lint = fl * fm < 0 
    uint = fm * fu < 0 
    
    slint = "-" if lint else "+"
    suint = "-" if uint else "+"

    print("Evaluamos los 3 puntos dados \n\\begin{itemize}")
    print(f"\t \\item $f(x_l) = f({F(xl)}) = {F(f(xl))} {sl} 0 $")
    print(f"\t \\item $f(x_m) = f({F(xm)}) = {F(f(xm))} {sm} 0 $")
    print(f"\t \\item $f(x_u) = f({F(xu)}) = {F(f(xu))} {su} 0 $")
    print("\\end{itemize}")

    print("\\textbf{\\textit{Evaluamos los signos en los dos intervalos}} ")
    
    print("\\begin{itemize}")
    print(f"\\item Intervalo inferior:  $({F(xl)}, {F(xm)})$: $f(x_l) \\cdot f(x_m) \\propto {pl} \\cdot {pm} = {slint}$")
    print(f"\\item Intervalo superior:  $({F(xm)}, {F(xu)})$: $f(x_m) \\cdot f(x_u) \\propto {pm} \\cdot {pu} = {suint}$")
    print("\\end{itemize}")

    print("\\vspace{0.5cm}")
    
    # It will return the next interval to check 
    if lint and not uint:
        veredict = f"Utilizar el intervalo inferior $({F(xl)}, {F(xm)})$"
        el = xl 
        eu = xm 
    elif uint and not lint:
        veredict = f"Utilizar el intervalo superior $({F(xm)}, {F(xu)})$"
        el = xl 
        eu = xm 
    else:
        veredict = "Cambiar el intervalo inicial" 
        el = 0 
        eu = 0

    err = abs( xl - xm )

    print("\\textbf{Veredicto para la iteración:}")
    print(veredict)
    print(f"\\\\ Con un error $E = | {F(eu)} - {F(el)} | = {F(err)}$")

    # It will return the next interval to check 
    if lint and not uint:
        return xl, xm
    elif uint and not lint:
        return xm, xu
    else:
        print("WE CHOSE AN IMPROPER INITIAL INTERVAL")
        return False  



# We'll now define Newton-Rhapson method
def Newton(x, n):        
    
    # We create the evaluations of f and f' in the given x 
    fx = f(x)
    fpx = fp(x)

    xn1 = x - fx / fpx

    print("Punto inicial: $x_{{ {n} }} = ", F(x), "$")
    print("\\\\ Evaluamos el siguiente punto")
    print("\\begin{align*}")
    print(f"x_{{ {n+1} }} &= x_{{ {n} }} - \\frac{{ f(x_{{ {n} }}) }}{{ f'(x_{{ {n} }}) }} \\\\")
    print(f"&= {F(x)} - \\frac{{ {F(f(x))} }}{{ {F(fp(x)) } }} \\\\")
    print(f"&= {xn1}")
    print("\\end{align*}")
    
    err = abs( (xn1 - x) / xn1)
    m = int(2.0 - log10(err * 2))


    print(f"\\\\ Con un error $E = \\left| \\frac{{ x_{{ {n+1} }} -  x_{{ {n} }}}}{{ x_{{ {n+1} }} }} \\right| = \\left| \\frac{{ {F(xn1)} - {F(x)} }}{{ {F(xn1)} }} \\right| = {F(err)}$, con un m máximo $m = {m}$.")

    return xn1



# Same as we just did but with Secand method
def secant(xn, xnm1, n):

    # Evaluate the function at the two previous points
    fxn = f(xn)
    fxnm1 = f(xnm1)

    # Secant update
    xn1 = xn - fxn * (xn - xnm1) / (fxn - fxnm1)

    print(f"Puntos iniciales: $x_{{{n-1}}} = {F(xnm1)}$, $x_{{{n}}} = {F(xn)}$")
    print("\\\\ Evaluamos el siguiente punto")
    print("\\begin{align*}")
    print(f"x_{{ {n+1} }} &= x_{{ {n} }} - \\frac{{ f(x_{{ {n} }}) (x_{{ {n} }} - x_{{ {n-1} }}) }}{{ f(x_{{ {n} }}) - f(x_{{ {n-1} }}) }} \\\\")
    print(f"&= {F(xn)} - \\frac{{ {F(fxn)}({F(xn)} - {F(xnm1)}) }}{{ {F(fxn)} - {F(fxnm1)} }} \\\\")
    print(f"&= {F(xn)} - \\frac{{ {F(fxn*(xn-xnm1))} }}{{ {F(fxn - fxnm1)} }} \\\\")
    print(f"&= {F(xn1)}")
    print("\\end{align*}")

    # Relative error
    err = abs((xn1 - xn) / xn1)
    m = int(2.0 - log10(err * 2))

    print(f"\\\\ Con un error $E = \\left| \\frac{{ x_{{ {n+1} }} - x_{{ {n} }}}}{{ x_{{ {n+1} }} }} \\right| "
          f"= \\left| \\frac{{ {F(xn1)} - {F(xn)} }}{{ {F(xn1)} }} \\right| = {F(err)}$, "
          f"con un m máximo $m = {m}$.")

    return xn1




a= secant(0.05, 0.049, 0)
a1= secant(a, 0.05, 1)
a2= secant(a1, a, 2)
