# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : June 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import warnings
from matplotlib.offsetbox import AnchoredText
import json

# TODO: program is currently not saving
# print info:
print_values = True
print_parameters = True
print_progress = True
print_file_name = True

variables = ["$\Delta$t", "tol", "$h_{init}$", "$h_{div}$", "A", "B"]
print_variables = ["dt", "tol", "h_init", "h_div", "A", "B"]
variable = 0 # in {0, 1, ..., 5}

values = [2, 4]
# values = [2**-2, 2**-1, 2**0, 2*1, 2**2, 2**3] # dt
# values = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2] # tol
# values = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2] # tol (larger range)
# values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1] # init
# values = [2, 4, 6, 8, 10] # divisor
# values = [2, 3, 4, 5, 6, 7, 8, 9, 10] # divisor (higher res)
# values = np.arange(2, 20 + .5, .5).tolist() # divisor (higher res 2)
# values = [2, 4, 6 ,8 ,10, 12, 14, 16, 18, 20] # divsor (larger range 1)
# values = [2, 4, 6 ,8 ,10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] # divsor (larger range 2)
# values = [1e-2, 1e-1, 1e0, ] # A

###############################################################################
# x : initial-ODE | y : Adjoint | u : Control
# A : weight for i-dependent term in J | B : weight for u-dependent term im J
# h : stepsize in u

# default values
dt = 4
tol = 1e-6
h_init = 1e-2
h_div = 4
A = 1e1
B = 1e0
parameters = [dt, tol, h_init, h_div, A, B]

last_iteration = -1 # if last_iteration >=1 the calculations stop after that many iterations

# minimal allowed stepsize in u
h_min = 1e-13
# constant for Armijo-condition
Armijo = 1e-3
# length of fit/start of projection
t_u = 270
# k_min ^= nminimal k value for MPRK
k_min = 0.04
x_0 = [0.007, 0.993, 0.3 - k_min]

ALPHA = 0.03
TAU = 3.58
DEL = 4
# datapoints[0] == time (int), datapoints[1] == i meassured
datapoints = np.load("datapoints.npy")
t_0 = datapoints[0, 0]
t_end = datapoints[0, len(datapoints[0]) - 1]

# print info:
def print_info(print_val, print_param):
    if print_val:
        print(print_variables[variable] + " in {" + ", ".join(str(value) for value in values) + "}")
    if print_param:
        for i in range(len(print_variables)):
            if i != variable:
                print("    " + print_variables[i] + " = " + str(parameters[i]))
    return
print_info(print_values, print_parameters)

# General functions
def dU(k, x_k, y_k, u_k):
    y1, y2, y3 = y_k
    x1, x2, x3 = x_k
    return y3 / (2 * B) - u_k
def J(I, u):
    value = 0
    for i in range(u_sum):
        value += (-A * (I[i] - Pi[i]) ** 2 - B * u[i] ** 2) * dt
    return value
def projJ(I, u):
    value = 0
    for i in range(u_sum, t_sum):
        value += (-A * (I[i] - Pi[i]) ** 2 - B * u[i] ** 2) * dt
    return value
def K(I):
    value = 0
    for i in range(u_sum):
        value += (I[i] - Pi[i]) ** 2 * dt
    return value 

# RK functions
# I = X1/(X2 * (X3 + k_min))
X1, X2, X3, Y1, Y2, Y3, U, Pi = sp.symbols("X1, X2, X3, Y1, Y2, Y3, U, Pi")
Symbols = X1, X2, X3, Y1, Y2, Y3, U, Pi

dX1_dt = X1 / TAU * (1 - 1 / (DEL * X2 * (X3 + k_min)))
dX2_dt = ALPHA * (1 - X2) - X1
dX3_dt = U
r = -A * (X1 / (X2 * (X3 + k_min)) - Pi) ** 2 - B * U ** 2
H = dX1_dt * Y1 + dX2_dt * Y2 + dX3_dt * Y3 + r
dY1_dt = -sp.diff(H, X1)
dY2_dt = -sp.diff(H, X2)
dY3_dt = -sp.diff(H, X3)

# numpy / lambdification
dx1_dt = sp.lambdify([Symbols], dX1_dt, "numpy")
dx2_dt = sp.lambdify([Symbols], dX2_dt, "numpy")
dx3_dt = sp.lambdify([Symbols], dX3_dt, "numpy")
dy1_dt = sp.lambdify([Symbols], dY1_dt, "numpy")
dy2_dt = sp.lambdify([Symbols], dY2_dt, "numpy")
dy3_dt = sp.lambdify([Symbols], dY3_dt, "numpy")

def Xprime(k, x_k):
    x1, x2, x3 = x_k
    if k % 1 == 0:
        u_k = u[k]
    else:
        k = int(k)
        u_k = (u[k] + u[k + 1]) / 2
    vals_k = np.append(x_k, (0, 0, 0, u_k, 0))
    dx1dt = dx1_dt(vals_k)
    dx2dt = dx2_dt(vals_k)
    dx3dt = dx3_dt(vals_k)
    return np.array([dx1dt, dx2dt, dx3dt])
def Yprime(k, x_k, y_k):
    y1, y2, y3 = y_k
    x1, x2, x3 = x_k
    if k % 1 == 0:
        u_k = u[k]
        pi = Pi[k]
    else:
        k = int(k)
        u_k = (u[k] + u[k + 1]) / 2
        pi = (Pi[k] + Pi[k + 1]) / 2
    vals_k = np.append(x_k, y_k)
    vals_k = np.append(vals_k, (u_k, pi))
    dy1dt = dy1_dt(vals_k)
    dy2dt = dy2_dt(vals_k)
    dy3dt = dy3_dt(vals_k)
    return np.array([dy1dt, dy2dt, dy3dt])

# MPRK functions
# productive and destructive terms
def p1(k, x_k):
    x1, x2, x3 = x_k
    p11 = x1/TAU
    return np.array([p11, 0, 0])
def d1(k, x_k):
    x1, x2, x3 = x_k
    d11 = x1/(TAU*DEL*x2*(x3 + k_min))
    return np.array([d11, 0, 0])
def p2(k, x_k):
    x1, x2, x3 = x_k
    p22 = ALPHA * (1 - x2)
    return np.array([0, p22, 0])
def d2(k, x_k):
    x1, x2, x3 = x_k
    d21 = x1 
    return np.array([d21, 0, 0])
def p3(k, x_k):
    x1, x2, x3 = x_k
    if k % 1 == 0:
        u_k = u[k]
    else:
        k = int(k)
        u_k = (u[k] + u[k + 1]) / 2
    p33 = 0
    if u_k >= 0: p33 = u_k
    return np.array([0, 0, p33])
def d3(k, x_k):
    x1, x2, x3 = x_k
    if k % 1 == 0:
        u_k = u[k]
    else:
        k = int(k)
        u_k = (u[k] + u[k + 1]) / 2
    d33 = 0
    if u_k < 0: d33 = -u_k
    return np.array([0, 0, d33])
    
# x^(1)
def X_1(k, x_k):
    x1, x2, x3 = x_k
    P1 = p1(k, x_k)
    D1 = d1(k, x_k)
    P2 = p2(k, x_k)
    D2 = d2(k, x_k)
    P3 = p3(k, x_k)
    D3 = d3(k, x_k)

    a11 = P1[0]/x1 - sum(D1)/x1
    a12 = P1[1]/x2
    a13 = P1[2]/x3
    a21 = P2[0]/x1
    a22 = P2[1]/x2 - sum(D2)/x2
    a23 = P2[2]/x3
    a31 = P3[0]/x1
    a32 = P3[1]/x2
    a33 = P3[2]/x3 - sum(D3)/x3    
    A = np.matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    A = np.eye(3) - dt * A
    x_1 = np.linalg.solve(A, x_k)
    return x_1
# x^(n+1)
def X_n1(k, x_k):
    x1, x2, x3 = x_k
    x_1 = X_1(k, x_k)
    x1, x2, x3 = x_1
    P1 = p1(k, x_k) + p1(k + 1, x_1)
    D1 = d1(k, x_k) + d1(k + 1, x_1)
    P2 = p2(k, x_k) + p2(k + 1, x_1)
    D2 = d2(k, x_k) + d2(k + 1, x_1)
    P3 = p3(k, x_k) + p3(k + 1, x_1)
    D3 = d3(k, x_k) + d3(k + 1, x_1)

    a11 = P1[0]/x1 - sum(D1)/x1
    a12 = P1[1]/x2
    a13 = P1[2]/x3
    a21 = P2[0]/x1
    a22 = P2[1]/x2 - sum(D2)/x2
    a23 = P2[2]/x3
    a31 = P3[0]/x1
    a32 = P3[1]/x2
    a33 = P3[2]/x3 - sum(D3)/x3    
    A = np.matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    A = np.eye(3) - dt/2 * A
    X_n1 = np.linalg.solve(A, x_k)
    return X_n1

RK_J = []
MPRK_J = []
RK_projJ = []
MPRK_projJ = []
RK_Iterations = []
MPRK_Iterations = []
RK_K = []
MPRK_K = []
for value in values:
    if variable == 0:
        dt = value
    elif variable == 1:
        tol = value
    elif variable == 2:
        h_init = value
    elif variable == 3:
        h_div = value
    elif variable == 4:
        A = value
    elif variable == 5:
        B = value
    
    # time
    t_sum = int((t_end - t_0) / dt) + 1
    t = np.linspace(t_0, t_end, t_sum)
    u_sum = int((t_u - t_0) / dt) + 1
    # Pi (Infectous measured, interpolated)
    Pi = np.interp(t, datapoints[0], datapoints[1])
    
    # RK Initialisation
    if True:
        new_J = -np.inf
        old_J = -np.inf
        x = np.zeros((t_sum, 3))
        x[0] = x_0
        y = np.zeros((t_sum, 3))
        u = np.zeros(t_sum)
        old_u = u.copy()
        
        # Forward
        for i in range(t_sum - 1):
            x_i = x[i]
            r1 = Xprime(i, x_i)
            r2 = Xprime(i + 1 / 2, x_i + dt / 2 * r1)
            r3 = Xprime(i + 1 / 2, x_i + dt / 2 * r2)
            r4 = Xprime(i + 1, x_i + dt * r3)
            x_i1 = x_i + (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
            x[i + 1] = x_i1
        # Infectious
        I = x[:, 0] / (x[:, 1] * (x[:, 2] + k_min))
        
        # Backward
        for i in range(1, u_sum):
            j = u_sum - i
            t_j = t[j]
            x_j = x[j]
            y_j = y[j]
    
            r1 = Yprime(j, x_j, y_j)
            r2 = Yprime(j - 1 / 2, x_j, y_j - dt / 2 * r1)
            r3 = Yprime(j - 1 / 2, x_j, y_j - dt / 2 * r2)
            r4 = Yprime(j - 1, x_j, y_j - dt * r3)
            y_j_1 = y_j - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
    
            y[j - 1] = y_j_1
    
        # Sweep
        del_u = np.zeros(t_sum)
        for i in range(u_sum):
            x_i = x[i]
            u_i = u[i]
            y_i = y[i]
            del_u[i] = dU(i, x_i, y_i, u_i)
        old_J = J(I, u)
        del_J = 0
        projection_J = projJ(I, u)
        
        error = 1
        iterations = 0
        step = h_init
        done = False
    # RK Iteration
    while error >= tol and not (done):
        while new_J <= old_J + Armijo * del_J * step or np.isnan(new_J):  # while (step too large)
            u = old_u.copy() + step * del_u.copy()
            old_step = step
            # Forward
            for i in range(t_sum - 1):
                x_i = x[i]
                r1 = Xprime(i, x_i)
                r2 = Xprime(i + 1 / 2, x_i + dt / 2 * r1)
                r3 = Xprime(i + 1 / 2, x_i + dt / 2 * r2)
                r4 = Xprime(i + 1, x_i + dt * r3)
                x_i1 = x_i + (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
                x[i + 1] = x_i1
            # Infectious
            with warnings.catch_warnings(): # TODO
                warnings.simplefilter("ignore")
                I = x[:, 0] / (x[:, 1] * (x[:, 2] + k_min))
                new_J = J(I, u)
            del_J = abs(old_J - new_J)
            if step <= h_min:
                print("(MPRK) No improvement possible at iteration " + str(iterations) 
                      + " (dt = " + str(dt) 
                      + ", tol = " + str(tol)  
                      + ", h_init = " + str(h_init) 
                      + ", h_div = " + str(h_div) 
                      + ", A = " + str(A)
                      + ", B = " + str(B) + ")")
                done = True
                break
            else: step = step / h_div
        if not done:
            iterations += 1
            error = del_J
            old_J = new_J
            projection_J = projJ(I, u)
            step = h_init
            old_u = u.copy()
        if iterations == last_iteration:
            error = 0
        
        # Backward
        for i in range(1, u_sum):
            j = u_sum - i
            t_j = t[j]
            x_j = x[j]
            y_j = y[j]
    
            r1 = Yprime(j, x_j, y_j)
            r2 = Yprime(j - 1 / 2, x_j, y_j - dt / 2 * r1)
            r3 = Yprime(j - 1 / 2, x_j, y_j - dt / 2 * r2)
            r4 = Yprime(j - 1, x_j, y_j - dt * r3)
            y_j_1 = y_j - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
            
            y[j - 1] = y_j_1
    
        # Sweep
        del_u = np.zeros(t_sum)
        for i in range(u_sum):
            x_i = x[i]
            u_i = u[i]
            y_i = y[i]
            del_u[i] = dU(i, x_i, y_i, u_i)
        for i in range(u_sum, t_sum):
            del_u[i] = 0
    RK_Iterations.append(iterations)
    RK_J.append(old_J)
    RK_projJ.append(projection_J)
    RK_K.append(K(I))
    
    # MPRK Initialisation
    if True:
        new_J = -np.inf
        old_J = -np.inf
        x = np.zeros((t_sum, 3))
        x[0] = x_0
        y = np.zeros((t_sum, 3))
        u = np.zeros(t_sum)
        old_u = u.copy()
        
        # Forward
        for i in range(t_sum - 1):
            x_i = x[i]
            x_i1 = X_n1(i, x_i)
            x[i + 1] = x_i1
        # Infectious
        I = x[:, 0] / (x[:, 1] * (x[:, 2] + k_min))
        
        # Backward
        for i in range(1, u_sum):
            j = u_sum - i
            t_j = t[j]
            x_j = x[j]
            y_j = y[j]
    
            r1 = Yprime(j, x_j, y_j)
            r2 = Yprime(j - 1 / 2, x_j, y_j - dt / 2 * r1)
            r3 = Yprime(j - 1 / 2, x_j, y_j - dt / 2 * r2)
            r4 = Yprime(j - 1, x_j, y_j - dt * r3)
            y_j_1 = y_j - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
    
            y[j - 1] = y_j_1
    
        # Sweep
        del_u = np.zeros(t_sum)
        for i in range(u_sum):
            x_i = x[i]
            u_i = u[i]
            y_i = y[i]
            del_u[i] = dU(i, x_i, y_i, u_i)
        old_J = J(I, u)
        del_J = 0
        projection_J = projJ(I, u)
        
        error = 1
        iterations = 0
        step = h_init
        done = False
    # MPRK Iteration
    while error >= tol and not (done):
        while new_J <= old_J + Armijo * del_J * step or np.isnan(new_J):  # while (step too large)
            u = old_u.copy() + step * del_u.copy()
            old_step = step
            # Forward
            for i in range(t_sum - 1):
                x_i = x[i]
                with warnings.catch_warnings(): # TODO
                    warnings.simplefilter("ignore")
                    x_i1 = X_n1(i, x_i)
                x[i + 1] = x_i1
            # Infectious
            with warnings.catch_warnings(): # TODO
                warnings.simplefilter("ignore")
                I = x[:, 0] / (x[:, 1] * (x[:, 2] + k_min))
                new_J = J(I, u)
            del_J = abs(old_J - new_J)
            if step <= h_min: 
                print("(MPRK) No improvement possible at iteration " + str(iterations) 
                      + " (dt = " + str(dt) 
                      + ", tol = " + str(tol)  
                      + ", h_init = " + str(h_init) 
                      + ", h_div = " + str(h_div) 
                      + ", A = " + str(A)
                      + ", B = " + str(B) + ")")
                done = True
                break
            else: step = step / h_div
        if not done:
            iterations += 1
            error = del_J
            old_J = new_J
            projection_J = projJ(I, u)
            step = h_init
            old_u = u.copy()
        if iterations == last_iteration:
            error = 0
        
        # Backward
        for i in range(1, u_sum):
            j = u_sum - i
            t_j = t[j]
            x_j = x[j]
            y_j = y[j]
    
            r1 = Yprime(j, x_j, y_j)
            r2 = Yprime(j - 1 / 2, x_j, y_j - dt / 2 * r1)
            r3 = Yprime(j - 1 / 2, x_j, y_j - dt / 2 * r2)
            r4 = Yprime(j - 1, x_j, y_j - dt * r3)
            y_j_1 = y_j - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
            
            y[j - 1] = y_j_1
    
        # Sweep
        del_u = np.zeros(t_sum)
        for i in range(u_sum):
            x_i = x[i]
            u_i = u[i]
            y_i = y[i]
            del_u[i] = dU(i, x_i, y_i, u_i)
        for i in range(u_sum, t_sum):
            del_u[i] = 0
    MPRK_Iterations.append(iterations)
    MPRK_J.append(old_J)
    MPRK_projJ.append(projection_J)
    MPRK_K.append(K(I))
    
    if print_progress: print("Done with " + variables[variable] + " = " + str(value)) # TODO

fixed_parts = ["dt = " + str(dt), 
               "tol = " + str(tol), 
               "h_init = " + str(h_init), 
               "h_div = " + str(h_div), 
               "A = " + str(A),
               "B = " + str(B)]
fixed = "\n".join(fixed_parts[:variable] + fixed_parts[variable + 1:])

for RK_plt, MPRK_plt in zip([RK_Iterations, RK_J, RK_K, RK_projJ],
                            [MPRK_Iterations, MPRK_J, MPRK_K, MPRK_projJ]):
    fig, ax = plt.subplots()
    at = AnchoredText(fixed, pad = 0.5, borderpad = 2, 
                      loc="lower right", bbox_to_anchor=(180, -16))
    at.patch.set_boxstyle("round,pad=0,rounding_size=0.4")
    ax.add_artist(at)
    ax.plot(values, RK_plt, label="RK")
    ax.plot(values, MPRK_plt, label="MPRK")
    ax.legend(loc="best")
    ax.set_xlabel(variables[variable])
    if variable == 0: ax.set_xscale("log", base = 2)
    elif variable == 1 or variable == 2 or variable > 3: ax.set_xscale("log", base = 10)
    ax.grid()
    if RK_plt == RK_Iterations:
        ax.set_title("# Iterations dependent on " + variables[variable])
    elif RK_plt == RK_J:
        ax.set_title("J dependent on " + variables[variable])
    elif RK_plt == RK_K:
        ax.set_title("K dependent on " + variables[variable])
    elif RK_plt == RK_projJ:
        ax.set_title("J on $[t_u, t_{end}]$ dependent on " + variables[variable])
    plt.tight_layout(pad = 3)
    plt.show()

# save Results
fixed_parts[variable] = variables[variable] + " = " + str(
    values[0]) + "-" + str(values[1]) + "--" + str(values[-1])
result_name = variables[variable] + "|" + "|".join(fixed_parts)
file_name = "fbs_data/" + result_name + ".json"
result = {
    "variable" : variable,
    "variables" : variables,
    "parameters" : {
        variables[0] : dt,
        variables[1] : tol,
        variables[2] : h_init,
        variables[3] : h_div,
        variables[4] : A,
        variables[5] : B,
    },
    "values" : values,
    "data" : {
        "RK": {
            "Iterations" : RK_Iterations,
            "fit_j" : RK_J,
            "konjJ" : RK_projJ,
            "fit Error" : RK_K,
        },
        "MPRK": {
            "Iterations" : MPRK_Iterations,
            "fit_j" : MPRK_J,
            "konjJ" : MPRK_projJ,
            "fit Error" : MPRK_K,
        }
    }
}

result_json = json.dumps(result)
with open(file_name, "w") as fp:
    json.dump(result_json, fp)

if print_file_name: 
    print("Results saved in " + file_name)