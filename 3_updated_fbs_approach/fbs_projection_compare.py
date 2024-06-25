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

# nur iterations anschauen ignoriert laufzeit der einzelnen iterations durch wiederholte teilung von 
    # step_size. 
    # Laufzeit vergleichen? Als mittelwert über zB 5 durchgänge.

# dimension \in [0, 1, 2, 3] ~= [dt, tolerance, init_step, step_divisor]
# const ~= [t_u, smallest_step, Armijo, k_min, A, B]
variable = 4
variables = ["dt", "tolerance", "initial stepsize", "stepsize divisor", "A", "B"] # 0, 1, 2, 3, 4, 5

# values = [2**-2, 2**-1, 2**0, 2*1, 2**2, 2**3] # dt
# values = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2] # tolerance
# values = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2] # tolerance (larger range)
# values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1] # init
# values = [2, 4, 6, 8, 10] # divisor
# values = [2, 3, 4, 5, 6, 7, 8, 9, 10] # divisor (higher res)
# values = np.arange(2, 20 + .5, .5).tolist() # divisor (higher res 2)
# values = [2, 4, 6 ,8 ,10, 12, 14, 16, 18, 20] # divsor (larger range 1)
# values = [2, 4, 6 ,8 ,10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] # divsor (larger range 2)
values = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4] # A

# default values
dt = 4
# tolerance in FBS
tolerance = 1e-6
# step-size in u
init_step = 1e-2
step_divisor = 4
# A: weight in J for i-term, B: weight in J for u-term
A = 1e1
B = 1e0

# alternative Abbruchbedingung
last_iteration = -1

if True:
    smallest_step = 1e-13
    Armijo = 1e-3
    # length of fit/start of projection
    t_u = 270
    # k_min ^= nminimal k value for MPRK
    k_min = 0.04
    x0 = [0.007, 0.993, 0.3 - k_min] # nu_0 value outside of whats in the paper (0.003 +/- 0.002)
    
    ALPHA = 0.03
    TAU = 3.58
    DEL = 4
    # datapoints[0] == time (int), datapoints[1] == i meassured =~ Pi
    datapoints = np.load("datapoints.npy")
    t_0 = datapoints[0, 0]
    t_end = datapoints[0, datapoints[0].size - 1]

# print info:
def print_info(print_val, print_param):
    if print_val:
        print(variables[variable] + " e {" + ", ".join(str(value) for value in values) + "}")
    if print_param:
        if variable != 0:
            print("    " + "dt = " + str(dt))
        if variable != 1:
            print("    " + "tolerance = " + str(tolerance))
        if variable != 2:
            print("    " + "initial stepsize = " + str(init_step))
        if variable != 3:
            print("    " + "stepsize divisor = " + str(step_divisor))
        if variable != 4:
            print("    " + "A = " + str(A))
        if variable != 5:
            print("    " + "B = " + str(B))
    return
print_info(print_values, print_parameters)

# x: ODE-Größen | y: Adjoint | u: Control
# x = [x1, x2, x3, y1, y2, y3, u]
# General Functions
if True:
    def dU(k, xk, uk, yk):
        y1, y2, y3 = yk
        x1, x2, x3 = xk
        return y3 / (2 * B) - uk
    def fitJ(I, kont):
        value = 0
        for i in range(u_sum):
            value += (-A * (I[i] - Pi[i]) ** 2 - B * kont[i] ** 2) * dt
        return value
    def projectJ(I, kont):
        value = 0
        for i in range(u_sum, t_sum):
            value += (-A * (I[i] - Pi[i]) ** 2 - B * kont[i] ** 2) * dt
        return value
    def fit_Error(I):
        value = 0
        for i in range(u_sum):
            value += (I[i] - Pi[i]) ** 2 * dt
        return value 
# RK Functions
if True:
    # I = X1/(X2 * X3)
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

    def Xprime(k, xk):
        x1, x2, x3 = xk
        if k % 1 == 0:
            uk = kont[k]
        else:
            k = int(k)
            uk = (kont[k] + kont[k + 1]) / 2
        sol_k = np.append(xk, (0, 0, 0, uk, 0))
        dx1dt = dx1_dt(sol_k)
        dx2dt = dx2_dt(sol_k)
        dx3dt = dx3_dt(sol_k)
        return np.array([dx1dt, dx2dt, dx3dt])
    def Yprime(k, xk, yk):
        y1, y2, y3 = yk
        x1, x2, x3 = xk
        if k % 1 == 0:
            uk = kont[k]
            pi = Pi[k]
        else:
            k = int(k)
            uk = (kont[k] + kont[k + 1]) / 2
            pi = (Pi[k] + Pi[k + 1]) / 2
        sol_k = np.append(xk, yk)
        sol_k = np.append(sol_k, (uk, pi))
        dy1dt = dy1_dt(sol_k)
        dy2dt = dy2_dt(sol_k)
        dy3dt = dy3_dt(sol_k)
        return np.array([dy1dt, dy2dt, dy3dt])
# MPRK Functions
if True:
    # productive and destructive terms
    if True:
        def p1(k, xk):
            x1, x2, x3 = xk
            p11 = x1/TAU
            return np.array([p11, 0, 0])
        def d1(k, xk):
            x1, x2, x3 = xk
            d11 = x1/(TAU*DEL*x2*(x3 + k_min))
            return np.array([d11, 0, 0])
        def p2(k, xk):
            x1, x2, x3 = xk
            p22 = ALPHA * (1 - x2)
            return np.array([0, p22, 0])
        def d2(k, xk):
            x1, x2, x3 = xk
            d21 = x1 
            return np.array([d21, 0, 0])
        def p3(k, xk):
            x1, x2, x3 = xk
            if k % 1 == 0:
                uk = kont[k]
            else:
                k = int(k)
                uk = (kont[k] + kont[k + 1]) / 2
            p33 = 0
            if uk >= 0: p33 = uk
            return np.array([0, 0, p33])
        def d3(k, xk):
            x1, x2, x3 = xk
            if k % 1 == 0:
                uk = kont[k]
            else:
                k = int(k)
                uk = (kont[k] + kont[k + 1]) / 2
            d33 = 0
            if uk < 0: d33 = -uk
            return np.array([0, 0, d33])
    # x^(1)
    def X_1(k, xk):
        x1, x2, x3 = xk
        P1 = p1(k, xk)
        D1 = d1(k, xk)
        P2 = p2(k, xk)
        D2 = d2(k, xk)
        P3 = p3(k, xk)
        D3 = d3(k, xk)
    
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
        A  = np.eye(3) - dt * A
        x_1 = np.linalg.solve(A, xk)
        return x_1
    # x^(n+1)
    def X_nt1(k, xk):
        x1, x2, x3 = xk
        x_1 = X_1(k, xk)
        x1, x2, x3 = x_1
        P1 = p1(k, xk) + p1(k + 1, x_1)
        D1 = d1(k, xk) + d1(k + 1, x_1)
        P2 = p2(k, xk) + p2(k + 1, x_1)
        D2 = d2(k, xk) + d2(k + 1, x_1)
        P3 = p3(k, xk) + p3(k + 1, x_1)
        D3 = d3(k, xk) + d3(k + 1, x_1)
    
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
        A  = np.eye(3) - dt/2 * A
        x_nt1 = np.linalg.solve(A, xk)
        return x_nt1

# [list of fitJs, list of projectionJs]
RK_J = [[], []]
MPRK_J = [[], []] # TODO fit error
RK_Iterations = []
MPRK_Iterations = []
RK_fit_Error = []
MPRK_fit_Error = []
ts = []
RK_kappa = []
MPRK_kappa = []
for value in values:
    if variable == 0:
        dt = value
    elif variable == 1:
        tolerance = value
    elif variable == 2:
        init_step = value
    elif variable == 3:
        step_divisor = value
    elif variable == 4:
        A = value
    elif variable == 5:
        B = value
    
    # time
    t_sum = int((t_end - t_0) / dt) + 1
    t = np.linspace(t_0, t_end, t_sum)
    u_sum = int((t_u - t_0) / dt) + 1
    # Pi (infectous measured, interpolated)
    Pi = np.interp(t, datapoints[0], datapoints[1])
    
    # RK Initialisation
    if True:
        newJ = -np.inf
        oldJ = -np.inf
        sol = np.zeros((t_sum, 3))
        sol[0] = x0
        adj = np.zeros((t_sum, 3))
        kont = np.zeros(t_sum)
        oldkont = kont.copy()
        
        # forward
        for i in range(t_sum - 1):
            xi = sol[i]
            r1 = Xprime(i, xi)
            r2 = Xprime(i + 1 / 2, xi + dt / 2 * r1)
            r3 = Xprime(i + 1 / 2, xi + dt / 2 * r2)
            r4 = Xprime(i + 1, xi + dt * r3)
            xi1 = xi + (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
            sol[i + 1] = xi1
        # infectuous
        I = sol[:, 0] / (sol[:, 1] * (sol[:, 2] + k_min))
        
        # backward
        for i in range(1, u_sum):
            j = u_sum - i
            tj = t[j]
            xj = sol[j]
            yj = adj[j]
    
            r1 = Yprime(j, xj, yj)
            r2 = Yprime(j - 1 / 2, xj, yj - dt / 2 * r1)
            r3 = Yprime(j - 1 / 2, xj, yj - dt / 2 * r2)
            r4 = Yprime(j - 1, xj, yj - dt * r3)
            yj_1 = yj - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
    
            adj[j - 1] = yj_1
    
        # sweep
        delkont = np.zeros(t_sum)
        for i in range(u_sum):
            xi = sol[i]
            ui = kont[i]
            yi = adj[i]
            delkont[i] = dU(i, xi, ui, yi)
        oldJ = fitJ(I, kont)
        delJ = 0
        projectionJ = projectJ(I, kont)
        
        error = 1
        iterations = 0
        step = init_step
        done = False
    # RK Iteration
    while error >= tolerance and not (done):
        while newJ <= oldJ + Armijo * delJ * step or np.isnan(newJ):  # while (step too large)
            kont = oldkont.copy() + step * delkont.copy()
            old_step = step
            # forward
            for i in range(t_sum - 1):
                xi = sol[i]
                r1 = Xprime(i, xi)
                r2 = Xprime(i + 1 / 2, xi + dt / 2 * r1)
                r3 = Xprime(i + 1 / 2, xi + dt / 2 * r2)
                r4 = Xprime(i + 1, xi + dt * r3)
                xi1 = xi + (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
                sol[i + 1] = xi1
            # infectuous
            with warnings.catch_warnings(): # TODO
                warnings.simplefilter("ignore")
                I = sol[:, 0] / (sol[:, 1] * (sol[:, 2] + k_min))
                newJ = fitJ(I, kont)
            delJ = abs(oldJ - newJ)
            if step <= smallest_step:
                print("(RK) Zu kleine Stepsize bei Iteration " + str(
                    iterations) + " (dt = " + str(dt) + ", tolerance = " + str(
                        tolerance) + ", init_step = " + str(init_step) + ", step_divisor = " + str(
                            step_divisor) + ", A = " + str(A) + ", B = " + str(B) + ")")
                done = True
                break
            else: step = step / step_divisor
        if not done:
            iterations += 1
            error = delJ
            oldJ = newJ
            projectionJ = projectJ(I, kont)
            step = init_step
            oldkont = kont.copy()
        if iterations == last_iteration:
            error = 0
        
        # backward
        for i in range(1, u_sum):
            j = u_sum - i
            tj = t[j]
            xj = sol[j]
            yj = adj[j]
    
            r1 = Yprime(j, xj, yj)
            r2 = Yprime(j - 1 / 2, xj, yj - dt / 2 * r1)
            r3 = Yprime(j - 1 / 2, xj, yj - dt / 2 * r2)
            r4 = Yprime(j - 1, xj, yj - dt * r3)
            yj_1 = yj - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
            
            adj[j - 1] = yj_1
    
        # sweep
        delkont = np.zeros(t_sum)
        for i in range(u_sum):
            xi = sol[i]
            ui = kont[i]
            yi = adj[i]
            delkont[i] = dU(i, xi, ui, yi)
        for i in range(u_sum, t_sum):
            delkont[i] = 0
    RK_Iterations.append(iterations)
    RK_J[0].append(oldJ)
    RK_J[1].append(projectionJ)
    RK_fit_Error.append(fit_Error(I))
    RK_kappa.append(sol[:, 2].tolist())
    
    # MPRK Initialisation
    if True:
        newJ = -np.inf
        oldJ = -np.inf
        sol = np.zeros((t_sum, 3))
        sol[0] = x0
        adj = np.zeros((t_sum, 3))
        kont = np.zeros(t_sum)
        oldkont = kont.copy()
        
        # forward
        for i in range(t_sum - 1):
            xi = sol[i]
            xi1 = X_nt1(i, xi)
            sol[i + 1] = xi1
        # infectuous
        I = sol[:, 0] / (sol[:, 1] * (sol[:, 2] + k_min))
        
        # backward
        for i in range(1, u_sum):
            j = u_sum - i
            tj = t[j]
            xj = sol[j]
            yj = adj[j]
    
            r1 = Yprime(j, xj, yj)
            r2 = Yprime(j - 1 / 2, xj, yj - dt / 2 * r1)
            r3 = Yprime(j - 1 / 2, xj, yj - dt / 2 * r2)
            r4 = Yprime(j - 1, xj, yj - dt * r3)
            yj_1 = yj - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
    
            adj[j - 1] = yj_1
    
        # sweep
        delkont = np.zeros(t_sum)
        for i in range(u_sum):
            xi = sol[i]
            ui = kont[i]
            yi = adj[i]
            delkont[i] = dU(i, xi, ui, yi)
        oldJ = fitJ(I, kont)
        delJ = 0
        projectionJ = projectJ(I, kont)
        
        error = 1
        iterations = 0
        step = init_step
        done = False
    # MPRK Iteration
    while error >= tolerance and not (done):
        while newJ <= oldJ + Armijo * delJ * step or np.isnan(newJ):  # while (step too large)
            kont = oldkont.copy() + step * delkont.copy()
            old_step = step
            # forward
            for i in range(t_sum - 1):
                xi = sol[i]
                with warnings.catch_warnings(): # TODO
                    warnings.simplefilter("ignore")
                    xi1 = X_nt1(i, xi)
                sol[i + 1] = xi1
            # infectuous
            with warnings.catch_warnings(): # TODO
                warnings.simplefilter("ignore")
                I = sol[:, 0] / (sol[:, 1] * (sol[:, 2] + k_min))
                newJ = fitJ(I, kont)
            delJ = abs(oldJ - newJ)
            if step <= smallest_step: 
                print("(MPRK) Zu kleine Stepsize bei Iteration " + str(
                    iterations) + " (dt = " + str(dt) + ", tolerance = " + str(
                        tolerance)  + ", init_step = " + str(init_step) + ", step_divisor = " + str(
                            step_divisor) + ")")
                done = True
                break
            else: step = step / step_divisor
        if not done:
            iterations += 1
            error = delJ
            oldJ = newJ
            projectionJ = projectJ(I, kont)
            step = init_step
            oldkont = kont.copy()
        if iterations == last_iteration:
            error = 0
        
        # backward
        for i in range(1, u_sum):
            j = u_sum - i
            tj = t[j]
            xj = sol[j]
            yj = adj[j]
    
            r1 = Yprime(j, xj, yj)
            r2 = Yprime(j - 1 / 2, xj, yj - dt / 2 * r1)
            r3 = Yprime(j - 1 / 2, xj, yj - dt / 2 * r2)
            r4 = Yprime(j - 1, xj, yj - dt * r3)
            yj_1 = yj - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
            
            adj[j - 1] = yj_1
    
        # sweep
        delkont = np.zeros(t_sum)
        for i in range(u_sum):
            xi = sol[i]
            ui = kont[i]
            yi = adj[i]
            delkont[i] = dU(i, xi, ui, yi)
        for i in range(u_sum, t_sum):
            delkont[i] = 0
    MPRK_Iterations.append(iterations)
    MPRK_J[0].append(oldJ)
    MPRK_J[1].append(projectionJ)
    MPRK_fit_Error.append(fit_Error(I))
    MPRK_kappa.append(sol[:, 2].tolist())
    
    ts.append(t.tolist())
    
    if print_progress: print("Done with " + variables[variable] + " = " + str(value)) # TODO

fixed_parts = ["dt = " + str(dt), 
               "tolerance = " + str(tolerance), 
               "initial stepsize = " + str(init_step), 
               "stepsize divisor = " + str(step_divisor), 
               "A = " + str(A),
               "B = " + str(B)]
fixed = '\n'.join(fixed_parts[:variable] + fixed_parts[variable + 1:])

for RK_plt, MPRK_plt in zip([RK_Iterations, RK_J[0], RK_fit_Error, RK_J[1]],
                            [MPRK_Iterations, MPRK_J[0], MPRK_fit_Error, MPRK_J[1]]):
    fig, ax = plt.subplots()
    at = AnchoredText(fixed, pad = 0.5, borderpad = 2, 
                      loc="lower right", bbox_to_anchor=(180, -16))
    at.patch.set_boxstyle("round,pad=0,rounding_size=0.4")
    ax.add_artist(at)
    ax.plot(values, RK_plt, label="RK")
    ax.plot(values, MPRK_plt, label="MPRK")
    ax.legend(loc="best")
    ax.set_xlabel(variables[variable])
    if variable == 0: ax.set_xscale('log', base = 2)
    elif variable == 1 or variable == 2 or variable > 3: ax.set_xscale('log', base = 10)
    ax.grid()
    if RK_plt == RK_Iterations:
        ax.set_title("# Iterationen in Abhängigkeit von " + variables[variable])
    elif RK_plt == RK_J[0]:
        ax.set_title("J im fit in Abhängigkeit von " + variables[variable])
    elif RK_plt == RK_fit_Error:
        ax.set_title("Fehler im fit in Abhängigkeit von " + variables[variable])
    elif RK_plt == RK_J[1]:
        ax.set_title("J in der Projektion in Abhängigkeit von " + variables[variable])
    plt.tight_layout(pad = 3)
    plt.show()

# save Results
fixed_parts[variable] = variables[variable] + " = " + str(
    values[0]) + "-" + str(values[1]) + "--" + str(values[-1])
result_name = variables[variable] + "|" + "|".join(fixed_parts)
file_name = "fbs_data/" + result_name + '.json'
result = {
    "variable" : variable,
    "variables" : variables,
    "parameters" : {
        variables[0] : dt,
        variables[1] : tolerance,
        variables[2] : init_step,
        variables[3] : step_divisor,
        variables[4] : A,
        variables[5] : B,
    },
    "values" : values,
    "data" : {
        "RK": {
            "Iterations" : RK_Iterations,
            "fitJ" : RK_J[0],
            "konjJ" : RK_J[1],
            "fit Error" : RK_fit_Error,
        },
        "MPRK": {
            "Iterations" : MPRK_Iterations,
            "fitJ" : MPRK_J[0],
            "konjJ" : MPRK_J[1],
            "fit Error" : MPRK_fit_Error,
        },
    },
    "kappa" : {
        "time" : ts,
        "RK" : RK_kappa,
        "MPRK" : MPRK_kappa,
    },
}

result_json = json.dumps(result)
with open(file_name, 'w') as fp:
    json.dump(result_json, fp)

if print_file_name: print(result_name)