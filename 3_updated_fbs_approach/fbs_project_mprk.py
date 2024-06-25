# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : May 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import warnings

nth_plot = 500
last_iteration = -1
positivity = True

dt = 8
target_error = 1e-8
t_u = 270 #normally: 270
init_step = 1e-2
step_divisor = 10
smallest_step = 1e-13
Armijo = 1e-3

# A: weight in J for i-term | B: weight in J for u-term
k_min = 0.04 #with init=1e-2, divisor 2: 
    #   0.02 enough for dt=1, 0.04 enough for dt=6, 0.06 enough for dt=10, 0.2 enought for dt=20
x0 = [0.007, 0.993, 0.3 - k_min] # nu_0 value outside of whats in the paper (0.003 +/- 0.002)
A = 1e1
B = 1e0
if True:
    ALPHA = 0.03
    TAU = 3.58
    DEL = 4

# datapoints[0] == time (int), datapoints[1] == i meassured =~ Pi
datapoints = np.load("datapoints.npy")

# time
if True:
    t_0 = datapoints[0, 0]
    t_end = datapoints[0, datapoints[0].size - 1]
    t_sum = int((t_end - t_0) / dt) + 1
    t = np.linspace(t_0, t_end, t_sum)
    u_sum = int((t_u - t_0) / dt) + 1

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
    
# Initialisation
if True:
    # Pi (infectous measured, interpolated)
    Pi = np.interp(t, datapoints[0], datapoints[1])
    
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
        if positivity:
            xi1 = X_nt1(i, xi)
        else:
            r1 = Xprime(i, xi)
            r2 = Xprime(i + 1 / 2, xi + dt / 2 * r1)
            r3 = Xprime(i + 1 / 2, xi + dt / 2 * r2)
            r4 = Xprime(i + 1, xi + dt * r3)  # this causes the num_err
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
    for i in range(u_sum, t_sum):
        delkont[i] = 0
    oldJ = fitJ(I, kont)
    delJ = 0
    projectionJ = projectJ(I, kont)
    
# Iteration
error = 1
iterations = 0
step = init_step
done = False
while error >= target_error and not (done):
    while newJ <= oldJ + Armijo * delJ * step or np.isnan(newJ):  # while (step too large)
        kont = oldkont.copy() + step * delkont.copy()
        old_step = step

        # forward
        for i in range(t_sum - 1):
            xi = sol[i]
            if positivity:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xi1 = X_nt1(i, xi)
            else:
                r1 = Xprime(i, xi)
                r2 = Xprime(i + 1 / 2, xi + dt / 2 * r1)
                r3 = Xprime(i + 1 / 2, xi + dt / 2 * r2)
                r4 = Xprime(i + 1, xi + dt * r3)  # this causes the num_err
                xi1 = xi + (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
            sol[i + 1] = xi1
        # infectuous
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            I = sol[:, 0] / (sol[:, 1] * (sol[:, 2] + k_min))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            newJ = fitJ(I, kont)
        delJ = abs(oldJ - newJ)
        if step <= smallest_step: 
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
    
    # plots
    if iterations % nth_plot == 0 and not done:
        # Value Funktion
        print("At iteration " + str(iterations) + ": fitJ = " + str(oldJ) 
              + " and projectionJ = " + str(projectionJ))
        
        # plot kappa
        plt.axvline(x = t_u, color = 'b', linestyle='dashed',label = 'end of fit')
        plt.plot(t, sol[:, 2] + k_min, 'g', label='$\kappa$(t)')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.title('$\kappa$ at iteration ' + str(iterations))
        plt.show()

        # plot Pi
        plt.axvline(x = t_u, color = 'b', linestyle='dashed',label = 'end of fit')
        plt.plot(t, Pi[:], "darkorange", label="$\pi$_t")
        plt.plot(t, I[:], "k", label="i")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.grid()
        plt.title("i at Iteration " + str(iterations))
        plt.show()

        # plot kont
        plt.axvline(x = t_u, color = 'b', linestyle='dashed',label = 'end of fit')
        plt.plot(t, kont, 'violet', label='u(t)')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.title('u at iteration ' + str(iterations))
        plt.show()
        end = True

print("Konvergenz nach " + str(iterations) + " Iterationen mit fitJ = " + str(newJ) 
      + " und projectionJ = " + str(projectionJ))
# plot kappa
plt.axvline(x = t_u, color = 'b', linestyle='dashed',label = 'end of fit')
plt.plot(t, sol[:, 2] + k_min, "g", label="$\kappa$(t)")
plt.legend(loc="best")
plt.xlabel("t")
plt.grid()
plt.title(r"Solution for $\kappa$")
plt.show()

# plot Pi
plt.axvline(x = t_u, color = 'b', linestyle='dashed',label = 'end of fit')
plt.plot(t, Pi, "darkorange", label="$\pi_{t}$")
plt.plot(t, I, "k", label="i(t)")
plt.legend(loc="best")
plt.xlabel("t")
plt.grid()
plt.title("Solution for i")
plt.show()

# plot kont
plt.axvline(x = t_u, color = 'b', linestyle='dashed',label = 'end of fit')
plt.plot(t, kont, 'violet', label='u(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.title('Solution for u')
plt.show()   