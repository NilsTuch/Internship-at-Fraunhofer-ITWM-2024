# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : May 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import warnings

# x : initial-ODE | y : Adjoint | u : Control
# A : weight for i-dependent term in J | B : weight for u-dependent term im J
# h : stepsize in u

show_nth = 20  # every show_nth iteration is ploted
patankar = True # = True for MPRK, = False for RK

dt = 4
tol = 1e-6
h_init = 1e-2
h_div = 10

h_min = 1e-13
t_u = 270
Armijo = 1e-3
k_min = 0.04
x_0 = [0.007, 0.993, 0.3 - k_min]
A = 1e1
B = 1e0
ALPHA = 0.03
TAU = 3.58
DEL = 4

# datapoints[0] == time (int), datapoints[1] == i meassured
datapoints = np.load("datapoints.npy")

# time
t_0 = datapoints[0, 0]
t_end = datapoints[0, -1]
t_sum = int((t_end - t_0) / dt) + 1
t = np.linspace(t_0, t_end, t_sum)
u_sum = int((t_u - t_0) / dt) + 1
    
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
def get_u_k(k):
    if k % 1 == 0:
        u_k = u[k]
    else:
        k = int(k)
        u_k = (u[k] + u[k + 1]) / 2
    return u_k
def get_pi_k(k):
    if k % 1 == 0:
        pi = Pi[k]
    else:
        k = int(k)
        pi = (Pi[k] + Pi[k + 1]) / 2
    return pi

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
    u_k = get_u_k(k)
    
    vals_k = np.append(x_k, (0, 0, 0, u_k, 0))
    
    dx1dt = dx1_dt(vals_k)
    dx2dt = dx2_dt(vals_k)
    dx3dt = dx3_dt(vals_k)
    
    return np.array([dx1dt, dx2dt, dx3dt])
def Yprime(k, x_k, y_k):
    y1, y2, y3 = y_k
    x1, x2, x3 = x_k
    u_k = get_u_k(k)
    pi_k = get_pi_k(k)
    
    vals_k = np.append(x_k, y_k)
    vals_k = np.append(vals_k, (u_k, pi_k))
    
    dy1dt = dy1_dt(vals_k)
    dy2dt = dy2_dt(vals_k)
    dy3dt = dy3_dt(vals_k)
    
    return np.array([dy1dt, dy2dt, dy3dt])
def RK_forward(k, x_k):
    r1 = Xprime(k, x_k)
    r2 = Xprime(k + 1 / 2, x_k + dt / 2 * r1)
    r3 = Xprime(k + 1 / 2, x_k + dt / 2 * r2)
    r4 = Xprime(k + 1, x_k + dt * r3)
    
    x_k1 = x_k + (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
    return x_k1
def RK_backward(k, x_k, y_k):
    r1 = Yprime(k, x_k, y_k)
    r2 = Yprime(k - 1 / 2, x_k, y_k - dt / 2 * r1)
    r3 = Yprime(k - 1 / 2, x_k, y_k - dt / 2 * r2)
    r4 = Yprime(k - 1, x_k, y_k - dt * r3)
    
    y_k_1 = y_k - (dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4))
    return y_k_1

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
    u_k = get_u_k(k)
    p33 = 0
    if u_k >= 0: p33 = u_k
    return np.array([0, 0, p33])
def d3(k, x_k):
    x1, x2, x3 = x_k
    u_k = get_u_k(k)
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

# Initialisation ##############################################################
# Pi (Infectous measured, interpolated)
Pi = np.interp(t, datapoints[0], datapoints[1])

new_J = -np.inf
old_J = -np.inf
x = np.zeros((t_sum, 3))
x[0] = x_0
y = np.zeros((t_sum, 3))
u = np.zeros(t_sum)
old_u = u

# Forward
for i in range(t_sum - 1):
    x_i = x[i]
    if patankar:
        x_i1 = X_n1(i, x_i)
    else:
        x_i1 = RK_forward(i, x_i)
    x[i + 1] = x_i1

# Infectious
I = x[:, 0] / (x[:, 1] * (x[:, 2] + k_min))

# Backward
for i in range(1, u_sum):
    j = u_sum - i
    x_j = x[j]
    y_j = y[j]
    y_j_1 = RK_backward(j, x_j, y_j)
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
old_J = J(I, u)
del_J = 0
projection_J = projJ(I, u)
    
# Iteration ###################################################################
error = 1
iterations = 0
h_n = h_init
done = False
while error >= tol and not (done):
    # determine h_nsize for u
    while new_J <= old_J + Armijo * del_J * h_n or np.isnan(new_J):
        u = old_u + h_n * del_u

        # Forward
        for i in range(t_sum - 1):
            x_i = x[i]
            if patankar:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x_i1 = X_n1(i, x_i)
            else:
                x_i1 = RK_forward(i, x_i)
            x[i + 1] = x_i1
        
        # Infectious
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            I = x[:, 0] / (x[:, 1] * (x[:, 2] + k_min))
            
            new_J = J(I, u)
        del_J = abs(old_J - new_J)
        
        if h_n <= h_min: 
            done = True
            break
        else: 
            h_n = h_n / h_div

    if not done:
        iterations += 1
        error = del_J
        old_J = new_J
        projection_J = projJ(I, u)
        h_n = h_init
        old_u = u
    
    # Backward
    for i in range(1, u_sum):
        j = u_sum - i
        x_j = x[j]
        y_j = y[j]
        y_j_1 = RK_backward(j, x_j, y_j)
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
    
    # plots
    if iterations % show_nth == 0 and not done:
        # Value Funktion
        print("J = " + str(old_J) + " after " + str(iterations) + " iterations")
        
        # plot kappa
        plt.axvline(x = t_u, color = "b", linestyle="dashed",label = "end of fit")
        plt.plot(t, x[:, 2] + k_min, "g", label="$\kappa$(t)")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.grid()
        plt.title("$\kappa$ at iteration " + str(iterations))
        plt.show()

        # plot Pi
        plt.axvline(x = t_u, color = "b", linestyle="dashed",label = "end of fit")
        plt.plot(t, Pi[:], "darkorange", label="$\pi$_t")
        plt.plot(t, I[:], "k", label="i(t)")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.grid()
        plt.title("i at Iteration " + str(iterations))
        plt.show()

        # plot u
        plt.axvline(x = t_u, color = "b", linestyle="dashed",label = "end of fit")
        plt.plot(t, u, "violet", label="u(t)")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.grid()
        plt.title("u at iteration " + str(iterations))
        plt.show()
        end = True

if not done:
    print("Reached tol after " + str(iterations) + " iterations  (J = " + str(new_J) 
      + r" and J on [t_u, t_{end}] = " + str(projection_J) + ")")
else:
    print("No further improvement after " + str(iterations) + " iterations")
# plot kappa
plt.axvline(x = t_u, color = "b", linestyle="dashed",label = "end of fit")
plt.plot(t, x[:, 2] + k_min, "g", label="$\kappa$(t)")
plt.legend(loc="best")
plt.xlabel("t")
plt.grid()
plt.title("Solution for $\kappa$")
plt.show()

# plot Pi
plt.axvline(x = t_u, color = "b", linestyle="dashed",label = "end of fit")
plt.plot(t, Pi, "darkorange", label="$\pi_{t}$")
plt.plot(t, I, "k", label="i(t)")
plt.legend(loc="best")
plt.xlabel("t")
plt.grid()
plt.title("Solution for i")
plt.show()

# plot u
plt.axvline(x = t_u, color = "b", linestyle="dashed",label = "end of fit")
plt.plot(t, u, "violet", label="u(t)")
plt.legend(loc="best")
plt.xlabel("t")
plt.grid()
plt.title("Solution for u")
plt.show()   