# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : May 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy as sp

# print info
show_nth = 1  # every show_nth iteration is ploted
last_iteration = -1 # if last_iteration >=1 the program stops after that many iterations

dt= 1
tol= 1e-16

x_0 = [.1, .9, .3]
y_end = [0, 0, 0]

t_0 = 0
t_end = 300
syn_amp = 0.05 # = 0.1 diverges, = 0.05 converges

#A: weight in J for x-term, B: weight in J for u-term
A = 1
B = 1
ALPHA = .03
TAU = 3.58
DEL = 4

t_sum = int((t_end - t_0)/dt) + 1
t = np.linspace(t_0, t_end, t_sum)

syn_kont = syn_amp * np.cos(t/(10 * np.pi))/(10 * np.pi)
kont = syn_kont.copy() 

def Xprime(k, xk):
    x1, x2, x3 = xk
    if k%1 == 0:
        uk = kont[k]
    else:
        k = int(k)
        uk = (kont[k] + kont[k + 1])/2
    dx1dt = x1/TAU * (1 - 1/(DEL * x2 * x3))
    dx2dt = ALPHA*(1 - x2) - x1
    dx3dt = uk
    return np.array([dx1dt, dx2dt, dx3dt])
def Yprime(k, xk, yk):
    y1, y2, y3 = yk
    x1, x2, x3 = xk
    if k%1 == 0:
        uk = kont[k]
        z = oldsol[k, 0]
    else:
        k = int(k)
        uk = (kont[k] + kont[k + 1])/2
        z = (oldsol[k, 0] + oldsol[k + 1, 0])/2
    dy1dt = y1/TAU * (1/(DEL * x2 * x3) - 1) + y2 + 2 * A * (x1 - z)
    dy2dt = -(x1 * y1)/(TAU * DEL * x2**2 * x3) + ALPHA * y2
    dy3dt = -(x1 * y1)/(TAU * DEL * x2 * x3**2)
    return np.array([dy1dt, dy2dt, dy3dt])
def U(tk, xk, uk, yk):
    y1, y2, y3 = yk
    x1, x2, x3 = xk
    return y3/(2*B*1)
def J(time, sol, kont):
    value = 0
    value += y_end[0] * sol[t_sum -1 , 0] + y_end[1] * sol[t_sum -1, 1] + y_end[2] * sol[t_sum -1, 2]
    for i in range(t_sum):
        value += (- A * (sol[i, 0] - oldsol[i, 0])**2 - B * kont[i]**2) * dt
    return value

# init ################################################################# init #
variables = 7
oldsol = np.zeros((t_sum, variables))
oldsol[0, :3] = x_0
oldsol[-1, 3:6] = y_end
oldsol[:, 6] = kont

for i in range(t_sum - 1):
    t_i = t[i]
    x_i = oldsol[i, :3]
    
    r1 = Xprime(i, x_i)
    r2 = Xprime(i + 1/2, x_i + dt/2 * r1)
    r3 = Xprime(i + 1/2, x_i + dt/2 * r2)
    r4 = Xprime(i + 1, x_i + dt * r3) # this causes the num_err
    x_1_i = x_i + (dt/6 * (r1 + 2*r2 + 2*r3 + r4))
    
    oldsol[i + 1, :3] = x_1_i

for i in range(1, t_sum):
    j = t_sum  - i
    t_j = t[j]
    x_j = oldsol[j, :3]
    y_j = oldsol[j, 3:6]
    
    r1 = Yprime(j, x_j, y_j)
    r2 = Yprime(j - 1/2, x_j, y_j - dt/2 * r1)
    r3 = Yprime(j - 1/2, x_j, y_j - dt/2 * r2)
    r4 = Yprime(j - 1, x_j, y_j - dt * r3)
    y_j_1 = y_j - (dt/6 * (r1 + 2*r2 + 2*r3 + r4))
    
    oldsol[j - 1, 3:6] = y_j_1
for i in range(t_sum):
    t_i = i
    x_i = oldsol[i, :3]
    u_i = oldsol[i, 6]
    y_i = oldsol[i, 3:6]
    oldsol[i, 6] = (oldsol[i, 6] + U(t_i, x_i, u_i, y_i))/2
syn_sol = oldsol.copy()
###############################################################################
# x = [x1, x2, x3, y1, y2, y3, u]
# Def Funktions
X1, X2, X3, Y1, Y2, Y3, U, X1s = sp.symbols("X1, X2, X3, Y1, Y2, Y3, U, X1s")
Symbols = X1, X2, X3, Y1, Y2, Y3, U, X1s
dX1_dt = X1/TAU * (1 - 1/(DEL * X2 * X3))
dX2_dt = ALPHA * (1 - X2) - X1
dX3_dt = U
r = -A * (X1 - X1s)**2 - B * U**2
H = dX1_dt * Y1 + dX2_dt * Y2 + dX3_dt * Y3 + r
dY1_dt = -sp.diff(H, X1)
dY2_dt = -sp.diff(H, X2)
dY3_dt = -sp.diff(H, X3)

dX1_dt_d = []
dX2_dt_d = []
dX3_dt_d = []
dY1_dt_d = []
dY2_dt_d = []
dY3_dt_d = []
for i in range(variables):
    dX1_dt_d.append(sp.diff(dX1_dt, Symbols[i]))
    dX2_dt_d.append(sp.diff(dX2_dt, Symbols[i]))
    dX3_dt_d.append(sp.diff(dX3_dt, Symbols[i]))
    dY1_dt_d.append(sp.diff(dY1_dt, Symbols[i]))
    dY2_dt_d.append(sp.diff(dY2_dt, Symbols[i]))
    dY3_dt_d.append(sp.diff(dY3_dt, Symbols[i]))
dH_dU = sp.diff(H, U)

#numpy / lambdification
dx1_dt = sp.lambdify([Symbols], dX1_dt,"numpy")
dx2_dt = sp.lambdify([Symbols], dX2_dt,"numpy")
dx3_dt = sp.lambdify([Symbols], dX3_dt,"numpy")
dy1_dt = sp.lambdify([Symbols], dY1_dt,"numpy")
dy2_dt = sp.lambdify([Symbols], dY2_dt,"numpy")
dy3_dt = sp.lambdify([Symbols], dY3_dt,"numpy")
dx1_dt_d = []
dx2_dt_d = []
dx3_dt_d = []
dy1_dt_d = []
dy2_dt_d = []
dy3_dt_d = []
dH_du = sp.lambdify([Symbols], dH_dU, "numpy")
for i in range(variables):
    dx1_dt_d.append(sp.lambdify([Symbols], dX1_dt_d[i],"numpy"))
    dx2_dt_d.append(sp.lambdify([Symbols], dX2_dt_d[i],"numpy"))
    dx3_dt_d.append(sp.lambdify([Symbols], dX3_dt_d[i],"numpy"))
    dy1_dt_d.append(sp.lambdify([Symbols], dY1_dt_d[i],"numpy"))
    dy2_dt_d.append(sp.lambdify([Symbols], dY2_dt_d[i],"numpy"))
    dy3_dt_d.append(sp.lambdify([Symbols], dY3_dt_d[i],"numpy"))

def getDf(sol):
    # A[equation, variable] = coeffitient
    # sol[i, var] = [x1, x2, x3]
    # fin diff ##################################################### fin diff #
    Df_fin_diff_seg = np.zeros((variables, t_sum, t_sum))
    for j in range(variables-1):
        for i in range(1, t_sum - 1): #dgl, adj t1,..,t_end-1
            Df_fin_diff_seg[j, i, i - 1] = -1
            Df_fin_diff_seg[j, i, i + 1] = 1
        if(j in [0, 1, 2]): #dgl t_0, t_end
            Df_fin_diff_seg[j, -1, - 1] = 3
            Df_fin_diff_seg[j, -1, - 2] = -4
            Df_fin_diff_seg[j, -1, - 3] = 1
            Df_fin_diff_seg[j, 0, 0] = 2*dt
        elif(j in [3, 4, 5]): #adj t_0, t_end
                Df_fin_diff_seg[j, 0, 0] = 3
                Df_fin_diff_seg[j, 0, 1] = -4
                Df_fin_diff_seg[j, 0, 2] = 1
                Df_fin_diff_seg[j, -1, -1] = 2*dt
    Df_fin_diff_seg = Df_fin_diff_seg/(2*dt)
    o = np.zeros((t_sum, t_sum))
    Df_fin_diff = np.block([
        [Df_fin_diff_seg[0], o, o, o, o, o, o],
        [o, Df_fin_diff_seg[1], o, o, o, o, o],
        [o, o, Df_fin_diff_seg[2], o, o, o, o],
        [o, o, o, Df_fin_diff_seg[3], o, o, o],
        [o, o, o, o, Df_fin_diff_seg[4], o, o],
        [o, o, o, o, o, Df_fin_diff_seg[5], o],
        [o, o, o, o, o, o, Df_fin_diff_seg[6]]])
    # dgl ############################################################### dgl #
    Df_dgl = np.zeros((t_sum*variables, t_sum*variables))
    for x1i in range(0, t_sum):
        x2i = x1i + t_sum
        x3i = x2i + t_sum
        y1i = x3i + t_sum
        y2i = y1i + t_sum
        y3i = y2i + t_sum
        u_i = y3i + t_sum
        sol_i = np.append(sol[x1i], 0)
        if x1i != 0:
            # x1
            Df_dgl[x1i, x1i] = dx1_dt_d[0](sol_i)
            Df_dgl[x1i, x2i] = dx1_dt_d[1](sol_i)
            Df_dgl[x1i, x3i] = dx1_dt_d[2](sol_i)
            # x2
            Df_dgl[x2i, x1i] = dx2_dt_d[0](sol_i)
            Df_dgl[x2i, x2i] = dx2_dt_d[1](sol_i)
            # x3
            Df_dgl[x3i, u_i] = dx3_dt_d[6](sol_i)
        if x1i != t_sum -1:
            #y1
            Df_dgl[y1i, x1i] = dy1_dt_d[0](sol_i)
            Df_dgl[y1i, x2i] = dy1_dt_d[1](sol_i)
            Df_dgl[y1i, x3i] = dy1_dt_d[2](sol_i)
            Df_dgl[y1i, y1i] = dy1_dt_d[3](sol_i)
            Df_dgl[y1i, y2i] = dy1_dt_d[4](sol_i)
            #y2
            Df_dgl[y2i, x1i] = dy2_dt_d[0](sol_i)
            Df_dgl[y2i, x2i] = dy2_dt_d[1](sol_i)
            Df_dgl[y2i, x3i] = dy2_dt_d[2](sol_i)
            Df_dgl[y2i, y1i] = dy2_dt_d[3](sol_i)
            Df_dgl[y2i, y2i] = dy2_dt_d[4](sol_i)
            #y3
            Df_dgl[y3i, x1i] = dy3_dt_d[0](sol_i)
            Df_dgl[y3i, x2i] = dy3_dt_d[1](sol_i)
            Df_dgl[y3i, x3i] = dy3_dt_d[2](sol_i)
            Df_dgl[y3i, y1i] = dy3_dt_d[3](sol_i)
        #u
        Df_dgl[u_i, y3i] = 1
        Df_dgl[u_i, u_i] = -2*B
        # Df_dgl[u_i,:] *= 100
    # combining
    Df = Df_fin_diff - Df_dgl
    return Df

def getf(sol):
    # fin diff ##################################################### fin diff #
    f_fin_diff = np.zeros((variables, t_sum))
    for j in range(variables - 1):
        for i in range(1, t_sum - 1):
            f_fin_diff[j, i] = sol[i + 1, j] - sol[i - 1, j] # x1 t1,..,t_end-1
        if(j in [0, 1, 2]): #dgl t_0, t_end
            f_fin_diff[j, -1] = 1*sol[-3, j] - 4*sol[-2, j] + 3*sol[-1, j] # x1 t_end
        elif(j in [3, 4, 5]): #adj t_0, t_end
            f_fin_diff[j, 0] = 1*sol[2, j] - 4*sol[1, j] + 3*sol[0, j] # x1 t_end
    f_fin_diff = f_fin_diff/(2*dt)
    f_fin_diff = f_fin_diff.reshape(t_sum*variables)
    # dgl ############################################################### dgl #
    f_dgl = np.zeros((variables, t_sum))
    # t1,..,t_end-1
    for i in range(0, t_sum):
        # x1, x2, x3, y1, y2, y3, u = sol[i]
        x1s = syn_sol[i, 0]
        sol_i = np.append(sol[i], x1s)
        if i != 0:
            f_dgl[0, i] = dx1_dt(sol_i) # x1
            f_dgl[1, i] = dx2_dt(sol_i) # x2
            f_dgl[2, i] = dx3_dt(sol_i) #x3
        if i != t_sum - 1:
            f_dgl[3, i] = dy1_dt(sol_i) #y1
            f_dgl[4, i] = dy2_dt(sol_i) #y2
            f_dgl[5, i] = dy3_dt(sol_i) #y3
        f_dgl[6, i] = dH_du(sol_i) #u
    # combining
    f_dgl = f_dgl.reshape(t_sum*variables)
    f = f_fin_diff - f_dgl
    return f

sol = oldsol.copy()

iterations = 0
error = 1
while error > tol:
    oldsol = sol.copy()
    Df = getDf(sol)
    f = getf(sol)

    delsol = np.linalg.solve(Df, -f).reshape((t_sum, variables), order='F')
    sol += delsol
    
    error = 1
    var_error = 0
    for i in range(variables):
        error = max(var_error, max(abs(delsol[:, i])))
    iterations += 1
    if(iterations == last_iteration): error = 0
    
    if iterations % show_nth == 0:
        # print
        sumf = 0
        for fi in f:
            sumf += abs(fi)
        print("Sum of f is ", sumf, " at iteration ", iterations)
        newJ = J(t, sol, sol[:, 6])
        print('J = ', str(newJ))
        
        # plot nu, sus, k
        plt.plot(t, sol[:, 0], 'b', label='$\\nu(t)$')
        plt.plot(t, sol[:, 1], 'r', label='s(t)')
        plt.plot(t, sol[:, 2], 'g', label='$\kappa(t)$')
        plt.plot(t, syn_sol[:, 0], 'cyan')
        plt.plot(t, syn_sol[:, 1], 'coral')
        plt.plot(t, syn_sol[:, 2], 'palegreen')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.title('Finite-Difference-Method at iteration ' + str(iterations))
        plt.show()
        
# plot nu, sus, k
plt.plot(t, sol[:, 0], 'b', label='$\\nu(t)$')
plt.plot(t, sol[:, 1], 'r', label='s(t)')
plt.plot(t, sol[:, 2], 'g', label='$\kappa(t)$')
plt.plot(t, syn_sol[:, 0], 'cyan')
plt.plot(t, syn_sol[:, 1], 'coral')
plt.plot(t, syn_sol[:, 2], 'palegreen')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.title('Finite-Difference-Method Solution for $(\\nu(t), s(t), \kappa(t))^T$')
plt.show()
        
# plot kont
plt.plot(t, syn_kont, 'purple', label='$u_{syn}(t)$')
plt.plot(t, sol[:, 6], 'violet', label='u(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.title('Finite-Difference-Method Solution for u')
plt.show()