# -*- coding: utf-8 -*-
"""
Author  : nils.t_03@gmail.com
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
syn_amp = 0.05 # e.g. 0.1 diverges, 0.05 converges

#A: weight in J for x-term, B: weight in J for u-term
A = 1
B = 1
ALPHA = .03
TAU = 3.58
DEL = 4

tsum = int((t_end - t_0)/dt) + 1
t = np.linspace(t_0, t_end, tsum)

syn_kont = np.zeros(tsum)
for i in range(tsum):
    syn_kont[i] += syn_amp * np.cos(i*dt/(10*np.pi))/(10*np.pi)
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
    u_syn = syn_kont[tk]
    y1, y2, y3 = yk
    x1, x2, x3 = xk
    return y3/(2*B*1)# + u_syn
def J(time, sol, kont):
    value = 0
    value += y_end[0] * sol[tsum -1 , 0] + y_end[1] * sol[tsum -1, 1] + y_end[2] * sol[tsum -1, 2]
    for i in range(tsum):
        value += (- A * (sol[i, 0] - oldsol[i, 0])**2 - B * kont[i]**2) * dt
    return value

# init ################################################################# init #
variables = 7
oldsol = np.zeros((tsum, variables))
oldsol[0, :3] = x_0
oldsol[-1, 3:6] = y_end
oldsol[:, 6] = kont
for i in range(tsum - 1):
    ti = t[i]
    xi = oldsol[i, :3]
    
    r1 = Xprime(i, xi)
    r2 = Xprime(i + 1/2, xi + dt/2 * r1)
    r3 = Xprime(i + 1/2, xi + dt/2 * r2)
    r4 = Xprime(i + 1, xi + dt * r3) # this causes the num_err
    xx1i = xi + (dt/6 * (r1 + 2*r2 + 2*r3 + r4))
    
    oldsol[i + 1, :3] = xx1i

# infectuous
# Inf = sol[:, 0]/(kont * sol[:, 1])

for i in range(1, tsum):
    j = tsum  - i
    tj = t[j]
    xj = oldsol[j, :3]
    yj = oldsol[j, 3:6]
    
    r1 = Yprime(j, xj, yj)
    r2 = Yprime(j - 1/2, xj, yj - dt/2 * r1)
    r3 = Yprime(j - 1/2, xj, yj - dt/2 * r2)
    r4 = Yprime(j - 1, xj, yj - dt * r3)
    yj_1 = yj - (dt/6 * (r1 + 2*r2 + 2*r3 + r4))
    
    oldsol[j - 1, 3:6] = yj_1
for i in range(tsum):
    ti = i
    xi = oldsol[i, :3]
    ui = oldsol[i, 6]
    yi = oldsol[i, 3:6]
    oldsol[i, 6] = (oldsol[i, 6] + U(ti, xi, ui, yi))/2
syn_sol = oldsol.copy()

oldsol[:, 6] = kont*0
# Value Funktion
newJ = J(t, oldsol, kont)
print('J = ' + str(newJ))
plt.plot(t, oldsol[:, 0], 'b', label='x1(t)')
plt.plot(t, oldsol[:, 1], 'r', label='x2(t)')
plt.plot(t, oldsol[:, 2], 'g', label='x3(t)')
plt.title("FWB initial Lösung")
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

# plt.plot(t_init, oldsol[:, 3], 'b', label='y1(t)')
# plt.plot(t_init, oldsol[:, 4], 'r', label='y2(t)')
# plt.plot(t_init, oldsol[:, 5], 'g', label='y3(t)')
# plt.title("FWB initial Lösung")
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.grid()
# plt.show()

# plt.plot(t_init, oldsol[:, 6], 'b', label='u(t)')
# plt.title("FWB initial Lösung")
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.grid()
# plt.show()
###############################################################################
# x = [x1, x2, x3, y1, y2, y3, u]
# Def Funktions
if True:
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
    Df_fin_diff_seg = np.zeros((variables, tsum, tsum))
    for j in range(variables-1):
        for i in range(1, tsum - 1): #dgl, adj t1,..,t_end-1
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
    o = np.zeros((tsum, tsum))
    Df_fin_diff = np.block([
        [Df_fin_diff_seg[0], o, o, o, o, o, o],
        [o, Df_fin_diff_seg[1], o, o, o, o, o],
        [o, o, Df_fin_diff_seg[2], o, o, o, o],
        [o, o, o, Df_fin_diff_seg[3], o, o, o],
        [o, o, o, o, Df_fin_diff_seg[4], o, o],
        [o, o, o, o, o, Df_fin_diff_seg[5], o],
        [o, o, o, o, o, o, Df_fin_diff_seg[6]]])
    # dgl ############################################################### dgl #
    Df_dgl = np.zeros((tsum*variables, tsum*variables))
    for x1i in range(0, tsum):
        x2i = x1i + tsum
        x3i = x2i + tsum
        y1i = x3i + tsum
        y2i = y1i + tsum
        y3i = y2i + tsum
        ui = y3i + tsum
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
            Df_dgl[x3i, ui] = dx3_dt_d[6](sol_i)
        if x1i != tsum -1:
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
        Df_dgl[ui, y3i] = 1
        Df_dgl[ui, ui] = -2*B
        # Df_dgl[ui,:] *= 100
    # combining
    Df = Df_fin_diff - Df_dgl
    return Df, Df_dgl, Df_fin_diff
def getf(sol):
    # fin diff ##################################################### fin diff #
    f_fin_diff = np.zeros((variables, tsum))
    for j in range(variables - 1):
        for i in range(1, tsum - 1):
            f_fin_diff[j, i] = sol[i + 1, j] - sol[i - 1, j] # x1 t1,..,t_end-1
        if(j in [0, 1, 2]): #dgl t_0, t_end
            f_fin_diff[j, -1] = 1*sol[-3, j] - 4*sol[-2, j] + 3*sol[-1, j] # x1 t_end
        elif(j in [3, 4, 5]): #adj t_0, t_end
            f_fin_diff[j, 0] = 1*sol[2, j] - 4*sol[1, j] + 3*sol[0, j] # x1 t_end
    f_fin_diff = f_fin_diff/(2*dt)
    f_fin_diff = f_fin_diff.reshape(tsum*variables)
    # dgl ############################################################### dgl #
    f_dgl = np.zeros((variables, tsum))
    # t1,..,t_end-1
    for i in range(0, tsum):
        # x1, x2, x3, y1, y2, y3, u = sol[i]
        x1s = syn_sol[i, 0]
        sol_i = np.append(sol[i], x1s)
        if i != 0:
            f_dgl[0, i] = dx1_dt(sol_i) # x1
            f_dgl[1, i] = dx2_dt(sol_i) # x2
            f_dgl[2, i] = dx3_dt(sol_i) #x3
        if i != tsum - 1:
            f_dgl[3, i] = dy1_dt(sol_i) #y1
            f_dgl[4, i] = dy2_dt(sol_i) #y2
            f_dgl[5, i] = dy3_dt(sol_i) #y3
        f_dgl[6, i] = dH_du(sol_i) #u
    # combining
    f_dgl = f_dgl.reshape(tsum*variables)
    f = f_fin_diff - f_dgl
    return f, f_dgl, f_fin_diff

sol = oldsol.copy() * .9#np.ones((tsum, variables))*.1 #
# sol[:, 6] = sol[:, 6]*0
# sol[0] = x_0

iterations = 0
error = 1
while error > tol:
    oldsol = sol.copy()
    Df, Df_dgl, Df_fin_diff = getDf(sol)
    f, f_dgl, f_fin_diff = getf(sol)

    delsol = np.linalg.solve(Df, -f).reshape((tsum, variables), order='F')
    # delsol[:, 6] = delsol[:, 6]*1e-12*0
    sol += delsol
    
    for i in range(variables):
        error = max(abs(delsol[:, i]))
    error = 1
    iterations += 1
    if(iterations == last_iteration): error = 0
    
    if iterations % show_nth == 0:
        # plot nu, sus, k
        plt.plot(t, sol[:, 0], 'b', label='nu(t)')
        plt.plot(t, sol[:, 1], 'r', label='sus(t)')
        plt.plot(t, sol[:, 2], 'g', label='kappa(t)')
        plt.plot(t, syn_sol[:, 0], 'cyan')
        plt.plot(t, syn_sol[:, 1], 'coral')
        plt.plot(t, syn_sol[:, 2], 'palegreen')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.title('Fin Diff Solution at iteration ' + str(iterations))
        plt.show()
        
        # # plot k
        # plt.plot(t[:int(75/dt)], sol[:int(75/dt), 2], 'g', label='kappa(t)')
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.grid()
        # plt.title('Fin Diff Solution at iteration ' + str(iterations))
        # plt.show()
        
        # # plot adjoint
        # plt.plot(t, sol[:, 3], 'cyan', label='y1(t)')
        # plt.plot(t, sol[:, 4], 'coral', label='y2(t)')
        # plt.plot(t, sol[:, 5], 'palegreen', label='y3(t)')
        # plt.plot(t, syn_sol[:, 3], 'b')
        # plt.plot(t, syn_sol[:, 4], 'r')
        # plt.plot(t, syn_sol[:, 5], 'g')
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.grid()
        # plt.title('Fin Diff Adjoint at ' + str(iterations) + ' with dt ' + str(dt))
        # plt.show()
        
        # # plot kont
        # plt.plot(t, sol[:, 6], 'violet', label='u(t)')
        # plt.plot(t, oldsol[:, 6], 'k')
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.grid()
        # plt.title('Fin Diff Solution for u at iteration ' + str(iterations))
        # plt.show()

        sumf = 0
        for fi in f:
            sumf += abs(fi)
        print("Sum of f is ", sumf, " at iteration ", iterations)
        newJ = J(t, sol, sol[:, 6])
        print('J = ', str(newJ))