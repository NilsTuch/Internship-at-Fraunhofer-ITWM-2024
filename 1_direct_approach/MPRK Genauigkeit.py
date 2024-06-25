# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : April 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

iterations = 4 # dt in {10**-0, 10^-1, ...,10**-iterat_ions}
positivity = 0 # 0 for RK, 1 for MPRK

x_0 = [0.003, 1]
t_0 = 0
t_end = 1000

tau = 3.58
a = 0.003
r = 4

def f(x, t):
    nu, sus= x
    dydt = np.array([1/tau * (1 - 1/(r * sus))*nu, a * (1 - sus) - nu])
    return dydt

def p11(x, t):
    nu, sus = x
    dnudt = 1/tau * nu
    return dnudt

def p12(x, t):
    nu, sus = x
    dnudt = 0
    return dnudt

def d11(x, t):
    nu, sus = x
    dnudt = 0
    return dnudt

def d12(x, t):
    nu, sus = x
    dnudt = 1/(tau * r * sus) * nu
    return dnudt

def p21(x, t):
    nu, sus = x
    dsusdt = a * (1)
    return dsusdt

def p22(x, t):
    nu, sus = x
    dsusdt = 0
    return dsusdt

def d21(x, t):
    nu, sus = x
    dsusdt = nu
    return dsusdt

def d22(x, t):
    nu, sus = x
    dsusdt = a* sus
    return dsusdt

ref = np.load('reference.npy')
diff_sum = np.zeros((2, iterations))
dts = []

for j in range(0, iterations):
    dt = 10**-j
    dts.append(dt)
    tsum = int((t_end - t_0)/dt) + 1
    t = np.linspace(t_0, t_end, tsum)

    sol = np.zeros((2,tsum))
    sol[0, 0] = x_0[0]
    sol[1, 0] = x_0[1]

    for i in range(1, tsum):
        t_i = t[i]
        nu = sol[0, i-1]
        sus = sol[1, i-1]
        x_i_1 = [nu, sus]
        
        if positivity == 1:
            a11 = 1 - dt/x_i_1[0] * (p11(x_i_1, t_i)
                                    - d11(x_i_1, t_i) 
                                    - d12(x_i_1, t_i))
            a12 = -dt/x_i_1[1] * p12(x_i_1, t_i)
            a21 = -dt/x_i_1[0] * p21(x_i_1, t_i)
            a22 = 1 - dt/x_i_1[1] * (p22(x_i_1, t_i)
                                    - d21(x_i_1, t_i)
                                    - d22(x_i_1, t_i))
            A = np.matrix([[a11,a12],[a21,a22]])
            x1 = np.linalg.solve(A, x_i_1)
            
            b11 = 1 - dt/(2 * x1[0]) * ((p11(x_i_1, t_i) + p11(x1, t_i))
                                          - (d11(x_i_1, t_i) + d11(x1, t_i))
                                          - (d12(x_i_1, t_i) + d12(x1, t_i)))
            b12 = -dt/(2 * x1[1]) * (p12(x_i_1, t_i) + p12(x1, t_i))
            b21 = -dt/(2 * x1[0]) * (p21(x_i_1, t_i) + p21(x1, t_i))
            b22 = 1 - dt/(2 * x1[1]) * ((p22(x_i_1, t_i) + p22(x1, t_i))
                                          - (d21(x_i_1, t_i) + d21(x1, t_i))
                                          - (d22(x_i_1, t_i) + d22(x1, t_i)))
            B = np.matrix([[b11, b12], [b21, b22]])
            x_i = np.linalg.solve(B, x_i_1)
        else:
            r1 = f(x_i_1, t_i)
            r2 = f(x_i_1 + dt/2 * r1, t_i + dt/2)
            r3 = f(x_i_1 + dt/2 * r2, t_i + dt/2)
            r4 = f(x_i_1 + dt * r3, t_i + dt)
            x_i = x_i_1 + (dt/6 * (r1 + 2*r2 + 2*r3 + r4))
            
        sol[0][i] = x_i[0]
        sol[1][i] = x_i[1]
    
    tdiff_ref = int((len(ref[2,:]) - 1)/(ref[2, (len(ref[2,:]) - 1)] - ref[2,0]))
    tdiff = int(1/dt)
    diff = np.zeros((2, t_end + 1))
    
    
    for i in range(t_0, t_end + 1):
        diff[0, i] = (sol[0, i * tdiff] - ref[0, i * tdiff_ref])**2
        diff_sum[0, j] += diff[0, i]
        diff[1, i] = (sol[1, i * tdiff] - ref[1, i * tdiff_ref])**2
        diff_sum[1, j] += diff[1, i]
        
    plt.plot(range(0, t_end + 1), diff[0,:]  , 'b', label='error in nu(t)')
    plt.plot(range(0, t_end + 1), diff[1,:]  , 'g', label='error in prot(t)')
    if positivity == 1:
        plt.title("RMPRK Verfahren punktweiser Fehler mit dt = " + str(dt))
    else:
        plt.title("Runge-Kutta Verfahren punktweiser Fehler mit dt = " + str(dt))
    
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.yscale('log')
    plt.show()

plt.loglog(dts, diff_sum[0, :], 'navy', label='kummulierter Fehler in nu')
plt.loglog(dts, diff_sum[1, :], 'darkgreen', label='kummulierter Fehler in prot')
if positivity == 1:
    plt.title("MPRK Verfahren kumulierter Fehler respektive dt")
else:
    plt.title("RK Verfahren kumulierter Fehler respektive dt")
plt.legend(loc='best')
plt.xlabel('dt')
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.show()