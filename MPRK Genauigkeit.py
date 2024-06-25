# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:32:22 2024

@author: Tuchscheerer
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

# TODO: Vector/Matrix Lsg einbauen? (effizienter)
# Generell analysen: wie ist der unterschied Euler zu Patankar Euler in Genauigkeit, Schnelligkeit, max dt

iterations = 3
#dt von 10**-0 bis 10**-dts
positivity = 1

y0 = [0.003, 1]
t0 = 0
tend = 1000

tau = 3.58
a = 0.003
r = 4

def epi(y, t, tau, a, r):
    nu, sus = y
    dydt = np.array([1/tau * (1 - 1/(r * sus))*nu, a * (1 - sus) - nu])
    return dydt

def p11(y, t, tau, a ,r):
    nu, sus = y
    dnudt = 1/tau * nu
    return dnudt

def p12(y, t, tau, a ,r):
    nu, sus = y
    dnudt = 0
    return dnudt

def d11(y, t, tau, a ,r):
    nu, sus = y
    dnudt = 0
    return dnudt

def d12(y, t, tau, a ,r):
    nu, sus = y
    dnudt = 1/(tau * r * sus) * nu
    return dnudt

def p21(y, t, tau, a ,r):
    nu, sus = y
    dsusdt = a * (1)
    return dsusdt

def p22(y, t, tau, a ,r):
    nu, sus = y
    dsusdt = 0
    return dsusdt

def d21(y, t, tau, a ,r):
    nu, sus = y
    dsusdt = nu
    return dsusdt

def d22(y, t, tau, a ,r):
    nu, sus = y
    dsusdt = a* sus
    return dsusdt

ref = np.load('reference.npy')
diff_sum = np.zeros((2, iterations + 1))
dts = []

for j in range(0, iterations + 1):
    dt = 10**-j
    dts.append(dt)
    tsum = int((tend - t0)/dt) + 1
    t = np.linspace(t0, tend, tsum)

    sol = np.zeros((2,tsum))
    sol[0, 0] = y0[0]
    sol[1, 0] = y0[1]

    for i in range(1, tsum):
        ti = t[i]
        nu = sol[0, i-1]
        sus = sol[1, i-1]
        yi_1 = [nu, sus]
        
        if positivity == 1:
            a11 = 1 - dt/yi_1[0] * (p11(yi_1, ti, tau, a, r)
                                    - d11(yi_1, ti, tau, a, r) 
                                    - d12(yi_1, ti, tau, a, r))
            a12 = -dt/yi_1[1] * p12(yi_1, ti, tau, a, r)
            a21 = -dt/yi_1[0] * p21(yi_1, ti, tau, a, r)
            a22 = 1 - dt/yi_1[1] * (p22(yi_1, ti, tau, a, r)
                                    - d21(yi_1, ti, tau, a, r)
                                    - d22(yi_1, ti, tau, a, r))
            # TODO: is A correct? Does it need to be transposed?
            A = np.matrix([[a11,a12],[a21,a22]])
            y1 = np.linalg.solve(A, yi_1)
            
            b11 = 1 - dt/(2 * y1[0]) * ((p11(yi_1, ti, tau, a, r) + p11(y1, ti, tau, a, r))
                                          - (d11(yi_1, ti, tau, a, r) + d11(y1, ti, tau, a, r))
                                          - (d12(yi_1, ti, tau, a, r) + d12(y1, ti, tau, a, r)))
            b12 = -dt/(2 * y1[1]) * (p12(yi_1, ti, tau, a, r) + p12(y1, ti, tau, a, r))
            b21 = -dt/(2 * y1[0]) * (p21(yi_1, ti, tau, a, r) + p21(y1, ti, tau, a, r))
            b22 = 1 - dt/(2 * y1[1]) * ((p22(yi_1, ti, tau, a, r) + p22(y1, ti, tau, a, r))
                                          - (d21(yi_1, ti, tau, a, r) + d21(y1, ti, tau, a, r))
                                          - (d22(yi_1, ti, tau, a, r) + d22(y1, ti, tau, a, r)))
            B = np.matrix([[b11, b12], [b21, b22]])
            yi = np.linalg.solve(B, yi_1)
        else:
            r1 = np.array(epi(yi_1, ti, tau, a, r))
            r2 = np.array(epi(yi_1 + dt/2 * r1, ti + dt/2, tau, a, r))
            r3 = np.array(epi(yi_1 + dt/2 * r2, ti + dt/2, tau, a, r))
            r4 = np.array(epi(yi_1 + dt * r3, ti + dt, tau, a, r))
            yi = np.array(yi_1) + (dt/6 * (r1 + 2*r2 + 2*r3 + r4))
            
        sol[0][i] = yi[0]
        sol[1][i] = yi[1]
    
    tdiff_ref = int((len(ref[2,:]) - 1)/(ref[2, (len(ref[2,:]) - 1)] - ref[2,0]))
    tdiff = int(1/dt)
    diff = np.zeros((2, tend + 1))
    
    
    for i in range(t0, tend + 1):
        diff[0, i] = (sol[0, i * tdiff] - ref[0, i * tdiff_ref])**2
        diff_sum[0, j] += diff[0, i]
        diff[1, i] = (sol[1, i * tdiff] - ref[1, i * tdiff_ref])**2
        diff_sum[1, j] += diff[1, i]
        
    plt.plot(range(0, tend + 1), diff[0,:]  , 'b', label='error in nu(t)')
    plt.plot(range(0, tend + 1), diff[1,:]  , 'g', label='error in prot(t)')
    if positivity == 1:
        plt.title("RMPRK Verfahren punktweiser Fehler mit dt = " + str(dt))
    else:
        plt.title("Runge-Kutta Verfahren punktweiser Fehler mit dt = " + str(dt))
    
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    #plt.yscale('log')
    plt.show()

plt.loglog(dts, diff_sum[0, :], 'navy', label='kummulierter Fehler in nu')
plt.loglog(dts, diff_sum[1, :], 'darkgreen', label='kummulierter Fehler in prot')
if positivity == 1:
    plt.title("MPRK Verfahren kumulierter Fehler respektive dt")
else:
    plt.title("RK Verfahren kumulierter Fehler respektive dt")
plt.legend(loc='best')
plt.xlabel('dt') # TODO
plt.grid()
plt.yscale('log')
#plt.xscale('log')
plt.show()