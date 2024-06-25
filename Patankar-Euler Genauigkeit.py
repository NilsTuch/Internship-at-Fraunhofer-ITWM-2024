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

iterations = 4
#dt von 10**-0 bis 10**-iterations
positivity = 0
show_1 = 0

y0 = [0.003, 1]
t0 = 0
tend = 1000

tau = 3.58
a = 0.003
r = 4

def epi(y, t, tau, a, r):
    nu, sus = y
    dydt = np.array([1/tau * (1 - 1/(r * sus))*nu, nu - a * (1 -sus)])
    return dydt

def P1(y, t, tau, a ,r):
    nu, sus = y
    dnudt_pos = 1/tau * nu
    return dnudt_pos

def P2(y, t, tau, a ,r):
    nu, sus = y
    dsusdt_pos = a * (1 - sus)
    return dsusdt_pos

def D1(y, t, tau, a ,r):
    nu, sus = y
    dnudt_pos = 1/(tau * r * sus) * nu
    return dnudt_pos

def D2(y, t, tau, a ,r):
    nu, sus = y
    dsusdt_pos = nu
    return dsusdt_pos

ref = np.load('reference.npy')
diff_sum = np.zeros((2, iterations))
dts = []

for j in range(0, iterations):
    dt = 10**-(j + 1 - show_1)
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
            sol[0, i] = (nu + dt * P1(yi_1, ti, tau, a, r))/(1 + dt * D1(yi_1, ti, tau, a, r)/nu)
            sol[1, i] = (sus + dt * P2(yi_1, ti, tau, a, r))/(1 + dt * D2(yi_1, ti, tau, a, r)/sus)
        else:
            sol[0, i] = nu + dt * (P1(yi_1, ti, tau, a, r) - D1(yi_1, ti, tau, a, r))
            sol[1, i] = sus + dt * (P2(yi_1, ti, tau, a, r) - D2(yi_1, ti, tau, a, r))
    
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
        plt.title("Patankar - Euler Verfahren punktweiser Fehler mit dt = " + str(dt))
    else:
        plt.title("Euler Verfahren punktweiser Fehler mit dt = " + str(dt))
    
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    #plt.yscale('log')
    plt.show()

plt.loglog(dts, diff_sum[0, :], 'navy', label='kummulierter Fehler in nu')
plt.loglog(dts, diff_sum[1, :], 'darkgreen', label='kummulierter Fehler in prot')
if positivity == 1:
    plt.title("Patankar - Euler Verfahren kumulierter Fehler respektive dt")
else:
    plt.title("Euler Verfahren kumulierter Fehler respektive dt")
plt.legend(loc='best')
plt.xlabel('dt') # TODO
plt.grid()
plt.yscale('log')
#plt.xscale('log')
plt.show()