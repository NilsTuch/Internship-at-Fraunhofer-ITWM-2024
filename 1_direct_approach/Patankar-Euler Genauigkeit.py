# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : April 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

iterations = 3 # dt in {10**-0, 10^-1, ...,10**-iterat_ions}
positivity = 1 # 0 for RK, 1 for MPRK

x_0 = [0.003, 1]
t_0 = 0
t_end = 1000

tau = 3.58
a = 0.003
r = 4

def f(x, t):
    nu, sus = x
    dydt = np.array([1/tau * (1 - 1/(r * sus))*nu, nu - a * (1 -sus)])
    return np.array(dydt)

def P1(x, t):
    nu, sus = x
    dnudt_pos = 1/tau * nu
    return dnudt_pos

def P2(x, t):
    nu, sus = x
    dsusdt_pos = a * (1 - sus)
    return dsusdt_pos

def D1(x, t):
    nu, sus = x
    dnudt_pos = 1/(tau * r * sus) * nu
    return dnudt_pos

def D2(x, t):
    nu, sus = x
    dsusdt_pos = nu
    return dsusdt_pos

reference = np.load('reference.npy')
diff_sum = np.zeros((2, iterations))
dts = []

for j in range(0, iterations):
    dt = 10**-j
    dts.append(dt)
    t_sum = int((t_end - t_0)/dt) + 1
    t = np.linspace(t_0, t_end, t_sum)

    sol = np.zeros((2,t_sum))
    sol[0, 0] = x_0[0]
    sol[1, 0] = x_0[1]

    for i in range(1, t_sum):
        t_i = t[i]
        nu = sol[0, i-1]
        sus = sol[1, i-1]
        x_i_1 = [nu, sus]
        
        if positivity == 1:
            sol[0, i] = (nu + dt * P1(x_i_1, t_i))/(1 + dt * D1(x_i_1, t_i)/nu)
            sol[1, i] = (sus + dt * P2(x_i_1, t_i))/(1 + dt * D2(x_i_1, t_i)/sus)
        else:
            sol[0, i] = nu + dt * (P1(x_i_1, t_i) - D1(x_i_1, t_i))
            sol[1, i] = sus + dt * (P2(x_i_1, t_i) - D2(x_i_1, t_i))
    
    tdiff_reference = int((len(reference[2,:]) - 1)/(reference[2, (len(reference[2,:]) - 1)] - reference[2,0]))
    tdiff = int(1/dt)
    diff = np.zeros((2, t_end + 1))
    
    
    for i in range(t_0, t_end + 1):
        diff[0, i] = (sol[0, i * tdiff] - reference[0, i * tdiff_reference])**2
        diff_sum[0, j] += diff[0, i]
        diff[1, i] = (sol[1, i * tdiff] - reference[1, i * tdiff_reference])**2
        diff_sum[1, j] += diff[1, i]
        
    plt.plot(range(0, t_end + 1), diff[0,:]  , 'b', label='error in nu(t)')
    plt.plot(range(0, t_end + 1), diff[1,:]  , 'g', label='error in prot(t)')
    if positivity == 1:
        plt.title("Patankar - Euler Verfahren punktweiser Fehler mit dt = " + str(dt))
    else:
        plt.title("Euler Verfahren punktweiser Fehler mit dt = " + str(dt))
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.yscale('log')
    plt.show()

plt.loglog(dts, diff_sum[0, :], 'navy', label='kummulierter Fehler in nu')
plt.loglog(dts, diff_sum[1, :], 'darkgreen', label='kummulierter Fehler in prot')
if positivity == 1:
    plt.title("Patankar - Euler Verfahren kumulierter Fehler respekt_ive dt")
else:
    plt.title("Euler Verfahren kumulierter Fehler respekt_ive dt")
plt.legend(loc='best')
plt.xlabel('dt')
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.show()