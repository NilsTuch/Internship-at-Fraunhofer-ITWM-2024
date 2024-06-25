# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:32:22 2024

@author: Tuchscheerer
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

eps = 10**-6

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

diff = eps + 1
iteration = 0
dt = .3

tsum = int(tend/dt) + 1
t = np.linspace(t0, tend, tsum)

sol = np.zeros((3,tsum))
sol[0, 0] = y0[0]
sol[1, 0] = y0[1]
sol[2, 0] = t0

for i in range(1, tsum):
    ti = t[i]
    nu = sol[0, i-1]
    sus = sol[1, i-1]
    yi_1 = [nu, sus]
    r1 = np.array(epi(yi_1, ti, tau, a, r))
    r2 = np.array(epi(yi_1 + dt/2 * r1, ti + dt/2, tau, a, r))
    r3 = np.array(epi(yi_1 + dt/2 * r2, ti + dt/2, tau, a, r))
    r4 = np.array(epi(yi_1 + dt * r3, ti + dt, tau, a, r))
    yi = np.array(yi_1) + (dt/6 * (r1 + 2*r2 + 2*r3 + r4))
    
    sol[0][i] = yi[0]
    sol[1][i] = yi[1]  
    sol[2, i] = ti
        
last_sol = sol

while diff > eps:
    diff = 0
    iteration += 1
    dt= 10**-iteration
    
    tsum = int(tend/dt) + 1
    t = np.linspace(t0, tend, tsum)

    sol = np.zeros((3,tsum))
    sol[0, 0] = y0[0]
    sol[1, 0] = y0[1]
    sol[2, 0] = t0
    for i in range(1, tsum):
        ti = t[i]
        nu = sol[0, i-1]
        sus = sol[1, i-1]
        yi_1 = [nu, sus]
        
        r1 = np.array(epi(yi_1, ti, tau, a, r))
        r2 = np.array(epi(yi_1 + dt/2 * r1, ti + dt/2, tau, a, r))
        r3 = np.array(epi(yi_1 + dt/2 * r2, ti + dt/2, tau, a, r))
        r4 = np.array(epi(yi_1 + dt * r3, ti + dt, tau, a, r))
        yi = np.array(yi_1) + (dt/6 * (r1 + 2*r2 + 2*r3 + r4))
        
        sol[0][i] = yi[0]
        sol[1][i] = yi[1]
        sol[2, i] = ti
    for i in range(0, int(tsum/10)):
        loc_diff = abs(sol[0, 10*i] - last_sol[0, i])
        diff = max(diff, loc_diff)
    last_sol = sol
    print("Für dt = " + str(dt) + " beträgt die Veränderung zur vorherigen Lösung " + str(diff))

np.save('reference', sol)