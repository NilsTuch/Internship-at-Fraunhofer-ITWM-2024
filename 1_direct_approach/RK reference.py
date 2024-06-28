# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : April 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

tol = 10**-6

x_0 = [0.003, 1]
t_0 = 0
t_end = 1000

tau = 3.58
a = 0.003
r = 4

def f(x_i, t):
    nu, s = x_i
    dxdt = np.array([1/tau * (1 - 1/(r * s))*nu, a * (1 - s) - nu])
    return np.array(dxdt)

error = tol + 1
iteration = 0
dt = .3

t_sum = int(t_end/dt) + 1
t = np.linspace(t_0, t_end, t_sum)

sol = np.zeros((3,t_sum))
sol[0, 0] = x_0[0]
sol[1, 0] = x_0[1]
sol[2, 0] = t_0

for i in range(1, t_sum):
    t_i = t[i]
    nu = sol[0, i-1]
    s = sol[1, i-1]
    x_i_1 = [nu, s]
    r1 = f(x_i_1, t_i)
    r2 = f(x_i_1 + dt/2 * r1, t_i + dt/2)
    r3 = f(x_i_1 + dt/2 * r2, t_i + dt/2)
    r4 = f(x_i_1 + dt * r3, t_i + dt)
    x_i = x_i_1 + dt/6 * (r1 + 2*r2 + 2*r3 + r4)
    
    sol[0][i] = x_i[0]
    sol[1][i] = x_i[1]  
    sol[2, i] = t_i
        
last_sol = sol

while error > tol:
    error = 0
    iteration += 1
    dt= 10**-iteration
    
    t_sum = int(t_end/dt) + 1
    t = np.linspace(t_0, t_end, t_sum)

    sol = np.zeros((3,t_sum))
    sol[0, 0] = x_0[0]
    sol[1, 0] = x_0[1]
    sol[2, 0] = t_0
    for i in range(1, t_sum):
        t_i = t[i]
        nu = sol[0, i-1]
        s = sol[1, i-1]
        x_i_1 = [nu, s]
        
        r1 = f(x_i_1, t_i)
        r2 = f(x_i_1 + dt/2 * r1, t_i + dt/2)
        r3 = f(x_i_1 + dt/2 * r2, t_i + dt/2)
        r4 = f(x_i_1 + dt * r3, t_i + dt)
        x_i = x_i_1 + dt/6 * (r1 + 2*r2 + 2*r3 + r4)
        
        sol[0][i] = x_i[0]
        sol[1][i] = x_i[1]
        sol[2, i] = t_i
    for i in range(0, int(t_sum/10)):
        error_at_i = abs(sol[0, 10*i] - last_sol[0, i])
        error = max(error, error_at_i)
    last_sol = sol
    print("Für dt = " + str(dt) + " beträgt die Veränderung zur vorherigen Lösung " + str(error))

np.save('reference', sol)