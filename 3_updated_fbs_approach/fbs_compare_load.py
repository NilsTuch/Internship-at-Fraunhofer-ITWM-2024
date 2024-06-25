# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : June 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.offsetbox import AnchoredText
import json

# Bericht auf deutsch => programm auf deutsch?
# wie sollte ich die plots bennenen?
# y-achse beschriften?

# more than 2 vars possible?

# nur iterations anschauen ignoriert laufzeit der einzelnen iterations durch wiederholte teilung von 
    # step_size. 
    # Laufzeit vergleichen? Als mittelwert über zB 5 durchgänge.

dataset_names = [
    # "dt|dt = 0.25-0.5--8|tolerance = 1e-08|initial stepsize = 0.01|stepsize divisor = 10|A = 10.0|B = 1.0",
    
    "A|dt = 4|tolerance = 1e-06|initial stepsize = 0.01|stepsize divisor = 4|A = 0.01-0.1--10000.0|B = 1.0",
    
    # "A|dt = 2|tolerance = 0.001|initial stepsize = 0.01|stepsize divisor = 10|A = 0.01-0.1--10000.0|B = 1.0",
    # "A|dt = 2|tolerance = 0.0001|initial stepsize = 0.01|stepsize divisor = 10|A = 0.01-0.1--10000.0|B = 1.0",
    
    # "A|dt = 2|tolerance = 1e-05|initial stepsize = 0.01|stepsize divisor = 10|A = 0.01-0.1--10000.0|B = 1.0",
    # "A|dt = 2|tolerance = 1e-06|initial stepsize = 0.01|stepsize divisor = 10|A = 0.01-0.1--10000.0|B = 1.0",
    
    
    ]

# first_variable = 2
second_variable = 1 # TODO
kappas_together = True
variables = ["§\Delta$t", "tolerance", "initial stepsize", "stepsize divisor, A, A"] # 0, 1, 2, 3, 4, 5

def get_2nd_var(possible_vars):
    x = -1
    for var in possible_vars:
        if any(var.count(x) < len(var) for x in var):
            x = possible_vars.index(var)
    return x

# positioning
if True: # TODO
    legend_x = .9
    legend_y = .7
    legend_width = .3
    legend_height = .2
    
    text_x = 400
    text_y = 70
    text_width = .3
    text_height = .2

directory_name = "fbs_data"
iter_fig, iter_ax = plt.subplots()
fitJ_fig, fitJ_ax = plt.subplots()
fitError_fig, fitError_ax = plt.subplots()
projJ_fig, projJ_ax = plt.subplots()
figs = [iter_fig, fitJ_fig, fitError_fig, projJ_fig,]
axs = [iter_ax, fitJ_ax, fitError_ax, projJ_ax,]
if kappas_together: kappa_fig, kappa_ax = plt.subplots()
else: kappa_figs, kappa_axs = [], []
    
A = 10
B = 1

dts = []
tolerances = []
init_steps = []
step_divisors = []
As = []
Bs = []
all_parameters = [dts, tolerances, init_steps, step_divisors, As, Bs]

list_parameters = []
list_values = []
list_RK_data = []
list_MPRK_data = []
list_ts = []
list_RK_kappas = []
list_MPRK_kappas = []
    
for dataset_name in dataset_names:
    # loading the json
    file_name = directory_name + "/" + dataset_name + ".json"
    file = open(file_name)
    data = json.load(file)
    data = json.loads(data)
    file.close()
    
    if True:
        variable = data["variable"]
        variables = data["variables"]
        
        parameters = data["parameters"]
        dt, tolerance = parameters["dt"], parameters["tolerance"]
        init_step, step_divisor = parameters["initial stepsize"], parameters["stepsize divisor"]
        A, B = parameters["A"], parameters["B"]
        
        values = data["values"]
        results = data["data"]
        RK_data, MPRK_data = [results["RK"], results["MPRK"]]
        kappa = data["kappa"]
        t = kappa["time"]
        RK_kappas = kappa["RK"]
        MPRK_kappas = kappa["MPRK"]
        
        dts.append(dt)
        tolerances.append(tolerance)
        init_steps.append(init_step)
        step_divisors.append(step_divisor)
        As.append(A)
        Bs.append(B)
        
        list_parameters.append(parameters)
        list_values.append(values)
        list_RK_data.append(RK_data)
        list_MPRK_data.append(MPRK_data)
        
        list_ts.append(t)
        list_RK_kappas.append(RK_kappas)
        list_MPRK_kappas.append(MPRK_kappas)

# determine second varianle
second_variable = get_2nd_var([dts, tolerances, init_steps, step_divisors, As, Bs])

for i in range(len(list_values)):
    parameters = list_parameters[i]
    values = list_values[i]
    RK_data = list_RK_data[i]
    MPRK_data = list_MPRK_data[i]
    
    # ts = list_ts[i]
    RK_kappas = list_RK_kappas[i]
    MPRK_kapas = list_MPRK_kappas[i]
        
    
    # plots
    if True:
        if second_variable != -1:
            label_suffix = ", " + variables[second_variable] + " = "  # TODO
            label_suffix += str(parameters[variables[second_variable]])
        else: label_suffix = ""
        
        iter_ax.plot(values, RK_data["Iterations"], label="Computed with RK" + label_suffix)
        iter_ax.plot(values, MPRK_data["Iterations"], label="Computed with MPRK" + label_suffix)
        
        fitJ_ax.plot(values, RK_data["fitJ"], label="Computed with RK" + label_suffix)
        fitJ_ax.plot(values, MPRK_data["fitJ"], label="Computed with MPRK" + label_suffix)
        
        fitError_ax.plot(values, RK_data["fit Error"], label="Computed with RK" + label_suffix)
        fitError_ax.plot(values, MPRK_data["fit Error"], label="Computed with MPRK" + label_suffix)

        projJ_ax.plot(values, RK_data["konjJ"], label="Computed with RK" + label_suffix)
        projJ_ax.plot(values, MPRK_data["konjJ"], label="Computed with MPRK" + label_suffix)
        
        # for RK_kappa, MPRK_kappa, t, i in zip(RK_kappas, MPRK_kappas, ts, range(len(RK_kappas))):
        #     if not kappas_together:
        #         kappa_fig, kappa_ax = plt.subplots()
        #         kappa_figs.append(kappa_fig)
        #         kappa_axs.append(kappa_ax)
        #     label_suffix = ", " + variables[variable] + " = " + str(values[i])
        #     kappa_ax.plot(t, RK_kappa, label = "RK_kappa" + label_suffix)
        #     kappa_ax.plot(t, MPRK_kappa, label = "MPRK_kappa" + label_suffix)
           
        #     kappa_fig.legend(loc = "upper left", bbox_to_anchor=(legend_x, legend_y, legend_width, legend_height))
        #     kappa_ax.set_xlabel(variables[variable])
        #     kappa_ax.grid()
        #     kappa_ax.set_title("kappa in Abhängigkeit von " + variables[variable])

fixed_parts = ["dt = " + str(dt), 
               "tolerance = " + str(tolerance), 
               "initial stepsize = " + str(init_step), 
               "stepsize divisor = " + str(step_divisor),
               "A = " + str(A),
               "B = " + str(B)]
if second_variable != -1: 
    fixed_parts[second_variable] = variables[second_variable] + " = see legend"
fixed = '\n'.join(fixed_parts[:variable] + fixed_parts[variable + 1:])

for fig, ax in zip(figs, axs):
    at = AnchoredText(fixed, pad = 0.5, borderpad = 0, 
                      loc="upper left", bbox_to_anchor=(text_x, text_y, text_width, text_height))
    at.patch.set_edgecolor('lightgrey')
    at.patch.set_boxstyle("round,pad=0,rounding_size=0.2")
    # fig.add_artist(at)
    
    ax.legend(loc="best")
    # fig.legend(loc = "upper left", bbox_to_anchor=(legend_x, legend_y, legend_width, legend_height))
    ax.set_xlabel(variables[variable])
    if variable == 0: ax.set_xscale('log', base = 2)
    elif variable == 1 or variable == 2 or variable > 3: ax.set_xscale('log', base = 10)
    ax.grid()
    # ax.set_yscale('symlog')
    # plt.tight_layout(pad = 3)
iter_ax.set_yscale('log', base = 10)
iter_ax.set_title("# Iterations dependent on " + variables[variable])
fitJ_ax.set_title("J dependent on " + variables[variable])
# fitError_ax.set_yscale('log')
fitError_ax.set_title("K dependent on " + variables[variable])
projJ_ax.set_title("J on the projection dependent on " + variables[variable])