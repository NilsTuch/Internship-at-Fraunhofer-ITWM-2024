# -*- coding: utf-8 -*-
"""
Author  : nils.t03@gmail.com
Created : June 2024
"""

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import json

show_parameter_box = True
legend_outside_plot = True

dataset_names = [ # without "data/" adn ".json"
    "A|dt = 4|tol = 0.0001|h_init = 0.01|h_div= 4|A = 0.1-1--10|B = 1.0",
    "A|dt = 4|tol = 0.0001|h_init = 0.01|h_div= 4|A = 0.1-1--10|B = 10.0"
    ]

variables = ["$\Delta$t", "tol", "$h_{init}$", "$h_{div}$", "A", "B"] # 0, 1, 2, 3, 4, 5

# positioning of parameter box
legend_x = .9
legend_y = .7
legend_width = .3
legend_height = .2
text_x = 400
text_y = 70
text_width = .3
text_height = .2

directory_name = "data"
iter_fig, iter_ax = plt.subplots()
J_fig, J_ax = plt.subplots()
K_fig, K_ax = plt.subplots()
projJ_fig, projJ_ax = plt.subplots()
figs = [iter_fig, J_fig, K_fig, projJ_fig,]
axs = [iter_ax, J_ax, K_ax, projJ_ax,]

dts = []
tolerances = []
init_steps = []
step_divisors = []
As = []
Bs = []
all_parameters = [dts, tolerances, init_steps, step_divisors, As, Bs]

list_parameters = []
list_parameter_range = []
list_RK_data = []
list_MPRK_data = []
    
for dataset_name in dataset_names:
    # loading json
    file_name = directory_name + "/" + dataset_name + ".json"
    file = open(file_name)
    data = json.load(file)
    data = json.loads(data)
    file.close()
    
    if True:
        first_chosen_par = data["chosen_parameter"]
        variables = data["variables"]
        
        parameters = data["parameters"]
        dt  = parameters["dt"]
        tolerance = parameters["tol"]
        init_step = parameters["h_init"]
        step_divisor = parameters["h_div"]
        A = parameters["A"]
        B = parameters["B"]
        
        parameter_range = data["parameter_range"]
        results = data["data"]
        RK_data = results["RK"]
        MPRK_data = results["MPRK"]
        
        dts.append(dt)
        tolerances.append(tolerance)
        init_steps.append(init_step)
        step_divisors.append(step_divisor)
        As.append(A)
        Bs.append(B)
        
        list_parameters.append(parameters)
        list_parameter_range.append(parameter_range)
        list_RK_data.append(RK_data)
        list_MPRK_data.append(MPRK_data)

# determine second varianle
def get_second_chosen_par(possible_vars):
    x = -1
    for var in possible_vars:
        if any(var.count(x) < len(var) for x in var):
            x = possible_vars.index(var)
    return x
second_chosen_par = get_second_chosen_par([dts, tolerances, init_steps, step_divisors, As, Bs])

for i in range(len(list_parameter_range)):
    parameters = list_parameters[i]
    parameter_range = list_parameter_range[i]
    RK_data = list_RK_data[i]
    MPRK_data = list_MPRK_data[i]
    
    # plots
    if second_chosen_par != -1:
        label_suffix = ", " + variables[second_chosen_par] + " = "  # TODO
        label_suffix += str(parameters[variables[second_chosen_par]])
    else: label_suffix = ""
    RK_label = "Computed with RK" + label_suffix
    MPRK_label = "Computed with MPRK" + label_suffix
    
    for ax, RK_graph, MPRK_graph in zip(axs, 
                                        [RK_data["Iterations"], RK_data["J"], RK_data["K"], RK_data["projJ"]],
                                        [MPRK_data["Iterations"], MPRK_data["J"], MPRK_data["K"], MPRK_data["projJ"]]):
        ax.plot(parameter_range, RK_graph, label = RK_label)
        ax.plot(parameter_range, MPRK_graph, label = MPRK_label)

parameter_text_parts = ["dt = " + str(dt), 
               "tolerance = " + str(tolerance), 
               "initial stepsize = " + str(init_step), 
               "stepsize divisor = " + str(step_divisor),
               "A = " + str(A),
               "B = " + str(B)]
if second_chosen_par != -1: 
    parameter_text_parts[second_chosen_par] = variables[second_chosen_par] + " = see legend"
parameter_text = '\n'.join(parameter_text_parts[:first_chosen_par] + parameter_text_parts[first_chosen_par + 1:])

for fig, ax in zip(figs, axs):
    if show_parameter_box:
        at = AnchoredText(parameter_text, pad = 0.5, borderpad = 0, 
                          loc="upper left", bbox_to_anchor=(text_x, text_y, text_width, text_height))
        at.patch.set_edgecolor('lightgrey')
        at.patch.set_boxstyle("round,pad=0,rounding_size=0.2")
        fig.add_artist(at)
    
    if legend_outside_plot:
        fig.legend(loc = "upper left", bbox_to_anchor=(legend_x, legend_y, legend_width, legend_height))
    else:
        ax.legend(loc="best")
    
    ax.set_xlabel(variables[first_chosen_par])
    if first_chosen_par == 0: ax.set_xscale('log', base = 2)
    elif first_chosen_par in (1, 2, 4, 5): ax.set_xscale('log')
    ax.grid()

iter_ax.set_yscale('log', base = 10)
iter_ax.set_title("# Iterations dependent on " + variables[first_chosen_par])
J_ax.set_title("J dependent on " + variables[first_chosen_par])
K_ax.set_title("K dependent on " + variables[first_chosen_par])
projJ_ax.set_title("J on the projection dependent on " + variables[first_chosen_par])