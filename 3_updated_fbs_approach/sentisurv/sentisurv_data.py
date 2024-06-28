#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:09:08 2024

@author: tuchscheerer
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

s = pd.read_csv('sentisurv.csv')

convert = {'test_date': np.datetime64,
           'town': str,
           'val_test': int,
           'pos': int,
           'n_pos': int,
           'phase': int
           }
s = s.astype(convert)

time = np.unique(s['test_date'])
t0 = time[0]
rel_time = []
for current in time:
    rel_time.append(int(pd.Timedelta(current - t0).days))
    
pos = []
n_pos = []
total = []

for t in time:
    h = s[s['test_date']==t]
    p = sum(h['pos'])
    npo = sum(h['n_pos'])
    n = sum(h['val_test'])
    pos.append(p)
    n_pos.append(npo)
    total.append(n)

pos = np.asarray(pos)
n_pos = np.asarray(n_pos)
total = np.asarray(total)
# data[0] == time, data[1] == i meassured
data = [rel_time, pos/total]
np.save('sentisurv_data', data)

# plt.plot(time, pos/total)
# plt.xticks(rotation = 90)
# plt.grid()