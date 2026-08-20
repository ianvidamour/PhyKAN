#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:17:36 2026

@author: ian
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

Vin = np.load('Input Voltages.npy')
Vout = np.load('Output Voltages.npy')
params = np.load('Device Parameters.npy')

Nrun = 20000
Nsamp = 200

X = np.zeros((Nrun*Nsamp, 6))
Y = np.zeros((Nrun*Nsamp, 1))

for run in range(Nrun):
    for samp in range(Nsamp):
        X[run*200+samp, 0] = Vin[run, samp]
        X[run*200+samp, 1:] = params[run]
        Y[run*200+samp] = Vout[run, samp]
        
#%%
np.save('MemdiodeModelX.npy', X)
np.save('MemdiodeModelY.npy', Y)