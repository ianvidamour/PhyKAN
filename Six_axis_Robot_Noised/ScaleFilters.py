#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:33:23 2025

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn
device='cpu'
torch.set_default_device(device)
from PhyKAN_Util import PhyKAN
import os.path
import os
import matplotlib.pyplot as plt
run = 0

transfer_folder = './Lab_Data/' 
Netsizes = [2, 3, 4, 5, 7, 9, 10, 12, 15, 20]
frequency_scale = np.load(transfer_folder+'Frequency Scale.npy')
low_ind = 271
high_ind = 7293
x_in = torch.from_numpy(np.linspace(0, 1, num=high_ind-low_ind))
loss_fn = nn.MSELoss()
normalised_mismatches = []
model_mismatches = []
for run in range(10):
    for N in Netsizes:
        edge_data = np.load(transfer_folder+f'FW Kin 2L Chirped Transfer, N{N}, {run}.npy', allow_pickle=True)
        model = torch.load(transfer_folder+f'Noised Forward Kinematics Model 2L, N={N} run {run}, noise_k=0.1.pt', weights_only=False, map_location='cpu')
        data = []
        model_data = []
        filter_data = []
        
        for l, layer in enumerate(model.filter_params):
            activations = np.zeros((layer.shape[0], layer.shape[1], high_ind-low_ind))
            for pre in range(layer.shape[0]):
                for post in range(layer.shape[1]):
                    activation = model.band_pass_single(x_in, layer[pre, post], 0).cpu().detach().numpy()
                    single_filters = model.band_pass_returnfilts(x_in, layer[pre,post]).cpu().detach().numpy()
                    experimental_acts = edge_data[l][pre, post, low_ind:high_ind]
                    freqs = frequency_scale[low_ind:high_ind]
                    biases = np.ones_like(experimental_acts)
                    in_mat = np.stack((biases, experimental_acts), axis=1)
                    solve = np.linalg.lstsq(in_mat, activation)
                    c, m = solve[0]
                    scaled_exp_data = experimental_acts*m + c
                    MSE = loss_fn(torch.from_numpy(scaled_exp_data), torch.from_numpy(activation))
                    activations[pre, post] = scaled_exp_data
                    plt.figure()
                    plt.title(MSE.item())
                    plt.plot(activation, label='Model')
                    plt.plot(scaled_exp_data, label='Experiment')
                    plt.legend()
                    plt.show()
                    model_mismatches.append(MSE.item())
                    normalised_mismatches.append(MSE.item()/np.abs(activation).mean())
            data.append(activations)
            model_data.append(activation)
        arr_obj = np.empty(len(data), dtype=object)
        arr_obj[:] = data
        np.save(transfer_folder+'Noised FW Kin Scaled Filters 2L, N '+str(N)+' Run '+str(run)+'.npy', arr_obj, allow_pickle=True)

