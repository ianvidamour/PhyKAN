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
from PhyKAN_Util_actorcritic import PhyKAN_actor
import os.path
import os
run = 0

model_folder = os.getcwd()+'/HPC_Data/'
shape = 10
run = 0

experimental_data = np.load('Experimental Transfer, Cartpole, N'+str(shape)+', '+str(run)+'.npy', allow_pickle=True)
x_in = torch.from_numpy(np.linspace(0, 1, 50))
model = torch.load(model_folder+'/Actor Network, CartPole Continous 2L, N=10, Run=0.pt', weights_only=False, map_location='cpu')
data = []
model_data = []
filter_data = []
import matplotlib.pyplot as plt
for l, layer in enumerate(model.filter_params):
    activations = np.zeros((layer.shape[0], layer.shape[1], 50))
    for pre in range(layer.shape[0]):
        for post in range(layer.shape[1]):
            activation = model.band_pass_single(x_in, layer[pre, post], 0).cpu().detach().numpy()
            single_filters = model.band_pass_returnfilts(x_in, layer[pre,post]).cpu().detach().numpy()
            experimental_acts = experimental_data[l][pre, post]
            bestmse = 1
            bestm = 0
            bestc = 0
            for m in np.arange(3, 7.5, 0.02):
                for c in np.arange(-0.5, 0.5, 0.0005):
                    scaled_act = (experimental_acts+c)*m 
                    mse = np.mean(np.abs((activation - scaled_act).flatten()))
                    if mse < bestmse:
                        bestmse = mse
                        bestm = m
                        bestc = c
            
            scaled_data = (experimental_acts+bestc)*bestm 
            activations[pre, post] = scaled_data
            plt.figure()
            plt.title('Run '+str(run)+', Layer '+str(l)+', Pre '+str(pre)+', Post '+str(post))
            #plt.ylim(-1, 1)
            plt.plot(activation, label='Model')
            plt.plot(scaled_data, label='Experiment')
            plt.legend()
            args = np.argsort(layer[pre, post][:, 0].detach().numpy())
            for i in range(6):
                plt.plot(single_filters[args[i]], alpha=0.5)
            plt.show()
    data.append(activations)
    model_data.append(activation)
    filter_data.append(single_filters)
np.save('Cartpole Edges Test Transfer Prewarped (Experiment), N '+str(shape)+' Run '+str(run)+'.npy', np.asarray(data, dtype='object'), allow_pickle=True)

