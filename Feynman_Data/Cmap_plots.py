#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:03:01 2025

@author: ian
"""

import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from PhyKAN_Util import PhyKAN
import os
from Dimensionless_Feynmann_name import return_function
from Dimensionless_Feynmann import return_function as rf

path = os.getcwd()
data_folder = '/FeynmanData/'
model_folder = '/Trained Models/'
lossfn = nn.MSELoss()

#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable
function = 16
shape = 2
name = return_function(function)
data = np.load(path+data_folder+'//'+name+'.npy')
Xin = data[:, :-1]
Yin = data[:, -1]

Xnorm = np.zeros_like(Xin)
norm_Xs = np.zeros((Xin.shape[1]))
xmins = np.zeros((Xin.shape[1]))
for i in range(Xin.shape[1]):
    norm_Xs[i] = np.amax((Xin[:, i]))
    xmins[i] = np.amin(Xin[:, i])
    Xnorm[:, i] = Xin[:, i]/np.amax(np.abs(Xin[:, i]))
norm_Ys = np.amax(np.abs(Yin))
Ynorm = Yin/np.amax(np.abs(Yin))

Xtrain, Xtest = np.split(Xnorm, [80000])
Ytrain, Ytest = np.split(Ynorm[:, None], [80000])

Xtrain = torch.from_numpy(Xtrain).float()
Xtest = torch.from_numpy(Xtest).float()
Ytrain = torch.from_numpy(Ytrain).float()
Ytest = torch.from_numpy(Ytest).float()

model = torch.load(path+model_folder+'Feynmann Function '+str(function)+', shape '+str(shape)+'_Rerun.pt', weights_only=False, map_location='cpu')

if Xtest.shape[1]==2:
    xs = np.linspace(xmins[0]/norm_Xs[0], 1, 100)
    ys = np.linspace(xmins[1]/norm_Xs[1], 1, 100)
    X, Y = np.meshgrid(xs, ys)
    name, f, ranges, f_shape = rf(function)
    inputs = torch.tensor([X.flatten(), Y.flatten()]).T
    prediction = model.forward(inputs)[-1].detach()
    finputs = inputs*norm_Xs
    true = f(finputs)/norm_Ys
    fig = plt.figure(figsize=(7,3), dpi=1200)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_xlabel('$X_{0}$', fontsize=14)
    ax1.set_ylabel('$X_{1}$', fontsize=14)
    ax1.set_title('KAN (Simulation)')
    ax2.set_title('True Function')
    ax2.set_xlabel('$X_{0}$', fontsize=14)
    ax2.set_ylabel('$X_{1}$', fontsize=14)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    
    pcm = ax1.pcolormesh(xs, ys, prediction.numpy().reshape(100, 100), cmap='inferno')
    pcm2 = ax2.pcolormesh(xs, ys, true.numpy().reshape(100, 100), cmap='inferno')
    #Create and remove the colorbar for the first subplot
    cbar1 = fig.colorbar(pcm, cax = cax1)
    fig.delaxes(fig.axes[2])

    #Create second colorbar
    cbar2 = fig.colorbar(pcm2, cax = cax2)
    cbar2.set_label('$f(X_{0}, X_{1})$', fontsize=14)
    #cbar2.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()

    plt.show()
    

