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
def sig(x):
    return 1/(1+np.exp(-x/0.5))
    
class Experimental_interpolation_model():
    def __init__(self, edge_responses, edge_mask, sigmoid_s=0.5):
        self.Nlayers = len(edge_responses)
        self.edge_mask = []
        for layer in edge_mask:
            self.edge_mask.append(layer.numpy())
        self.edge_responses = edge_responses
        self.N_disc = len(edge_responses[0][0,0])-1
        self.sigmoid_s = sigmoid_s
        
    def linear_interpolate(self, edge_response, input_data):
        discsteps = self.N_disc
        steps = np.asarray(np.floor((input_data*discsteps)), dtype='int')
        remainder = discsteps*(input_data - (steps/discsteps))
        lows = edge_response[steps]
        highs = edge_response[steps + 1]
        difference = highs - lows
        interpolate = lows+difference*remainder
        return interpolate
        
    def forward(self, input_data):
        batch_size = input_data.shape[0]
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(np.zeros((batch_size, self.edge_responses[i].shape[0])))
            outputs.append(np.zeros((batch_size, self.edge_responses[i].shape[1])))
        inputs[0] = self.sig(input_data)
        # Pass through model
        for layer in range(self.Nlayers):
            shape = [self.edge_responses[layer].shape[0], self.edge_responses[layer].shape[1]]
            for pre in range(shape[0]):
                for post in range(shape[1]):
                    outputs[layer][:, post] += self.linear_interpolate(self.edge_responses[layer][pre, post], inputs[layer][:, pre])*self.edge_mask[layer][pre, post]
            if layer < self.Nlayers-1:
                inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        return inputs, outputs

    def sig(self, x):
        return 1/(1+np.exp(-x/self.sigmoid_s))

    
def linear_interpolate(edge_response, input_data):
    discsteps = 50
    steps = np.asarray(np.floor((input_data*discsteps)), dtype='int')
    remainder = discsteps*(input_data - (steps/discsteps))
    lows = edge_response[steps]
    highs = edge_response[steps + 1]
    difference = highs - lows
    interpolate = lows+difference*remainder
    return interpolate

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
import os
from Dimensionless_Feynmann_name import return_function
from Dimensionless_Feynmann import return_function as rf

path = os.getcwd()
data_folder = '/FeynmanData/'
model_folder = '/Trained Models/'
experimental_folder = '/ExperimentalData/'
lossfn = nn.MSELoss()


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
experimental_data = np.load(path+experimental_folder+'Feynman Edges Test Transfer Prewarped (Experiment), Shape '+str(shape)+' Function '+str(function)+'.npy', allow_pickle=True)
edge_mask = model.edge_mask
experimental_KAN = Experimental_interpolation_model(experimental_data, edge_mask)
if Xtest.shape[1]==2:
    xs = np.linspace(xmins[0]/norm_Xs[0], 1, 100)
    ys = np.linspace(xmins[1]/norm_Xs[1], 1, 100)
    X, Y = np.meshgrid(xs, ys)
    name, f, ranges, f_shape = rf(function)
    inputs = torch.tensor([X.flatten(), Y.flatten()]).T
    finputs = inputs*norm_Xs
    experimental_inputs, experimental_outputs = prediction = experimental_KAN.forward(inputs.numpy())
    prediction = experimental_outputs[-1][:, 0]
    true = f(finputs)/norm_Ys
    fig = plt.figure(figsize=(7,3), dpi=1200)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_xlabel('$X_{0}$', fontsize=14)
    ax1.set_ylabel('$X_{1}$', fontsize=14)
    ax1.set_title('KAN (Hardware)')
    ax2.set_title('True Function')
    ax2.set_xlabel('$X_{0}$', fontsize=14)
    ax2.set_ylabel('$X_{1}$', fontsize=14)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    
    pcm = ax1.pcolormesh(xs, ys, prediction.reshape(100, 100), cmap='inferno')
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
    

