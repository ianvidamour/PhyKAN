#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:01:33 2026

@author: ian
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device='cpu'
torch.set_default_device(device)


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
        inputs[0] = input_data
        # Pass through model
        for layer in range(self.Nlayers):
            shape = [self.edge_responses[layer].shape[0], self.edge_responses[layer].shape[1]]
            for pre in range(shape[0]):
                for post in range(shape[1]):
                    outputs[layer][:, post] += self.linear_interpolate(self.edge_responses[layer][pre, post], inputs[layer][:, pre])*self.edge_mask[layer][pre, post]
            if layer < self.Nlayers-1:
                inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        return outputs

    def sig(self, x):
        return 1/(1+np.exp(-x/self.sigmoid_s))

    
def linear_interpolate(edge_response, input_data, discsteps=7022):
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

device='cpu'
torch.set_default_device(device)

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')


edge_folder = './Lab_Data/'
input_folder = './Input_Data/'


noise_k = 0.1

prenoise_joint_angles = np.load(input_folder+'PreNoise_6a_joint_angles_noise_k='+str(noise_k)+'.npy')
true_effector_locations = np.load(input_folder+'True_6a_effector_locations_noise_k='+str(noise_k)+'.npy')
Xin, x_test  = np.split(prenoise_joint_angles, [15000])
Yin, y_test  = np.split(true_effector_locations, [15000])
y_test = torch.from_numpy(y_test)
loss_function = nn.MSELoss()

Ns = [2, 3, 4, 5, 7, 9, 10, 12, 15, 20]
performance_out = np.zeros((10, 5))
for run in range(5):
    for _, N in enumerate(Ns):
        np.random.seed(run)
        edges = np.load(edge_folder+'Noised FW Kin Scaled Filters 2L, N '+str(N)+' Run '+str(run)+'.npy', allow_pickle=True)
        model = torch.load(edge_folder+f'Noised Forward Kinematics Model 2L, N={N} run {run}, noise_k=0.1.pt', weights_only=False, map_location='cpu')
        edge_mask = model.edge_mask
        data = []
        Nunits = 0
        for layer in edge_mask:
            Nunits += layer.sum()
        
        experimental_KAN = Experimental_interpolation_model(edges, edge_mask)
        model_pred = experimental_KAN.forward(x_test)[-1]
        accuracy = loss_function(torch.from_numpy(model_pred), y_test)
        performance_out[_, run] = accuracy.item()
np.save('Forward Kinematics PhyKAN 2L Experimental Accuracies noise_k=0.1.npy', performance_out)

        
accuracies_out2 = np.load('Forward Kinematics PhyKAN 2L Simulation Accuracies noise_k=0.1.npy')
parameters_out2 = np.load('Forward Parameter Counts, PhyKAN 2L.npy')

np.save('Forward Kinematics PhyKAN 2L Experimental Parameters noise_k=0.1.npy', parameters_out2[:10, :5])

accuracies_outMLP2L = np.load('Forward Kinematics MLP 2L Accuracies noise_k=0.1.npy')
parameters_outMLP2L = np.load('Forward Parameter Counts, MLP 2L.npy')

import scipy.stats as stats
plt.figure(dpi=1200, figsize=(4,3))
plt.ylim(1e-5, 0.1)
plt.title('Forward 2L Prenoised to true (Noise k = '+str(noise_k)+')', fontsize=14)
plt.loglog(parameters_out2.mean(axis=1), stats.gmean(accuracies_out2, axis=1), color='red', label='Software PhyKAN', marker='o')
plt.fill_between(parameters_out2.mean(axis=1), stats.gmean(accuracies_out2, axis=1)*stats.gstd(accuracies_out2, axis=1), stats.gmean(accuracies_out2, axis=1)/stats.gstd(accuracies_out2, axis=1), lw=0, alpha=0.2, color='red')
plt.loglog(parameters_outMLP2L.mean(axis=1), stats.gmean(accuracies_outMLP2L, axis=1), color='black', label='Software MLP', marker='o')
plt.fill_between(parameters_outMLP2L.mean(axis=1), stats.gmean(accuracies_outMLP2L, axis=1)*stats.gstd(accuracies_outMLP2L, axis=1), stats.gmean(accuracies_outMLP2L, axis=1)/stats.gstd(accuracies_outMLP2L, axis=1), lw=0, alpha=0.2, color='black')
plt.loglog(parameters_out2.mean(axis=1)[:10], performance_out.mean(axis=1), label='Experimental PhyKAN', marker='x', color='xkcd:brick')
plt.fill_between(parameters_out2[:10].mean(axis=1), stats.gmean(performance_out, axis=1)*stats.gstd(performance_out, axis=1), stats.gmean(performance_out, axis=1)/stats.gstd(performance_out, axis=1), lw=0, alpha=0.2, color='red')
plt.xlabel('Trainable Parameters', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend(frameon=False, ncols=1, fontsize=12)
plt.show()
        