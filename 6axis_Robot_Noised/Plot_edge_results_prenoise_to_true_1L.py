#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:26:58 2025

@author: ian
"""

'''
For these plots, the model was trained input data of pre-noised joint angles, with targets of post-noise effector locations.
Noise is applied to the joint angles and is uniformly distributed in the range +- noise_k radians
Plots here evaluate the ability to fit the *noised* locations from un-noised joint angles
'''

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from PhyKAN_Util import PhyKAN

class Net(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.relu()
        out = self.fc2(out)
        return out


def edges_KAN_1L(N):
    # Trainables = 18 * (in * hidden + hidden * hidden + hidden * out)
    return (6 * N + N * 3)


def edges_MLP_1L(N):
    # Trainables = out + 2*hidden (biases) + in * hidden + hidden * hidden + hidden * out (weights)
    return (6 * N + N * 3)

import os
path = os.getcwd()


data_folder = '/HPC_Data/'
input_folder = '/Input_Data/'


noise_k = 0.1
        
true_joint_angles = np.load(path+input_folder+'PreNoise_6a_joint_angles_noise_k='+str(noise_k)+'.npy')
true_effector_locations = np.load(path+input_folder+'True_6a_effector_locations_noise_k='+str(noise_k)+'.npy')
Xin, x_test  = np.split(true_joint_angles, [15000])
Yin, y_test  = np.split(true_effector_locations, [15000])
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)
loss_function = nn.MSELoss()

Netsizes = [2, 3, 4, 5, 7, 9, 10, 12, 15, 20]
subfolder = '/'+str(noise_k)+'/'
accuracies_out2 = np.zeros((10, 10))
parameters_out2 = np.zeros((10, 10))
for i, N in enumerate(Netsizes):
    for run in range(10):
        model_KAN = torch.load(path+data_folder+subfolder+'Noised Forward Kinematics Model 1L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location='cpu')
        model_pred = model_KAN.forward(x_test, input_sigmoid=False)[-1]
        accuracy = loss_function(model_pred, y_test)
        parameters_out2[i, run] = edges_KAN_1L(N)
        accuracies_out2[i, run] = accuracy.item()
np.save('Forward Kinematics PhyKAN 1L Simulation Accuracies noise_k=0.1.npy', accuracies_out2)
np.save('Forward Parameter Counts, PhyKAN 1L.npy', parameters_out2)
        



Netsizes = [5, 7, 9, 10, 12, 15, 20, 50, 100, 150, 200, 250]
accuracies_outMLP2L = np.zeros((12, 10))
parameters_outMLP2L = np.zeros((12, 10))        
for i, N in enumerate(Netsizes):
    for run in range(10):
        model = torch.load(path+data_folder+subfolder+'Noised Forward Kinematics MLP 1L, N='+str(N)+' run = '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location='cpu')
        model_pred = model.forward(x_test)
        accuracy = loss_function(model_pred, y_test)
        parameters_outMLP2L[i, run] = edges_MLP_1L(N)
        accuracies_outMLP2L[i, run] = accuracy.item()
np.save('Forward Kinematics MLP Accuracies 1L noise_k=0.1.npy', accuracies_outMLP2L)
np.save('Forward Parameter Counts, MLP 1L.npy', parameters_outMLP2L)        

        
import scipy.stats as stats
plt.figure(dpi=1200, figsize=(4,3))
plt.title('Forward 1L, Prenoised to true (Noise k = '+str(noise_k)+')', fontsize=14)
plt.loglog(parameters_out2.mean(axis=1), stats.gmean(accuracies_out2, axis=1), color='red', label='Software PhyKAN', marker='o')
plt.fill_between(parameters_out2.mean(axis=1), stats.gmean(accuracies_out2, axis=1)*stats.gstd(accuracies_out2, axis=1), stats.gmean(accuracies_out2, axis=1)/stats.gstd(accuracies_out2, axis=1), lw=0, alpha=0.2, color='red')
plt.loglog(parameters_outMLP2L.mean(axis=1), stats.gmean(accuracies_outMLP2L, axis=1), color='black', label='Software MLP', marker='o')
plt.fill_between(parameters_outMLP2L.mean(axis=1), stats.gmean(accuracies_outMLP2L, axis=1)*stats.gstd(accuracies_outMLP2L, axis=1), stats.gmean(accuracies_outMLP2L, axis=1)/stats.gstd(accuracies_outMLP2L, axis=1), lw=0, alpha=0.2, color='black')
plt.xlabel('Number of Edges', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend(frameon=False, ncols=1, fontsize=12)
plt.show()

