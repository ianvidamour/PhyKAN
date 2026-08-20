#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:26:58 2025

@author: ian
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from PhyKAN_Util import PhyKAN

Netsizes = [2, 3, 4, 5, 7, 9, 10, 12, 15, 20]
exp_inds = [1, 3, 5, 8, 10, 11, 13]
masked_edges = np.zeros((10, 7))
accuracies_out = np.zeros((10, 7))
parameters_out = np.zeros((10, 7))

def trainables_KAN_1L(N):
    # Trainables = 18 * (in * hidden + hidden * out)
    return 18 * (6 * N + N * 3)

def trainables_KAN_2L(N):
    # Trainables = 18 * (in * hidden + hidden * hidden + hidden * out)
    return 18 * (6 * N + N * N + N * 3)

def trainables_MLP_1L(N):
    # Trainables = out + hidden (biases) + in * hidden + hidden * out (weights)
    return (N + 3) + (6 * N + N * 3)

def trainables_MLP_2L(N):
    # Trainables = out + 2*hidden (biases) + in * hidden + hidden * hidden + hidden * out (weights)
    return (N + N + 3) + (6 * N + N * N + N * 3)



import os
path = os.getcwd()
folder = '/HPC_Data/'
for i, N in enumerate(Netsizes):
    for run in range(7):
        accuracies = np.load(path+folder+'Noised Training Accuracies 1L, , N='+str(N)+' run '+str(run)+'.npy')
        accuracies_out[i, run] = np.average(accuracies[-10:])
        model = torch.load(path+folder+'Noised Forward Kinematics Model 1L, N='+str(N)+' run '+str(run)+'.pt', weights_only=False)
        mask = model.edge_mask
        # masked = 0
        # for layer in mask:
        #     masked += (layer-1).abs().sum().item()
        # masked_edges[i, run] = masked
        parameters_out[i, run] = trainables_KAN_1L(N)
        


masked_edges2 = np.zeros((10, 7))
accuracies_out2 = np.zeros((10, 7))
parameters_out2 = np.zeros((10, 7))


for i, N in enumerate(Netsizes):
    for run in range(7):
        accuracies = np.load(path+folder+'Noised Training Accuracies 2L, , N='+str(N)+' run '+str(run)+'.npy')
        accuracies_out2[i, run] = np.average(accuracies[-10:])
        model = torch.load(path+folder+'Noised Forward Kinematics Model 2L, N='+str(N)+' run '+str(run)+'.pt', weights_only=False)
        mask = model.edge_mask
        masked = 0
        for layer in mask:
            masked += (layer-1).abs().sum().item()
        masked_edges2[i, run] = masked
        parameters_out2[i, run] = trainables_KAN_2L(N)
        

        
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.relu()
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)
        return out

Netsizes = [5, 7, 9, 10, 12, 15, 20, 50, 100]
accuracies_outMLP = np.zeros((9, 7))
parameters_outMLP = np.zeros((9, 7))
accuracies_outMLP2L = np.zeros((9, 7))
parameters_outMLP2L = np.zeros((9, 7))



for i, N in enumerate(Netsizes):
    for run in range(7):
        accuracies = np.load(path+folder+'Noised MLP Accuracies 1L, N='+str(N)+' run ='+str(run)+'.npy')
        accuracies_outMLP[i, run] = np.average(accuracies[-10:])
        parameters_outMLP[i, run] = trainables_MLP_1L(N)
        accuracies = np.load(path+folder+'Noised MLP Accuracies 2L, N='+str(N)+' run ='+str(run)+'.npy')
        accuracies_outMLP2L[i, run] = np.average(accuracies[-10:])
        parameters_outMLP2L[i, run] = trainables_MLP_2L(N)
        

# accuracy_1L = np.load('Transferred Accuracies 1L.npy').T

# parameters_experimental_1L = parameters_out[exp_inds]
# accuracy_2L = np.load('Transferred Accuracies 2L.npy').T
# parameters_experimental_2L = parameters_out2[exp_inds]

import scipy.stats as stats
plt.figure(dpi=1200, figsize=(4,3))
plt.title('Forward Kinematics 1L (Noised)', fontsize=14)
plt.ylim(-0.0001, 0.004)
plt.semilogx(parameters_out.mean(axis=1), stats.gmean(accuracies_out, axis=1), color='red', label='Software PhyKAN', marker='o')
plt.fill_between(parameters_out.mean(axis=1), stats.gmean(accuracies_out, axis=1)*stats.gstd(accuracies_out, axis=1), stats.gmean(accuracies_out, axis=1)/stats.gstd(accuracies_out, axis=1), lw=0, alpha=0.2, color='red')
# plt.loglog(parameters_experimental_1L.mean(axis=1), stats.gmean(accuracy_1L, axis=1), color='xkcd:crimson', label='Hardware PhyKAN', marker='x')
# plt.fill_between(parameters_experimental_1L[:, :5].mean(axis=1), stats.gmean(accuracy_1L, axis=1)*stats.gstd(accuracy_1L, axis=1), stats.gmean(accuracy_1L, axis=1)/stats.gstd(accuracy_1L, axis=1), lw=0, alpha=0.2, color='xkcd:crimson')
plt.semilogx(parameters_outMLP.mean(axis=1), stats.gmean(accuracies_outMLP, axis=1), color='black', label='Software MLP', marker='o')
plt.fill_between(parameters_outMLP.mean(axis=1), stats.gmean(accuracies_outMLP, axis=1)*stats.gstd(accuracies_outMLP, axis=1), stats.gmean(accuracies_outMLP, axis=1)/stats.gstd(accuracies_outMLP, axis=1), lw=0, alpha=0.2, color='black')
plt.xlabel('Trainable Parameters', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend(frameon=False, ncols=1, fontsize=12)
plt.show()
#%%


plt.figure(dpi=1200, figsize=(4,3))
plt.ylim(-0.0001, 0.004)
plt.title('Forward Kinematics 2L (Noised)', fontsize=14)
plt.semilogx(parameters_out2.mean(axis=1), stats.gmean(accuracies_out2, axis=1), color='red', label='Software PhyKAN', marker='o')
plt.fill_between(parameters_out2.mean(axis=1), stats.gmean(accuracies_out2, axis=1)*stats.gstd(accuracies_out2, axis=1), stats.gmean(accuracies_out2, axis=1)/stats.gstd(accuracies_out2, axis=1), lw=0, alpha=0.2, color='red')
# plt.loglog(parameters_experimental_2L.mean(axis=1), stats.gmean(accuracy_2L, axis=1), color='xkcd:crimson', label='Hardware PhyKAN', marker='x')
# plt.fill_between(parameters_experimental_2L.mean(axis=1), stats.gmean(accuracy_2L, axis=1)*stats.gstd(accuracy_2L, axis=1), stats.gmean(accuracy_2L, axis=1)/stats.gstd(accuracy_2L, axis=1), lw=0, alpha=0.2, color='xkcd:crimson')
plt.semilogx(parameters_outMLP2L.mean(axis=1), stats.gmean(accuracies_outMLP2L, axis=1), color='black', label='Software MLP', marker='o')
plt.fill_between(parameters_outMLP2L.mean(axis=1), stats.gmean(accuracies_outMLP2L, axis=1)*stats.gstd(accuracies_outMLP2L, axis=1), stats.gmean(accuracies_outMLP2L, axis=1)/stats.gstd(accuracies_outMLP2L, axis=1), lw=0, alpha=0.2, color='black')
plt.xlabel('Trainable Parameters', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend(frameon=False, ncols=1, fontsize=12)
plt.show()

