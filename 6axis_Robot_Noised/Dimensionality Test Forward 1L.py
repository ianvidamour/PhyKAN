#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:39:41 2026

@author: ian
"""



import numpy as np
import torch
import torch.nn as nn
device='cpu'
torch.set_default_device(device)
from PhyKAN_Util import PhyKAN
from dimensionality import intrinsic_dimension

def whiten_data(Z,covariance_bias=True):
    
    # Check the type of Z and convert it to a numpy array if necessary
    if isinstance(Z, torch.Tensor):
        Z_np = Z.numpy()
    elif isinstance(Z, np.matrix):
        Z_np = np.array(Z)
    else:  # assuming Z is a numpy array
        Z_np = Z
          
    # Calculate the mean and subtract it
    mean = np.mean(Z_np, axis=0)
    Z_centered = Z_np - mean

    # Compute the covariance matrix
    cov_matrix = np.cov(Z_centered, rowvar=0, bias=covariance_bias) # should be the same   

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Clip the eigenvalues to be non-negative
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

    # Compute the diagonal matrix of inverse square roots of eigenvalues
    epsilon = 1e-12
    whiten_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + epsilon))

    # Whitening: decorrelate and scale features
    Z_whitened = Z_centered @ whiten_matrix
    
    Z_whitened = torch.from_numpy(Z_whitened)

    return Z_whitened, eigenvectors, eigenvalues

def trainables_KAN_1L(N):
    # Trainables = 18 * (in * hidden + hidden * hidden + hidden * out)
    return 18 * (6 * N + N * 3)


def trainables_MLP_1L(N):
    # Trainables = out + 2*hidden (biases) + in * hidden + hidden * hidden + hidden * out (weights)
    return (N + 3) + (6 * N + N * 3)

root_folder = './HPC_Data/'
Noise_ks = [0.1, 0.05, 0.01]
noise_k = 0.1
subfolder = str(noise_k)+'/'

Xdata = np.load('./Input_Data/PreNoise_6a_joint_angles_noise_k='+str(noise_k)+'.npy')
Ydata = np.load('./Input_Data/True_6a_effector_locations_noise_k='+str(noise_k)+'.npy')

Xin, x_test  = np.split(Xdata, [15000])
Yin, y_test  = np.split(Ydata, [15000])
idxs = np.arange(0, len(Xin), 1, dtype='int')
np.random.seed(0)
np.random.shuffle(idxs)
shuffled_X = Xin[idxs]
shuffled_Y = Yin[idxs]


x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

Netsizes = [2, 3, 4, 5, 7, 9, 10, 12, 15, 20]
accuracies_out = np.zeros((10, 10))
parameters_out = np.zeros((10, 10))
dims_out_1l = np.zeros((10, 10, 3))
        

lossfn = torch.nn.MSELoss()


for i, N in enumerate(Netsizes):
    for run in range(10):
        model = torch.load(root_folder+subfolder+'Noised Forward Kinematics Model 1L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location='cpu')
        activations = model.forward(x_test, input_sigmoid=False)
        outputs = activations[-1]
        accuracies_out[i, run] = lossfn(outputs, y_test)
        # Whiten data
        whitened_inputs, eig, vec = whiten_data(x_test.detach())
        whitened_hiddens1, eig, vec = whiten_data(activations[0].detach())
        whitened_hiddens2, eig, vec = whiten_data(activations[1].detach())
        # Calculate intrinsic dimensionality
        input_dimensionality = intrinsic_dimension(whitened_inputs.T)
        hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
        hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
        dims_out_1l[i, run, 1] = hidden_dimensionality1
        dims_out_1l[i, run, 2] = hidden_dimensionality2
        dims_out_1l[i, run, 0] = input_dimensionality
        parameters_out[i, run] = trainables_KAN_1L(N)



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

Netsizes = [5, 7, 9, 10, 12, 15, 20, 50, 100, 150, 200, 250]
accuracies_outMLP = np.zeros((12, 10))
parameters_outMLP = np.zeros((12, 10))
dims_out_mlp = np.zeros((12, 10, 3))




for i, N in enumerate(Netsizes):
    for run in range(10):
        model_1l = torch.load(root_folder+subfolder+'Noised Forward Kinematics MLP 1L, N='+str(N)+' run = '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location='cpu')
        pred_1l = model_1l.forward(x_test)
        loss = lossfn(pred_1l, y_test)
        accuracies_outMLP[i, run] = loss.item()
        parameters_outMLP[i, run] = trainables_MLP_1L(N)
        l1_acts = model_1l.fc1(x_test).relu()
        l2_acts = model_1l.fc2(l1_acts).relu()
        activations = [l1_acts, l2_acts]
        # Whiten data
        whitened_inputs, eig, vec = whiten_data(x_test.detach())
        whitened_hiddens1, eig, vec = whiten_data(activations[0].detach())
        whitened_hiddens2, eig, vec = whiten_data(activations[1].detach())
        # Calculate intrinsic dimensionality
        input_dimensionality = intrinsic_dimension(whitened_inputs.T)
        hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
        hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
        dims_out_mlp[i, run, 1] = hidden_dimensionality1
        dims_out_mlp[i, run, 2] = hidden_dimensionality2
        dims_out_mlp[i, run, 0] = input_dimensionality
      

import matplotlib.pyplot as plt
#%%
fig, axes = plt.subplots(2,1, figsize=(5,6))
axes[0].set_title('Forward Kinematics 1L, Hidden Layer')
# axes[0].axvline(3250, color='black', ls=':')
# axes[1].axvline(3250, color='black', ls=':')
axes[0].loglog(parameters_out.mean(axis=1), dims_out_1l[:, :, 1].mean(axis=1), color='red', label='PhyKAN')
axes[0].fill_between(parameters_out.mean(axis=1), dims_out_1l[:, :, 1].mean(axis=1)-dims_out_1l[:, :, 1].std(axis=1), dims_out_1l[:, :, 1].mean(axis=1)+dims_out_1l[:, :, 1].std(axis=1), color='red', alpha=0.2)

axes[1].loglog(parameters_out.mean(axis=1), accuracies_out.mean(axis=1), color='red', label='PhyKAN')
axes[1].fill_between(parameters_out.mean(axis=1), accuracies_out.mean(axis=1)-accuracies_out.std(axis=1), accuracies_out.mean(axis=1)+accuracies_out.std(axis=1), color='red', alpha=0.2)

axes[0].loglog(parameters_outMLP.mean(axis=1), dims_out_mlp[:, :, 1].mean(axis=1), color='black', label='MLP')
axes[0].fill_between(parameters_outMLP.mean(axis=1), dims_out_mlp[:, :, 1].mean(axis=1)-dims_out_mlp[:, :, 1].std(axis=1), dims_out_mlp[:, :, 1].mean(axis=1)+dims_out_mlp[:, :, 1].std(axis=1), color='black', alpha=0.2)

axes[1].loglog(parameters_outMLP.mean(axis=1), accuracies_outMLP.mean(axis=1), color='black', label='MLP')
axes[1].fill_between(parameters_outMLP.mean(axis=1), accuracies_outMLP.mean(axis=1)-accuracies_outMLP.std(axis=1), accuracies_outMLP.mean(axis=1)+accuracies_outMLP.std(axis=1), color='black', alpha=0.2)

#axes[0].set_xlabel('Trainable Parameters')
axes[1].set_xlabel('Trainable Parameters')
axes[0].set_ylabel('Intrinsic Dimensionality')
axes[1].set_ylabel('Mean Squared Error')
axes[0].set_yticks([2, 3, 4, 6, 8, 10, 12], labels=['2', '3', '4', '6', '8', '10', '12'])
plt.legend()
plt.show()


fig, axes = plt.subplots(2,1, figsize=(5,6))
axes[0].set_title('Noise: 0.1, Output Layer')
# axes[0].axvline(3250, color='black', ls=':')
# axes[1].axvline(3250, color='black', ls=':')
axes[0].loglog(parameters_out.mean(axis=1), dims_out_1l[:, :, 2].mean(axis=1), color='red', label='PhyKAN')
axes[0].fill_between(parameters_out.mean(axis=1), dims_out_1l[:, :, 2].mean(axis=1)-dims_out_1l[:, :, 2].std(axis=1), dims_out_1l[:, :, 2].mean(axis=1)+dims_out_1l[:, :, 2].std(axis=1), color='red', alpha=0.2)

axes[1].loglog(parameters_out.mean(axis=1), accuracies_out.mean(axis=1), color='red', label='PhyKAN')
axes[1].fill_between(parameters_out.mean(axis=1), accuracies_out.mean(axis=1)-accuracies_out.std(axis=1), accuracies_out.mean(axis=1)+accuracies_out.std(axis=1), color='red', alpha=0.2)

axes[0].loglog(parameters_outMLP.mean(axis=1), dims_out_mlp[:, :, 2].mean(axis=1), color='black', label='MLP')
axes[0].fill_between(parameters_outMLP.mean(axis=1), dims_out_mlp[:, :, 2].mean(axis=1)-dims_out_mlp[:, :, 2].std(axis=1), dims_out_mlp[:, :, 2].mean(axis=1)+dims_out_mlp[:, :, 2].std(axis=1), color='black', alpha=0.2)

axes[1].loglog(parameters_outMLP.mean(axis=1), accuracies_outMLP.mean(axis=1), color='black', label='MLP')
axes[1].fill_between(parameters_outMLP.mean(axis=1), accuracies_outMLP.mean(axis=1)-accuracies_outMLP.std(axis=1), accuracies_outMLP.mean(axis=1)+accuracies_outMLP.std(axis=1), color='black', alpha=0.2)

#axes[0].set_xlabel('Trainable Parameters')
axes[1].set_xlabel('Trainable Parameters')
axes[0].set_ylabel('Intrinsic Dimensionality')
axes[1].set_ylabel('Mean Squared Error')
axes[0].set_yticks([2, 3, 4, 6, 8, 10, 12], labels=['2', '3', '4', '6', '8', '10', '12'])
plt.legend()
plt.show()


