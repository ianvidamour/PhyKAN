#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:26:58 2025

@author: ian
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

Netsizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20]
masked_edges = np.zeros((14, 10))
intrinsic_dimensions1L = np.zeros((14, 10, 3))
accuracies_out = np.zeros((14, 10))
parameters_out = np.zeros((14, 10))


def trainables_KAN_1L(Netsize, mask):
    return 9*Netsize*6 - 3*mask

Xdata = np.load('6a_effector_locations.npy')

idxs = np.arange(0, len(Xdata), 1, dtype='int')
np.random.shuffle(idxs)
device='cpu'
lossfn = nn.MSELoss()
shuffled_X = Xdata[idxs]

x_train, x_val = np.split(shuffled_X, [12000])


x_train = torch.from_numpy(x_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)
import os


for i, N in enumerate(Netsizes):
    for run in range(10):
        accuracies = np.load(os.getcwd()+'//Inverse_Kinematics/Inverse Training Accuracies 1L, , N='+str(N)+' run '+str(run)+'.npy')
        accuracies_out[i, run] = np.average(accuracies[-10:])
        model = torch.load(os.getcwd()+'//Inverse_Kinematics/Inverse Kinematics Model 1L, N='+str(N)+' run '+str(run)+'.pt', weights_only=False)
        activations = model.forward(x_val)
        # Whiten data
        whitened_inputs, eig, vec = whiten_data(x_val.cpu())
        whitened_hiddens1, eig, vec = whiten_data(activations[0].detach().cpu())
        whitened_hiddens2, eig, vec = whiten_data(activations[1].detach().cpu())

        # Calculate intrinsic dimensionality
        input_dimensionality = intrinsic_dimension(whitened_inputs.T)
        hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
        hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
        intrinsic_dimensions1L[i, run] = [input_dimensionality, hidden_dimensionality1, hidden_dimensionality2]

        locs = model.fw_kin(activations[-1])
        loss = lossfn(locs, x_val)
        mask = model.edge_mask
        masked = 0
        for layer in mask:
            masked += (layer-1).abs().sum().item()
        masked_edges[i, run] = masked
        parameters_out[i, run] = trainables_KAN_1L(N, masked)
        
def trainables_KAN_2L(Netsize, mask):
    return 9*Netsize*6 - 3*mask + 18*Netsize**2

Netsizes = [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 20, 25, 30, 35]
intrinsic_dimensions2L = np.zeros((16, 10, 4))
masked_edges2 = np.zeros((16, 10))
accuracies_out2 = np.zeros((16, 10))
parameters_out2 = np.zeros((16, 10))




for i, N in enumerate(Netsizes):
    for run in range(10):
        try:
            accuracies = np.load(os.getcwd()+'//Inverse_Kinematics/Inverse Training Accuracies 2L, N='+str(N)+' run '+str(run)+'.npy')
            accuracies_out2[i, run] = np.average(accuracies[-10:])
            model = torch.load(os.getcwd()+'//Inverse_Kinematics/Inverse Kinematics Model 2L, N='+str(N)+' run '+str(run)+'.pt', weights_only=False)
            activations = model.forward(x_val)
            # Whiten data
            whitened_inputs, eig, vec = whiten_data(x_val.cpu())
            whitened_hiddens1, eig, vec = whiten_data(activations[0].detach().cpu())
            whitened_hiddens2, eig, vec = whiten_data(activations[1].detach().cpu())
            whitened_hiddens3, eig, vec = whiten_data(activations[2].detach().cpu())
            # Calculate intrinsic dimensionality
            input_dimensionality = intrinsic_dimension(whitened_inputs.T)
            hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
            hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
            hidden_dimensionality3 = intrinsic_dimension(whitened_hiddens3.T)
            intrinsic_dimensions2L[i, run] = [input_dimensionality, hidden_dimensionality1, hidden_dimensionality2, hidden_dimensionality3]
            locs = model.fw_kin(activations[-1])
            loss = lossfn(locs, x_val)
            mask = model.edge_mask
            masked = 0
            for layer in mask:
                masked += (layer-1).abs().sum().item()
            masked_edges2[i, run] = masked
            parameters_out2[i, run] = trainables_KAN_2L(N, masked)
        except:
            parameters_out2[i, run] = trainables_KAN_2L(N, masked)
            continue

import torch.nn as nn
class Net_ik(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, load_model=None):
        super(Net_ik, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc12 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.lossfn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(params=self.parameters(), lr=1e-3)
        if load_model!=None:
            self.load_model = torch.load(load_model, weights_only=False)

    def forward(self, x):
        out = self.fc1(x)
        out = out.relu()
        out = self.fc12(out)
        out = out.relu()
        out = self.fc2(out)
        return out
    
    def train_ik(self, x):
        self.optimiser.zero_grad()
        im_pred = self.forward(x)
        pred_loc = self.fw_kin(im_pred)
        loss = self.lossfn(pred_loc, x)
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
    def return_matrix(self, thetak, ak, dk, alphak, batch_size):
        cos = lambda x: torch.cos(x)
        sin = lambda x: torch.sin(x)
        T = torch.zeros((batch_size, 4, 4))
        # define denavit hartenberg transformations
        T[:, 0, 0] = cos(thetak)
        T[:, 0, 1] = -cos(alphak)*sin(thetak)
        T[:, 0, 2] = sin(alphak)*sin(thetak)
        T[:, 0, 3] = ak*cos(thetak)
        T[:, 1, 0] = sin(thetak)
        T[:, 1, 1] = cos(alphak)*cos(thetak)
        T[:, 1, 2] = -sin(alphak)*cos(thetak)
        T[:, 1, 3] = ak*sin(thetak)
        T[:, 2, 1] = sin(alphak)
        T[:, 2, 2] = cos(alphak)
        T[:, 2, 3] = dk
        T[:, 3, 3] = 1
        return T
    
    def fw_kin(self, angles):
        batch_size = angles.shape[0]
        # define lengths
        l1 = 0.290
        l2 = 0.270
        l3 = 0.070
        l4 = 0.134
        l5 = 0.168
        l6 = 0.072
        # take each angle
        q1 = angles[:, 0]
        q2 = angles[:, 1]
        q3 = angles[:, 2]
        q4 = angles[:, 3]
        q5 = angles[:, 4]
        q6 = angles[:, 5]
        # scale to ranges
        q1 = 5.76*(q1-0.5)
        q2 = 3.84*(q1-0.5)
        q3 = 1.22+(q3*0.7)
        q4 = 5.58*(q4-0.5)
        q5 = 4.18*(q5-0.5)
        q6 = 6.28*q6
        # generate denavit hartenberg transformations
        T10 = self.return_matrix(q1, torch.zeros(batch_size), torch.ones(batch_size)*l1, -torch.pi/2 * torch.ones(batch_size), batch_size)
        T21 = self.return_matrix(q2, torch.ones(batch_size)*l2, torch.zeros(batch_size), torch.zeros(batch_size), batch_size)
        T32 = self.return_matrix(q3, torch.ones(batch_size)*l3, torch.ones(batch_size)*l4, -torch.pi/2*torch.ones(batch_size), batch_size)
        T43 = self.return_matrix(q4, torch.zeros(batch_size), torch.ones(batch_size)*l5, torch.ones(batch_size)*torch.pi/2, batch_size)
        T54 = self.return_matrix(q5, torch.ones(batch_size)*l6, torch.zeros(batch_size), torch.ones(batch_size)*-torch.pi/2, batch_size)
        T65 = self.return_matrix(q6, torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size), batch_size)
        # Work through links
        T20 = torch.bmm(T10, T21)
        T30 = torch.bmm(T20, T32)
        T40 = torch.bmm(T30, T43)
        T50 = torch.bmm(T40, T54)
        T60 = torch.bmm(T50, T65)
        locs = T60[:, :-1, -1]
        return locs
    
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

Netsizes = [5, 7, 9, 10, 15, 20, 50, 100, 250]
accuracies_outMLP = np.zeros((9, 10))
parameters_outMLP = np.zeros((9, 10))
accuracies_outMLP2L = np.zeros((9, 10))
parameters_outMLP2L = np.zeros((9, 10))
intrinsic_dim_mlp2L = np.zeros((9, 10, 4))

def trainables_MLP_1L(Netsize):
    return 9*Netsize + 9 + Netsize

def trainables_MLP_2L(Netsize):
    return 9*Netsize + Netsize**2 + 9 + 2*Netsize


for i, N in enumerate(Netsizes):
    for run in range(10):
        model_2l = torch.load(os.getcwd()+'//Inverse_Kinematics/Inverse Kinematics Model MLP 2L, N='+str(N)+' run '+str(run)+'.pt', weights_only=False)
        pred_2l = model_2l.forward(x_val)
        hiddens_1 = model_2l.fc1(x_val).relu()
        hiddens_2 = model_2l.fc12(hiddens_1).relu()
        locs = model.fw_kin(pred_2l)
        loss = lossfn(locs, x_val)
        accuracies_outMLP2L[i, run] = loss.item()
        parameters_outMLP2L[i, run] = trainables_MLP_2L(N)
        
        # Whiten data
        whitened_inputs, eig, vec = whiten_data(x_val.cpu())
        whitened_hiddens1, eig, vec = whiten_data(hiddens_1.detach().cpu())
        whitened_hiddens2, eig, vec = whiten_data(hiddens_2.detach().cpu())
        whitened_hiddens3, eig, vec = whiten_data(pred_2l.detach().cpu())
        # Calculate intrinsic dimensionality
        input_dimensionality = intrinsic_dimension(whitened_inputs.T)
        hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
        hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
        hidden_dimensionality3 = intrinsic_dimension(whitened_hiddens3.T)
        intrinsic_dim_mlp2L[i, run] = [input_dimensionality, hidden_dimensionality1, hidden_dimensionality2, hidden_dimensionality3]
        
#%%
        
class Net_ik(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, load_model=None):
        super(Net_ik, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.lossfn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(params=self.parameters(), lr=1e-3)
        if load_model!=None:
            self.load_model = torch.load(load_model, weights_only=False)

    def forward(self, x):
        out = self.fc1(x)
        out = out.relu()
        out = self.fc2(out)
        return out
    
    def train_ik(self, x):
        self.optimiser.zero_grad()
        im_pred = self.forward(x)
        pred_loc = self.fw_kin(im_pred)
        loss = self.lossfn(pred_loc, x)
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
    def return_matrix(self, thetak, ak, dk, alphak, batch_size):
        cos = lambda x: torch.cos(x)
        sin = lambda x: torch.sin(x)
        T = torch.zeros((batch_size, 4, 4))
        # define denavit hartenberg transformations
        T[:, 0, 0] = cos(thetak)
        T[:, 0, 1] = -cos(alphak)*sin(thetak)
        T[:, 0, 2] = sin(alphak)*sin(thetak)
        T[:, 0, 3] = ak*cos(thetak)
        T[:, 1, 0] = sin(thetak)
        T[:, 1, 1] = cos(alphak)*cos(thetak)
        T[:, 1, 2] = -sin(alphak)*cos(thetak)
        T[:, 1, 3] = ak*sin(thetak)
        T[:, 2, 1] = sin(alphak)
        T[:, 2, 2] = cos(alphak)
        T[:, 2, 3] = dk
        T[:, 3, 3] = 1
        return T
    
    def fw_kin(self, angles):
        batch_size = angles.shape[0]
        # define lengths
        l1 = 0.290
        l2 = 0.270
        l3 = 0.070
        l4 = 0.134
        l5 = 0.168
        l6 = 0.072
        # take each angle
        q1 = angles[:, 0]
        q2 = angles[:, 1]
        q3 = angles[:, 2]
        q4 = angles[:, 3]
        q5 = angles[:, 4]
        q6 = angles[:, 5]
        # scale to ranges
        q1 = 5.76*(q1-0.5)
        q2 = 3.84*(q1-0.5)
        q3 = 1.22+(q3*0.7)
        q4 = 5.58*(q4-0.5)
        q5 = 4.18*(q5-0.5)
        q6 = 6.28*q6
        # generate denavit hartenberg transformations
        T10 = self.return_matrix(q1, torch.zeros(batch_size), torch.ones(batch_size)*l1, -torch.pi/2 * torch.ones(batch_size), batch_size)
        T21 = self.return_matrix(q2, torch.ones(batch_size)*l2, torch.zeros(batch_size), torch.zeros(batch_size), batch_size)
        T32 = self.return_matrix(q3, torch.ones(batch_size)*l3, torch.ones(batch_size)*l4, -torch.pi/2*torch.ones(batch_size), batch_size)
        T43 = self.return_matrix(q4, torch.zeros(batch_size), torch.ones(batch_size)*l5, torch.ones(batch_size)*torch.pi/2, batch_size)
        T54 = self.return_matrix(q5, torch.ones(batch_size)*l6, torch.zeros(batch_size), torch.ones(batch_size)*-torch.pi/2, batch_size)
        T65 = self.return_matrix(q6, torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size), batch_size)
        # Work through links
        T20 = torch.bmm(T10, T21)
        T30 = torch.bmm(T20, T32)
        T40 = torch.bmm(T30, T43)
        T50 = torch.bmm(T40, T54)
        T60 = torch.bmm(T50, T65)
        locs = T60[:, :-1, -1]
        return locs
    
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
    
intrinsic_dim_mlp1L = np.zeros((9, 10, 3))
for i, N in enumerate(Netsizes):
    for run in range(10):
        model_1l = torch.load(os.getcwd()+'//Inverse_Kinematics/Inverse Kinematics Model MLP 1L, N='+str(N)+' run '+str(run)+'.pt', weights_only=False)
        pred_1l = model_1l.forward(x_val)
        locs = model.fw_kin(pred_1l)
        loss = lossfn(locs, x_val)
        accuracies_outMLP[i, run] = loss.item()
        parameters_outMLP[i, run] = trainables_MLP_1L(N)
        pred_1l = model_1l.forward(x_val)
        hiddens_1 = model_1l.fc1(x_val).relu()
        
        # Whiten data
        whitened_inputs, eig, vec = whiten_data(x_val.cpu())
        whitened_hiddens1, eig, vec = whiten_data(hiddens_1.detach().cpu())
        whitened_hiddens2, eig, vec = whiten_data(pred_1l.detach().cpu())
        # Calculate intrinsic dimensionality
        input_dimensionality = intrinsic_dimension(whitened_inputs.T)
        hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
        hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
        intrinsic_dim_mlp1L[i, run] = [input_dimensionality, hidden_dimensionality1, hidden_dimensionality2]
        


#%%
fig, ax1 = plt.subplots(dpi=1200, figsize=(6,4))
ax2 = plt.twinx(ax1)
plt.title('Inverse Kinematics 1L', fontsize=14)
ax1.set_ylim(0, 0.01)
ax1.semilogx(parameters_out.mean(axis=1), np.mean(accuracies_out, axis=1), color='red', label='1 Hidden Layer (Software KAN)', marker='o')
ax1.fill_between(parameters_out.mean(axis=1), np.mean(accuracies_out, axis=1)+np.std(accuracies_out, axis=1), np.mean(accuracies_out, axis=1)-np.std(accuracies_out, axis=1), lw=0, alpha=0.2, color='red')
ax1.semilogx(parameters_outMLP.mean(axis=1), np.mean(accuracies_outMLP, axis=1), color='black', label='1 Hidden Layer (Software MLP)', marker='o')
ax1.fill_between(parameters_outMLP.mean(axis=1), np.mean(accuracies_outMLP, axis=1)+np.std(accuracies_outMLP, axis=1), np.mean(accuracies_outMLP, axis=1)-np.std(accuracies_outMLP, axis=1), lw=0, alpha=0.2, color='black')
ax2.semilogx(parameters_out.mean(axis=1), np.mean(intrinsic_dimensions1L[:, :, 1], axis=1), color='red', marker='+', ls=':')
ax2.fill_between(parameters_out.mean(axis=1), np.mean(intrinsic_dimensions1L[:, :, 1], axis=1)+np.std(intrinsic_dimensions1L[:, :, 1], axis=1), np.mean(intrinsic_dimensions1L[:, :, 1], axis=1)-np.std(intrinsic_dimensions1L[:, :, 1], axis=1), lw=0, alpha=0.2, color='red')
ax2.semilogx(parameters_outMLP.mean(axis=1), np.mean(intrinsic_dim_mlp1L[:, :, 1], axis=1), color='black', marker='+', ls=':')
ax2.fill_between(parameters_outMLP.mean(axis=1), np.mean(intrinsic_dim_mlp1L[:, :, 1], axis=1)+np.std(intrinsic_dim_mlp1L[:, :, 1], axis=1), np.mean(intrinsic_dim_mlp1L[:, :, 1], axis=1)-np.std(intrinsic_dim_mlp1L[:, :, 1], axis=1), lw=0, alpha=0.2, color='black')
ax1.set_xlabel('Trainable Parameters', fontsize=14)
ax1.set_ylabel('Mean Squared Error', fontsize=14)
ax2.set_ylabel('Intrinsic Dimensionality', fontsize=14)
plt.legend(frameon=False, ncols=1)
plt.show()


fig, ax1 = plt.subplots(dpi=1200, figsize=(6,4))
ax2 = plt.twinx(ax1)
plt.title('Inverse Kinematics 2L', fontsize=14)
ax1.set_ylim(0, 0.01)
ax1.semilogx(parameters_out2.mean(axis=1), np.mean(accuracies_out2, axis=1), color='red', label='1 Hidden Layer (Software KAN)', marker='o')
ax1.fill_between(parameters_out2.mean(axis=1), np.mean(accuracies_out2, axis=1)+np.std(accuracies_out2, axis=1), np.mean(accuracies_out2, axis=1)-np.std(accuracies_out2, axis=1), lw=0, alpha=0.2, color='red')
ax1.semilogx(parameters_outMLP2L.mean(axis=1), np.mean(accuracies_outMLP, axis=1), color='black', label='1 Hidden Layer (Software MLP)', marker='o')
ax1.fill_between(parameters_outMLP2L.mean(axis=1), np.mean(accuracies_outMLP2L, axis=1)+np.std(accuracies_outMLP2L, axis=1), np.mean(accuracies_outMLP2L, axis=1)-np.std(accuracies_outMLP2L, axis=1), lw=0, alpha=0.2, color='black')
ax2.semilogx(parameters_out2.mean(axis=1), np.mean(intrinsic_dimensions2L[:, :, 1], axis=1), color='red', marker='+', ls=':')
ax2.fill_between(parameters_out2.mean(axis=1), np.mean(intrinsic_dimensions2L[:, :, 1], axis=1)+np.std(intrinsic_dimensions2L[:, :, 1], axis=1), np.mean(intrinsic_dimensions2L[:, :, 1], axis=1)-np.std(intrinsic_dimensions2L[:, :, 1], axis=1), lw=0, alpha=0.2, color='red')
ax2.semilogx(parameters_outMLP2L.mean(axis=1), np.mean(intrinsic_dim_mlp2L[:, :, 1], axis=1), color='black', marker='+', ls=':')
ax2.fill_between(parameters_outMLP2L.mean(axis=1), np.mean(intrinsic_dim_mlp2L[:, :, 1], axis=1)+np.std(intrinsic_dim_mlp2L[:, :, 1], axis=1), np.mean(intrinsic_dim_mlp2L[:, :, 1], axis=1)-np.std(intrinsic_dim_mlp2L[:, :, 1], axis=1), lw=0, alpha=0.2, color='black')
ax1.set_xlabel('Trainable Parameters', fontsize=14)
ax1.set_ylabel('Mean Squared Error', fontsize=14)
ax2.set_ylabel('Intrinsic Dimensionality', fontsize=14)
plt.legend(frameon=False, ncols=1)
plt.show()

