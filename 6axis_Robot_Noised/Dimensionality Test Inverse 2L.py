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

def trainables_KAN_2L(N):
    # Trainables = 18 * (in * hidden + hidden * hidden + hidden * out)
    return 18 * (6 * N + N * N + N * 3)


def trainables_MLP_2L(N):
    # Trainables = out + 2*hidden (biases) + in * hidden + hidden * hidden + hidden * out (weights)
    return (N + N + 6) + (6 * N + N * N + N * 3)

class Net_ik(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, load_model=None):
        super(Net_ik, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc12 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.lossfn = nn.MSELoss()
        self.optimiser = torch.optim.Adam([{'params':self.fc1.parameters()}, {'params':self.fc12.parameters()}, {'params':self.fc2.parameters()}], lr=1e-4, weight_decay=1e-5)
        if load_model!=None:
            self.load_model = load_model

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
        pred_loc = self.load_model.forward(im_pred)
        loss = self.lossfn(pred_loc, x)
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
    def train_ik_noised(self, x, noise_k):
        self.optimiser.zero_grad()
        im_pred = self.forward(x)
        pred_loc = self.noised_fw_kin(im_pred, noise_k)
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
        q2 = 3.84*(q2-0.5)
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
    
    def noised_fw_kin(self, angles, noise_k):
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
        q2 = 3.84*(q2-0.5)
        q3 = 1.22+(q3*0.7)
        q4 = 5.58*(q4-0.5)
        q5 = 4.18*(q5-0.5)
        q6 = 6.28*q6
        # Generate noise
        q1_noise = 2*(torch.rand(q1.shape)-0.5) * noise_k
        q2_noise = 2*(torch.rand(q2.shape)-0.5) * noise_k
        q3_noise = 2*(torch.rand(q3.shape)-0.5) * noise_k
        q4_noise = 2*(torch.rand(q4.shape)-0.5) * noise_k
        q5_noise = 2*(torch.rand(q5.shape)-0.5) * noise_k
        q6_noise = 2*(torch.rand(q6.shape)-0.5) * noise_k
        # Add noise
        q1 = q1 + q1_noise
        q2 = q2 + q2_noise
        q3 = q3 + q3_noise
        q4 = q4 + q4_noise
        q5 = q5 + q5_noise
        q6 = q6 + q6_noise
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

root_folder = './HPC_Data/'
Noise_ks = [0.1, 0.05, 0.01]
noise_k = 0.1
subfolder = str(noise_k)+'/'

Ydata = np.load('./Input_Data/PreNoise_6a_joint_angles_noise_k='+str(noise_k)+'.npy')
Xdata = np.load('./Input_Data/True_6a_effector_locations_noise_k='+str(noise_k)+'.npy')

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
dims_out_2l = np.zeros((10, 10, 4))
        

lossfn = torch.nn.MSELoss()


for i, N in enumerate(Netsizes):
    for run in range(10):
        model = torch.load(root_folder+subfolder+'Noised Inverse Kinematics Model 2L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location='cpu')
        activations = model.forward(x_test)
        outputs = activations[-1]
        locs = model.fw_kin(outputs)
        accuracies_out[i, run] = lossfn(locs, x_test)
        # Whiten data
        whitened_inputs, eig, vec = whiten_data(x_test.detach())
        whitened_hiddens1, eig, vec = whiten_data(activations[0].detach())
        whitened_hiddens2, eig, vec = whiten_data(activations[1].detach())
        whitened_hiddens3, eig, vec = whiten_data(activations[2].detach())
        # Calculate intrinsic dimensionality
        input_dimensionality = intrinsic_dimension(whitened_inputs.T)
        hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
        hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
        hidden_dimensionality3 = intrinsic_dimension(whitened_hiddens3.T)
        dims_out_2l[i, run, 1] = hidden_dimensionality1
        dims_out_2l[i, run, 2] = hidden_dimensionality2
        dims_out_2l[i, run, 3] = hidden_dimensionality3
        dims_out_2l[i, run, 0] = input_dimensionality
        parameters_out[i, run] = trainables_KAN_2L(N)




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

Netsizes = [5, 7, 9, 10, 12, 15, 20, 50, 100, 150, 200, 250]
accuracies_outMLP = np.zeros((12, 10))
parameters_outMLP = np.zeros((12, 10))
dims_out_mlp = np.zeros((12, 10, 4))




for i, N in enumerate(Netsizes):
    for run in range(10):
        model_2l = torch.load(root_folder+subfolder+'Noised Inverse Kinematics Model MLP 2L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location='cpu')
        pred_2l = model_2l.forward(x_test)
        locs = model_2l.fw_kin(pred_2l)
        loss = lossfn(locs, x_test)
        accuracies_outMLP[i, run] = loss.item()
        parameters_outMLP[i, run] = trainables_MLP_2L(N)
        l1_acts = model_2l.fc1(x_test).relu()
        l2_acts = model_2l.fc12(l1_acts).relu()
        l3_acts = model_2l.fc2(l2_acts)
        activations = [l1_acts, l2_acts, l3_acts]
        # Whiten data
        whitened_inputs, eig, vec = whiten_data(x_test.detach())
        whitened_hiddens1, eig, vec = whiten_data(activations[0].detach())
        whitened_hiddens2, eig, vec = whiten_data(activations[1].detach())
        whitened_hiddens3, eig, vec = whiten_data(activations[2].detach())
        # Calculate intrinsic dimensionality
        input_dimensionality = intrinsic_dimension(whitened_inputs.T)
        hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
        hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
        hidden_dimensionality3 = intrinsic_dimension(whitened_hiddens3.T)
        dims_out_mlp[i, run, 1] = hidden_dimensionality1
        dims_out_mlp[i, run, 2] = hidden_dimensionality2
        dims_out_mlp[i, run, 3] = hidden_dimensionality3
        dims_out_mlp[i, run, 0] = input_dimensionality
      

import matplotlib.pyplot as plt
#%%
fig, axes = plt.subplots(2,1, figsize=(5,6))
axes[0].set_title('Inverse Kinemeatics 2L, Hidden Layer 1')
axes[0].loglog(parameters_out.mean(axis=1), dims_out_2l[:, :, 1].mean(axis=1), color='red', label='PhyKAN')
axes[0].fill_between(parameters_out.mean(axis=1), dims_out_2l[:, :, 1].mean(axis=1)-dims_out_2l[:, :, 1].std(axis=1), dims_out_2l[:, :, 1].mean(axis=1)+dims_out_2l[:, :, 1].std(axis=1), color='red', alpha=0.2)

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
axes[0].set_title('Inverse Kinemeatics 2L, Hidden Layer 2')
axes[0].loglog(parameters_out.mean(axis=1), dims_out_2l[:, :, 2].mean(axis=1), color='red', label='PhyKAN')
axes[0].fill_between(parameters_out.mean(axis=1), dims_out_2l[:, :, 2].mean(axis=1)-dims_out_2l[:, :, 2].std(axis=1), dims_out_2l[:, :, 2].mean(axis=1)+dims_out_2l[:, :, 2].std(axis=1), color='red', alpha=0.2)

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

fig, axes = plt.subplots(2,1, figsize=(5,6))
axes[0].set_title('Noise: 0.1, Output Layer')
axes[0].axvline(1200, color='black', ls=':')
axes[1].axvline(1200, color='black', ls=':')
axes[0].loglog(parameters_out.mean(axis=1), dims_out_2l[:, :, 3].mean(axis=1), color='red', label='PhyKAN')
axes[0].fill_between(parameters_out.mean(axis=1), dims_out_2l[:, :, 3].mean(axis=1)-dims_out_2l[:, :, 3].std(axis=1), dims_out_2l[:, :, 3].mean(axis=1)+dims_out_2l[:, :, 3].std(axis=1), color='red', alpha=0.2)

axes[1].loglog(parameters_out.mean(axis=1), accuracies_out.mean(axis=1), color='red', label='PhyKAN')
axes[1].fill_between(parameters_out.mean(axis=1), accuracies_out.mean(axis=1)-accuracies_out.std(axis=1), accuracies_out.mean(axis=1)+accuracies_out.std(axis=1), color='red', alpha=0.2)

axes[0].loglog(parameters_outMLP.mean(axis=1), dims_out_mlp[:, :, 3].mean(axis=1), color='black', label='MLP')
axes[0].fill_between(parameters_outMLP.mean(axis=1), dims_out_mlp[:, :, 3].mean(axis=1)-dims_out_mlp[:, :, 3].std(axis=1), dims_out_mlp[:, :, 3].mean(axis=1)+dims_out_mlp[:, :, 3].std(axis=1), color='black', alpha=0.2)

axes[1].loglog(parameters_outMLP.mean(axis=1), accuracies_outMLP.mean(axis=1), color='black', label='MLP')
axes[1].fill_between(parameters_outMLP.mean(axis=1), accuracies_outMLP.mean(axis=1)-accuracies_outMLP.std(axis=1), accuracies_outMLP.mean(axis=1)+accuracies_outMLP.std(axis=1), color='black', alpha=0.2)

axes[0].set_xlabel('Trainable Parameters')
axes[1].set_xlabel('Trainable Parameters')
axes[0].set_ylabel('Intrinsic Dimensionality')
axes[1].set_ylabel('Mean Squared Error')
axes[0].set_yticks([2, 3, 4, 6, 8, 10, 12], labels=['2', '3', '4', '6', '8', '10', '12'])
plt.legend()
plt.show()
