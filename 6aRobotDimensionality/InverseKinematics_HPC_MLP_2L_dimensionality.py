#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:17:04 2024

@author: ian
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PhyKAN_Util import *
device='cuda'
torch.set_default_device(device)
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

class Net_ik(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, lr=1e-3):
        super(Net_ik, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.lossfn = nn.MSELoss()

    def forward(self, x, returnacts=False):
        h1 = self.fc1(x)
        a1 = h1.relu()
        h2 = self.fc2(a1)
        a2 = h2.relu()
        out = self.fc3(a2)
        if returnacts == False:
            return out
        else:
            return [a1, a2, out]
    
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
    def __init__(self, input_dim, hidden, output_dim, lr=1e-3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.lossfn = nn.MSELoss()

    def forward(self, x, returnacts=False):
        h1 = self.fc1(x)
        a1 = h1.relu()
        h2 = self.fc2(a1)
        a2 = h2.relu()
        out = self.fc3(a2)
        if returnacts == False:
            return out
        else:
            return [a1, a2, out]
    

def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')

import sys
inp = int(sys.argv[1])
Netsizes = [5, 7, 9, 10, 12, 15, 20, 50, 100, 150, 200, 250]
N = Netsizes[inp]
nfilt = 6

Ntr = 20000
Ncheck = 10
saved_weights = []
saved_thresholds = []
accuracies = []
Nprune = 2000

intrinsic_dim_inputs = np.zeros((Ntr//Ncheck))
intrinsic_dim_hiddens1 = np.zeros((Ntr//Ncheck))
intrinsic_dim_hiddens2 = np.zeros((Ntr//Ncheck))
intrinsic_dim_hiddens3 = np.zeros((Ntr//Ncheck))


shape = [3, N, N, 6]
for run in range(10):
    model = Net_ik(3, N, 6)
    
    Xdata = np.load('6a_effector_locations.npy')
    
    idxs = np.arange(0, len(Xdata), 1, dtype='int')
    np.random.seed(0)
    np.random.shuffle(idxs)
    
    shuffled_X = Xdata[idxs]
    
    x_train, x_val = np.split(shuffled_X, [12000])
    
    x_train = torch.from_numpy(x_train).float().to(device)
    x_val = torch.from_numpy(x_val).float().to(device)
    
    for i in range(Ntr):
        Xs, Ys = gen_samples(x_train, x_train, 500)
        loss = model.train_ik(Xs)
        if i%Ncheck == 0:
            print(loss)
            with torch.no_grad():
                
                # Measure Intrinsic Dimensionality
                inputs = x_val
                activations = model.forward(inputs, returnacts=True)
                true_loc = model.fw_kin(activations[-1])
                # Whiten data
                whitened_inputs, eig, vec = whiten_data(inputs.detach().cpu())
                whitened_hiddens1, eig, vec = whiten_data(activations[0].detach().cpu())
                whitened_hiddens2, eig, vec = whiten_data(activations[1].detach().cpu())
                whitened_hiddens3, eig, vec = whiten_data(activations[2].detach().cpu())
                # Calculate intrinsic dimensionality
                input_dimensionality = intrinsic_dimension(whitened_inputs.T)
                hidden_dimensionality1 = intrinsic_dimension(whitened_hiddens1.T)
                hidden_dimensionality2 = intrinsic_dimension(whitened_hiddens2.T)
                hidden_dimensionality3 = intrinsic_dimension(whitened_hiddens3.T)
                intrinsic_dim_hiddens1[i//Ncheck] = hidden_dimensionality1
                intrinsic_dim_hiddens2[i//Ncheck] = hidden_dimensionality2
                intrinsic_dim_hiddens3[i//Ncheck] = hidden_dimensionality3
                intrinsic_dim_inputs[i//Ncheck] = input_dimensionality
            loss = model.lossfn(true_loc, x_val)
            print(loss.item())
            accuracies.append(loss.cpu().detach().numpy())
    
    intrinsic_dims = np.zeros((3, Ntr//Ncheck))
    intrinsic_dims[0] = intrinsic_dim_inputs
    intrinsic_dims[1] = intrinsic_dim_hiddens1
    intrinsic_dims[2] = intrinsic_dim_hiddens2
    intrinsic_dims[3] = intrinsic_dim_hiddens3
    
    torch.save(model, 'Inverse Kinematics Model MLP 2L, N='+str(N)+' run '+str(run)+'.pt')
    np.save('Inverse Training Accuracies MLP 2L, N='+str(N)+' run '+str(run)+'.npy', accuracies)
    np.save('Intrinsic Dims MLP 2L, N='+str(N)+' run '+str(run)+'.npy', intrinsic_dims)
