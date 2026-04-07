#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:17:04 2024

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
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')



# Number of hidden nodes
N = 50
# Filters per edge
nfilt = 6

# Number of training iterations
Ntr = 20000
# How many iterations per check of dimensionality
Ncheck = 10
# Output for accuracies
accuracies = []

# How often to prune weak connections
Nprune = 2000

# Define KAN shape
shape = [3, N, N, 6]

# Initialise model
model = PhyKAN(shape, nfilt, low_vals, high_vals, lr=1e-2)

# Load data
Xdata = np.load('6a_effector_locations.npy')
# Random shuffling of samples. Note that this is only in numpy random number generation, so the model's starting weights
# are initialised randomly every run
idxs = np.arange(0, len(Xdata), 1, dtype='int')
np.random.seed(0)
np.random.shuffle(idxs)
shuffled_X = Xdata[idxs]

# Split into train test and validation shets
x_train, x_val, x_test = np.split(shuffled_X, [12000, 15000])
x_train = torch.from_numpy(x_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)

# Blank output matrices to record ID
intrinsic_dim_inputs = np.zeros((Ntr//Ncheck))
intrinsic_dim_hiddens1 = np.zeros((Ntr//Ncheck))
intrinsic_dim_hiddens2 = np.zeros((Ntr//Ncheck))
intrinsic_dim_hiddens3 = np.zeros((Ntr//Ncheck))

# Identifier to test multiple initialisations
run = 0
for i in range(Ntr):
    # Random sample of minibatch
    Xs, Ys = gen_samples(x_train, x_train, 50)
    # Strong penalty early in training to encourage sparsity
    if i < Ntr-20000:
        loss = model.train_ik(Xs, Xs, penalty=True, lamda=1e-5)
    # Weak penalty later in training to minimise error
    else:
        loss = model.train_ik(Xs, Xs, penalty=True, lamda=1e-9, discrete=True)
    if i%Ncheck == 0:
        model.sched1.step()
        print(loss)
        # Commenting out this block that was previously used for pruning
        
        # if i%Nprune==0 and i > 10000 and i < Ntr-5000:
        #     model.prune_edges()
        
        # Pass validation set with no gradient tracking 
        with torch.no_grad():
            prediction = model.forward(x_val)[-1]
            true_loc = model.fw_kin(prediction)
        # Evaluate validation loss
        loss = model.lossfn(true_loc, x_val)
        print(loss.item())
        # Append accuracies to be saved
        accuracies.append(loss.cpu().detach().numpy())
        # Measure Intrinsic Dimensionality
        inputs = x_val
        activations = model.forward(inputs)
        # Whiten data
        whitened_inputs, eig, vec = whiten_data(inputs.cpu())
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
        print('Iteration: '+str(i))
        print('Loss : '+str(loss))
        print('Input Intrinsic Dimensionality: '+str(input_dimensionality))
        print('Hidden Intrinsic Dimensionality 1: '+str(hidden_dimensionality1))
        print('Hidden Intrinsic Dimensionality 2: '+str(hidden_dimensionality2))
        print('Hidden Intrinsic Dimensionality 3: '+str(hidden_dimensionality3))

# Organise each layer's ID into matrix for single save file
intrinsic_dims = np.zeros((4, Ntr//Ncheck))
intrinsic_dims[0] = intrinsic_dim_inputs
intrinsic_dims[1] = intrinsic_dim_hiddens1
intrinsic_dims[2] = intrinsic_dim_hiddens2
intrinsic_dims[3] = intrinsic_dim_hiddens3

# Save the model and ID data
torch.save(model, 'Inverse Kinematics Model 2L No pruning, N='+str(N)+' run '+str(run)+'.pt')
np.save('Inverse Training Accuracies 2L No pruning, N='+str(N)+' run '+str(run)+'.npy', accuracies)
np.save('Intrinsic Dims No pruning, N='+str(N)+' run '+str(run)+'.npy', intrinsic_dims)
