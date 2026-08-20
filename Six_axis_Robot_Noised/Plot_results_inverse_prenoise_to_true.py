#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:26:58 2025

@author: ian
"""


import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from PhyKAN_Util import PhyKAN

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


def trainables_KAN_2L(N):
    # Trainables = 18 * (in * hidden + hidden * hidden + hidden * out)
    return 18 * (6 * N + N * N + N * 3)


def trainables_MLP_2L(Netsize):
    # Trainables = out + 2*hidden (biases) + in * hidden + hidden * hidden + hidden * out (weights)
    return (N + N + 3) + (6 * N + N * N + N * 3)

import os
path = os.getcwd()


data_folder = '/HPC_Data/'
input_folder = '/Input_Data/'


noise_k = 0.1

true_effector_locations = np.load(path+input_folder+'True_6a_effector_locations_noise_k='+str(noise_k)+'.npy')
Xin, x_test  = np.split(true_effector_locations, [15000])
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
        model_KAN = torch.load(path+data_folder+subfolder+'Noised Inverse Kinematics Model 2L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location='cpu')
        model_pred = model_KAN.forward(x_test)[-1]
        model_locs = model_KAN.fw_kin(model_pred)
        accuracy = loss_function(model_locs, y_test)
        parameters_out2[i, run] = trainables_KAN_2L(N)
        accuracies_out2[i, run] = accuracy.item()
np.save('Inverse Kinematics PhyKAN 2L Simulation Accuracies noise_k=0.1.npy', accuracies_out2)
np.save('Inverse Parameter Counts, PhyKAN 2L.npy', parameters_out2)

        



Netsizes = [5, 7, 9, 10, 12, 15, 20, 50, 100, 150, 200, 250]
accuracies_outMLP2L = np.zeros((12, 10))
parameters_outMLP2L = np.zeros((12, 10))        
for i, N in enumerate(Netsizes):
    for run in range(10):
        model = torch.load(path+data_folder+subfolder+'Noised Inverse Kinematics Model MLP 2L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location='cpu')
        model_pred = model.forward(x_test)
        model_locs = model.fw_kin(model_pred)
        accuracy = loss_function(model_locs, y_test)
        parameters_outMLP2L[i, run] = trainables_MLP_2L(N)
        accuracies_outMLP2L[i, run] = accuracy.item()
np.save('Inverse Kinematics MLP 2L Accuracies noise_k=0.1.npy', accuracies_outMLP2L)
np.save('Inverse Parameter Counts, MLP 2L.npy', parameters_outMLP2L)
        

        
import scipy.stats as stats
plt.figure(dpi=1200, figsize=(4,3))
plt.title('Inverse 2L, Prenoised to true (Noise k = '+str(noise_k)+')', fontsize=14)
plt.loglog(parameters_out2.mean(axis=1), stats.gmean(accuracies_out2, axis=1), color='red', label='Software PhyKAN', marker='o')
plt.fill_between(parameters_out2.mean(axis=1), stats.gmean(accuracies_out2, axis=1)*stats.gstd(accuracies_out2, axis=1), stats.gmean(accuracies_out2, axis=1)/stats.gstd(accuracies_out2, axis=1), lw=0, alpha=0.2, color='red')
plt.loglog(parameters_outMLP2L.mean(axis=1), stats.gmean(accuracies_outMLP2L, axis=1), color='black', label='Software MLP', marker='o')
plt.fill_between(parameters_outMLP2L.mean(axis=1), stats.gmean(accuracies_outMLP2L, axis=1)*stats.gstd(accuracies_outMLP2L, axis=1), stats.gmean(accuracies_outMLP2L, axis=1)/stats.gstd(accuracies_outMLP2L, axis=1), lw=0, alpha=0.2, color='black')
plt.xlabel('Trainable Parameters', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend(frameon=False, ncols=1, fontsize=12)
plt.show()

