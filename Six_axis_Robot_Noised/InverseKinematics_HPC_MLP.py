#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:17:04 2024

@author: ian
"""
import numpy as np
import torch
import torch.nn as nn
device='cuda'
torch.set_default_device(device)

class Net_ik(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, load_model=None):
        super(Net_ik, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.lossfn = nn.MSELoss()
        self.optimiser = torch.optim.Adam([{'params':self.fc1.parameters()}, {'params':self.fc2.parameters()}], lr=1e-4)
        if load_model!=None:
            self.load_model = load_model.eval()


    def forward(self, x):
        out = self.fc1(x)
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
    

def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')

import sys
inp = int(sys.argv[1])

nfilt = 6

Ntr = 50000
Ncheck = 1000
saved_weights = []
saved_thresholds = []
accuracies = []
Nprune = 2000


Netsizes = [5, 7, 9, 10, 12, 15, 20, 50, 100, 150, 200, 250]
N = Netsizes[inp%12]
run = inp//12
noise_k = 0.1

forward_model = torch.load('./HPC_Data/0.1/Noised Forward Kinematics MLP 1L, N='+str(N)+' run = '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location=device)
model = Net_ik(3, N, 6, load_model=forward_model)
Xdata = np.load(f'True_6a_effector_locations_noise_k={noise_k}.npy')
Xin, x_test  = np.split(Xdata, [15000])
idxs = np.arange(0, len(Xin), 1, dtype='int')
np.random.seed(run)
np.random.shuffle(idxs)
shuffled_X = Xin[idxs]

x_train, x_val = np.split(shuffled_X, [12000])
x_train = torch.from_numpy(x_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)


count = 0
prev_val_loss = 1
for i in range(Ntr):
    Xs, Ys = gen_samples(x_train, x_train, 500)
    loss = model.train_ik(Xs)
    if i%Ncheck == 0:
        print(loss)
        with torch.no_grad():
            prediction = model.forward(x_val)
            true_loc = model.fw_kin(prediction)
        val_loss = model.lossfn(true_loc, x_val)
        if val_loss.item() > prev_val_loss:
            count += 1
        else:
            prev_val_loss = np.copy(val_loss.cpu())
            count -= 0.2
            count = np.max(count, 0)
        print(val_loss.item())
        accuracies.append(val_loss.cpu().detach().numpy())
        if count >= 3:
            print('Stopped Early')
            break

torch.save(model, './HPC_Data/0.1/Noised Inverse Kinematics Model MLP 1L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt')
np.save('./HPC_Data/0.1/Noised Inverse Training Accuracies MLP 1L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.npy', accuracies)
