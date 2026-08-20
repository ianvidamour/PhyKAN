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
device='cpu'
torch.set_default_device(device)


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
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

import sys
inp = int(sys.argv[1])

Netsizes = [5, 7, 9, 10, 12, 15, 20, 50, 100, 150, 200, 250]
N = Netsizes[inp%12]
run = (inp%120)//12
noise_ks = [0.1, 0.05, 0.01]
noise_k = noise_ks[inp//120]


Ntr = 100000
Ncheck = 2000
saved_weights = []
saved_thresholds = []
accuracies = []

lossfn = nn.MSELoss()
Xdata = np.load('PreNoise_6a_joint_angles_noise_k='+str(noise_k)+'.npy')
Ydata = np.load('Noised_6a_effector_locations_noise_k='+str(noise_k)+'.npy')
Xin, x_test  = np.split(Xdata, [15000])
Yin, y_test  = np.split(Ydata, [15000])
idxs = np.arange(0, len(Xin), 1, dtype='int')
np.random.seed(run)
np.random.shuffle(idxs)
shuffled_X = Xin[idxs]
shuffled_Y = Yin[idxs]

x_train, x_val = np.split(shuffled_X, [12000])
x_train = torch.from_numpy(x_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)

y_train, y_val = np.split(shuffled_Y, [12000])
y_train = torch.from_numpy(y_train).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)


accuracies = []
shape = [6, N, 3]
model = Net(6, N, 3)
optimiser = optim.Adam(model.parameters(), lr=1e-3)


for i in range(Ntr):
    optimiser.zero_grad()
    Xs, Ys = gen_samples(x_train, y_train, 500)
    pred = model.forward(Xs)
    loss = lossfn(pred, Ys)
    loss.backward()
    optimiser.step()
    if i%Ncheck == 0:
        with torch.no_grad():
            prediction = model.forward(x_val)
        loss = lossfn(y_val, prediction)
        print(loss.item())
        accuracies.append(loss.cpu().detach().numpy())
np.save('Noised MLP Accuracies 2L, N='+str(N)+' run ='+str(run)+', noise_k='+str(noise_k)+'.pt', accuracies)
torch.save(model, 'Noised Forward Kinematics MLP 2L, N='+str(N)+' run = '+str(run)+', noise_k='+str(noise_k)+'.pt')
