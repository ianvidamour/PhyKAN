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
device='cuda'
torch.set_default_device(device)


class Net(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.relu()
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)
        out = out.relu()
        return out
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

import sys
inp = 59

Netsizes = [5, 7,9, 10, 12, 15, 20, 50, 100, 150, 200, 250]
N = Netsizes[inp%7]


if inp//21== 0:
    thresholding=True
else:
    thresholding=False

shape = [6, N, N, 3]

Ntr = 100000
Ncheck = 1000
saved_weights = []
saved_thresholds = []
accuracies = []

lossfn = nn.MSELoss()


for N in Netsizes:
    model = Net(6, N, N, 3)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    Xdata = np.load('Forward Kinematics Inputs.npy')
    Ydata = np.load('Forward Kinematics Targets.npy')
    
    idxs = np.arange(0, len(Xdata), 1, dtype='int')
    np.random.shuffle(idxs)
    
    shuffled_X = Xdata[idxs]
    shuffled_Y = Ydata[idxs]
    
    x_train, x_val = np.split(shuffled_X, [12000])
    y_train, y_val = np.split(shuffled_Y, [12000])
    
    x_train = torch.from_numpy(x_train).float().to(device)
    x_val = torch.from_numpy(x_val).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    
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
    
    torch.save(model, 'Forward Kinematics MLP, N='+str(N)+'.pt')
