#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:48:57 2026

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn
device='cuda'
torch.set_default_device(device)
from MemKAN_Util import MemKAN
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample


import sys
inp = int(sys.argv[1])


Netsizes = [12, 15, 20]
N = Netsizes[inp//10]
run = inp%10
noise_k = 0.1
nfilt = 6

Ntr = 100000
Ncheck = 1000
saved_weights = []
saved_thresholds = []
accuracies = []
Nprune = 2000

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

shape = [6, N, N, 3]

Ntr = 100000
Ncheck = 1000


memdiode_model = torch.load('MemdiodeMLP, 2HL 125 nodes.pt', weights_only=False, map_location=device)
model = MemKAN(shape, 3, memdiode_model, lr=1e-2)

for i in range(Ntr):
    Xs, Ys = gen_samples(x_train, y_train, 500)
    loss = model.train(Xs, Ys, penalty=True, lamda=1e-5)
    if i%Ncheck == 0:
        with torch.no_grad():
            prediction = model.forward(x_val)[-1]
        loss = model.lossfn(y_val, prediction)
        print(loss.item())

torch.save(model, 'Noised Forward Kinematics Model Memristor 2L, N='+str(N)+', run '+str(run)+', noise_k='+str(noise_k)+'.pt')