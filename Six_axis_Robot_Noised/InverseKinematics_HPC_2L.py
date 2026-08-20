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
from PhyKAN_Util_noised import PhyKAN
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')

import sys
inp = int(sys.argv[1])

noise_k = 0.1
Netsizes = [2, 3, 4, 5, 7, 9, 10, 12, 15, 20]
N = Netsizes[inp%10]
run = inp//10

nfilt = 6


Ntr = 50000
Ncheck = 1000
saved_weights = []
saved_thresholds = []
accuracies = []
Nprune = 2000


shape = [3, N, N, 6]



forward_model = torch.load('Noised Forward Kinematics Model 2L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt', weights_only=False, map_location=device)
model = PhyKAN(shape, nfilt, low_vals, high_vals, lr=1e-2, load_model=forward_model)

Xdata = np.load(f'True_6a_effector_locations_noise_k={noise_k}.npy')
Xin, x_test  = np.split(Xdata, [15000])
idxs = np.arange(0, len(Xin), 1, dtype='int')
np.random.seed(run)
np.random.shuffle(idxs)
shuffled_X = Xin[idxs]

x_train, x_val = np.split(shuffled_X, [12000])
x_train = torch.from_numpy(x_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)


for i in range(Ntr):
    Xs, Ys = gen_samples(x_train, x_train, 500)
    if i < Ntr-20000:
        loss = model.train_ik(Xs, Xs, penalty=True, lamda=1e-5)
    else:
        loss = model.train_ik(Xs, Xs, penalty=True, lamda=1e-9, discrete=True)
    if i%Ncheck == 0:
        model.sched1.step()
        print(loss)
        if i%Nprune==0 and i > 20000 and i < Ntr-20000:
            model.prune_edges(threshold=1e-3)
        with torch.no_grad():
            prediction = model.forward(x_val)[-1]
            true_loc = model.fw_kin(prediction)
        loss = model.lossfn(true_loc, x_val)
        print(loss.item())
        accuracies.append(loss.cpu().detach().numpy())


torch.save(model, 'Noised Inverse Kinematics Model 2L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.pt')
np.save('Noised Inverse Training Accuracies 2L, N='+str(N)+' run '+str(run)+', noise_k='+str(noise_k)+'.npy', accuracies)
