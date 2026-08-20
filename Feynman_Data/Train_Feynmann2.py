#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:08:55 2025

@author: ian
"""
import numpy as np
import torch
import torch.nn as nn
device='cpu'
torch.set_default_device(device)
from PhyKAN_Util import PhyKAN
from Dimensionless_Feynmann_name import return_function

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')

import sys

inp = int(float(sys.argv[1]))

function = inp%27
N = inp//27

folder = 'FeynmanData'
name = return_function(function)
data = np.load(folder+'//'+name+'.npy')


Xin = data[:, :-1]
Yin = data[:, -1]

Xnorm = np.zeros_like(Xin)
for i in range(Xin.shape[1]):
    Xnorm[:, i] = Xin[:, i]/np.amax(np.abs(Xin[:, i]))
Ynorm = Yin/np.amax(Yin)

Xtrain, Xtest = np.split(Xnorm, [80000])
Ytrain, Ytest = np.split(Ynorm[:, None], [80000])

Xtrain = torch.from_numpy(Xtrain).float()
Xtest = torch.from_numpy(Xtest).float()
Ytrain = torch.from_numpy(Ytrain).float()
Ytest = torch.from_numpy(Ytest).float()

if N == 0:
    shape = [Xtrain.shape[1], 5, 1]
    model = PhyKAN(shape, 6, low_vals, high_vals, thresholding=False, lr=1e-2)
if N == 1:
    shape = [Xtrain.shape[1], 5, 5, 1]
    model = PhyKAN(shape, 6, low_vals, high_vals, thresholding=False, lr=1e-2)
if N == 2:
    shape = [Xtrain.shape[1], 5, 5, 5, 1]
    model = PhyKAN(shape, 6, low_vals, high_vals, thresholding=False, lr=1e-2)
if N == 3:
    shape = [Xtrain.shape[1], 5, 5, 5, 5, 1]
    model = PhyKAN(shape, 6, low_vals, high_vals, thresholding=False, lr=1e-2)
    

Ntrain = 200000
Nprune = 5000
prunemax = 150000
prunemin =  20000

lossfn = nn.MSELoss()
import matplotlib.pyplot as plt
for i in range(Ntrain):
    ks = np.random.randint(0, 80000, 2500)
    Xs = Xtrain[ks]
    Ys = Ytrain[ks]
    if i < 150000:
    	loss = model.train(Xs, Ys, lamda=1e-4, discrete=False)
    else:
    	loss = model.train(Xs, Ys, lamda=1e-9)
    if i%1000 == 0:
        if prunemin<i<prunemax and i%Nprune == 0:
            model.prune_edges()
            model.sched1.step()
        with torch.no_grad():
            prediction = model.forward(Xtest)[-1]
            MSE = lossfn(prediction, Ytest)
            print(MSE)
            plt.figure()
            plt.plot(prediction.numpy(), Ytest.numpy(), marker='.', lw=0)
            plt.show()
            
torch.save(model, 'Feynmann Function '+str(function)+', shape '+str(N)+'_Rerun.pt')

print('End MSE: '+str(MSE))


