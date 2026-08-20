#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:39:52 2026

@author: ian
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
device = 'cuda'
X = np.load('MemdiodeModelX.npy')
Y = np.load('MemdiodeModelY.npy')

def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

Nsplits = 100
Ntrain = 3000000
Nval = 500000
Ntest = 500000
Niter = 500000
Nbatch = 5000
for k in range(10):
    inds = np.arange(0, len(X), 1, dtype='int')
    np.random.shuffle(inds)
    X = X[inds]
    Y = Y[inds]
    X_train, X_val, X_test = torch.split(torch.from_numpy(X).float().to(device), [Ntrain, Nval, Ntest])
    Y_train, Y_val, Y_test = torch.split(torch.from_numpy(Y).float().to(device), [Ntrain, Nval, Ntest])
    for N in [50, 75, 100, 125, 150, 175, 200, 250, 300]:
        
        model = nn.Sequential(nn.Linear(6, N), nn.ReLU(), nn.Linear(N, N), nn.ReLU(), nn.Linear(N, 1)).to(device)
        optimiser = optim.Adam(params=model.parameters(), lr=1e-4)
        lossfn = nn.MSELoss()
        
        for i in range(Niter):
            Xin, Yin = gen_samples(X_train, Y_train, Nbatch)
            optimiser.zero_grad()
            pred = model.forward(Xin)
            loss = lossfn(pred, Yin)
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                if i%1000 == 0:
                    running_loss = 0
                    for split in range(Nsplits):
                        pred = model.forward(X_val[split*int(Nval/Nsplits):(split+1)*int(Nval/Nsplits)])
                        loss = lossfn(pred, Y_val[split*int(Nval/Nsplits):(split+1)*int(Nval/Nsplits)])
                        running_loss += loss.item()/Nsplits
                    print(N, i, running_loss)
    
        with torch.no_grad():
            final_loss = 0
            for split in range(Nsplits):
                pred = model.forward(X_test[split*int(Ntest/Nsplits):(split+1)*int(Ntest/Nsplits)])
                loss = lossfn(pred, Y_test[split*int(Ntest/Nsplits):(split+1)*int(Ntest/Nsplits)])
                final_loss += loss.item()/Nsplits
            print(N, 'Test Accuracy:', running_loss)
        plt.figure()
        plt.plot(pred.cpu().detach().numpy(), Y_test[-5000:].cpu().detach().numpy(), lw=0, marker='.', alpha=0.2)
        plt.show()
        torch.save(model, 'MemdiodeMLP, N='+str(N)+', run='+str(k)+'.pt')
        np.save('MemdiodeMLP Accuracy, N='+str(N)+', run='+str(k)+'.pt', running_loss)
                
#%%
