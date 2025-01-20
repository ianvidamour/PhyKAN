#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:16:30 2024

@author: ian
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device='cuda'
torch.set_default_device(device)

class MLP(nn.Module):
    def __init__(self, NetShape):
        super(MLP, self).__init__()
        # Find number of layers
        self.Nlayers = len(NetShape)
        self.NetShape = NetShape
        self.parameters = []
        for layers in range(self.Nlayers-1):
            scaling = torch.sqrt(torch.tensor(2/(NetShape[layers]+NetShape[layers+1])))
            self.parameters.append((scaling*torch.randn((NetShape[layers], NetShape[layers+1]))).requires_grad_())
        self.optimiser = optim.Adam(params=self.params, lr=1e-3)
        self.lossfn = nn.BCEWithLogitsLoss()
        
    def forward(self, Xin):
        x = torch.matmul(Xin, self.parameters[0]).sigmoid()
        for layer in range(self.Nlayers-2):
            x = torch.matmul(x, self.parameters[layer+1]).sigmoid()
        return x
            
    def train(self, Xin, Yin):
        self.optimiser.zero_grad()
        prediction = self.forward(Xin)
        loss = self.lossfn(prediction, Yin)
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device='cuda')
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

import sys
#run  = int(float(sys.argv[1]))
run = 5
Nin = 15

train_inputs = torch.load('Encoded train inputs '+str(Nin)+'.pt')
train_targets = torch.load('Training Targets'+str(Nin)+'.pt')
test_inputs = torch.load('Encoded test inputs'+str(Nin)+'.pt')
test_targets = torch.load('Test Targets'+str(Nin)+'.pt')

train_inputs = train_inputs.to(device)
train_targets = train_targets.to(device)
test_inputs = test_inputs.to(device)
test_targets = test_targets.to(device)

nodes = [5, 10, 15, 20, 25, 50, 100]
nhiddens = [1, 2, 3]

node = nodes[run%7]
nhidden = nhiddens[run//7]

if nhidden == 1:
    shape = [15, node, 10]
elif nhidden ==2:
    shape = [15, node, node, 10]
else:
    shape = [15, node, node, node, 10]
MLP = MLP(shape)
Ntr = 1000000
Ncheck = 1000
accuracies = np.zeros(int(Ntr/Ncheck)+1)
for i in range(Ntr):
    Xsample, Ysample = gen_samples(train_inputs, train_targets, 500)
    loss = MLP.train(Xsample, Ysample)
    if i %Ncheck == 0:
        print(loss)
        with torch.no_grad():
            prediction0 = MLP.forward(test_inputs[:5000])
            prediction1 = MLP.forward(test_inputs[5000:])
        prediction = torch.concatenate((prediction0, prediction1))
        correct = 0
        for i in range(10000):
            if torch.argmax(prediction[i])==torch.argmax(test_targets[i]):
                correct += 1
        print('Accuracy: ',correct/10000)
        accuracies[i//Ncheck] = correct/10000
np.save('Accuracies, '+str(node)+' nodes, '+str(nhidden)+' layers.npy', correct/10000)


