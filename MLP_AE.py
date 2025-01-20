#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:27:00 2024

@author: ian
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import os
if torch.cuda.is_available()==True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device='cuda'
train_inputs = torch.load('Encoded train inputs.pt').to('cuda')
train_labels = torch.load('Training Targets.pt').to('cuda')
test_inputs = torch.load('Encoded test inputs.pt').to('cuda')
test_labels = torch.load('Test Targets.pt').to('cuda')


class Network(nn.Module):
    def __init__(self, NetShape):
        super(Network, self).__init__()
        # Encoder
        self.h1 = nn.Linear(30, NetShape[0])
        self.h2 = nn.Linear(NetShape[0], NetShape[1])
        # Output
        self.o1 = nn.Linear(NetShape[1], 10)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.o1(x)
        return x


def train(Network, optimiser, loss_func, Xin, Yin):
    optimiser.zero_grad()
    prediction = Network.forward(Xin)
    loss = loss_func(prediction, Yin)
    loss.backward()
    optimiser.step()
    return loss.item()
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device='cuda')
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

nodes = [5, 10, 15, 20, 25, 50, 100]
for node in nodes:
    NN = Network([node, node])
    optimiser = optim.Adam(NN.parameters(), lr=1e-3)
    lf = nn.MSELoss()
    
    for i in range(1000000):
        Xsample, Ysample = gen_samples(train_inputs, train_labels, 500)
        loss = train(NN, optimiser, lf, Xsample, Ysample)
        if i %10000 == 0:
            print('Loss: ',loss)
            test_prediction = NN.forward(test_inputs)
            correct=0
            for i in range(10000):
                if torch.argmax(test_prediction[i]) == torch.argmax(test_labels[i]):
                    correct += 1
            correct/=10000
            print('Accuracy: ', correct)