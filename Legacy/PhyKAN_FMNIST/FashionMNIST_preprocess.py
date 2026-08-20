#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:59:47 2024

@author: ian
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import os

if torch.cuda.is_available()==True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device='cuda'

train_data = datasets.FashionMNIST(train=True, download=False, root=os.getcwd())
test_data = datasets.FashionMNIST(train=False, download=False, root=os.getcwd())

train_inputs = train_data.train_data.reshape(60000, 784)/255
train_labels = train_data.train_labels

test_inputs = test_data.train_data.reshape(10000, 784)/255
test_labels = test_data.train_labels

train_targets = torch.zeros((60000,10))
test_targets = torch.zeros((10000,10))

train_inputs = train_inputs.to('cuda')

for i in range(60000):
    train_targets[i, train_labels[i]] = 1
    if i < 10000:
        test_targets[i, test_labels[i]] = 1

class AutoEncoder(nn.Module):
    def __init__(self, img_size, lr_size):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.e1 = nn.Linear(img_size * img_size, 500)
        self.e2 = nn.Linear(500, 250)
        # Latent Representation
        self.lr = nn.Linear(250, lr_size)
        # Decoder
        self.d1 = nn.Linear(lr_size, 250)
        self.d2 = nn.Linear(250, 500)
        # Output
        self.o1 = nn.Linear(500, img_size * img_size)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        # Latent Representation
        x = torch.sigmoid(self.lr(x))
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.o1(x)
        return x
    
    def forward_encode(self, x):
        # Encoder
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        # Latent Representation
        x = torch.sigmoid(self.lr(x))
        return x

import sys
latentrep = int(float(sys.argv[1]))
encoder = AutoEncoder(28, latentrep)
optimiser = optim.Adam(encoder.parameters(), lr=1e-3)
lf = nn.MSELoss()

def train(encoder, optimiser, loss_func, Xin, Yin):
    optimiser.zero_grad()
    prediction = encoder.forward(Xin)
    loss = loss_func(prediction, Yin)
    loss.backward()
    optimiser.step()
    return loss.item()
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device='cuda')
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

for i in range(200000):
    Xsample, Ysample = gen_samples(train_inputs, train_inputs, 500)
    loss = train(encoder, optimiser, lf, Xsample, Ysample)
    if i %10000 == 0:
        print(loss)
        
#%%
train_inputs_encoded = encoder.forward_encode(train_inputs.cuda())
test_inputs_encoded = encoder.forward_encode(test_inputs.cuda())

torch.save(train_inputs_encoded, 'Encoded train inputs '+str(latentrep)+'.pt')
torch.save(test_inputs_encoded, 'Encoded test inputs'+str(latentrep)+'.pt')      
torch.save(train_targets, 'Training Targets'+str(latentrep)+'.pt')
torch.save(test_targets, 'Test Targets'+str(latentrep)+'.pt')
    
    
