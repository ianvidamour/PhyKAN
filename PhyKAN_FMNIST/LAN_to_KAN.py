#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:39:21 2024

@author: ian
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
device='cuda'
torch.set_default_device(device)

class PhyLAN_KAN(nn.Module):
    def __init__(self, LANshape, KANshape, filters_per_unit, lr_decay=0.99, sigmoid_s=0.5):
        super().__init__()
        # Find number of layers
        self.Nlayers_LAN = len(LANshape)-1
        self.Nlayers_KAN = len(KANshape)-1
        self.LANshape = LANshape
        self.KANshape = KANshape
        # Create parameters for each layer
        self.filter_params = []
        self.weights = []
        for i in range(self.Nlayers_LAN):
            layer_weights = 2*(torch.rand((LANshape[i], LANshape[i+1]))-0.5)*(torch.sqrt(torch.tensor(6/(LANshape[i]+LANshape[i+1]))))
            layer_weights.requires_grad=True
            low_params = torch.linspace(-1, 1, filters_per_unit).unsqueeze(0).unsqueeze(-1)
            layer_params = torch.tile(low_params, (LANshape[i+1], 1, 2))
            noise = (torch.rand(layer_params.shape)-0.5)*0.01
            layer_params = layer_params + noise
            gains = torch.rand((LANshape[i+1], filters_per_unit, 1))/filters_per_unit
            params = torch.cat((gains, layer_params), dim=2)
            params.requires_grad=True
            self.filter_params.append(params)
            self.weights.append(layer_weights)
        self.edge_mask = []
        for i in range(self.Nlayers_KAN):
            low_params = torch.linspace(-1, 1, filters_per_unit).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            layer_params = torch.tile(low_params, (KANshape[i], KANshape[i+1], 1, 2))
            noise = (torch.rand(layer_params.shape)-0.5)*0.01
            layer_params = layer_params + noise
            gains = torch.rand((KANshape[i], KANshape[i+1], filters_per_unit, 1))/filters_per_unit
            params = torch.cat((gains, layer_params), dim=3)
            params.requires_grad=True
            self.filter_params.append(params)
            layer_mask = torch.ones((KANshape[i], KANshape[i+1]))
            self.edge_mask.append(layer_mask)
        # Initialise optimisers
        self.filter_optim = torch.optim.Adam(params=self.filter_params, lr=1e-4)
        self.weight_optim = torch.optim.Adam(params=self.weights, lr=1e-3)
        self.sched1 = torch.optim.lr_scheduler.ExponentialLR(self.weight_optim, 0.99)
        self.sched2 = torch.optim.lr_scheduler.ExponentialLR(self.filter_optim, 0.99)
        self.lossfn = nn.BCEWithLogitsLoss()
        self.sigmoid_s = sigmoid_s
        
    def sig(self, x):
        return 1/(1+torch.exp(-x/self.sigmoid_s))
        
        
    # Redefined for torch & assuming trainable parameters
    def band_pass_LAN(self, Xin, params):
        # Bound parameters
        params= self.sig(params)
        # Encode inputs to frequency space
        freq = 10**(3.69+2*Xin.relu()).unsqueeze(-1)
        # Allow bounded gains to be both positive and negative
        gain = 10*(params[:,:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies from parameters
        fc_low = 10**(3.5+3*params[:,:, 1].unsqueeze(0))
        fc_high = 10**(3.5+3*params[:,:, 2].unsqueeze(0))
        # Convert cutoff frequency to product of resistance and capacitance
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        # Calculate steady state response with respect to drive frequency
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        # Sum across units within each edge
        return Hout.sum(dim=-1)
    
    def band_pass_KAN(self, Xin, params, mask):
        # Bound parameters
        params= self.sig(params)
        # Encode inputs to frequency space
        freq = 10**(3.69+2*Xin.relu()).unsqueeze(-1).unsqueeze(-1)
        # Allow bounded gains to be both positive and negative
        gain = 10*(params[:,:,:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies from parameters
        fc_low = 10**(3.5+3*params[:,:,:, 1].unsqueeze(0))
        fc_high = 10**(3.5+3*params[:,:,:, 2].unsqueeze(0))
        # Convert cutoff frequency to product of resistance and capacitance
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        # Calculate steady state response with respect to drive frequency
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        # Apply mask
        Hout = Hout * mask.unsqueeze(0).unsqueeze(-1)
        # Sum across units within each edge, and across multiple inputs to each node
        return Hout.sum(dim=1).sum(dim=-1)
    
    def forward(self, Xin):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs_LAN = []
        outputs_LAN = []
        for i in range(self.Nlayers_LAN):
            inputs_LAN.append(torch.zeros((batch_size, self.LANshape[i])))
            outputs_LAN.append(torch.zeros((batch_size, self.LANshape[i+1])))
        inputs_LAN[0] = Xin
        # Pass through model
        for layer in range(self.Nlayers_LAN):
            weighted_inputs = torch.matmul(inputs_LAN[layer], self.weights[layer])
            outputs_LAN[layer][:, :] = self.band_pass_LAN(self.sig(weighted_inputs), self.filter_params[layer])
            if layer < self.Nlayers_LAN-1:
                inputs_LAN[layer+1] = outputs_LAN[layer]
        inputs = []
        outputs = []
        for i in range(self.Nlayers_KAN):
            inputs.append(torch.zeros((batch_size, self.KANshape[i])))
            outputs.append(torch.zeros((batch_size, self.KANshape[i+1])))
        inputs[0][:, :] = self.sig((outputs_LAN[-1]))
        # Pass through model
        for layer in range(self.Nlayers_KAN):
            outputs[layer][:, :] = self.band_pass_KAN(inputs[layer], self.filter_params[layer+self.Nlayers_LAN], self.edge_mask[layer])
            if layer < self.Nlayers_KAN-1:
                inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        return outputs[-1]
    
    def train(self, Xin, Yin):
        # Reset optimisers
        self.filter_optim.zero_grad()
        self.weight_optim.zero_grad()
        # Pass through Model
        pred = self.forward(Xin)
        # Calculate loss
        loss = self.lossfn(pred, Yin)
        # Backward pass
        loss.backward()
        # Update parameters
        self.filter_optim.step()
        self.weight_optim.step()
        return loss.item()


import torchvision.datasets as datasets
import os

if torch.cuda.is_available()==True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device='cuda'

train_data = datasets.FashionMNIST(train=True, download=True, root=os.getcwd())
test_data = datasets.FashionMNIST(train=False, download=True, root=os.getcwd())

train_inputs = train_data.train_data.reshape(60000, 784)/255
train_labels = train_data.train_labels

test_inputs = test_data.train_data.reshape(10000, 784)/255
test_labels = test_data.train_labels

train_targets = torch.zeros((60000,10))
test_targets = torch.zeros((10000,10))

train_inputs = train_inputs.to('cuda')
test_inputs = test_inputs.to('cuda')

for i in range(60000):
    train_targets[i, train_labels[i]] = 1
    if i < 10000:
        test_targets[i, test_labels[i]] = 1
        
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device='cuda')
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample
        
model = PhyLAN_KAN([784, 100, 20], [20, 10], 6)
from tqdm import tqdm
for i in tqdm(range(100000)):
    Xs, Ys = gen_samples(train_inputs, train_targets, 500)
    loss = model.train(Xs, Ys)
    if i%1000 == 0 and i > 0:
        print(loss)
        with torch.no_grad():
            prediction = model.forward(test_inputs[:10000])
        correct = 0
        for i in range(10000):
            if torch.argmax(prediction[i])==torch.argmax(test_targets[i]):
                correct +=1
        accuracy = correct/10000
        model.sched1.step()
        model.sched2.step()
        print('Accuracy: ', accuracy)

