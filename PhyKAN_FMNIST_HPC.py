#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:39:21 2024

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn

class PhyKAN(nn.Module):
    def __init__(self, KANshape, filters_per_unit, thresholding=True):
        super().__init__()
        # Find number of layers
        self.Nlayers = len(KANshape)-1
        self.KANshape = KANshape
        # Create parameters for each layer
        self.filter_params = []
        self.thresholds = []
        for i in range(self.Nlayers):
            layer_params = 4*(torch.rand((KANshape[i], KANshape[i+1], filters_per_unit, 3))-0.5)
            layer_params.requires_grad=True
            self.filter_params.append(layer_params)
            thresholds = torch.zeros((KANshape[i],)).requires_grad_()
            self.thresholds.append(thresholds)
        self.optim = torch.optim.Adam(params=self.filter_params+self.thresholds, lr=1e-3)
        self.lossfn = nn.MSELoss()
        self.thresholding=thresholding
        
    # Redefined for torch & assuming trainable parameters
    def band_pass(self, Xin, params):
        # Bound parameters
        params=params.sigmoid()
        # Normalise input of 0-1 to frequencies of 100 Hz-100 kHz
        freq = 10**(3.69+2*Xin).unsqueeze(-1).unsqueeze(-1)
        # For 'sensible' initialisation, we'd like parameters that make sense for gain, f_low, and f_high:
        # Set gain between +/- 5X
        gain = 10*(params[:,:,:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies between 10 Hz and 1 MHz
        fc_low = 10+(10**(0.5+5*params[:,:,:, 1].unsqueeze(0)))
        fc_high = 10+(10**(0.5+5*params[:,:,:, 2].unsqueeze(0)))
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        return Hout.sum(dim=1).sum(dim=-1)
    
    # Redefined for torch & assuming trainable parameters
    def band_pass_single(self, Xin, params):
        # Bound parameters
        params=params.sigmoid()
        # Normalise input of 0-1 to frequencies of 100 Hz-100 kHz
        freq = 10**(3.69+2*Xin).unsqueeze(-1).unsqueeze(-1)
        # For 'sensible' initialisation, we'd like parameters that make sense for gain, f_low, and f_high:
        # Set gain between +/- 5X
        gain = 10*(params[:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies between 10 Hz and 1 MHz
        fc_low = 10+(10**(0.5+5*params[:, 1].unsqueeze(0)))
        fc_high = 10+(10**(0.5+5*params[:, 2].unsqueeze(0)))
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        return Hout.sum(dim=1).sum(dim=-1)
    
    def forward(self, Xin):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(torch.zeros((batch_size, self.KANshape[i])))
            outputs.append(torch.zeros((batch_size, self.KANshape[i+1])))
        if self.thresholding==True:
            inputs[0] = (Xin-self.thresholds[0]).relu()
        else:
            inputs[0][:, :] = Xin
        # Pass through model
        for layer in range(self.Nlayers):
            outputs[layer][:, :] = self.band_pass(inputs[layer], self.filter_params[layer])      
            if layer < self.Nlayers-1:
                if self.thresholding==True:
                    inputs[layer+1][:, :] = (outputs[layer][:, :].sigmoid()-self.thresholds[layer+1]).relu()
                else:
                    inputs[layer+1][:, :] = outputs[layer][:, :].sigmoid()
        return outputs[-1]
    
    def train(self, Xin, Yin, l1=True, entropy=True, laml1=1e-3, lament=1e-3):
        self.optim.zero_grad()
        pred = self.forward(Xin)
        loss = self.lossfn(pred, Yin)
        if l1==True:
            l1_penalty = 0
            for layer in range(self.Nlayers):
                l1_penalty = l1_penalty + self.l1_layer(layer)
            loss = loss + l1_penalty * laml1
        if entropy==True:
            entropy_penalty = 0
            for layer in range(self.Nlayers):
                entropy_penalty = l1_penalty + self.entropy_layer(layer)
            loss = loss + entropy_penalty * lament
        loss.backward()
        self.optim.step()
        return(loss.item())

    def l1_node(self, l, i, j, n=100):
        xin = torch.tile(torch.linspace(0,1,n), (self.KANshape[0], 1)).T
        l1 = (1/n)*torch.sum(torch.abs(self.band_pass_single(xin, self.filter_params[l][i,j])))
        return l1

    def l1_layer(self, l):
        layer = self.filter_params[l]
        nin = layer.shape[0]
        nout = layer.shape[1]
        l1_layer = 0
        for i in range(nin):
            for j in range(nout):
                l1_layer = l1_layer + self.l1_node(l, i, j)
        return l1_layer

    def entropy_layer(self, l):
        layer = self.filter_params[l]
        nin = layer.shape[0]
        nout = layer.shape[1]
        entropy_layer = 0
        l1_layer = self.l1_layer(l)
        for i in range(nin):
            for j in range(nout):
                entropy_layer = entropy_layer + self.l1_node(l, i, j)/l1_layer * torch.log(self.l1_node(l, i, j)/l1_layer)
        entropy_layer = entropy_layer * -1
        return entropy_layer
    
    def prune(self, Nin, Nsamp, threshold=0.01):
        xs = torch.rand(Nsamp, Nin)
        outputs = self.forward_return_acts(xs)
        for layer in range(self.Nlayers-1):
            layerouts = torch.abs(outputs[layer]).mean(axis=0)
            for node, activation in enumerate(layerouts):
                if activation < threshold:
                    self.reshape(layer, node)
                    
    def reshape(self, layer, node):
        newshape = []
        for l in range(self.Nlayers+1):
            if l != layer+1:
                newshape.append(self.KANshape[l])
            else:
                newshape.append(self.KANshape[l]-1)
        new_params = []
        flag = 0
        for l in range(self.Nlayers):
            if flag == 0:
                if l != layer:
                    new_params.append(self.filter_params[l])
                else:
                    layer_params=[]
                    for i in range(self.KANshape[l+1]):
                        if i != node:
                            layer_params.append(self.filter_params[l][:, i, :, :])
                    layer_params = torch.stack(layer_params, dim=1)
                    new_params.append(layer_params)
                    flag = 1
            else:
                layer_params=[]
                for i in range(self.KANshape[l]):
                    if i != node:
                        layer_params.append(self.filter_params[l][i, :, :, :])
                layer_params = torch.stack(layer_params, dim=0)
                new_params.append(layer_params)
                flag = 0
        self.KANshape = newshape
        self.filter_params = new_params
        print('Pruned: Layer '+str(layer)+', Node '+str(node))
        print('New Shape: '+str(self.KANshape))
                
                        
    def forward_return_acts(self, Xin):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(torch.zeros((batch_size, self.KANshape[i])))
            outputs.append(torch.zeros((batch_size, self.KANshape[i+1])))
        inputs[0][:, :] = Xin
        # Pass through model
        for layer in range(self.Nlayers):
            outputs[layer][:, :] = self.band_pass(inputs[layer], self.filter_params[layer])      
            if layer < self.Nlayers-1:
                inputs[layer+1][:, :] = outputs[layer][:, :].sigmoid()
        return outputs
    

if torch.cuda.is_available()==True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device='cuda'
train_inputs = torch.load('Encoded train inputs 15.pt').to('cuda')
train_labels = torch.load('Training Targets15.pt').to('cuda')
test_inputs = torch.load('Encoded test inputs15.pt').to('cuda')
test_labels = torch.load('Test Targets15.pt').to('cuda')
train_inputs_np = train_inputs.cpu().detach().numpy()

def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device='cuda')
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

import sys
import os
run = int(float(sys.argv[1]))

nodes = [5, 10, 15, 20, 25, 50, 100]
nfilts = [1, 2, 3, 4, 5, 6]
nhiddens = [1, 2, 3]

node = nodes[run%7]
nfilt = nfilts[(run//7)%6]
nhidden = nhiddens[run//42]

if nhidden == 1:
    shape = [15, node, 10]
elif nhidden ==2:
    shape = [15, node, node, 10]
else:
    shape = [15, node, node, node, 10]
    
model = PhyKAN(shape, nfilt)
Ntr = 1000000
Ncheck = 1000
accuracies = np.zeros(int(Ntr/Ncheck)+1)
for i in range(Ntr):
    Xs, Ys = gen_samples(train_inputs, train_labels, 500)
    loss = model.train(Xs, Ys, l1=False, entropy=False)
    if i%Ncheck == 0:
        #model.prune(20, 1000, threshold=0.1)
        print(loss)
        prediction = model.forward(test_inputs)
        correct = 0
        for k in range(10000):
            if torch.argmax(prediction[k])==torch.argmax(test_labels[k]):
                correct +=1
        accuracy = correct/10000
        print('Accuracy: ', accuracy)
        accuracies[i//Ncheck] = accuracy
torch.save(model, 'Fashion MNIST Model, '+str(node)+' nodes, '+str(nhidden)+' layers, '+str(nfilt)+' filters.pt')
np.save('Accuracies, '+str(node)+' nodes, '+str(nhidden)+' layers, '+str(nfilt)+' filters.npy', accuracies)
