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
            thresholds = (0.1*torch.randn((KANshape[i],))).requires_grad_()
            self.thresholds.append(thresholds)
        self.optim = torch.optim.Adam(params=self.filter_params, lr=1e-3)
        self.threshold_optim = torch.optim.Adam(params=self.thresholds, lr=1e-5)
        self.lossfn = nn.BCEWithLogitsLoss()
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
                    inputs[layer+1][:, :] = (outputs[layer][:, :]-self.thresholds[layer+1]).relu()
                else:
                    inputs[layer+1][:, :] = outputs[layer][:, :].sigmoid()
        return outputs[-1]
    
    def train(self, Xin, Yin, penalty=True, lamda=1e-3):
        # Reset optimisers
        self.optim.zero_grad()
        self.threshold_optim.zero_grad()
        # Pass through Model
        pred = self.forward(Xin)
        # Calculate loss
        loss = self.lossfn(pred, Yin)
        # Add penalty terms
        if penalty==True:
            penalty_loss = lamda * self.penalty()
            loss = loss + penalty_loss
        # Backward pass
        loss.backward()
        # Update parameters
        self.optim.step()
        self.threshold_optim.step()
        return(loss.item())

    def penalty(self, n=1000):
        # Create input range to sample edge behaviour
        xin = torch.rand((n, self.KANshape[0]))
        # Get activations
        activations = self.forward_return_acts(xin)
        # Set up variable
        penalty = 0
        # Loop over layers
        for layer in range(self.Nlayers-1):
            # Calculate l1 penalties
            l1_in = (1/n) * activations[layer].abs().sum(dim=0)
            l1_out = (1/n) * activations[layer+1].abs().sum(dim=0)
            l1_layer = l1_in.sum() + l1_out.sum()
            # Calculate entropy penalties
            entropy_in = -1 * (l1_in/l1_layer) * torch.log(l1_in/l1_layer)
            entropy_out = -1 * (l1_out/l1_layer) * torch.log(l1_out/l1_layer)
            entropy_layer = entropy_in.sum() + entropy_out.sum()
            # Combine terms
            penalty = penalty + l1_layer + entropy_layer
        return penalty
    
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

model = PhyKAN([15, 10, 10, 10, 10], 2, thresholding=False)
from tqdm import tqdm
for i in tqdm(range(1000000)):
    Xs, Ys = gen_samples(train_inputs, train_labels, 500)
    loss = model.train(Xs, Ys, lamda=1e-5)
    if i%1000 == 0:
        #model.prune(20, 1000, threshold=0.1)
        print(loss)
        prediction = model.forward(test_inputs[:1000])
        correct = 0
        for i in range(1000):
            if torch.argmax(prediction[i])==torch.argmax(test_labels[i]):
                correct +=1
        accuracy = correct/1000
        print('Accuracy: ', accuracy)
        

#%%
xin  = torch.linspace(0, 1, 1000)
filter_params = model.filter_params
print(len(filter_params))

for l, layer in enumerate(filter_params):
    for n, node in enumerate(layer):
        params = node[0]
        print(params)
        yf = model.band_pass_single(xin, params)
        plt.figure()
        plt.title('Layer: '+str(l)+', Node: '+str(n))
        plt.plot(xin.cpu().detach().numpy(), yf.sigmoid().cpu().detach().numpy())
        plt.show()