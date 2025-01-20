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
    def __init__(self, KANshape, filters_per_unit, thresholding=False, lr_decay=0.99, sigmoid_s=0.5):
        super().__init__()
        # Find number of layers
        self.Nlayers = len(KANshape)-1
        self.KANshape = KANshape
        # Create lists to store parameters for each layer
        self.filter_params = []
        self.thresholds = []
        self.edge_mask = []
        for i in range(self.Nlayers):
            # Evenly distribute filter properties across the n filters per unit
            filt_params = torch.linspace(-1, 1, filters_per_unit).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            # Tile to shape of network
            layer_params = torch.tile(filt_params, (KANshape[i], KANshape[i+1], 1, 2))
            # Add some random perturbation to the corner frequencies so units do not start identically
            noise = (torch.rand(layer_params.shape)-0.5)*0.01
            layer_params = layer_params + noise
            # Add a gain to each sub-unit
            gains = torch.rand((KANshape[i], KANshape[i+1], filters_per_unit, 1))/filters_per_unit
            # Combine all physical parameters to a single tensor and mark for gradients
            params = torch.cat((gains, layer_params), dim=3)
            params.requires_grad=True
            # Add to list for given layer
            self.filter_params.append(params)
            # Initialise mask for pruning of network
            layer_mask = torch.ones((KANshape[i], KANshape[i+1]))
            self.edge_mask.append(layer_mask)
            # Add optional tuneable input threshold for each edge
            if thresholding==True:
                thresholds = 0.25*torch.rand(KANshape[i])
                thresholds.requires_grad=True
            else:
                thresholds = torch.zeros(KANshape[i], requires_grad=True)
            self.thresholds.append(thresholds)
        # Initialise optimisers
        self.filter_optim = torch.optim.Adam(params=self.filter_params, lr=1e-3)
        self.threshold_optim = torch.optim.Adam(params=self.thresholds, lr=1e-5)
        # Add a learning rate scheduler for the filter parameters
        self.sched1 = torch.optim.lr_scheduler.ExponentialLR(self.filter_optim, 0.99)
        # BCE Loss for classification, MSE for regression
        self.lossfn = nn.BCEWithLogitsLoss()
        self.thresholding=thresholding
        self.sigmoid_s = sigmoid_s
        
    # A redefined sigmoid function to maintain larger gradients over a bigger range
    def sig(self, x):
        return 1/(1+torch.exp(-x/self.sigmoid_s))
          
    # Redefined for torch & assuming trainable parameters
    def band_pass(self, Xin, params, mask):
        # Bound parameters
        params= self.sig(params)
        # Encode inputs to frequency space
        freq = 10**(3.69+2*Xin).unsqueeze(-1).unsqueeze(-1)
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

    # Variant of the above function used for pruning edges
    # (returns input/output response of each edge in a layer when a linearly spaced input is passed)
    def band_pass_edges(self, Xin, params):
        # Bound parameters
        params= self.sig(params)
        # Encode inputs to frequency space
        freq = 10**(3.69+2*Xin).unsqueeze(-1).unsqueeze(-1)
        # Allow bounded gains to be both positive and negative
        gain = 10*(params[:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies from parameters
        fc_low = 10**(3.5+3*params[:, 1].unsqueeze(0))
        fc_high = 10**(3.5+3*params[:, 2].unsqueeze(0))
        # Convert cutoff frequency to product of resistance and capacitance
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        # Calculate steady state response with respect to drive frequency
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        # Sum across units within each edge
        return Hout.sum(dim=-1)
    
    # Similar to the above, returns outputs for just one edge, useful for plotting
    def band_pass_single(self, Xin, params, threshold):
        # Threshold input
        Xin = (Xin-threshold).relu()
        # Bound parameters
        params=self.sig(params)
        # Normalise input of 0-1 to frequencies of 100 Hz-100 kHz
        freq = 10**(3.69+2*Xin).unsqueeze(-1).unsqueeze(-1)
        # For 'sensible' initialisation, we'd like parameters that make sense for gain, f_low, and f_high:
        # Set gain between +/- 5X
        gain = 10*(params[:, 0].unsqueeze(0)-0.5)
        # Set cutoff frequencies between 10 Hz and 1 MHz
        fc_low = 10**(3.5+3*params[:, 1].unsqueeze(0))
        fc_high = 10**(3.5+3*params[:, 2].unsqueeze(0))
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        return Hout.sum(dim=1).sum(dim=-1)
    
    # Forward pass through the model, returns activations on every node in the network
    def forward(self, Xin):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(torch.zeros((batch_size, self.KANshape[i])))
            outputs.append(torch.zeros((batch_size, self.KANshape[i+1])))
        if self.thresholding==True:
            inputs[0][:, :] = self.sig((Xin-self.thresholds[0]))
        else:
            inputs[0][:,:] = self.sig((Xin))
        # Pass through model
        for layer in range(self.Nlayers):
            outputs[layer][:, :] = self.band_pass(inputs[layer], self.filter_params[layer], self.edge_mask[layer])
            if layer < self.Nlayers-1:
                if self.thresholding==True:
                    inputs[layer+1][:, :] = self.sig(outputs[layer][:, :]+self.thresholds[layer+1])
                else:
                    inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        return outputs
    
    # Simple function to train the network given input/output pairings
    def train(self, Xin, Yin, penalty=True, lamda=1e-3):
        # Reset optimisers
        self.filter_optim.zero_grad()
        self.threshold_optim.zero_grad()
        # Pass through Model
        pred = self.forward(Xin)
        # Calculate loss
        loss_np = self.lossfn(pred[-1], Yin)
        # Add penalty terms
        if penalty==True:
            penalty_loss = lamda * self.penalty(pred)
            loss = loss_np + penalty_loss
        else:
            loss = loss_np
        # Backward pass
        loss.backward()
        # Update parameters
        self.filter_optim.step()
        self.threshold_optim.step()
        
        return loss.item(), loss_np.item()

    # Penalty term for the network. Aims to penalise high average activations across the edges (l1 term)
    # while also encouraging specialisation of information in fewer edges to promote sparsity (entropy term)
    def penalty(self, activations, n=1000):
        # Create input range to sample edge behaviour
        xin = torch.rand((n, self.KANshape[0]))
        # Set up variable
        penalty = 0
        # Loop over layers
        for layer in range(self.Nlayers-1):
            # Calculate l1 penalties 
            l1_in = (1/n) * activations[layer].abs().sum(dim=0)/self.KANshape[layer]
            l1_out = (1/n) * activations[layer+1].abs().sum(dim=0)/self.KANshape[layer+1]
            l1_layer = l1_in.sum() + l1_out.sum()
            # Calculate entropy penalties (ReLU/ small addition is to prevent NaNs pruned edges)
            entropy_in = -1 * (l1_in/l1_layer) * torch.log(1e-9+ (l1_in/l1_layer).relu())
            entropy_out = -1 * (l1_out/l1_layer) * torch.log(1e-9 + (l1_out/l1_layer).relu())
            entropy_layer = (entropy_in.sum() + entropy_out.sum())
            # Combine terms
            penalty = penalty + l1_layer + entropy_layer
        return penalty
        
    def prune(self, threshold = 0.01):
        xs = torch.rand(10000, self.KANshape[0])
        outputs = self.forward_return_acts(xs)
        for layer in range(self.Nlayers-1):
            layerouts = torch.abs(outputs[layer]).mean(axis=0)
            for node, activation in enumerate(layerouts):
                if activation < threshold:
                    self.reshape(layer, node)

    def prune_edges(self, threshold = 0.05):
        xs = torch.linspace(0, 1, 1000)
        for layer in range(self.Nlayers-1):
            for pre in range(self.KANshape[layer]):
                for post in range(self.KANshape[layer+1]):
                    if self.edge_mask[layer][pre, post] == 0:
                        continue
                    edges = self.band_pass_edges(xs, self.filter_params[layer][pre, post])
                    if torch.abs(edges).mean() < threshold:
                        self.edge_mask[layer][pre, post] = 0
                        print('Pruning edge, layer '+str(layer)+' '+str(pre)+' '+str(post))
            
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
                


    

device = 'cpu'
torch.set_default_device(device)
train_inputs = torch.load('Encoded train inputs 15.pt', map_location=device)
train_labels = torch.load('Training Targets15.pt', map_location=device)
test_inputs = torch.load('Encoded test inputs15.pt', map_location=device)
test_labels = torch.load('Test Targets15.pt', map_location=device)
train_inputs_np = train_inputs.cpu().detach().numpy()

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
    
model = PhyKAN(shape, nfilt, thresholding=False)

def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample

for i in range(200000):
    Xs, Ys = gen_samples(train_inputs, train_labels, 500)
    loss = model.train(Xs, Ys, penalty=True, lamda=1e-5)
    if i%1000 == 0:
        model.prune_edges()
        model.sched1.step()
        print(loss)
        prediction = model.forward(test_inputs[:1000])[-1]
        correct = 0
        for i in range(1000):
            if torch.argmax(prediction[i])==torch.argmax(test_labels[i]):
                correct +=1
        accuracy = correct/1000
        print('Accuracy: ', accuracy)
        
torch.save(model, 'Trained FMNIST model AE '+str(node)+'nodes, '+str(nfilt)+' filters, '+str(nhidden)+' layers.pt')
np.save('Model Accuracy AE '+str(node)+'nodes, '+str(nfilt)+' filters, '+str(nhidden)+' layers.npy', accuracy)
        
