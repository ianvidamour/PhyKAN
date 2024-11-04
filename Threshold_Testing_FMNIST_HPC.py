#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:57:36 2024

@author: ian
"""
import numpy as np
import torch
import torch.nn as nn
device='cuda'
torch.set_default_device(device)

class PhyKAN(nn.Module):
    def __init__(self, KANshape, filters_per_unit, thresholding=True):
        super().__init__()
        # Find number of layers
        self.Nlayers = len(KANshape)-1
        self.KANshape = KANshape
        # Create parameters for each layer
        self.filter_params = []
        self.gains = []
        self.thresholds = []
        for i in range(self.Nlayers):
            layer_params = torch.rand((KANshape[i], KANshape[i+1], filters_per_unit, 2))
            layer_params.requires_grad=True
            gains = torch.rand((KANshape[i], KANshape[i+1], filters_per_unit))
            gains.requires_grad=True
            self.filter_params.append(layer_params)
            self.gains.append(gains)
            if thresholding==True:
                thresholds = 0.25*torch.randn(KANshape[i])
                thresholds.requires_grad=True
            else:
                thresholds = torch.zeros(KANshape[i], requires_grad=True)
            self.thresholds.append(thresholds)
        # Initialise optimisers
        self.gain_optim = torch.optim.Adam(params=self.gains, lr=1e-3)
        self.filter_optim = torch.optim.Adam(params=self.filter_params, lr=1e-3)
        self.threshold_optim = torch.optim.Adam(params=self.thresholds, lr=1e-5)
        self.sched1 = torch.optim.lr_scheduler.ExponentialLR(self.gain_optim, 0.97)
        self.sched2 = torch.optim.lr_scheduler.ExponentialLR(self.filter_optim, 0.97)
        self.lossfn = nn.BCEWithLogitsLoss()
        self.thresholding=thresholding
        
        
    # Redefined for torch & assuming trainable parameters
    def band_pass(self, Xin, params, gains):
        # Bound parameters
        params=params.clamp(0, 1)
        gains=gains.clamp(0, 1)
        # Encode inputs to frequency space
        freq = 10**(3.69+2*Xin).unsqueeze(-1).unsqueeze(-1)
        # Allow bounded gains to be both positive and negative
        gain = 10*(gains.unsqueeze(0)-0.5)
        # Set cutoff frequencies from parameters
        fc_low = 10**(0.5+5*params[:,:,:, 0].unsqueeze(0))
        fc_high = 10**(0.5+5*params[:,:,:, 1].unsqueeze(0))
        # Convert cutoff frequency to product of resistance and capacitance
        RC_low = (2*torch.pi*fc_low)**-1
        RC_high = (2*torch.pi*fc_high)**-1
        # Calculate steady state response with respect to drive frequency
        Hout = gain*torch.abs((2j*torch.pi*freq*RC_low/(1+2j*torch.pi*freq*RC_low))*(1/(1+2j*torch.pi*freq*RC_high)))
        # Sum across units within each edge, and across multiple inputs to each node
        return Hout.sum(dim=1).sum(dim=-1)
    
    # Redefined for torch & assuming trainable parameters
    def band_pass_single(self, Xin, params, gain, threshold):
        # Threshold input
        Xin = (Xin-threshold).relu()
        # Bound parameters
        params=params.clamp(0, 1)
        gain=gain.clamp(0, 1)
        # Normalise input of 0-1 to frequencies of 100 Hz-100 kHz
        freq = 10**(3.69+2*Xin).unsqueeze(-1).unsqueeze(-1)
        # For 'sensible' initialisation, we'd like parameters that make sense for gain, f_low, and f_high:
        # Set gain between +/- 5X
        gain = 10*(gain.unsqueeze(0)-0.5)
        # Set cutoff frequencies between 10 Hz and 1 MHz
        fc_low = 10**(0.5+5*params[:,:,:, 0].unsqueeze(0))
        fc_high = 10**(0.5+5*params[:,:,:, 1].unsqueeze(0))
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
            inputs[0][:, :] = (Xin-self.thresholds[0]).clamp(0, 1)
        else:
            inputs[0][:,:] = Xin.clamp(0, 1)
        # Pass through model
        for layer in range(self.Nlayers):
            outputs[layer][:, :] = self.band_pass(inputs[layer], self.filter_params[layer], self.gains[layer])      
            if layer < self.Nlayers-1:
                if self.thresholding==True:
                    inputs[layer+1][:, :] = (outputs[layer][:, :]+self.thresholds[layer+1]).clamp(0, 1)
                else:
                    inputs[layer+1][:, :] = outputs[layer][:, :].clamp(0, 1)
        return outputs[-1]
    
    def train(self, Xin, Yin, penalty=True, lamda=1e-3):
        # Reset optimisers
        self.filter_optim.zero_grad()
        self.gain_optim.zero_grad()
        self.threshold_optim.zero_grad()
        # Pass through Model
        pred = self.forward(Xin)
        # Calculate loss
        loss_np = self.lossfn(pred, Yin)
        # Add penalty terms
        if penalty==True:
            penalty_loss = lamda * self.penalty()
            loss = loss_np + penalty_loss
        else:
            loss = loss_np
        # Backward pass
        loss.backward()
        # Update parameters
        self.filter_optim.step()
        self.gain_optim.step()
        self.threshold_optim.step()
        
        return loss.item(), loss_np.item()

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
            entropy_layer = (1/n)*(entropy_in.sum() + entropy_out.sum())
            # Combine terms
            penalty = penalty + l1_layer + entropy_layer
        return penalty
        
    def prune(self, threshold = 0.01):
        xs = torch.rand(10000, 2)
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
        if self.thresholding==True:
            inputs[0][:, :] = (Xin-self.thresholds[0]).clamp(0, 1)
        else:
            inputs[0][:,:] = Xin.clamp(0, 1)
        # Pass through model
        for layer in range(self.Nlayers):
            outputs[layer][:, :] = self.band_pass(inputs[layer], self.filter_params[layer], self.gains[layer])      
            if layer < self.Nlayers-1:
                if self.thresholding==True:
                    inputs[layer+1][:, :] = (outputs[layer][:, :]+self.thresholds[layer+1]).clamp(0, 1)
                else:
                    inputs[layer+1][:, :] = outputs[layer][:, :].clamp(0, 1)
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
#inp = int(float(sys.argv[1]))
inp = 40
def rescale_inputs(Xin, ranges):
    Xout = torch.zeros_like(Xin)
    for i, ran in enumerate(ranges):
        low = ran[0]
        range = ran[1] - ran[0]
        Xout[:, i] = Xin[:, i]*range - low
    return Xout


if inp % 3 == 0:
    shape = [15, 20, 20, 20, 10]
    name = 'Large'
elif inp%3 == 1:
    shape = [15, 20, 20, 10]
    name = 'Small'
else:
    shape = [15, 25, 20, 10]
    name = 'Tailored'
    
if (inp//3)%2==0:
    thresholding=True
else:
    thresholding=False

if (inp//6)%2==0:
    penalty = True
else:
    penalty = False
shape = [15, 75, 20, 10]
nfilt = 2
model = PhyKAN(shape, nfilt, thresholding=False)
Ntr = 1000000
Ncheck = 1000
accuracies = np.zeros(int(Ntr/Ncheck)+1)
saved_weights = []
saved_thresholds = []
for i in range(Ntr):
    Xs, Ys = gen_samples(train_inputs, train_labels, 500)
    loss = model.train(Xs, Ys, penalty=True, lamda=1e-4)
    if i%Ncheck == 0:
        model.sched1.step()
        model.sched2.step()
        #model.prune(20, 1000, threshold=0.1)
        print(loss)
        with torch.no_grad():
            prediction = model.forward(test_inputs)
        correct = 0
        for k in range(10000):
            if torch.argmax(prediction[k])==torch.argmax(test_labels[k]):
                correct +=1
        accuracy = correct/10000
        print('Accuracy: ', accuracy)
        accuracies[i//Ncheck] = accuracy
        weights = []
        gains = []
        thresholds= [] 
        for gain in model.gains: gains.append(torch.clone(gain).cpu().detach().numpy())
        for weight in model.filter_params: weights.append(torch.clone(weight).cpu().detach().numpy())
        for threshold in model.thresholds: thresholds.append(torch.clone(threshold).cpu().detach().numpy())
        saved_weights.append(weights)
        saved_thresholds.append(thresholds)
np.save('Classification Filter Params, shape='+str(name)+', penalty='+str(penalty)+', threshold='+str(thresholding)+'.npy', np.asarray(saved_weights, dtype='object'))
np.save('Classification Gain Params, shape='+str(name)+', penalty='+str(penalty)+', threshold='+str(thresholding)+'.npy', np.asarray(saved_weights, dtype='object'))
np.save('Classification Accuracies, shape='+str(name)+', penalty='+str(penalty)+', threshold='+str(thresholding)+'.npy', np.asarray(accuracies))
np.save('Classification Thresholds, shape='+str(name)+', penalty='+str(penalty)+', threshold='+str(thresholding)+'.npy', np.asarray(saved_thresholds, dtype='object'))



