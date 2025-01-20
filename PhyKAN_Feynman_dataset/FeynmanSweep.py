# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:37:57 2024

@author: Ian
"""
import numpy as np
import torch
import torch.nn as nn
device='cpu'
torch.set_default_device(device)

class PhyKAN(nn.Module):
    def __init__(self, KANshape, filters_per_unit, thresholding=False, lr_decay=0.99, sigmoid_s=0.5):
        super().__init__()
        # Find number of layers
        self.Nlayers = len(KANshape)-1
        self.KANshape = KANshape
        # Create parameters for each layer
        self.filter_params = []
        self.thresholds = []
        self.edge_mask = []
        for i in range(self.Nlayers):
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
            if thresholding==True:
                thresholds = 0.25*torch.randn(KANshape[i])
                thresholds.requires_grad=True
            else:
                thresholds = torch.zeros(KANshape[i], requires_grad=True)
            self.thresholds.append(thresholds)
        # Initialise optimisers
        self.filter_optim = torch.optim.Adam(params=self.filter_params, lr=1e-3)
        self.threshold_optim = torch.optim.Adam(params=self.thresholds, lr=1e-5)
        self.sched1 = torch.optim.lr_scheduler.ExponentialLR(self.filter_optim, 0.99)
        self.lossfn = nn.MSELoss()
        self.thresholding=thresholding
        self.sigmoid_s = sigmoid_s
        
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
    
    # Redefined for torch & assuming trainable parameters
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
        return outputs[-1]
    
    def train(self, Xin, Yin, penalty=True, lamda=1e-3):
        # Reset optimisers
        self.filter_optim.zero_grad()
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
                

    def forward_return_acts(self, Xin):
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
    

def return_function(number):
    pi = torch.tensor(torch.pi)
    
    if number == 0:
        f = lambda x: torch.exp(-x[:,[0]]**2/(2*x[:,[1]]**2))/torch.sqrt(2*pi*x[:,[1]]**2)
        ranges = [[-1,1],[0.5,2]]
        ideal_shape = [2, 2, 1, 1]
        
    if number == 1:
        f = lambda x: torch.exp(-(x[:,[0]]-x[:,[1]])**2/(2*x[:,[2]]**2))/torch.sqrt(2*pi*x[:,[2]]**2)
        ranges = [[-1.5,1.5],[-1.5,1.5],[0.5,2]]
        ideal_shape = [3, 2, 2, 1, 1]
        
    if number == 2:
        f = lambda x: x[:,[0]]/((x[:,[1]]-1)**2+(x[:,[2]]-x[:,[3]])**2+(x[:,[4]]-x[:,[5]])**2)
        ranges = [[-1,1],[-1,-0.5],[-1,-0.5],[0.5,1],[-1,-0.5],[0.5,1]]
        ideal_shape = [6, 4, 2, 1, 1]
        
    if number == 3:
        f = lambda x: x[:,[0]]*(x[:,[1]]+x[:,[2]]*x[:,[3]]*torch.sin(x[:,[4]]))
        ranges = [[-1,1],[-1,1],[-1,1],[-1,1],[0,2*pi]]
        ideal_shape = [5, 2, 2, 1]
        
    if number == 4:
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]*(1/x[:,[4]]-1/x[:,[3]])
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]
        ideal_shape = [5, 2, 1]
        
    if number == 5:
        f = lambda x: (x[:,[0]] - x[:,[1]]*x[:,[2]])/torch.sqrt(1-x[:,[1]]**2/x[:,[3]]**2)
        ranges = [[-1,1],[-1,1],[-1,1],[1,2]]
        ideal_shape = [4, 2, 1, 1]
        
    if number == 6:
        f = lambda x: x[:,[0]]*x[:,[1]]/(1+x[:,[0]]*x[:,[1]]/x[:,[2]]**2)
        ranges = [[-0.8,0.8],[-0.8,0.8],[1,2]]
        ideal_shape = [3, 2, 2, 2, 2, 1]
        
    if number == 7:
        f = lambda x: (x[:,[0]]*x[:,[1]]+x[:,[2]]*x[:,[3]])/(x[:,[0]]+x[:,[2]])
        ranges = [[0.5,1],[-1,1],[0.5,1],[-1,1]]
        ideal_shape = [4, 2, 2, 1, 1]
        
    if number == 8:
        f = lambda x: torch.arcsin(x[:,[0]]*torch.sin(x[:,[1]]))
        ranges = [[0,0.99],[0,2*pi]]
        ideal_shape = [2, 2, 2, 1, 1]
        
    if number == 9:
        f = lambda x: 1/(1/x[:,[0]]+x[:,[2]]/x[:,[1]])
        ranges = [[0.5,2],[1,2],[0.5,2]]
        ideal_shape = [3, 2, 1, 1]
        
    if number == 10:
        f = lambda x: torch.sqrt(x[:,[0]]**2+x[:,[1]]**2-2*x[:,[0]]*x[:,[1]]*torch.cos(x[:,[2]]-x[:,[3]]))
        ranges = [[-1,1],[-1,1],[0,2*pi],[0,2*pi]]
        ideal_shape = [4, 2, 2, 3, 2, 1, 1]
        
    if number == 11:
        f = lambda x: x[:,[0]] * torch.sin(x[:,[1]]*x[:,[2]]/2)**2 / torch.sin(x[:,[2]]/2)**2
        ranges = [[0,1],[0,4],[0.4*pi,1.6*pi]]
        ideal_shape = [3, 2, 2, 3, 2, 1, 1]
        
    if number == 12:
        f = lambda x: torch.arcsin(x[:,[0]]/(x[:,[1]]*x[:,[2]]))
        ranges = [[-1,1],[1,1.5],[1,1.5]]
        ideal_shape = [3, 3, 2, 2, 1, 1]
        
    if number == 13:
        f = lambda x: x[:,[0]] + x[:,[1]] + 2*torch.sqrt(x[:,[0]]*x[:,[1]])*torch.cos(x[:,[2]])
        ranges = [[0.1,1],[0.1,1],[0,2*pi]]
        ideal_shape = [3, 1, 1]
        
    if number == 14:
        f = lambda x: x[:,[0]] * torch.exp(-x[:,[1]]*x[:,[2]]*x[:,[3]]/(x[:,[4]]*x[:,[5]]))
        ranges = [[0,1],[-1,1],[-1,1],[-1,1],[1,2],[1,2]]
        ideal_shape = [6, 3, 2, 1]
        
    if number == 15:
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]*torch.log(x[:,[4]]/x[:,[3]])
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]
        ideal_shape = [5, 1, 1]
        
    if number == 16:
        f = lambda x: x[:,[0]]*(torch.cos(x[:,[2]]*x[:,[3]])+x[:,[1]]*torch.cos(x[:,[2]]*x[:,[3]])**2)
        ranges = [[0,1],[0,1],[0,2*pi],[0,1]]
        ideal_shape = [4, 2, 3, 1]
        
    if number == 17:
        f = lambda x: x[:,[0]]*(x[:,[2]]-x[:,[1]])*x[:,[3]]/x[:,[4]]
        ranges = [[0,1],[0,1],[0,1],[0,1],[0.5,2]]
        ideal_shape = [5, 2, 1]
        
    if number == 18:
        f = lambda x: 3/(4*pi*x[:,[0]])*x[:,[1]]*x[:,[2]]/x[:,[5]]**5*torch.sqrt(x[:,[3]]**2+x[:,[4]]**2)
        ranges = [[0.5,2],[0,1],[0,1],[0,1],[0,1],[0.5,2]]
        ideal_shape = [6, 2, 1]
        
    if number == 19:
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*x[:,[2]]*torch.cos(x[:,[3]])/(x[:,[4]]*x[:,[5]]))
        ranges = [[0,1],[-1,1],[-1,1],[0,2*pi],[0.5,2],[0.5,2]]
        ideal_shape = [6, 2, 3, 1]
        
    if number == 20:
        f = lambda x: x[:,[0]]*x[:,[1]]/(1-x[:,[0]]*x[:,[1]]/3)*x[:,[2]]*x[:,[3]]
        ranges = [[0,1],[0,2],[0,1],[0,1]]
        ideal_shape = [4, 2, 1]
        
    if number == 21:
        f = lambda x: x[:,[0]]/(torch.exp(x[:,[1]]*x[:,[2]]/(x[:,[3]]*x[:,[4]]))+torch.exp(-x[:,[1]]*x[:,[2]]/(x[:,[3]]*x[:,[4]])))
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]
        ideal_shape = [5, 3, 1]
        
    if number == 22:
        f = lambda x: x[:,[0]]*x[:,[1]]/(x[:,[2]]*x[:,[3]]) + x[:,[0]]*x[:,[4]]*x[:,[5]]/(x[:,[6]]*x[:,[7]]**2*x[:,[2]]*x[:,[3]])
        ranges = [[0,1],[0,1],[0.5,2],[0.5,2],[0,1],[0,1],[0.5,2],[0.5,2]]
        ideal_shape = [8, 2, 3, 1, 1]
        
    if number == 23:
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]
        ranges = [[0,1],[0,1],[0,1],[0.5,2]]
        ideal_shape = [4, 1, 1]
        
    if number == 24:
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]*torch.sin((x[:,[4]]-x[:,[5]])*x[:,[2]]/2)**2/((x[:,[4]]-x[:,[5]])*x[:,[2]]/2)**2
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0,pi],[0,pi]]
        ideal_shape = [6, 2, 3, 1, 1]
        
    if number == 25:
        f = lambda x: x[:,[0]]*torch.sqrt(x[:,[1]]**2+x[:,[2]]**2+x[:,[3]]**2)
        ranges = [[0,1],[0,1],[0,1],[0,1]]
        ideal_shape = [4, 1, 1]
        
    if number == 26:
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*torch.cos(x[:,[2]]))
        ranges = [[0,1],[0,1],[0,2*pi]]
        ideal_shape = [3, 3, 1]
        
    return f, ranges, ideal_shape

import sys

#inp = int(float(sys.argv[1]))
inp = 68
function  = inp%27
N = function // 27

f, ranges, ideal_shape = return_function(function)

if N == 0:
    shape = ideal_shape
    model = PhyKAN(shape, 6, thresholding=False)
if N == 1:
    shape = [ideal_shape[0], 5, 1]
    model = PhyKAN(shape, 6, thresholding=False)
if N == 2:
    shape = [ideal_shape[0], 5, 5, 1]
    model = PhyKAN(shape, 6, thresholding=False)
if N == 3:
    shape = [ideal_shape[0], 5, 5, 5, 1]
    model = PhyKAN(shape, 6, thresholding=False)
if N == 4:
    shape = [ideal_shape[0], 5, 5, 5, 5, 1]
    model = PhyKAN(shape, 6, thresholding=False)
    
def sample_renormalise(samples, ranges):
    Nlen = len(ranges)
    for idx in range(Nlen):
        low = ranges[idx][0]
        high = ranges[idx][1]
        samples[:, idx] = samples[:, idx]*(high-low)+low
    return samples

Ntrain = 200000
Nprune = 5000
prunemax = 180000
prunemin =  20000


for i in range(Ntrain):
    Xs = torch.rand(2000, shape[0])
    rescaled_Xs = sample_renormalise(Xs, ranges)
    Ys = f(rescaled_Xs)
    loss = model.train(Xs, Ys, lamda=1e-4)
    if i%1000 == 0:
        print(loss)
        if prunemin<i<prunemax and i%Nprune == 0:
            model.prune_edges(threshold=0.1)
#%%
with torch.no_grad():
    Xs = torch.rand(100000, shape[0])
    rescaled_Xs = sample_renormalise(Xs, ranges)
    Ys = f(rescaled_Xs)
    MSE = model.lossfn(Xs, Ys)
print('End MSE: '+str(MSE))
torch.save(MSE, 'Function '+str(function)+' Shape '+str(N)+' Accuracy.pt')
torch.save(model, 'Function '+str(function)+' Shape '+str(N)+' Model.pt')

