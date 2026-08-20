# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:37:57 2024

@author: Ian
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
        self.optimiser = optim.Adam(params=self.parameters, lr=1e-3)
        self.lossfn = nn.MSELoss()
        
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
for inp in range(27):
    function  = inp%27
    N = function // 27
    
    f, ranges, ideal_shape = return_function(function)
    model = MLP([ideal_shape[0], 10, 10, 1])
        
    def sample_renormalise(samples, ranges):
        Nlen = len(ranges)
        for idx in range(Nlen):
            low = ranges[idx][0]
            high = ranges[idx][1]
            samples[:, idx] = samples[:, idx]*(high-low)+low
        return samples
    
    Ntrain = 150000
    Nprune = 5000
    prunemax = 180000
    prunemin =  20000
    
    
    for i in range(Ntrain):
        Xs = torch.rand(2000, ideal_shape[0])
        rescaled_Xs = sample_renormalise(Xs, ranges)
        Ys = f(rescaled_Xs)
        loss = model.train(Xs, Ys)
        if i%1000 == 0:
            print(loss)
    
    with torch.no_grad():
        Xs = torch.rand(100000, ideal_shape[0])
        rescaled_Xs = sample_renormalise(Xs, ranges)
        Ys = f(rescaled_Xs)
        prediction = model.forward(Xs)
        MSE = model.lossfn(prediction, Ys)
    print('End MSE: '+str(MSE))
    torch.save(MSE, 'MLP, Function '+str(function)+' Accuracy.pt')
    torch.save(model, 'MLP, Function '+str(function)+' Model.pt')

