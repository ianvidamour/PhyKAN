#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 12:53:32 2025

@author: ian
"""
import torch
def return_function(number):
    pi = torch.tensor(torch.pi)
    
    if number == 0:
        name = 'I.6.2'
        f = lambda x: torch.exp(-x[:,[1]]**2/(2*x[:,[0]]**2))/torch.sqrt(2*pi*x[:,[0]]**2)
        ranges = [[-1,1],[0.5,2]]
        ideal_shape = [2, 2, 1, 1]
        
    if number == 1:
        name = 'I.6.2b'
        f = lambda x: torch.exp(-(x[:,[0]]-x[:,[1]])**2/(2*x[:,[2]]**2))/torch.sqrt(2*pi*x[:,[2]]**2)
        ranges = [[-1.5,1.5],[-1.5,1.5],[0.5,2]]
        ideal_shape = [3, 2, 2, 1, 1]
        
    if number == 2:
        name = 'I.19.8'
        f = lambda x: x[:,[0]]/((x[:,[1]]-1)**2+(x[:,[2]]-x[:,[3]])**2+(x[:,[4]]-x[:,[5]])**2)
        ranges = [[-1,1],[-1,-0.5],[-1,-0.5],[0.5,1],[-1,-0.5],[0.5,1]]
        ideal_shape = [6, 4, 2, 1, 1]
        
    if number == 3:
        name = 'I.12.11'
        f = lambda x: 1 + x[:,[0]]*torch.sin(x[:,[1]])
        ranges = [[-1,1],[0,2*pi]]
        ideal_shape = [2, 2, 2, 1]
        
    if number == 4:
        name = 'I.13.12'
        f = lambda x: x[:,[0]]*(1/x[:,[1]]-1)
        ranges = [[0,1],[0.5,2]]
        ideal_shape = [2, 2, 1]
        
    if number == 5:
        name = 'I.15.3x'
        f = lambda x: (1-x[:,[0]])/torch.sqrt((1-x[:,[1]]**2))
        ranges = [[-1,1],[-1,1],[-1,1],[1,2]]
        ideal_shape = [2, 2, 1, 1]
        
    if number == 6:
        name = 'I.16.6'
        f = lambda x: (x[:,[0]]+x[:,[1]])/(1+x[:,[0]]*x[:,[1]])
        ranges = [[-0.8,0.8],[-0.8,0.8]]
        ideal_shape = [2, 2, 2, 2, 2, 1]
        
    if number == 7:
        name = 'I.18.4'
        f = lambda x: (1+x[:,[0]]*x[:,[1]])/(1+x[:,[0]])
        ranges = [[0.5,1],[0.5,1]]
        ideal_shape = [2, 2, 2, 1, 1]
        
    if number == 8:
        name = 'I.26.2'
        f = lambda x: torch.arcsin(x[:,[0]]*torch.sin(x[:,[1]]))
        ranges = [[0,0.99],[0,2*pi]]
        ideal_shape = [2, 2, 2, 1, 1]
        
    if number == 9:
        name = 'I.27.6'
        f = lambda x: 1/(1+x[:,[0]]*x[:,[1]])
        ranges = [[0.5,2],[0.5,2]]
        ideal_shape = [2, 2, 1, 1]
        
    if number == 10:
        name = 'I.29.16'
        f = lambda x: torch.sqrt(1+x[:,[0]]**2-2*x[:,[0]]*torch.cos(x[:,[1]]-x[:,[2]]))
        ranges = [[-1,1],[0,2*pi],[0,2*pi]]
        ideal_shape = [3, 2, 2, 3, 2, 1, 1]
        
    if number == 11:
        name = 'I.30.3'
        f = lambda x: torch.sin(x[:,[1]]*x[:,[0]]/2)**2 / torch.sin(x[:,[0]]/2)**2
        ranges = [[0,4],[0.4*pi,1.6*pi]]
        ideal_shape = [2, 3, 2, 2, 1, 1]
        
    if number == 12:
        name = 'I.30.5'
        f = lambda x: torch.arcsin(x[:,[0]]/(x[:,[1]]))
        ranges = [[-1,1],[1,1.5]]
        ideal_shape = [2, 3, 2, 1]
        
    if number == 13:
        name = 'I.37.4'
        f = lambda x: 1+ x[:,[0]] + 2*torch.sqrt(x[:,[0]])*torch.cos(x[:,[1]])
        ranges = [[0.1,1],[0,2*pi]]
        ideal_shape = [2, 1, 1]
        
    if number == 14:
        name = 'I.40.1'
        f = lambda x: x[:,[0]] * torch.exp(-x[:,[1]])
        ranges = [[0,1],[-1,1]]
        ideal_shape = [2, 1, 1]
        
    if number == 15:
        name = 'I.44.4'
        f = lambda x: x[:,[0]]*torch.log(x[:,[1]])
        ranges = [[0,1],[0,1]]
        ideal_shape = [2, 1, 1]
        
    if number == 16:
        name = 'I.50.26'
        f = lambda x: (torch.cos(x[:,[0]])+x[:,[1]]*torch.cos(x[:,[0]])**2)
        ranges = [[0,2*pi],[0,1]]
        ideal_shape = [2, 2, 3, 1]
        
    if number == 17:
        name = 'II.2.42'
        f = lambda x: (x[:,[0]]-1)*x[:,[1]]
        ranges = [[0,1],[0,1]]
        ideal_shape = [2, 2, 1]
        
    if number == 18:
        name = 'II.6.15a'
        f = lambda x: x[:[0]]/(4*pi)*torch.sqrt(x[:,[1]]**2+x[:,[2]]**2)
        ranges = [[0,1],[0,1],[0,1]]
        ideal_shape = [3, 2, 2, 2, 1]
        
    if number == 19:
        name = 'II.11.7'
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*torch.cos(x[:,[2]]))
        ranges = [[0,1],[-1,1],[0,2*pi]]
        ideal_shape = [3, 3, 3, 2, 2, 1]
        
    if number == 20:
        name = 'II.11.27'
        f = lambda x: x[:,[0]]*x[:,[1]]/(1-x[:,[0]]*x[:,[1]]/3)
        ranges = [[0,1],[0,2]]
        ideal_shape = [2, 2, 1, 2, 1]
        
    if number == 21:
        name = 'II.35.18'
        f = lambda x: x[:,[0]]/(torch.exp(x[:,[1]])+torch.exp(-x[:,[1]]))
        ranges = [[0,1],[0,1],[0,1]]
        ideal_shape = [5, 3, 1]
        
    if number == 22:
        name = 'II.36.38'
        f = lambda x: x[:,[0]]+x[:,[1]]*x[:,[2]]
        ranges = [[0,1],[0,1],[0,1]]
        ideal_shape = [3, 3, 1]
        
    if number == 23:
        name = 'II.38.3'
        f = lambda x: x[:,[0]]/x[:,[1]]
        ranges = [[0,1],[0,1]]
        ideal_shape = [2, 1, 1]
        
    if number == 24:
        name = 'III.9.52'
        f = lambda x: x[:,[0]]*torch.sin((x[:,[1]]-x[:,[2]])/2)**2/((x[:,[1]]-x[:,[2]])/2)**2
        ranges = [[0,1],[0,pi],[0,pi]]
        ideal_shape = [3, 2, 3, 1, 1]
        
    if number == 25:
        name = 'III.10.19'
        f = lambda x: torch.sqrt(1+x[:,[0]]**2+x[:,[1]]**2)
        ranges = [[0,1],[0,1]]
        ideal_shape = [2, 1, 1]
        
    if number == 26:
        name = 'III.17.37'
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*torch.cos(x[:,[2]]))
        ranges = [[0,1],[0,1],[0,2*pi]]
        ideal_shape = [3, 3, 3, 2, 2, 1]
        
    return name, f, ranges, ideal_shape