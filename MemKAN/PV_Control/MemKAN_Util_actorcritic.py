#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 15:02:08 2025

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn

class MemKAN_critic(nn.Module):
    def __init__(self, KANshape, units_per_edge, edge_model, sigmoid_s=0.5, lr=1e-3):
        super().__init__()
        # Find number of layers
        self.Nlayers = len(KANshape)-1
        self.KANshape = KANshape
        # Load model for edges
        self.edge_model = edge_model
        # Create parameters for each layer
        self.edge_params = []
        self.gains = []
        self.biases = []
        for i in range(self.Nlayers):
            gains = (torch.rand((KANshape[i], KANshape[i+1], units_per_edge))-0.5)/((KANshape[i] + KANshape[i+1]))
            biases = torch.zeros((KANshape[i], KANshape[i+1], units_per_edge))
            edge_constants = torch.rand((KANshape[i], KANshape[i+1], units_per_edge, 5))-0.5
            gains.requires_grad=True
            biases.requires_grad=True
            edge_constants.requires_grad=True
            self.edge_params.append(edge_constants)
            self.gains.append(gains)
            self.biases.append(biases)
        # Initialise optimisers
        self.edge_optim = torch.optim.Adam(params=self.edge_params+self.gains+self.biases, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.edge_optim, 0.99)
        self.lossfn = nn.MSELoss()
        self.sigmoid_s = sigmoid_s
    
    def sig(self, x):
        return 1/(1+torch.exp(-x/self.sigmoid_s))
    
    def mlp_edge_forward(self, Xin, params, gains, biases):
        # Get batch size
        batch_size = Xin.shape[0]
        din = params.shape[0]
        dout = params.shape[1]
        nunits = params.shape[2]
        # bound parameters
        params = self.sig(params)
        # Tile parameters to match shape of all edges/inputs, of shape [batch_size, d_in, d_out, nunits, 5]
        tiled_params = params.unsqueeze(0).tile(batch_size, 1, 1, 1, 1)
        # Tile inputs to all output edges, of shape [batch_size, d_in, d_out, nunits, 1]
        tiled_inputs = Xin.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).tile(1, 1, dout, nunits, 1)
        # Add biases
        biased_inputs = tiled_inputs + biases.unsqueeze(0).unsqueeze(-1)
        # Concatenate inputs to be first entry of final dimension for all edges, [batch_size, d_in, d_out, nunits, 6]
        mlp_inputs = torch.cat((biased_inputs, tiled_params), dim=-1)
        # Flatten to 2d
        flattened_inputs = torch.flatten(mlp_inputs, end_dim=3)
        # Pass through mlp
        edge_responses = self.edge_model.forward(flattened_inputs)
        # Reshape back to dims
        reshaped_output = edge_responses.reshape(batch_size, din, dout, nunits)
        # Apply gains
        scaled_output = reshaped_output * gains.unsqueeze(0)
        # Sum at edges
        summed_outputs = scaled_output.sum(dim=1).sum(dim=-1)
        return summed_outputs
    
    def forward(self, Xin, discrete=False):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(torch.zeros((batch_size, self.KANshape[i])))
            outputs.append(torch.zeros((batch_size, self.KANshape[i+1])))
       
        inputs[0][:,:] = self.sig((Xin))
        # Pass through model
        for layer in range(self.Nlayers):
            outputs[layer][:, :] = self.mlp_edge_forward(inputs[layer], self.edge_params[layer], self.gains[layer], self.biases[layer])
            if layer < self.Nlayers-1:
                inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        return outputs
    
    def train(self, Xin, Yin, penalty=True, lamda=1e-4):
        # Reset optimisers
        self.edge_optim.zero_grad()
        # Pass through Model
        activations = self.forward(Xin)
        pred = activations[-1]
        # Calculate loss
        loss_np = self.lossfn(pred, Yin)
        # Add penalty terms
        if penalty==True:
            penalty_loss = lamda * self.penalty(activations)
            loss = loss_np + penalty_loss
        else:
            loss = loss_np
        # Backward pass
        loss.backward()
        # Update parameters
        self.edge_optim.step()
        return loss.item()

    def penalty(self, activations):
        n = len(activations[0])
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
        
            

                
class MemKAN_actor(nn.Module):
    def __init__(self, KANshape, units_per_edge, edge_model, load_model=None, sigmoid_s=0.5, lr=1e-3):
        super().__init__()
        # Find number of layers
        self.Nlayers = len(KANshape)-1
        self.KANshape = KANshape
        # Load model for edges
        self.edge_model = edge_model
        # Create parameters for each layer
        self.edge_params = []
        self.gains = []
        self.biases = []
        for i in range(self.Nlayers):
            gains = (torch.rand((KANshape[i], KANshape[i+1], units_per_edge))-0.5)/((KANshape[i] + KANshape[i+1]))
            biases = torch.zeros((KANshape[i], KANshape[i+1], units_per_edge))
            edge_constants = torch.rand((KANshape[i], KANshape[i+1], units_per_edge, 5))-0.5
            gains.requires_grad=True
            biases.requires_grad=True
            edge_constants.requires_grad=True
            self.edge_params.append(edge_constants)
            self.gains.append(gains)
            self.biases.append(biases)
        # Initialise optimisers
        self.edge_optim = torch.optim.Adam(params=self.edge_params+self.gains+self.biases, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.edge_optim, 0.99)
        self.lossfn = nn.MSELoss()
        self.sigmoid_s = sigmoid_s
        if load_model!=None:
            self.load_model = load_model
    
    def sig(self, x):
        return 1/(1+torch.exp(-x/self.sigmoid_s))
    
    def mlp_edge_forward(self, Xin, params, gains, biases):
        # Get batch size
        batch_size = Xin.shape[0]
        din = params.shape[0]
        dout = params.shape[1]
        nunits = params.shape[2]
        # bound parameters
        params = self.sig(params)
        # Tile parameters to match shape of all edges/inputs, of shape [batch_size, d_in, d_out, nunits, 5]
        tiled_params = params.unsqueeze(0).tile(batch_size, 1, 1, 1, 1)
        # Tile inputs to all output edges, of shape [batch_size, d_in, d_out, nunits, 1]
        tiled_inputs = Xin.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).tile(1, 1, dout, nunits, 1)
        # Add biases
        biased_inputs = tiled_inputs + biases.unsqueeze(0).unsqueeze(-1)
        # Concatenate inputs to be first entry of final dimension for all edges, [batch_size, d_in, d_out, nunits, 6]
        mlp_inputs = torch.cat((biased_inputs, tiled_params), dim=-1)
        # Flatten to 2d
        flattened_inputs = torch.flatten(mlp_inputs, end_dim=3)
        # Pass through mlp
        edge_responses = self.edge_model.forward(flattened_inputs)
        # Reshape back to dims
        reshaped_output = edge_responses.reshape(batch_size, din, dout, nunits)
        # Apply gains
        scaled_output = reshaped_output * gains.unsqueeze(0)
        # Sum at edges
        summed_outputs = scaled_output.sum(dim=1).sum(dim=-1)
        return summed_outputs
    
    def forward(self, Xin, discrete=False):
        batch_size = len(Xin)
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(torch.zeros((batch_size, self.KANshape[i])))
            outputs.append(torch.zeros((batch_size, self.KANshape[i+1])))
       
        inputs[0][:,:] = self.sig((Xin))
        # Pass through model
        for layer in range(self.Nlayers):
            outputs[layer][:, :] = self.mlp_edge_forward(inputs[layer], self.edge_params[layer], self.gains[layer], self.biases[layer])
            if layer < self.Nlayers-1:
                inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        return outputs
    
    def train(self, Xin, Yin, penalty=True, lamda=1e-4):
        # Reset optimisers
        self.edge_optim.zero_grad()
        # Pass through Model
        activations = self.forward(Xin)
        pred = activations[-1]
        # Calculate loss
        loss_np = self.lossfn(pred, Yin)
        # Add penalty terms
        if penalty==True:
            penalty_loss = lamda * self.penalty(activations)
            loss = loss_np + penalty_loss
        else:
            loss = loss_np
        # Backward pass
        loss.backward()
        # Update parameters
        self.edge_optim.step()
        return loss.item()
    
    def train_actor_critic(self, Xin, penalty=True, stability_penalty=True, lamda=1e-4, stablam=1e-5, discrete=False):
        # Reset optimisers
        self.edge_optim.zero_grad()
        # Pass through Model
        activations = self.forward(Xin, discrete=discrete)
        actions = activations[-1]
        critic_input = torch.cat((actions, Xin), dim=1)
        q_values = self.load_model.forward(critic_input)[-1]
        loss = -1*q_values.mean()
        loss.backward()
        # Update parameters
        self.edge_optim.step()
        return loss.item()

    def penalty(self, activations):
        n = len(activations[0])
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
        
