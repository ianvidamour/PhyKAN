#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 11:58:51 2025

@author: ian
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from PhyKAN_Util_actorcritic import *

class CriticNet(nn.Module):
    def __init__(self, input_shape, lr, actfn = nn.ReLU(), lossfn = nn.MSELoss()):
        super().__init__()
        self.Nlayers = len(input_shape)-1
        self.weights = []
        self.biases = []
        for layer in range(self.Nlayers):
            # Xavier initialisation for weights
            weights = (2*np.sqrt(6/(input_shape[layer]+input_shape[layer+1])))*(torch.rand((input_shape[layer], input_shape[layer+1]))-0.5)
            biases = torch.zeros((input_shape[layer+1]))
            weights.requires_grad=True
            biases.requires_grad=True
            self.weights.append(weights)
            self.biases.append(biases)
        self.optimiser = optim.Adam(self.weights+self.biases, lr=lr)
        self.actfn = actfn
        self.lossfn = lossfn
        
    def forward(self, x):
        h = torch.matmul(x, self.weights[0]) + self.biases[0]
        activation = self.actfn(h) 
        for i in range(1, self.Nlayers):
            h = torch.matmul(activation, self.weights[i]) + self.biases[i]
            activation = self.actfn(h)
        return h
    
    def train(self, x, y):
        self.optimiser.zero_grad()
        ypred = self.forward(x)
        loss = self.lossfn(ypred, y)
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
# Policy network for predicting what action to take with custom training routine
class ActorNet(nn.Module):
    def __init__(self, input_shape, lr, critic_model, actfn=nn.ReLU()):
        super().__init__()
        self.Nlayers = len(input_shape)-1
        self.weights = []
        self.biases = []
        for layer in range(self.Nlayers):
            # Xavier initialisation for weights
            weights = (2*np.sqrt(6/(input_shape[layer]+input_shape[layer+1])))*(torch.rand((input_shape[layer], input_shape[layer+1]))-0.5)
            biases = torch.zeros((input_shape[layer+1]))
            weights.requires_grad=True
            biases.requires_grad=True
            self.weights.append(weights)
            self.biases.append(biases)
        self.optimiser = optim.Adam(self.weights+self.biases, lr=lr, weight_decay=1e-5)
        self.actfn = actfn
        self.critic_model = critic_model
        
    def forward(self, x):
        h = torch.matmul(x, self.weights[0]) + self.biases[0]
        activation = self.actfn(h) 
        for i in range(1, self.Nlayers):
            h = torch.matmul(activation, self.weights[i]) + self.biases[i]
            activation = self.actfn(h)
        return h
    
    def train(self, x):
        self.optimiser.zero_grad()
        action = self.forward(x)
        critic_in = torch.cat((action, x), dim=1)
        q_values = self.critic_model.forward(critic_in)
        loss = -1*q_values.mean() + 1e-4*action.abs().mean()
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
def state_normalisation(states):
    norm_states = np.copy(states)
    norm_states[0] = states[0]/200
    norm_states[1] = states[1]/9
    return norm_states
    
def step_power(action, Vin):
    Vnew = Vin + action
    dnew = action
    return Vnew, dnew

def find_power_curve(power_data, current_data, Vin):
    MPPT = 1000
    # Ensure Vin is a Python float (handle 0-d or 1-element arrays)
    Vin = np.asarray(Vin)
    if Vin.size > 1:
        Vin = Vin.flatten()[0]
    Vin = float(Vin)
    V_index = int(Vin*5)
    # Accept either a 2D (1, N) row or a 1D (N,) array for power/current
    pd = np.asarray(power_data)
    cd = np.asarray(current_data)
    if pd.ndim > 1:
        pd = pd[0]
    if cd.ndim > 1:
        cd = cd[0]
    if V_index > 998:
        V_index=998
    if V_index < 0:
        V_index = 0
    low_val_P= pd[V_index]
    low_val_I= cd[V_index]
    remainder = (Vin*5)%1
    difference_P = pd[V_index+1]-low_val_P
    difference_I = cd[V_index+1] - low_val_I
    power = low_val_P + remainder*difference_P
    current = low_val_I + remainder*difference_I
    ratio = power/MPPT
    return power, current, ratio

def return_reward(ratio, prev_ratio, Vin):
    r1 = ratio
    if ratio > prev_ratio:
        r2 = r1**2
    else:
        r2 = 0
    if 0<Vin<200:
        r3 = 0
    else:
        r3 = -1
    return r1+r2+r3

def step_agent(state, actor_model, power_data, current_data, ratio):
    inputs = np.zeros((1, len(state)))
    # Normalise state
    inputs[0, :] = state_normalisation(state)
    # Predict Q values
    activations = actor_model.forward(torch.from_numpy(inputs))
    output = activations[-1]
    action = output.detach().numpy()
    # Step the agent: use the first action component as scalar
    action_scalar = action.flat[0]
    Vnew, dnew = step_power(action_scalar, state[0])
    power, current, new_ratio = find_power_curve(power_data, current_data, Vnew)
    reward = return_reward(new_ratio, ratio, Vnew)
    new_state = np.copy(state)
    new_state[0] = Vnew
    new_state[1] = current
    new_state[2] = dnew
    new_state[3] = new_ratio
    return new_state, action, reward, new_ratio, power




device='cpu'
torch.set_default_device(device)

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')

state_data = np.load('state_data.npy')
power_data = np.load('power_data.npy')
current_data = np.load('current_data.npy')

Ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 750, 1000]

performance_out = np.zeros((16, 10))

import os
for _, N in enumerate(Ns):
    for run in range(10):
        np.random.seed(run)
        actor_network = torch.load(os.getcwd()+'//HPC Data/MLP Actor network 2L, PV Control, N='+str(N)+', run '+str(run)+'.pt', weights_only=False)
        output_scores = np.zeros((100))
        for episode in range(100):
            power_plot = np.zeros((50))
            V_plot = np.zeros((50))
            k = int(np.random.randint(0, 2000))
            start_V = np.random.ranf()*200
            power, current, ratio = find_power_curve(power_data[k], current_data[k], start_V)
            state = np.array([start_V, current, 0, ratio])
            action_list=[]
            # Step the agent
            new_state, action, reward, new_ratio, power = step_agent(state, actor_network, power_data[k], current_data[k], ratio)
            state = new_state
            for step in range(50):
                power_plot[step] = power
                V_plot[step] = new_state[0]
                # Create inputs from state and action
                inputs = np.zeros((1, 4))
                inputs[0, :] = state_normalisation(state)
                # Step the agent
                new_state, action, reward, new_ratio, power = step_agent(state, actor_network, power_data[k], current_data[k], ratio)
                # Update state
                state = new_state
            power_plot/= np.amax(power_data[k])
            output_scores[episode] = power_plot.mean()
        performance_out[_, run] = output_scores.mean()
        print(output_scores.mean())

#%%
np.save('PV_MLP_scores.npy', performance_out)

plt.figure()
plt.plot(Ns, performance_out.mean(axis=1))
plt.fill_between(Ns, performance_out.mean(axis=1)-performance_out.std(axis=1), performance_out.mean(axis=1)+performance_out.std(axis=1), alpha=0.2)
plt.show()
