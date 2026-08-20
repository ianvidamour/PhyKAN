#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:04:08 2025

@author: ian
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device='cpu'
torch.set_default_device(device)


def sig(x):
    return 1/(1+np.exp(-x/0.5))
    
class Experimental_interpolation_model():
    def __init__(self, edge_responses, edge_mask, sigmoid_s=0.5):
        self.Nlayers = len(edge_responses)
        self.edge_mask = []
        for layer in edge_mask:
            self.edge_mask.append(layer.numpy())
        self.edge_responses = edge_responses
        self.N_disc = len(edge_responses[0][0,0])-1
        self.sigmoid_s = sigmoid_s
        
    def linear_interpolate(self, edge_response, input_data):
        discsteps = self.N_disc
        steps = np.asarray(np.floor((input_data*discsteps)), dtype='int')
        remainder = discsteps*(input_data - (steps/discsteps))
        lows = edge_response[steps]
        highs = edge_response[steps + 1]
        difference = highs - lows
        interpolate = lows+difference*remainder
        return interpolate
        
    def forward(self, input_data):
        batch_size = input_data.shape[0]
        # Initialise inputs to and outputs from each layer
        inputs = []
        outputs = []
        for i in range(self.Nlayers):
            inputs.append(np.zeros((batch_size, self.edge_responses[i].shape[0])))
            outputs.append(np.zeros((batch_size, self.edge_responses[i].shape[1])))
        inputs[0] = self.sig(input_data)
        # Pass through model
        for layer in range(self.Nlayers):
            shape = [self.edge_responses[layer].shape[0], self.edge_responses[layer].shape[1]]
            for pre in range(shape[0]):
                for post in range(shape[1]):
                    outputs[layer][:, post] += self.linear_interpolate(self.edge_responses[layer][pre, post], inputs[layer][:, pre])*self.edge_mask[layer][pre, post]
            if layer < self.Nlayers-1:
                inputs[layer+1][:, :] = self.sig(outputs[layer][:, :])
        return inputs, outputs

    def sig(self, x):
        return 1/(1+np.exp(-x/self.sigmoid_s))

    
def linear_interpolate(edge_response, input_data):
    discsteps = 50
    steps = np.asarray(np.floor((input_data*discsteps)), dtype='int')
    remainder = discsteps*(input_data - (steps/discsteps))
    lows = edge_response[steps]
    highs = edge_response[steps + 1]
    difference = highs - lows
    interpolate = lows+difference*remainder
    return interpolate

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

Netsizes = [5, 7, 9, 10, 12, 15, 20]
import os.path
import os

def state_normalisation(state):
    norm_state = np.zeros_like(state)
    norm_state[0] = state[0]/2.4 
    norm_state[1] = state[1]/3 
    norm_state[2] = state[2]/0.418 
    norm_state[3] = state[3]/5 
    return norm_state

device='cpu'
torch.set_default_device(device)

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')

from CartPole_env import CartpoleEnvironment
env = CartpoleEnvironment()
x, xdot, theta, thetadot, upright = env.reset()

model_folder = os.getcwd()+'/HPC_Data/'
run = 0
N = 10

edges = np.load('Cartpole Edges Test Transfer Prewarped (Experiment), N 10 Run 0.npy', allow_pickle=True)

x_disc = np.linspace(0, 1, 50)
x_in = torch.linspace(0, 1, 50)
model = torch.load(model_folder+'/Actor Network, CartPole Continous 2L, N=10, Run=0.pt', weights_only=False, map_location='cpu')
edge_mask = model.edge_mask
data = []
Nunits = 0
for layer in edge_mask:
    Nunits += layer.sum()

experimental_KAN = Experimental_interpolation_model(edges, edge_mask)
trial_length = np.zeros((500))
for episode in range(100):
    thetas = np.zeros((500))
    xs = np.zeros_like(thetas)
    fs = np.zeros_like(thetas)
    # initialise environment and create state vector
    x, xdot, theta, thetadot, upright = env.reset()
    state = np.array([x, xdot, theta, thetadot]).flatten()
    # Generate inputs from state and scale
    inputs = np.zeros((1, 4))
    inputs[0, :] = state_normalisation(state)
    # Predict Q values
    _, action = experimental_KAN.forward(inputs)
    action = action[-1][0]
    # Step the agent
    xnew, xdotnew, thetanew, thetadotnew, upright, reward = env.step_environment(action, x, xdot, theta, thetadot, upright)
    new_state = np.array([xnew, xdotnew, thetanew, thetadotnew]).flatten()
    state = new_state
    for step in range(500):
        if upright == True:
            # Generate inputs from action and state
            inputs = np.zeros((1, 4))
            inputs[0, :] = state_normalisation(state)
            # Predict Q values
            _, action = experimental_KAN.forward(inputs)
            action = action[-1][0]
            xnew, xdotnew, thetanew, thetadotnew, upright, reward = env.step_environment(action, xnew, xdotnew, thetanew, thetadotnew, upright)
            new_state = np.array([xnew, xdotnew, thetanew, thetadotnew]).flatten()
            state = new_state
            thetas[step] = thetanew[0]
            xs[step] = xnew[0]
            fs[step] = action
        else:
            x, xdot, theta, thetadot, upright = env.reset()
            break
    xbase = np.arange(0, 500, 1)
    trial_length[episode] = step
    print(episode, step)
    ylim_theta = np.amax(np.abs(thetas))
    ylim_fs = np.amax(np.abs(5*fs))
    fig, ax = plt.subplots(dpi=800, figsize=(3,2))
    ax2 = plt.twinx(ax)
    ax2.plot(thetas, lw=2, color='red', label='Theta')
    ax2.set_ylim(ylim_theta*-1.1, ylim_theta*1.1)
    ax.set_ylim(ylim_fs*-1.1, ylim_fs*1.1)
    ax.plot(fs*5, lw=2, color='black', label='F')
    ax.set_xlabel('Trial Timestep')
    ax2.set_ylabel('Pole Angle (Radians)')
    ax.set_ylabel('Force Applied (Newtons)', color='red')
    ax.tick_params(axis='y', colors='red')
    plt.show()
    
    ylim_theta = np.amax(np.abs(thetas[:100]))
    ylim_fs = np.amax(np.abs(5*fs[:100]))
    fig, ax = plt.subplots(dpi=800, figsize=(3,2))
    ax2 = plt.twinx(ax)
    ax2.plot(thetas[:100], lw=2, color='red', label='Theta')
    ax2.set_ylim(ylim_theta*-1.1, ylim_theta*1.1)
    ax.set_ylim(ylim_fs*-1.1, ylim_fs*1.1)
    ax.plot(5*fs[:100], lw=2, color='black', label='F')
    ax.set_xlabel('Trial Timestep')
    ax2.set_ylabel('Pole Angle (Radians)')
    ax.set_ylabel('Force Applied (Newtons)', color='red')
    ax.tick_params(axis='y', colors='red')
    plt.show()
    
    ylim_theta = np.amax(np.abs(thetas[100:]))
    ylim_fs = np.amax(np.abs(5*fs[100:]))
    fig, ax = plt.subplots(dpi=800, figsize=(3,2))
    ax2 = plt.twinx(ax)
    ax2.plot(xbase[100:], thetas[100:], lw=2, color='red', label='Theta')
    ax2.set_ylim(ylim_theta*-1.1, ylim_theta*1.1)
    ax.set_ylim(ylim_fs*-1.1, ylim_fs*1.1)
    ax.plot(xbase[100:], fs[100:]*5, lw=2, color='black', label='F')
    ax.set_xlabel('Trial Timestep')
    ax2.set_ylabel('Pole Angle (Radians)')
    ax.set_ylabel('Force Applied (Newtons)', color='red')
    ax.tick_params(axis='y', colors='red')
    plt.show()     

