#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:01:33 2026

@author: ian
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device='cpu'
torch.set_default_device(device)

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
        r3 = -10
    return r1+r2+r3

def step_agent(state, actor_model, power_data, current_data, ratio):
    inputs = np.zeros((1, len(state)))
    # Normalise state
    inputs[0, :] = state_normalisation(state)
    # Predict Q values
    activations = actor_model.forward((inputs))
    output = activations[-1]
    action = output
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

# Add experiences to buffer
def add_experience(replay_buffer, action, state, new_state, reward):
    # Add details to replay buffer
    buffer_entry = np.concatenate((action[0], state_normalisation(state), state_normalisation(new_state), [reward]))[None, :]
    replay_buffer = np.concatenate((replay_buffer, buffer_entry), axis=0)
    return replay_buffer


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
        if steps>=discsteps:
            return edge_response[discsteps]
        else:
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
        return outputs

    def sig(self, x):
        return 1/(1+np.exp(-x/self.sigmoid_s))

    
def linear_interpolate(edge_response, input_data, discsteps=7022):
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
device='cpu'
torch.set_default_device(device)

low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')

state_data = np.load('state_data.npy')
power_data = np.load('power_data.npy')
current_data = np.load('current_data.npy')

edge_folder = './PV_Transfer/'
model_folder = './HPC Data/'
run = 0
Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
performance_out = np.zeros((12, 10))

for _, N in enumerate(Ns):
    for run in range(10):
        np.random.seed(run)
        edges = np.load(edge_folder+'PV Control Scaled Filters, N '+str(N)+' Run '+str(run)+'.npy', allow_pickle=True)
        
        x_disc = np.linspace(0, 1, 50)
        x_in = torch.linspace(0, 1, 50)
        model = torch.load(model_folder+f'Actor network, PV Control, N={N}, run {run}.pt', weights_only=False, map_location='cpu')
        edge_mask = model.edge_mask
        data = []
        Nunits = 0
        for layer in edge_mask:
            Nunits += layer.sum()
        
        experimental_KAN = Experimental_interpolation_model(edges, edge_mask)
        output_scores = np.zeros((100))
        for episode in range(100):
            power_plot = np.zeros((50))
            V_plot = np.zeros((50))
            k = np.random.randint(0, 2000, 1)
            start_V = np.random.ranf()*200
            power, current, ratio = find_power_curve(power_data[k], current_data[k], start_V)
            state = np.array([start_V, current, 0, ratio])
            action_list=[]
            # Step the agent
            new_state, action, reward, new_ratio, power = step_agent(state, experimental_KAN, power_data[k], current_data[k], ratio)
            state = new_state
            for step in range(50):
                power_plot[step] = power
                V_plot[step] = new_state[0]
                # Create inputs from state and action
                inputs = np.zeros((1, 4))
                inputs[0, :] = state_normalisation(state)
                # Step the agent
                new_state, action, reward, new_ratio, power = step_agent(state, experimental_KAN, power_data[k], current_data[k], ratio)
                # Update state
                state = new_state
            power_plot/= np.amax(power_data[k])
            output_scores[episode] = power_plot.mean()
        performance_out[_, run] = output_scores.mean()
        print(output_scores.mean())
np.save('PV_Experimental_KAN_scores.npy', performance_out)
plt.figure()
plt.title('Experiment')
plt.plot(Ns, performance_out.mean(axis=1))
plt.fill_between(Ns, performance_out.mean(axis=1)-performance_out.std(axis=1), performance_out.mean(axis=1)+performance_out.std(axis=1), alpha=0.2)
plt.show()