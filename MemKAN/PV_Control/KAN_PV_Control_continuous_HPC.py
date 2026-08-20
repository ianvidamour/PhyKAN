#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:35:33 2025

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn
from MemKAN_Util_actorcritic import MemKAN_actor, MemKAN_critic


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
    if Vin.size > 0:
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
    # Step the agent
    Vnew, dnew = step_power(action, state[0])
    power, current, new_ratio = find_power_curve(power_data, current_data, Vnew)
    reward = return_reward(new_ratio, ratio, Vnew)
    new_state = np.copy(state)
    new_state[0] = np.clip(Vnew[0,0], a_min=0, a_max=200)
    new_state[1] = current
    new_state[2] = dnew[0,0]
    new_state[3] = new_ratio
    return new_state, action, reward, new_ratio, power

# Add experiences to buffer
def add_experience(replay_buffer, action, state, new_state, reward):
    # Add details to replay buffer
    buffer_entry = np.concatenate((action[0], state_normalisation(state), state_normalisation(new_state), [reward]))[None, :]
    replay_buffer = np.concatenate((replay_buffer, buffer_entry), axis=0)
    return replay_buffer

# Step through model, update model net via gradient descent, soft update to prediction net
def training_step(model, prediction_network, replay_buffer, Nactions, Nstates, batch_size=50):
    # Target prediction for network training
    # Sample from replay buffer
    ks = np.random.randint(0, len(replay_buffer), np.min([batch_size, len(replay_buffer)]))
    # Include most recent experience as part of batch
    ks = np.concatenate(([-1], ks))
    # Use update network to predict current Q values
    current_inputs = torch.from_numpy(replay_buffer[ks, Nactions:Nactions+Nstates])
    activations = model.forward(current_inputs)
    q_values = activations[-1]
    # Create table of actions
    actions = replay_buffer[ks, :Nactions]
    # Greedy action selection for Q learning
    selected_actions = np.argmax(actions, axis=1)
    # Create action mask
    mask = np.zeros_like(actions)
    for ind in range(len(mask)):
        mask[ind, selected_actions[ind]] = 1
    skipped_actions = np.where(mask==0)
    # Generate new inputs from model predictions
    new_states = torch.from_numpy(replay_buffer[ks, Nactions+Nstates:Nactions+2*Nstates])
    # Use prediction network for future q values
    new_q = prediction_network.forward(new_states)[-1]
    # Generate targets from reward + discounted future Q
    targets = (torch.from_numpy(replay_buffer[ks, -1]).unsqueeze(-1) + gamma * torch.amax(new_q, dim=1).unsqueeze(-1)).tile(1, Nactions).float()
    # Mask unselected actions by setting the delta to zero
    targets[skipped_actions[0], skipped_actions[1]] = torch.clone(q_values[skipped_actions[0], skipped_actions[1]])
    # Step model
    loss, rawloss, stabloss = model.train(current_inputs.float(), targets.float(), penalty=True, lamda=1e-4)
    # Soft update for prediction network
    with torch.no_grad():
        for l in range(prediction_network.Nlayers):
            prediction_network.filter_params[l] = prediction_network.filter_params[l] * (1-update_tau) + update_tau*model.filter_params[l]
    return loss, activations, current_inputs



device='cpu'
torch.set_default_device(device)



state_data = np.load('state_data.npy')
power_data = np.load('power_data.npy')
current_data = np.load('current_data.npy')


import sys

Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]

#inp= int(sys.argv[1])

  
# N = Ns[inp%12]
# run =  inp//12
N = 25
for run in range(10):
    actor_shape = [4, N, 1]
    critic_shape = [5, N, 1]
    memdiode_model = torch.load('MemdiodeMLP, 2HL 125 nodes.pt', weights_only=False, map_location=device)
    # Initialise networks
    critic_model = MemKAN_critic(critic_shape, 3, memdiode_model, lr=1e-2)
    actor_model = MemKAN_actor(actor_shape, 3, memdiode_model, load_model=critic_model, lr=1e-4)
    prediction_network = MemKAN_critic(critic_shape, 3, memdiode_model, lr=1e-2)
    for l in range(critic_model.Nlayers):
        prediction_network.edge_params[l] = torch.clone(critic_model.edge_params[l])
        prediction_network.gains[l] = torch.clone(critic_model.gains[l])
        prediction_network.biases[l] = torch.clone(critic_model.biases[l])
    
    gamma = 0.75
    
    tau = 1
    epsilon = 0.33
    Nactions = 1
    Nstate = 4
    replay_buffer = np.zeros((1, Nactions+Nstate*2+1))
    trial_lengths = np.zeros((10, 1000))
    
    N_episodes = 10000
    T = 1000
    Rs=np.zeros([N_episodes,T])
    update_tau = 0.001
    
     
    for episode in range(2000):
        power_plot = np.zeros((50))
        V_plot = np.zeros((50))
        k = np.random.randint(0, 2000, 1)
        start_V = np.random.ranf()*200
        power, current, ratio = find_power_curve(power_data[k], current_data[k], start_V)
        state = np.array([start_V, current, 0, ratio])
        action_list=[]
        # Step the agent
        new_state, action, reward, new_ratio, power = step_agent(state, actor_model, power_data[k], current_data[k], ratio)
        # Add details to replay buffer
        buffer_entry = np.concatenate((action[0], state_normalisation(state), state_normalisation(new_state), reward[None]))
        replay_buffer[0] = buffer_entry
        state = new_state
        Rs[episode, 0] = reward
        for step in range(50):
       	        power_plot[step] = power
       	        V_plot[step] = new_state[0]
       	        # Create inputs from state and action
       	        inputs = np.zeros((1, 4))
       	        inputs[0, :] = state_normalisation(state)
       	        # Step the agent
       	        new_state, action, reward, new_ratio, power = step_agent(state, actor_model, power_data[k], current_data[k], ratio)
       	        # Add to buffer
       	        buffer_entry = np.concatenate((action[0], state_normalisation(state), state_normalisation(new_state), reward[None]))[None, :]
       	        replay_buffer = np.concatenate((replay_buffer, buffer_entry), axis=0)
       	        # randomly sample from buffer
       	        ks = np.random.randint(0, len(replay_buffer), np.min([50, len(replay_buffer)]))
       	        ks = np.concatenate(([-1], ks))
       	        # Use update network to predict current Q values
       	        current_inputs = torch.from_numpy(replay_buffer[ks, :Nactions+Nstate])
       	        q_values = critic_model.forward(current_inputs)[-1]
       	        # Create table of actions
       	        actions = replay_buffer[ks, :Nactions]
       	        # Generate new inputs from model predictions
       	        new_actions = actor_model(torch.from_numpy(replay_buffer[ks, Nactions+Nstate:Nactions+Nstate*2]))[-1]
       	        new_inputs = torch.cat((new_actions, torch.from_numpy(replay_buffer[ks, Nactions+Nstate:Nactions+Nstate*2])), dim=1)
       	        # Use prediction network for future q values
       	        new_q = prediction_network.forward(new_inputs)[-1]
       	        targets = (torch.from_numpy(replay_buffer[ks, -1]).unsqueeze(-1) + gamma * new_q).float()
       	        critic_model.train(current_inputs, targets)
       	        with torch.no_grad():
       	            for l in range(prediction_network.Nlayers):
       	                prediction_network.edge_params[l] = prediction_network.edge_params[l] * (1-update_tau) + update_tau*critic_model.edge_params[l]
       	                prediction_network.gains[l] = prediction_network.gains[l] * (1-update_tau) + update_tau*critic_model.gains[l]
       	                prediction_network.biases[l] = prediction_network.biases[l] * (1-update_tau) + update_tau*critic_model.biases[l]
       	        actor_model.train_actor_critic(current_inputs[:, 1:])
       	        # Update state
       	        state = new_state
       	        Rs[episode, step] = reward            
        print(episode, Rs[episode].sum())
        torch.save(actor_model, 'Actor network 1e4, PV Control, N='+str(N)+', run '+str(run)+'.pt')
        torch.save(critic_model, 'Critic network 1e4, PV Control, N='+str(N)+', run '+str(run)+'.pt')
        torch.save(prediction_network, 'Prediction network 1e4, PV Control, N='+str(N)+', run '+str(run)+'.pt')
       	    
       	        
       	        
       	        
       	        
