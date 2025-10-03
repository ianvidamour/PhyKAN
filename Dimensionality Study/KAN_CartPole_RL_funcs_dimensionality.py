#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:35:33 2025

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn
from PhyKAN_Util import PhyKAN
import gym

def softmax(x, tau):
    e_x = np.exp(x/tau)
    return (e_x/e_x.sum()).flatten()

# Shape reward for improved learning
def reward_shaping(state, reward):
    x, x_dot, theta, theta_dot = state
    r = np.copy(reward)
    p1 = np.abs(x) # Penalise cart position, favouring centre
    p2 = 0.1*np.abs(x_dot) # Penalise cart velocity, favouring lower velocities
    p3 = np.abs(theta) # Penalise pole angle, favouring close to zero
    p4 = 0.5*np.abs(theta_dot) # Penalise angular velocity, favouring low velocities
    shaped_reward = r - p1 - p2 - p3 - p4
    return shaped_reward

# Normalise input of state to range -1, 1
def state_normalisation(state):
    norm_state = np.zeros_like(state)
    norm_state[0] = state[0]/2.4 
    norm_state[1] = state[1]/3 
    norm_state[2] = state[2]/0.418 
    norm_state[3] = state[3]/5 
    return norm_state

# Step through the model
def step_agent_softmax(state, model, softmax_tau=1):
    inputs = np.zeros((1, len(state)))
    # Normalise state
    inputs[0, :] = state_normalisation(state)#+np.random.ranf(state.shape)*0.05
    # Predict Q values
    activations = model.forward(torch.from_numpy(inputs))
    output = activations[-1]
    # Select action based on epsilon greedy polic
    action = np.zeros(len(output[0]))
    # Softmax policy
    probs = softmax(output.detach().numpy(), softmax_tau)
    choice = np.random.choice(np.arange(0, len(output[0]), 1, dtype='int'), p=probs)
    action[choice] = 1
    # Step the agent
    new_state, reward, terminated, truncated, info = env.step(np.argmax(action))
    return new_state, action, reward, terminated


# Step through the model
def step_agent_greedy(state, model):
    inputs = np.zeros((1, len(state)))
    # Normalise state
    inputs[0, :] = state_normalisation(state)#+np.random.ranf(state.shape)*0.05
    # Predict Q values
    activations = model.forward(torch.from_numpy(inputs))
    output = activations[-1]
    # Select action based on epsilon greedy polic
    action = np.zeros(len(output[0]))
    # Greedy policy
    action[np.argmax(output.detach().numpy())] = 1
    # Step the agent
    new_state, reward, terminated, truncated, info = env.step(np.argmax(action))
    return new_state, action, reward, terminated

# Add experiences to buffer
def add_experience(replay_buffer, action, state, new_state, reward):
    # Add details to replay buffer
    buffer_entry = np.concatenate((action, state_normalisation(state), state_normalisation(new_state), [reward]))[None, :]
    replay_buffer = np.concatenate((replay_buffer, buffer_entry), axis=0)
    return replay_buffer

# Step through model, update model net via gradient descent, soft update to prediction net
def training_step(model, prediction_network, replay_buffer, Nactions, Nstates, batch_size=500):
    # Target prediction for network training
    # Sample from replay buffer
    ks = np.random.randint(0, len(replay_buffer), np.min([500, len(replay_buffer)]))
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
    

from dimensionality import *
def whiten_data(Z,covariance_bias=True):
    
    # Check the type of Z and convert it to a numpy array if necessary
    if isinstance(Z, torch.Tensor):
        Z_np = Z.numpy()
    elif isinstance(Z, np.matrix):
        Z_np = np.array(Z)
    else:  # assuming Z is a numpy array
        Z_np = Z
          
    # Calculate the mean and subtract it
    mean = np.mean(Z_np, axis=0)
    Z_centered = Z_np - mean

    # Compute the covariance matrix
    cov_matrix = np.cov(Z_centered, rowvar=0, bias=covariance_bias) # should be the same   

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Clip the eigenvalues to be non-negative
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

    # Compute the diagonal matrix of inverse square roots of eigenvalues
    epsilon = 1e-12
    whiten_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + epsilon))

    # Whitening: decorrelate and scale features
    Z_whitened = Z_centered @ whiten_matrix
    
    Z_whitened = torch.from_numpy(Z_whitened)

    return Z_whitened, eigenvectors, eigenvalues

device='cpu'
torch.set_default_device(device)




# Initialise double Q networks

# Define shape for models
Nactions = 2
Nstates = 4
shape = [Nstates, 10, Nactions]
# Load look-up table for discretisation
low_vals = np.load('LowFilterVals.npy')
high_vals = np.load('HighFilterVals.npy')
# Model for predicting current Q values
model = PhyKAN(shape, 6, low_vals, high_vals)
# Model for predicting future q values
prediction_network = PhyKAN(shape, 6, low_vals, high_vals)
# Copy parameters from first model
for l in range(model.Nlayers):
    prediction_network.filter_params[l] = torch.clone(model.filter_params[l])
# Soft update for future q value network
update_tau = 0.001
# Number of episodes
Nepisodes = 1000
# Steps per episode
Nsteps = 500


# Discount factor
gamma = 0.95
# Softmax temperature
tau = 1
# Initialise buffer for experience replay
replay_buffer = np.zeros((1, 1+Nactions+Nstates*2))
# Initialise environment
env = gym.make('CartPole-v1')
state, info = env.reset()
terminated = False

# Outputs to save learning curves
trial_lengths = np.zeros((Nepisodes))
rewards_out = np.zeros((Nepisodes, Nsteps))
theta_prime_weights = np.zeros((Nepisodes, 6))
theta_zero_weights = np.zeros((Nepisodes, 6))
intrinsic_dim_inputs = np.zeros((Nepisodes))
intrinsic_dim_hiddens = np.zeros((Nepisodes))
# Script for training model
for episode in range(Nepisodes):
    for step in range(Nsteps):
        # If pole remains upright
        if terminated==False:
            # Step agent with softmax policy
            new_state, action, reward, terminated = step_agent_softmax(state, model, softmax_tau=tau)
            shaped_reward = reward_shaping(new_state, reward)
            # Add experience to replay buffer
            replay_buffer = add_experience(replay_buffer, action, state, new_state, shaped_reward)
            # Get rid of placeholder information in very first run
            if episode and step == 0:
                replay_buffer = replay_buffer[1][None, :]
            # Model training
            loss, activations, inputs = training_step(model, prediction_network, replay_buffer, Nactions, Nstates)
            # Track reward
            rewards_out[episode, step] = reward
            # Update state to new state
            state = new_state
        else:
            # Reset environment as pole has fell
            state, info = env.reset()
            terminated = False
            break
    # Greedy Run
    for step in range(Nsteps):
        if terminated == False:
            # Step agent with greedy policy
            new_state, action, reward, terminated = step_agent_greedy(state, model)
            state = new_state
        else:
            # Record length of greedy run
            trial_lengths[episode] = step
            # Reset environment
            state, info = env.reset()
            terminated = False
            break
    # Measure Intrinsic Dimensionality
    inputs = replay_buffer[:10000, Nactions:Nactions+Nstates]
    activations = model.forward(torch.from_numpy(inputs))
    # Whiten data
    whitened_inputs, eig, vec = whiten_data(inputs)
    whitened_hiddens, eig, vec = whiten_data(activations[0].detach())
    # Calculate intrinsic dimensionality
    input_dimensionality = intrinsic_dimension(whitened_inputs.T)
    hidden_dimensionality = intrinsic_dimension(whitened_hiddens.T)
    intrinsic_dim_hiddens[episode] = hidden_dimensionality
    intrinsic_dim_inputs[episode] = input_dimensionality
    print('Episode: '+str(episode)+', Trial Length (greedy) :'+str(step))
    print('Input Intrinsic Dimensionality: '+str(input_dimensionality))
    print('Hidden Intrinsic Dimensionality: '+str(hidden_dimensionality))
    # Track example weights as learning progresses
    theta_prime_weights[episode] = torch.clone(model.filter_params[0][0, 0, :, 2]).detach().numpy()
    theta_zero_weights[episode] = torch.clone(prediction_network.filter_params[0][0, 0, :, 2]).detach().numpy()
        
#%%
import matplotlib.pyplot as plt

plt.figure()
for i in range(6):
    if i == 0:
        plt.plot(theta_prime_weights[:, i], color='red', label='Theta prime')
        plt.plot(theta_zero_weights[:, i], color='black', label='Theta zero')
    else:
        plt.plot(theta_prime_weights[:, i], color='red')
        plt.plot(theta_zero_weights[:, i], color='black')
plt.legend()
plt.show()
