#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:35:33 2025

@author: ian
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device='cpu'
torch.set_default_device(device)

# Standard NN for estimating Q value of action
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
        loss = -1*q_values.mean()
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
# Normalise each of the state parameters to the same range    
def state_normalisation(state):
    norm_state = np.zeros_like(state)
    norm_state[0] = state[0]/2.4 
    norm_state[1] = state[1]/3 
    norm_state[2] = state[2]/0.418 
    norm_state[3] = state[3]/5 
    return norm_state
        
    
def gen_samples(Xin, Yin, batch_size):
    indices = torch.randint(len(Xin), (batch_size,), device=device)
    Xsample = Xin[indices]
    Ysample = Yin[indices]
    return Xsample, Ysample
    

from CartPole_env import CartpoleEnvironment

# Define environment and reset
env = CartpoleEnvironment()
x, xdot, theta, thetadot, upright = env.reset()



# Define number of nodes in hidden layers
N = 200
# Define rate of updating prediction network
update_tau = 0.001
# Define discount factor
gamma = 0.95
# Initialise critic network
critic_network = CriticNet([5, N, N, N, 1], 1e-3)
# Initialise prediction network, setting initial parameters as equal to those of the critic network
prediction_network = CriticNet([5, N, N, N, 1], 1e-3)
for l in range(critic_network.Nlayers):
    prediction_network.weights[l] = torch.clone(critic_network.weights[l])
    prediction_network.biases[l] = torch.clone(critic_network.biases[l])
# Initialise policy network
actor_network = ActorNet([4, N, N, N, 1], 1e-5, critic_network)
  
# Initialise replay buffer
replay_buffer = np.zeros((1, 10))

# Loop over episodes
for episode in range(1000):
    # First in, first out for replay buffer when exceeding a given length
    if len(replay_buffer)>100000:
        replay_buffer = replay_buffer[-10000:]
    # Store values of theta and x for plots
    thetas = np.zeros((500))
    xs = np.zeros_like(thetas)
    # Re-initialise environment at the start of each episode
    x, xdot, theta, thetadot, upright = env.reset()
    # Concatenate into state vector
    state = np.array([x, xdot, theta, thetadot]).flatten()
    # Create list for forces used
    action_list=[]
    # Generate inputs from state, normalising scales
    inputs = np.zeros((1, 4))
    inputs[0, :] = state_normalisation(state)
    action = actor_network.forward(torch.from_numpy(inputs).float())
    # Step the agent
    xnew, xdotnew, thetanew, thetadotnew, upright, reward = env.step_environment(action.squeeze().detach().numpy(), x, xdot, theta, thetadot, upright)
    new_state = np.array([xnew, xdotnew, thetanew, thetadotnew]).flatten()
    # Add details to replay buffer
    buffer_entry = np.concatenate(([action.squeeze().detach().numpy()], state_normalisation(state), state_normalisation(new_state), [reward]))
    replay_buffer[0] = buffer_entry
    # Update current state
    state = new_state
    for step in range(500):
        # If pole was upright from previous call
        if upright == True:
            # Create inputs from state and normalise
            inputs = np.zeros((1, 4))
            inputs[0, :] = state_normalisation(state)
            # Predict Q values
            action = actor_network.forward(torch.from_numpy(inputs).float())
            action_list.append(action.item())
            # Step the agent and find new state
            xnew, xdotnew, thetanew, thetadotnew, upright, reward = env.step_environment(action.detach().numpy(), xnew, xdotnew, thetanew, thetadotnew, upright)
            new_state = np.array([xnew, xdotnew, thetanew, thetadotnew]).flatten()
            # Add details to replay buffer
            buffer_entry = np.concatenate(([action.squeeze().detach().numpy()], state_normalisation(state), state_normalisation(new_state), reward.flatten()))[None, :]
            replay_buffer = np.concatenate((replay_buffer, buffer_entry), axis=0)
            # Target prediction for network training #
            # Randomly sample buffer for experience replay
            ks = np.random.randint(0, len(replay_buffer), np.min([50, len(replay_buffer)]))
            # Use update network to predict current Q values
            current_inputs = torch.from_numpy(replay_buffer[ks, :5])
            q_values = critic_network.forward(current_inputs.float())
            # Generate new inputs from model predictions
            new_states = torch.from_numpy(replay_buffer[ks, 5:9])
            new_action = actor_network.forward(new_states.float())
            new_q_in = torch.cat((new_action, new_states), dim=1)
            # Use prediction network for future q values
            new_q = prediction_network.forward(new_q_in.float())
            targets = (torch.from_numpy(replay_buffer[ks, -1]).unsqueeze(-1) + gamma * new_q).float()
            # Update critic network
            critic_loss = critic_network.train(current_inputs.float(), targets.float())
            # Soft update for prediction network
            with torch.no_grad():
                for l in range(prediction_network.Nlayers):
                    prediction_network.biases[l] = prediction_network.biases[l] * (1-update_tau) + update_tau*critic_network.biases[l]
                    prediction_network.weights[l] = prediction_network.weights[l] * (1-update_tau) + update_tau*critic_network.weights[l]
            # Update state
            state = new_state
            thetas[step] = thetanew[0]
            xs[step] = xnew[0]
            # step actor network
            actor_loss = actor_network.train(current_inputs[:, 1:].float())
        else:
            break
    print(episode, step)
    if episode%20==0:
        fig, ax1 = plt.subplots()
        plt.title('Epsiode: '+str(episode))
        ax2 = plt.twinx(ax1)
        ax1.set_ylabel('Pole angle, radians', color='tab:blue')
        ax1.set_ylim(-0.3, 0.3)
        ax2.set_ylabel('Force applied, Newtons', color='red')
        ax1.plot(thetas[:step])
        ax1.set_xlabel('Timestep')
        ax2.plot(np.array(action_list[:step])*5, color='red')
        ax2.set_ylim(-np.amax(np.abs(np.array(action_list[:step])*5)), np.amax(np.abs(np.array(action_list[:step])*5)))
        plt.show()
   