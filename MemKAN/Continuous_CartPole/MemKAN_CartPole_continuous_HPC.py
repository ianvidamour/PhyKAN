#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:35:33 2025

@author: ian
"""

import numpy as np
import torch
from MemKAN_Util_actorcritic import MemKAN_actor, MemKAN_critic

def state_normalisation(state):
    norm_state = np.zeros_like(state)
    norm_state[0] = state[0]/2.4 
    norm_state[1] = state[1]/3 
    norm_state[2] = state[2]/0.418 
    norm_state[3] = state[3]/5 
    return norm_state


device='cpu'
torch.set_default_device(device)


from CartPole_env import CartpoleEnvironment

env = CartpoleEnvironment()
x, xdot, theta, thetadot, upright = env.reset()


import sys
inp = int(sys.argv[1])

Ns = [3, 4]
N = Ns[inp%2]
run = inp//2

# Define shapes for each network
actor_shape = [4, N, N, N, 1]
critic_shape = [5, N, N, N, 1]

memdiode_model = torch.load('MemdiodeMLP, 2HL 125 nodes.pt', weights_only=False, map_location=device)

# Initialise networks
critic_model = MemKAN_critic(critic_shape, 3, memdiode_model, lr=1e-2)
actor_model = MemKAN_actor(actor_shape, 3, memdiode_model, load_model=critic_model, lr=1e-3)
prediction_network = MemKAN_critic(critic_shape, 3, memdiode_model, lr=1e-2)
for l in range(critic_model.Nlayers):
    prediction_network.edge_params[l] = torch.clone(critic_model.edge_params[l])
    prediction_network.gains[l] = torch.clone(critic_model.gains[l])
    prediction_network.biases[l] = torch.clone(critic_model.biases[l])

# Discount factor
gamma = 0.95
# Initialise buffer
replay_buffer = np.zeros((1, 10))
# Define rate of updating prediction network
update_tau = 0.001
#%%
trial_length = np.zeros((1000))


for episode in range(1000):
	# First in, first out for replay buffer when exceeding a given length
	if len(replay_buffer)>100000:
		replay_buffer = replay_buffer[-10000:]
	# store plot variables
	thetas = np.zeros((500))
	xs = np.zeros_like(thetas)
	action_list=[]
	# initialise environment and create state vector
	x, xdot, theta, thetadot, upright = env.reset()
	state = np.array([x, xdot, theta, thetadot]).flatten()
	# Generate inputs from state and scale
	inputs = np.zeros((1, 4))
	inputs[0, :] = state_normalisation(state)
	# Predict action from policy network
	action = actor_model.forward(torch.from_numpy(inputs))[-1]
	# Step the agent
	xnew, xdotnew, thetanew, thetadotnew, upright, reward = env.step_environment(action.squeeze().detach().numpy(), x, xdot, theta, thetadot, upright)
	new_state = np.array([xnew, xdotnew, thetanew, thetadotnew]).flatten()
	# Add details to replay buffer
	buffer_entry = np.concatenate(([action.squeeze().detach().numpy()], state_normalisation(state), state_normalisation(new_state), [reward]))
	replay_buffer[0] = buffer_entry
	# update state
	state = new_state
	for step in range(500):
		# While pole remains upright (in previous step)
		if upright == True:
			# Generate inputs from action and state
			inputs = np.zeros((1, 4))
			inputs[0, :] = state_normalisation(state)#+np.random.ranf(state.shape)*0.05
			# Predict Q values
			action = actor_model.forward(torch.from_numpy(inputs))[-1]
			action_list.append(action.item())
			# Step the agent
			xnew, xdotnew, thetanew, thetadotnew, upright, reward = env.step_environment(action.detach().numpy(), xnew, xdotnew, thetanew, thetadotnew, upright)
			new_state = np.array([xnew, xdotnew, thetanew, thetadotnew]).flatten()
			# Add details to replay buffer
			buffer_entry = np.concatenate(([action.squeeze().detach().numpy()], state_normalisation(state), state_normalisation(new_state), reward.flatten()))[None, :]
			replay_buffer = np.concatenate((replay_buffer, buffer_entry), axis=0)
			# Target prediction for network training
			ks = np.random.randint(0, len(replay_buffer), np.min([50, len(replay_buffer)]))
			# Use update network to predict current Q values
			current_inputs = torch.from_numpy(replay_buffer[ks, :5])
			q_values = critic_model.forward(current_inputs)[-1]
			# Generate new inputs from model predictions
			new_states = torch.from_numpy(replay_buffer[ks, 5:9])
			new_action = actor_model.forward(new_states)[-1]
			new_q_in = torch.cat((new_action, new_states), dim=1)
			# Use prediction network for future q values
			new_q = prediction_network.forward(new_q_in)[-1]
			targets = (torch.from_numpy(replay_buffer[ks, -1]).unsqueeze(-1) + gamma * new_q).float()
			# Step critic model
			loss = critic_model.train(current_inputs.float(), targets.float(), penalty=True, lamda=1e-4)
			# Soft update for prediction network
			with torch.no_grad():
				for l in range(prediction_network.Nlayers):
					prediction_network.edge_params[l] = prediction_network.edge_params[l] * (1-update_tau) + update_tau*critic_model.edge_params[l]
					prediction_network.gains[l] = prediction_network.gains[l] * (1-update_tau) + update_tau*critic_model.gains[l]
					prediction_network.biases[l] = prediction_network.biases[l] * (1-update_tau) + update_tau*critic_model.biases[l]
			# step actor model
			actor_model.train_actor_critic(current_inputs[:, 1:])
			# Update state
			state = new_state
			thetas[step] = thetanew[0]
			xs[step] = xnew[0]
			# actor_model.load_model=critic_model
		else:
			break
	print(episode, step)
	trial_length[episode] = step
	if episode%20==0:
		np.save('Trial Lengths, MemKAN CartPole Continous, N='+str(N)+', Run='+str(run)+'.npy', trial_length)
		torch.save(actor_model, 'Actor Network, MemKAN CartPole Continous, N='+str(N)+', Run='+str(run)+'.pt')
		torch.save(critic_model, 'Critic Network, MemKAN CartPole Continous, N='+str(N)+', Run='+str(run)+'.pt')


	
	
	
	
