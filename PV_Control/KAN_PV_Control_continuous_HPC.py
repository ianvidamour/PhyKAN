#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:35:33 2025

@author: ian
"""

from pathlib import Path
import sys
from PhyKAN_Util_actorcritic import PhyKAN_actor, PhyKAN_critic
import numpy as np
import torch
import torch.nn as nn
import os

def _to_vector(values, dtype=float):
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _to_scalar(values):
    return float(_to_vector(values)[0])


def state_normalisation(states):
    norm_states = _to_vector(states, dtype=float).copy()
    if norm_states.size >= 2:
        norm_states[0] = norm_states[0] / 200.0
        norm_states[1] = norm_states[1] / 9.0
    return norm_states


def step_power(action, Vin):
    action_val = _to_scalar(action)
    Vnew = _to_scalar(Vin) + action_val
    dnew = action_val
    return float(Vnew), float(dnew)


def _curve_series(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr.reshape(-1)


def find_power_curve(power_data, current_data, Vin):
    MPPT = 1000.0
    vin = _to_scalar(Vin)
    V_index = int(vin * 5.0)
    power_series = _curve_series(power_data)
    current_series = _curve_series(current_data)
    num_points = min(power_series.size, current_series.size)
    V_index = int(np.clip(V_index, 0, num_points - 2))

    low_val_P = power_series[V_index]
    low_val_I = current_series[V_index]
    remainder = (vin * 5.0) % 1.0
    difference_P = power_series[V_index + 1] - low_val_P
    difference_I = current_series[V_index + 1] - low_val_I
    power = low_val_P + remainder * difference_P
    current = low_val_I + remainder * difference_I
    ratio = power / MPPT
    return float(power), float(current), float(ratio)


def return_reward(ratio, prev_ratio, Vin):
    r1 = _to_scalar(ratio)
    prev_ratio_val = _to_scalar(prev_ratio)
    vin = _to_scalar(Vin)
    if r1 > prev_ratio_val:
        r2 = r1**2
    else:
        r2 = 0.0
    if 0 < vin < 200:
        r3 = 0.0
    else:
        r3 = -1.0
    return float(r1 + r2 + r3)


def _last_output(activations):
    if isinstance(activations, (list, tuple)):
        return activations[-1]
    return activations


def step_agent(state, actor_model, power_data, current_data, ratio):
    state_vec = _to_vector(state, dtype=float)
    inputs = np.zeros((1, state_vec.size), dtype=float)
    inputs[0, :] = state_normalisation(state_vec)

    activations = actor_model.forward(torch.from_numpy(inputs))
    output = _last_output(activations)
    output = output.detach().cpu().numpy().reshape(-1)
    action = np.array([_to_scalar(output)], dtype=float)

    Vnew, dnew = step_power(action[0], state_vec[0])
    power, current, new_ratio = find_power_curve(power_data, current_data, Vnew)
    reward = return_reward(new_ratio, ratio, Vnew)

    new_state = state_vec.copy()
    new_state[0] = np.clip(Vnew, a_min=0, a_max=200)
    new_state[1] = current
    new_state[2] = dnew
    new_state[3] = new_ratio
    return new_state, action, float(reward), float(new_ratio), float(power)


def add_experience(replay_buffer, action, state, new_state, reward):
    action_vec = _to_vector(action, dtype=float)
    buffer_entry = np.concatenate(
        (action_vec, state_normalisation(state), state_normalisation(new_state), np.array([_to_scalar(reward)], dtype=float))
    )[None, :]
    replay_buffer = np.concatenate((replay_buffer, buffer_entry), axis=0)
    return replay_buffer


def training_step(model, prediction_network, replay_buffer, Nactions, Nstates, batch_size=50):
    ks = np.random.randint(0, len(replay_buffer), np.min([batch_size, len(replay_buffer)]))
    ks = np.concatenate(([-1], ks))
    current_inputs = torch.from_numpy(replay_buffer[ks, Nactions:Nactions + Nstates])
    activations = model.forward(current_inputs)
    q_values = _last_output(activations)
    actions = replay_buffer[ks, :Nactions]
    selected_actions = np.argmax(actions, axis=1)
    mask = np.zeros_like(actions)
    for ind in range(len(mask)):
        mask[ind, selected_actions[ind]] = 1
    skipped_actions = np.where(mask == 0)
    new_states = torch.from_numpy(replay_buffer[ks, Nactions + Nstates:Nactions + 2 * Nstates])
    new_q = _last_output(prediction_network.forward(new_states))
    targets = (
        torch.from_numpy(replay_buffer[ks, -1]).unsqueeze(-1) + gamma * torch.amax(new_q, dim=1).unsqueeze(-1)
    ).tile(1, Nactions).float()
    targets[skipped_actions[0], skipped_actions[1]] = torch.clone(q_values[skipped_actions[0], skipped_actions[1]])
    loss, rawloss, stabloss = model.train(current_inputs.float(), targets.float(), penalty=True, lamda=1e-4)
    with torch.no_grad():
        for l in range(prediction_network.Nlayers):
            updated = prediction_network.filter_params[l].detach() * (1 - update_tau) + update_tau * model.filter_params[l].detach()
            prediction_network.filter_params[l] = nn.Parameter(updated)
    return loss, activations, current_inputs


if __name__ == '__main__':
    device = 'cpu'
    try:
        torch.set_default_device(device)
    except AttributeError:
        pass

    script_dir = Path(__file__).resolve().parent
    low_vals = np.load(script_dir / 'LowFilterVals.npy')
    high_vals = np.load(script_dir / 'HighFilterVals.npy')

    state_data = np.load(script_dir / 'state_data.npy')
    power_data = np.load(script_dir / 'power_data.npy')
    current_data = np.load(script_dir / 'current_data.npy')


    Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    inp = int(sys.argv[1])
    N = Ns[inp%12]
    run = inp//12
    
    # Add catch to see if file already created + skip (fix crashed runs)
    if os.path.isfile('Actor network, PV Control, N='+str(N)+', run '+str(run)+'.pt'):
        print('Run completed succesfully')
    else:
        
    
        actor_shape = [4, N, 1]
        critic_shape = [5, N, 1]
    
        critic_network = PhyKAN_critic(critic_shape, 6, low_vals, high_vals, lr=1e-2)
        actor_network = PhyKAN_actor(actor_shape, 6, low_vals, high_vals, load_model=critic_network, lr=1e-3)
        prediction_network = PhyKAN_critic(critic_shape, 6, low_vals, high_vals)
        for l in range(critic_network.Nlayers):
            prediction_network.filter_params[l] = nn.Parameter(torch.clone(critic_network.filter_params[l].detach()))
    
        gamma = 0.75
        tau = 1
        epsilon = 0.33
        Nactions = 1
        Nstate = 4
        replay_buffer = np.zeros((1, Nactions + Nstate * 2 + 1))
        trial_lengths = np.zeros((10, 1000))
    
        N_episodes = 2000
        T = 1000
        Rs = np.zeros([N_episodes, T])
        update_tau = 0.001
    
        for episode in range(N_episodes):
            power_plot = np.zeros((50))
            V_plot = np.zeros((50))
            k = int(np.random.randint(0, 2000))
            start_V = float(np.random.ranf() * 200.0)
            power, current, ratio = find_power_curve(power_data[k], current_data[k], start_V)
            state = np.array([start_V, current, 0.0, ratio], dtype=float)
            action_list = []
            new_state, action, reward, new_ratio, power = step_agent(state, actor_network, power_data[k], current_data[k], ratio)
            buffer_entry = np.concatenate(
                (action, state_normalisation(state), state_normalisation(new_state), np.array([reward], dtype=float))
            )
            replay_buffer[0] = buffer_entry
            state = new_state
            ratio = new_ratio
            Rs[episode, 0] = reward
            for step in range(50):
                power_plot[step] = power
                V_plot[step] = new_state[0]
                inputs = np.zeros((1, 4), dtype=float)
                inputs[0, :] = state_normalisation(state)
                new_state, action, reward, new_ratio, power = step_agent(state, actor_network, power_data[k], current_data[k], ratio)
                buffer_entry = np.concatenate(
                    (action, state_normalisation(state), state_normalisation(new_state), np.array([reward], dtype=float))
                )[None, :]
                replay_buffer = np.concatenate((replay_buffer, buffer_entry), axis=0)
                ks = np.random.randint(0, len(replay_buffer), np.min([50, len(replay_buffer)]))
                ks = np.concatenate(([-1], ks))
                current_inputs = torch.from_numpy(replay_buffer[ks, :Nactions + Nstate])
                q_values = critic_network.forward(current_inputs)[-1]
                actions = replay_buffer[ks, :Nactions]
                new_actions = actor_network(torch.from_numpy(replay_buffer[ks, Nactions + Nstate:Nactions + Nstate * 2]))[-1]
                new_inputs = torch.cat((new_actions, torch.from_numpy(replay_buffer[ks, Nactions + Nstate:Nactions + Nstate * 2])), dim=1)
                new_q = prediction_network.forward(new_inputs)[-1]
                targets = (torch.from_numpy(replay_buffer[ks, -1]).unsqueeze(-1) + gamma * new_q).float()
                critic_network.train(current_inputs, targets)
                with torch.no_grad():
                    for l in range(prediction_network.Nlayers):
                        updated = prediction_network.filter_params[l].detach() * (1 - update_tau) + update_tau * critic_network.filter_params[l].detach()
                        prediction_network.filter_params[l] = nn.Parameter(updated)
                actor_network.train_actor_critic(current_inputs[:, 1:])
                state = new_state
                ratio = new_ratio
                Rs[episode, step] = reward
            print(episode, Rs[episode].sum())
    
        torch.save(actor_network, 'Actor network, PV Control, N='+str(N)+', run '+str(run)+'.pt')
        torch.save(critic_network, 'Critic network, PV Control, N='+str(N)+', run '+str(run)+'.pt')
        torch.save(prediction_network, 'Prediction network, PV Control, N='+str(N)+', run '+str(run)+'.pt')
        
