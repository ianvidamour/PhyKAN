#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:24:41 2025

@author: ian
"""

import numpy as np
import torch

def return_matrix(thetak, ak, dk, alphak, batch_size):
    cos = lambda x: torch.cos(x)
    sin = lambda x: torch.sin(x)
    T = torch.zeros((batch_size, 4, 4))
    # define denavit hartenberg transformations
    T[:, 0, 0] = cos(thetak)
    T[:, 0, 1] = -cos(alphak)*sin(thetak)
    T[:, 0, 2] = sin(alphak)*sin(thetak)
    T[:, 0, 3] = ak*cos(thetak)
    T[:, 1, 0] = sin(thetak)
    T[:, 1, 1] = cos(alphak)*cos(thetak)
    T[:, 1, 2] = -sin(alphak)*cos(thetak)
    T[:, 1, 3] = ak*sin(thetak)
    T[:, 2, 1] = sin(alphak)
    T[:, 2, 2] = cos(alphak)
    T[:, 2, 3] = dk
    T[:, 3, 3] = 1
    return T

def fw_kin(angles):
    batch_size = angles.shape[0]
    # define lengths
    l1 = 0.290
    l2 = 0.270
    l3 = 0.070
    l4 = 0.134
    l5 = 0.168
    l6 = 0.072
    # take each angle
    q1 = angles[:, 0]
    q2 = angles[:, 1]
    q3 = angles[:, 2]
    q4 = angles[:, 3]
    q5 = angles[:, 4]
    q6 = angles[:, 5]
    # scale to ranges
    q1 = 5.76*(q1-0.5)
    q2 = 3.84*(q2-0.5)
    q3 = 1.22+(q3*0.7)
    q4 = 5.58*(q4-0.5)
    q5 = 4.18*(q5-0.5)
    q6 = 6.28*q6
    # generate denavit hartenberg transformations
    T10 = return_matrix(q1, torch.zeros(batch_size), torch.ones(batch_size)*l1, -torch.pi/2 * torch.ones(batch_size), batch_size)
    T21 = return_matrix(q2, torch.ones(batch_size)*l2, torch.zeros(batch_size), torch.zeros(batch_size), batch_size)
    T32 = return_matrix(q3, torch.ones(batch_size)*l3, torch.ones(batch_size)*l4, -torch.pi/2*torch.ones(batch_size), batch_size)
    T43 = return_matrix(q4, torch.zeros(batch_size), torch.ones(batch_size)*l5, torch.ones(batch_size)*torch.pi/2, batch_size)
    T54 = return_matrix(q5, torch.ones(batch_size)*l6, torch.zeros(batch_size), torch.ones(batch_size)*-torch.pi/2, batch_size)
    T65 = return_matrix(q6, torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size), batch_size)
    # Work through links
    T20 = torch.bmm(T10, T21)
    T30 = torch.bmm(T20, T32)
    T40 = torch.bmm(T30, T43)
    T50 = torch.bmm(T40, T54)
    T60 = torch.bmm(T50, T65)
    locs = T60[:, :-1, -1]
    return T60, locs

def noised_fw_kin(angles, noise_k):
    batch_size = angles.shape[0]
    # define lengths
    l1 = 0.290
    l2 = 0.270
    l3 = 0.070
    l4 = 0.134
    l5 = 0.168
    l6 = 0.072
    # take each angle
    q1 = angles[:, 0]
    q2 = angles[:, 1]
    q3 = angles[:, 2]
    q4 = angles[:, 3]
    q5 = angles[:, 4]
    q6 = angles[:, 5]
    # scale to ranges
    q1 = 5.76*(q1-0.5) 
    q2 = 3.84*(q2-0.5)
    q3 = 1.22+(q3*0.7)
    q4 = 5.58*(q4-0.5)
    q5 = 4.18*(q5-0.5)
    q6 = 6.28*q6
    # Generate noise
    q1_noise = 2*(torch.rand(q1.shape)-0.5) * noise_k
    q2_noise = 2*(torch.rand(q2.shape)-0.5) * noise_k
    q3_noise = 2*(torch.rand(q3.shape)-0.5) * noise_k
    q4_noise = 2*(torch.rand(q4.shape)-0.5) * noise_k
    q5_noise = 2*(torch.rand(q5.shape)-0.5) * noise_k
    q6_noise = 2*(torch.rand(q6.shape)-0.5) * noise_k
    # Add noise
    q1 = q1 + q1_noise
    q2 = q2 + q2_noise
    q3 = q3 + q3_noise
    q4 = q4 + q4_noise
    q5 = q5 + q5_noise
    q6 = q6 + q6_noise
    # generate denavit hartenberg transformations
    T10 = return_matrix(q1, torch.zeros(batch_size), torch.ones(batch_size)*l1, -torch.pi/2 * torch.ones(batch_size), batch_size)
    T21 = return_matrix(q2, torch.ones(batch_size)*l2, torch.zeros(batch_size), torch.zeros(batch_size), batch_size)
    T32 = return_matrix(q3, torch.ones(batch_size)*l3, torch.ones(batch_size)*l4, -torch.pi/2*torch.ones(batch_size), batch_size)
    T43 = return_matrix(q4, torch.zeros(batch_size), torch.ones(batch_size)*l5, torch.ones(batch_size)*torch.pi/2, batch_size)
    T54 = return_matrix(q5, torch.ones(batch_size)*l6, torch.zeros(batch_size), torch.ones(batch_size)*-torch.pi/2, batch_size)
    T65 = return_matrix(q6, torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size), batch_size)
    # Work through links
    T20 = torch.bmm(T10, T21)
    T30 = torch.bmm(T20, T32)
    T40 = torch.bmm(T30, T43)
    T50 = torch.bmm(T40, T54)
    T60 = torch.bmm(T50, T65)
    locs = T60[:, :-1, -1]
    # Stack noised angles for output
    noised_angles = torch.stack([q1, q2, q3, q4, q5, q6])
    return T60, locs, noised_angles

noise_k = 0.01

input_angles_true = torch.rand(20000, 6)
T60, true_locs = fw_kin(input_angles_true)
T60_N, noised_locs, noised_angles = noised_fw_kin(input_angles_true, noise_k)

np.save('PostNoise_6a_joint_angles_noise_k='+str(noise_k), noised_angles.cpu().detach().numpy())
np.save('PreNoise_6a_joint_angles_noise_k='+str(noise_k), input_angles_true)
np.save('Noised_6a_effector_locations_noise_k='+str(noise_k)+'.npy', noised_locs)
np.save('True_6a_effector_locations_noise_k='+str(noise_k)+'.npy', true_locs.cpu().detach().numpy())


    
    
    
    
    
    
    
