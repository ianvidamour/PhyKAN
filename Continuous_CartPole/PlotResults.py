#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:08:40 2025

@author: ian
"""

import numpy as np
import matplotlib.pyplot as plt
import os
Ns = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
from scipy.interpolate import make_interp_spline, BSpline




best_accs_KAN = []
best_stds_KAN = []

def exponential_average(signal, tau, dt=1):
   exp_avg = np.zeros_like(signal)
   alpha=dt/tau 
   for i in range(1, len(signal)):
       exp_avg[i] = (1-alpha)*exp_avg[i-1] + alpha * signal[i] 
   return exp_avg


folder = os.getcwd()+'/HPC_Data/'
for N in Ns:
    actor_shape = [4, N, N, N, 1]
    critic_shape = [4, N, N, N, 1]
    # try:
    tl = np.zeros((10, 1000))
    mas = np.zeros_like(tl)
    for run in range(10):
        tl[run] = np.load(folder+'Trial Lengths, CartPole Continous, N='+str(N)+', Run='+str(run)+'.npy')
        mas[run] = exponential_average(tl[run], 10)
    acs = np.amax(mas, axis=1)
    stds = acs.std()
    plt.figure(figsize=(6,4), dpi=600)
    plt.ylim(-5,505)
    plt.title('KAN N: '+str(N))
    for run in range(10):
        plt.plot(mas[run], label=str(run))
    plt.xlabel('Episodes')
    plt.ylabel('Trial Length')
    plt.legend()
    plt.show()
    best_accs_KAN.append(acs.mean())
    best_stds_KAN.append(stds)
    # except:
    #     continue
    
np.save('KAN best accs.npy', best_accs_KAN)
    
Ns = [20, 30, 40, 50, 75, 100, 125, 150, 200, 250]
folder = os.getcwd()+'/HPC_Data/'
best_accs_MLP = []
best_stds_MLP = []
for N in Ns:
    actor_shape = [4, N, N, N, 1]
    critic_shape = [4, N, N, N, 1]

    tl = np.zeros((20, 1000))
    mas = np.zeros_like(tl)
    for run in range(20):
        data = np.load(folder+'MLP Trial Lengths, CartPole Continous, N='+str(N)+', Run='+str(run)+'.npy')
        tl[run, :len(data)] = data
        mas[run] = exponential_average(tl[run], 10)
    acs = np.amax(mas, axis=1)
    stds = acs.std()
    # plt.figure(figsize=(6,4), dpi=600)
    # plt.ylim(-5,505)
    # plt.title('MLP N: '+str(N))
    # for run in range(20):
    #     plt.plot(mas[run], label=str(run))
    # plt.xlabel('Episodes')
    # plt.ylabel('Trial Length')
    # plt.legend()
    # plt.show()
    best_accs_MLP.append(acs.mean())
    best_stds_MLP.append(stds)

np.save('MLP best accs.npy', best_accs_MLP)
    
#%%
def trainables_KAN(shape):
    return 18*(shape[0]*shape[1]+shape[1]*shape[2]+shape[2]*shape[3]+shape[3]*shape[4]) 

def trainables_MLP(shape):
    return (shape[0]*shape[1]+shape[1]*shape[2]+shape[2]*shape[3]+shape[3]*shape[4])+shape[1]+shape[2]+shape[3]+shape[4]


trainables_kan = []
trainables_mlp = []

for N in [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]:
    trainables_kan.append(trainables_KAN([4, N, N, N, 1]))

for N in [20, 30, 40, 50, 75, 100, 125, 150, 200, 250]:
    trainables_mlp.append(trainables_MLP([4, N, N, N, 1]))

best_accs_KAN = np.array(best_accs_KAN)
best_stds_KAN = np.array(best_stds_KAN)
best_accs_MLP = np.array(best_accs_MLP)
best_stds_MLP = np.array(best_stds_MLP)

plt.figure(figsize=(3, 2), dpi=1200)
plt.title('3 Hidden Layers')
plt.ylim(0,515)
plt.semilogx(trainables_kan, best_accs_KAN, color='red', marker='.', label='PhyKAN', lw=2, ms=8)
plt.fill_between(trainables_kan, best_accs_KAN-best_stds_KAN, best_accs_KAN+best_stds_KAN, color='red', alpha=0.2, lw=0, interpolate=True)
plt.semilogx(trainables_mlp, best_accs_MLP, color='black', marker='.', label='MLP', lw=2, ms=8)
plt.fill_between(trainables_mlp, best_accs_MLP-best_stds_MLP, best_accs_MLP+best_stds_MLP, color='black', lw=0, alpha=0.2)
plt.xlabel('Trainable Parameters')
plt.ylabel('Average Upright Duration')
plt.legend(frameon=False, loc='lower right')
plt.show()