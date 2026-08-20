#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 11:29:13 2026

@author: ian
"""

import matplotlib.pyplot as plt
import numpy as np

KAN_data = np.load('PV_KAN_scores.npy')
MLP_data = np.load('PV_MLP_scores.npy')

N_kan = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20])
N_mlp = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200])

KAN_data = np.load('PV_KAN_scores.npy')
exp_data = np.load('PV_Experimental_KAN_scores.npy')
MLP_data = np.load('PV_MLP_scores.npy')[:len(N_mlp)]

def trainables_KAN(N):
    return 5*N*18

def trainables_MLP(N):
    return N**2 + 7*N + 1

trainables_kan = np.zeros_like(N_kan)
trainables_mlp = np.zeros_like(N_mlp)


for i, N in enumerate(N_kan):
    trainables_kan[i] = trainables_KAN(N)
    
for i, N in enumerate(N_mlp):
    trainables_mlp[i] = trainables_MLP(N)
    
plt.figure(dpi=800, figsize=(4, 3))
plt.semilogx(trainables_kan, KAN_data.mean(axis=1), color='red', lw=2, marker='o', label='PhyKAN (sim)')
plt.fill_between(trainables_kan, KAN_data.mean(axis=1)-KAN_data.std(axis=1), KAN_data.mean(axis=1)+KAN_data.std(axis=1), color='red', alpha=0.2, lw=0)
plt.semilogx(trainables_kan, exp_data.mean(axis=1), color='xkcd:brick', lw=2, marker='o', label='PhyKAN (exp)')
plt.fill_between(trainables_kan, exp_data.mean(axis=1)-exp_data.std(axis=1), exp_data.mean(axis=1)+exp_data.std(axis=1), color='xkcd:brick', alpha=0.2, lw=0)
plt.semilogx(trainables_mlp, MLP_data.mean(axis=1), color='black', lw=2, marker='o', label='MLP')
plt.fill_between(trainables_mlp, MLP_data.mean(axis=1)-MLP_data.std(axis=1), MLP_data.mean(axis=1)+MLP_data.std(axis=1), color='black', alpha=0.2, lw=0)
plt.legend()
plt.xlabel('Trainable Parameters')
plt.ylabel('Power Generation Ratio')
plt.show()