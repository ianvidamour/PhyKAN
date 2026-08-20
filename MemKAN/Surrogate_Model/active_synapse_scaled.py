#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:56:32 2026

@author: aguirref
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def safe_exp(x):
    return np.exp(np.clip(x, -100, 100))


class CircuitModel:
    def __init__(self, mode, params):
        self.Rl = params[0]*99+1
        self.R = params[1]*9+1
        self.Rs = (params[2]+0.01)/1.01
        self.alpha = params[3]*40+10
        self.Io = 10**(-7+2*params[4])
        self.mode = mode
        
    def residuals(self, unknowns, Vin):
        mode = self.mode
        Vref = 0.5

        if mode == 'LP':
            Vout, I_MR = unknowns

            eq1 = Vout - self.Rl * I_MR
            eq2 = I_MR - self.Io * safe_exp(self.alpha * (((Vref - Vin) - Vout) - I_MR * self.Rs))
            
            return np.array([
                eq1, eq2
            ], dtype=float)
        
        elif mode == 'BP':
            
            Vout, I_MR = unknowns

            eq1 = Vout - self.Rl * ((Vin - Vout) / self.R - I_MR)
            eq2 = I_MR - self.Io * safe_exp(self.alpha * ((Vout + Vin) - I_MR * self.Rs))
            
            return np.array([
                eq1, eq2
            ], dtype=float)
        
        elif mode == 'HP':
            
            Vout, I_MR = unknowns

            eq1 = Vout - self.Rl * I_MR
            eq2 = I_MR - self.Io * safe_exp(self.alpha * ((Vin - Vout) - I_MR * self.Rs))
            
            return np.array([
                eq1, eq2
            ], dtype=float)

    def solve(self, vin, initial_guess):
        res = least_squares(
            self.residuals,
            initial_guess,
            args=(vin,),
            method='trf',
            max_nfev=5000
        )

        if not res.success:
            raise RuntimeError(f"No convergió para Vin={vin}: {res.message}")

        return res.x

    def solve_sweep(self, vin_array, initial_guess):
        solutions = []
        guess = np.array(initial_guess, dtype=float)

        for vin in vin_array:
            sol = self.solve(vin, guess)
            solutions.append(sol)
            guess = sol.copy()

        return np.array(solutions)

Nruns = 20000
nsteps = 200
vins = np.zeros((Nruns, nsteps))
params = np.zeros((Nruns, 5))
vouts = np.zeros((Nruns, nsteps))
Iouts = np.zeros((Nruns, nsteps))

import tqdm
for i in tqdm.tqdm(range(Nruns)):
    vin = np.random.ranf(nsteps)
    vin = np.sort(vin)
    vins[i] = vin
    
    initial_guess = np.array([
        0,                  # Vout
        1e-6,                 # I
    ], dtype=float)
    
    _params = np.random.ranf(5)
    params[i] = _params
    model = CircuitModel('BP', _params)                                              # synapse declaration, the 1st parameter determines if it acts as a "Band pass", "Low pass" or "high pass" filter equivalent (insteaed of filtering based on frequency encoding, here the encoding is assumed to be done on the voltage amplitude of hte input signal)
    
    solutions = model.solve_sweep(vin, initial_guess)                         # synapse evaluation
    
    vout_array = solutions[:, 0]
    I_array = solutions[:, 1]
    if i % 100 == 0:
        plt.figure()
        plt.plot(vin, vout_array)
        plt.show()
    vouts[i] = vout_array
    Iouts[i] = I_array

np.save('Input Voltages.npy', vins)
np.save('Output Voltages.npy', vouts)
np.save('Output Currents.npy', Iouts)
np.save('Device Parameters.npy', params)


