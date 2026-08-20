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
        self.params = params
        self.mode = mode
        
    def residuals(self, unknowns, Vin):
        p = self.params
        mode = self.mode
        Vref = 0.5

        if mode == 'LP':
            Vout, I_MR = unknowns

            eq1 = Vout - p['Rl'] * I_MR
            eq2 = I_MR - p['Io'] * safe_exp(p['alpha'] * (((Vref - Vin) - Vout) - I_MR * p['Rs']))
            
            return np.array([
                eq1, eq2
            ], dtype=float)
        
        elif mode == 'BP':
            
            Vout, I_MR = unknowns

            eq1 = Vout - p['Rl'] * ((Vin - Vout) / p['R'] - I_MR)
            eq2 = I_MR - p['Io'] * safe_exp(p['alpha'] * ((Vout + Vin) - I_MR * p['Rs']))
            
            return np.array([
                eq1, eq2
            ], dtype=float)
        
        elif mode == 'HP':
            
            Vout, I_MR = unknowns

            eq1 = Vout - p['Rl'] * I_MR
            eq2 = I_MR - p['Io'] * safe_exp(p['alpha'] * ((Vin - Vout) - I_MR * p['Rs']))
            
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

fig_synapse, axs_synapse = plt.subplots(2, 2, squeeze = False)
fig_synapse.set_figheight(8)
fig_synapse.set_figwidth(8)

vin_array = np.concatenate([np.linspace(0, 0.5, 500)])

initial_guess = np.array([
    0,                  # Vout
    1e-6,                 # I
], dtype=float)


###############################################################################
params = {
    "Rl": 1e3,
    "R": 3.75,
    "Rs": 1,
    "alpha": 30, #This parameter determines the non-linearity of the synapse
    "Io": 2e-6
}

model = CircuitModel('BP', params)                                              # synapse declaration, the 1st parameter determines if it acts as a "Band pass", "Low pass" or "high pass" filter equivalent (insteaed of filtering based on frequency encoding, here the encoding is assumed to be done on the voltage amplitude of hte input signal)

solutions = model.solve_sweep(vin_array, initial_guess)                         # synapse evaluation

vout_array = solutions[:, 0]
I_array = solutions[:, 1]

axs_synapse[0, 0].plot(vin_array, vout_array)
axs_synapse[0, 0].set_xlabel("Vin (V)")
axs_synapse[0, 0].set_ylabel("Vout (V)")
axs_synapse[0, 0].set_title("Circuit transfer")
axs_synapse[0, 0].grid(True)
###############################################################################

params = {
    "Rl": 1e3,
    "R": 300,
    "Rs": 1,
    "alpha": 10,
    "Io": 2e-6
}

model = CircuitModel('LP', params)

solutions = model.solve_sweep(vin_array, initial_guess)

vout_array = solutions[:, 0]
I_array = solutions[:, 1]

axs_synapse[0, 1].plot(vin_array, vout_array)
axs_synapse[0, 1].set_xlabel("Vin (V)")
axs_synapse[0, 1].set_ylabel("Vout (V)")
axs_synapse[0, 1].set_title("Circuit transfer")
axs_synapse[0, 1].grid(True)
###############################################################################

params = {
    "Rl": 1e3,
    "R": 300,
    "Rs": 1,
    "alpha": 10,
    "Io": 2e-6
}

model = CircuitModel('HP', params)

solutions = model.solve_sweep(vin_array, initial_guess)

vout_array = solutions[:, 0]
I_array = solutions[:, 1]

axs_synapse[1, 0].plot(vin_array, vout_array)
#plt.plot(vin_array, I_array)

axs_synapse[1, 0].set_xlabel("Vin (V)")
axs_synapse[1, 0].set_ylabel("Vout (V)")
axs_synapse[1, 0].set_title("Circuit transfer")
axs_synapse[1, 0].grid(True)
###############################################################################

plt.tight_layout(pad=1)   
plt.show()
