#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:17:27 2025

@author: ian
"""

import numpy as np

class CartpoleEnvironment():
    # Re-initialise environment
    def reset(self):
        # Randomly initialise state
        x = 0.1*(np.random.ranf(1)-0.5)
        xdot = 0.1*(np.random.ranf(1)-0.5)
        theta = 0.1*(np.random.ranf(1)-0.5)
        thetadot = 0.1*(np.random.ranf(1)-0.5)
        # Return that the pole is upright
        upright = True
        return x[0], xdot[0], theta[0], thetadot[0], upright
        
    # Return angular acceleration of pole
    def return_thetadotdot(self, theta, thetadot, xdot, F):
        # Cart mass
        mc = 1
        # Pole mass
        m = 0.1
        # length to centre of gravity of the pole
        l = 0.5
        # Gravitational acceleration
        g = 9.8
        sin = lambda x: np.sin(x)
        cos = lambda x: np.cos(x)
        # Kinematic equations
        return (g*sin(theta) - cos(theta)*((F + m*l*sin(theta)*thetadot**2)/(mc+m)))/(l*(4/3 - (m*cos(theta)**2)/(mc+m)))
    
    # Find acceleration of the pole
    def return_xdotdot(self, theta, thetadot, thetadotdot, xdot, F):
        # Cart mass
        mc = 1
        # Pole mass
        m = 0.1
        # length to centre of gravity of the pole
        l = 0.5
        sin = lambda x: np.sin(x)
        cos = lambda x: np.cos(x)
        # Kinematic equations
        return (F+m*l*(sin(theta)*thetadot**2 - thetadotdot*cos(theta)))/(mc+m)
    
    # Euler integration of kinematic equations
    def integrate(self, F, x, xdot, theta, thetadot):
        # Scale force
        F = 5*F
        # Use kinematic equations to find second order derivatives
        thetadotdot = self.return_thetadotdot(theta, thetadot, xdot, F)
        xdotdot = self.return_xdotdot(theta, thetadot, thetadotdot, xdot, F)
        # Numerically integrate for first order derivatives
        thetadotnew = thetadot + 0.02*thetadotdot
        xdotnew = xdot + 0.02*xdotdot
        # Find position and velocity
        thetanew = theta + thetadotnew*0.02
        xnew = x + xdotnew*0.02
        return xnew, xdotnew, thetanew, thetadotnew
    
    # Return that the pole has fallen if values exceed the range
    def failure_check(self, x, xdot, theta, thetadot):
        # If outside of x range then fail
        if np.abs(x) > 2.4:
            return False
        # If pole drops then fail
        if np.abs(theta)>0.2095:
            return False
        # Else, continue
        else:
            return True
    
    # From an input force, calculate new state of the environment
    def step_environment(self, F, x, xdot, theta, thetadot, upright):
        # Step the environment with kinematic equations
        xnew, xdotnew, thetanew, thetadotnew = self.integrate(F, x, xdot, theta, thetadot)
        # Check for failure
        upright = self.failure_check(x, xdot, theta, thetadot)
        # Calculate reward
        if upright==True:
            reward = self.reward_shaping([xnew, xdotnew, thetanew, thetadotnew], 1)
        else:
            reward = np.array([-1.0])
        return xnew, xdotnew, thetanew, thetadotnew, upright, reward
    
    # Shape the reward, penalising for position, velocity, angle, and angular velocity
    def reward_shaping(self, state, reward):
        x, x_dot, theta, theta_dot = state
        r = np.copy(reward)
        p1 = np.abs(x) # Penalise cart position, favouring centre
        p2 = 0.1*np.abs(x_dot) # Penalise cart velocity, favouring lower velocities
        p3 = np.abs(theta) # Penalise pole angle, favouring close to zero
        p4 = 0.5*np.abs(theta_dot) # Penalise angular velocity, favouring low velocities
        shaped_reward = r - p1 - p2 - p3 - p4
        return shaped_reward
    

    
    
    