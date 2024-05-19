#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, B. Vantaggi (2024). 
Behavioral dynamic portfolio selection with S-shaped utility and epsilon-contaminations.
"""
"""
EXPLANATION OF THE CODE:
Plots the graph of the following function with respect to the variable eta:
    f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)
where:
    * eta: optimization variable
    * T: time horizon in periods
    * u: up factor for the binomial market
    * d: down factor for the binomial market
    * V0: initial endowment (either positive or negative)
    * r: risk-free interest rate per period
    * p: real-world probability parameter for the binomial market
    * epsilon_p: ambiguity parameter for gains
    * epsilon_m: ambiguity parameter for losses
    * alpha_p: constant relative risk aversion for gains
    * alpha_m: constant relative risk aversion for losses
    * lamb: scale parameter for losses
IMPORTANT: eta is assumed to range in [[V0]+, +infinity)
"""

import numpy as np
from behavioral_portfolio import f
from behavioral_portfolio import tolerance
import matplotlib.pyplot as plt


# Example 5
T = 3
epsilon_p = 0.05
alpha_p = 0.5
epsilon_m = 0.05
alpha_m = 0.7
p = 0.25
u = 2
d = 0.5
V0 = 5
V0p = max(V0, 0)
r = 0
lamb = 0.4

# Create the image for the plot
plt.figure(figsize=(7, 4))
plt.xlabel('$\eta$')
plt.ylabel(r'$f_{V_0,r,p,{\bf \epsilon},{\bf \alpha},\lambda}$')
plt.title('Optimal value as a function of $\eta$')


# Create the grid of eta values distinguishing eta = [V0]+ and adding a value
# in its right neighborhood
etas = np.sort(np.concatenate([
    np.arange(V0p, V0p + 10 + tolerance, 0.1),
    np.arange(V0p + 10, V0p + 400 + tolerance, 5),
    np.array([450, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 467.5, 
             467.7006, 468, 469, 470, 500]),
    np.arange(V0p + 500, V0p + 800 + tolerance, 50)
    ]))

etas[0] += tolerance * 10
etas = np.insert(etas,0,V0p)


# Values of f are appended to an empty list
f_val = []
    
# Compute f on the chosen grid
for eta in etas:
    print('Computing f(eta) for eta:', eta)
    diff, V_p, V_m, A, Ac = f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)
    f_val.append(diff)
                    

# Transform the list of f values in an array
f_val = np.array(f_val)

# Extract the initial point
eta_first = etas[0]
f_first = f_val[0]

# Extract the rest of points
etas = etas[1:]
f_val = f_val[1:]



# Plot the graph of f distinguishing the possible discontinuity at the initial point eta = [V0]+
plt.plot(etas, np.round(f_val, 6), color='red')
# Plot the initial point
plt.plot([eta_first], [f_first], marker="o", markersize=5, color='red')
# Plot the maximum
plt.plot([467.7006], [6.2064], marker="o", markersize=5, color='red')
# Save the plot as an image
plt.savefig('Example_5.png', dpi=300)
plt.show()