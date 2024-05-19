#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, B. Vantaggi (2024). 
Behavioral dynamic portfolio selection with S-shaped utility and epsilon-contaminations.
"""
"""
EXPLANATION OF THE CODE:
Maximizes the following function with respect to the variable eta using an
exhausitive search in a partition of a subinterval of [[V0]+, +infinity):
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


# Apply exhaustive search
a = 467.7001
b = 467.7009
step = 0.0001
etas = np.arange(a, b + tolerance, step)


print('EXHAUSTIVE SEARCH:')
print('Searching opt_eta in the interval [', a, ', ', b, '] with step', step)

# Find the maximizer of f using an exhaustive search
opt_diff = -np.Infinity
opt_V_p = {}
opt_V_m = 0
opt_A = ()
opt_Ac = ()
opt_eta = -np.Infinity

for eta in etas:
    diff, V_p, V_m, A, Ac = f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)

    if diff >= opt_diff:
        opt_diff = diff
        opt_V_p = V_p
        opt_V_m = V_m
        opt_A = A
        opt_Ac = Ac
        opt_eta = np.round(eta, 4)
                
                
            
print('\n\n*** PARAMETERS ***') 
print('T = ', T)
print('epsilon_p =', epsilon_p)
print('alpha_p =', alpha_p)
print('epsilon_m =', epsilon_m)
print('alpha_m =', alpha_m)
print('p =', p)
print('u =', u)
print('d =', d)
print('V0 =', V0)
print('V0p =', V0p)
print('r =', r)
print('lamb =', lamb)
print('\n\n*** REDUCED Optimal ***') 
print('opt_A:', opt_A,' opt_A^c:', opt_Ac)
print('opt_eta:', opt_eta)     
print('opt_diff:', opt_diff) 
print('opt_V_p:', opt_V_p)
print('opt_V_m:', opt_V_m)
