#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, B. Vantaggi (2024). 
Behavioral dynamic portfolio selection with S-shaped utility and epsilon-contaminations.
"""
"""
EXPLANATION OF THE CODE:
Maximizes the following function with respect to the variable eta using the
Golden-section search algorithm:
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
IMPORTANT: this must be used only when the maximum lies in ([V0]+, +infinity)
"""

import numpy as np
from behavioral_portfolio import f
from behavioral_portfolio import tolerance
from scipy import optimize


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


# Apply golden-section search
a = 467
b = 468

# Function to transform the maximization problem in a minimization problem
def g(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb):
    return -f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)[0]

print('GOLDEN-SECTION SEARCH:')
print('Searching opt_eta in the interval [', a, ', ', b, ']')


# Find the maximizer of f using the Golden-section search algorithm:
# * this must be used only when the maximum lies in ([V0]+, +infinity)
# * the case in which the maximum lies at eta = [V0]+ must be treated separtely
opt_eta = optimize.golden(g, args=(T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb), 
                           brack=(a, (a + b) / 2, b), tol=tolerance)
opt_eta = np.round(opt_eta, 4)

# Display the result
opt_diff, opt_V_p, opt_V_m, opt_A, opt_Ac = f(opt_eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)
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