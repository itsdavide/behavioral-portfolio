#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, B. Vantaggi (2024). 
Behavioral dynamic portfolio selection with S-shaped utility and epsilon-contaminations.
"""
"""
EXPLANATION OF THE CODE:
Optimization code that reduces the dynamic portfolio selection in a binomial
market into the maximization of the function of the variable eta:
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

import pyomo.environ as pyo
import scipy.special
import numpy as np

optimizer_path = 'PATH_TO_BONMIN'

tolerance = 0.0000001

def representation(T):
    As = []
    for k in range(T + 1):
        A = []
        for i in range(int(scipy.special.comb(T, k)) + 1):
            A.append(i)
        As.append(A)
    return As


def product(ar_list):
    if not ar_list:
        yield ()
    else:
        for a in ar_list[0]:
            for prod in product(ar_list[1:]):
                yield (a,)+prod


# Optimal solution of the positive part problem
def CH_Port_Pos(p, V0, eta, u, d, r, alpha, epsilon, T, c):   
    model = pyo.ConcreteModel()

    # Number of variables
    N = list(range(T + 1)) # 0, ..., T
    # Number of extreme points
    M = list(range(T + 2)) # 0, ..., T + 1
    # Check if the current event is Omega
    if sum(c[k] for k in N) == 2**T:
        M = list(range(T + 1)) # 0, ..., T
    

    q = ((1 + r) - d) / (u - d)
    
    
    def U(x):
        return x ** alpha

    # Generates the P probabilities
    P = {}
    tot = 0
    for k in N:
        P[k] = c[k] * p**k * (1-p)**(T - k)
        tot += P[k]
    # Generates the Q probabilities
    Q = {}
    tot = 0
    for k in N:
        Q[k] = c[k] * q**k * (1-q)**(T - k)
        tot += Q[k]
    # Generates the permuted P probabilities
    PP = {}
    for i in M:
        for j in N:
            if i == j and c[j] != 0 and M != T + 1:
                PP[i,j] = (1 - epsilon) * P[j] + epsilon
            else:
                PP[i,j] = (1 - epsilon) * P[j]
    
    model.N = pyo.Set(initialize=N)
    model.M = pyo.Set(initialize=M)

    model.PP = pyo.Param(model.M, model.N, initialize=PP)
    model.Q = pyo.Param(model.N, initialize=Q, mutable=True)
    model.V = pyo.Var(model.N, within=pyo.NonNegativeReals, bounds=(tolerance, np.Infinity), initialize=tolerance)
    model.C = pyo.Var(bounds=(tolerance, np.Infinity))

    def ConstrRule(model, i):
        return sum(model.PP[i, j] * U(model.V[j]) for j in model.N) >= model.C

    model.c = pyo.Constraint(model.M, rule=ConstrRule)

    model.d = pyo.Constraint(expr=sum(model.Q[i] * model.V[i] for i in model.N) - eta*((1+r)**T) == 0)

    model.o = pyo.Objective(expr = model.C, sense=pyo.maximize)
    
    
    status = pyo.SolverFactory(optimizer_path).solve(model)
    pyo.assert_optimal_termination(status)
    
    V={}
    
    for i in model.N:
        V[i] = round(pyo.value(model.V[i]), 4)
    
    return (V,  pyo.value(model.o))


# Optimal solution of the negative part problem
def CH_Port_Neg(p, V0, eta, u, d, r, alpha, epsilon, T, c):

    # Number of variables
    N = list(range(T + 1)) # 0, ..., T
    

    q = ((1 + r) - d) / (u - d)
    
    def U(x):
        return x ** alpha
    
    # Generates the P probabilities
    P = {}
    PAc = 0
    for k in N:
        P[k] = c[k] * p**k * (1-p)**(T - k)
        PAc += P[k]
    # Generates the Q probabilities
    Q = {}
    QAc = 0
    for k in N:
        Q[k] = c[k] * q**k * (1-q)**(T - k)
        QAc += Q[k]
    
    K = (1 + r)**T * (eta - V0) / QAc
    Choq = U(K) * ((1 - epsilon)*PAc + epsilon)
    
    return round(K, 4), Choq


# Optimal solution for a fixed value of eta in [[V0]+, +infinity)
def f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb):
    N = list(range(T + 1))
    
    # Generate the representations of subsets
    cart_prod = list(product(representation(T)))
    # Representation of Omega
    omega = tuple(int(scipy.special.comb(T, k)) for k in range(T + 1))
    
    opt_diff = -np.infty
    for i in range(len(cart_prod)):
        c = cart_prod[i]
        cc = tuple(map(lambda i, j: i - j, omega, c))
        opt_p = - np.Infinity
        V_p = {}
        opt_m = np.Infinity
        V_m = {}
        if sum(c[k] for k in N) == 0: # empty set
            if eta == 0:
                opt_p = 0
                V_p = {}
                for i in N:
                    V_p[i] = 0
            else:
                opt_p = - np.Infinity
                V_p = {}
            if opt_p == - np.Infinity or abs(eta - V0) < tolerance:
                opt_m = 0
                V_m = 0
            else:
                V_m, opt_m = CH_Port_Neg(p, V0, eta, u, d, r, alpha_m, epsilon_m, T, cc)
        elif sum(c[k] for k in N) == 2**T: # all space
            if abs(eta - V0) < tolerance:
                opt_m = 0
                V_m = 0
            else:
                opt_m = np.Infinity
                V_m = 0
            if opt_m == np.Infinity or eta == 0:
                opt_p = 0
                V_p = {}
                for i in N:
                    V_p[i] = 0
            else:
                V_p, opt_p = CH_Port_Pos(p, V0, eta, u, d, r, alpha_p, epsilon_p, T, c)
        elif sum(c[k] for k in N) != 0 and sum(c[k] for k in N) != 2**T:
            if eta == 0:
                opt_p = 0
                V_p = {}
                for i in N:
                    V_p[i] = 0
            else:
                V_p, opt_p = CH_Port_Pos(p, V0, eta, u, d, r, alpha_p, epsilon_p, T, c)
            if abs(eta - V0) < tolerance:
                opt_m = 0
                V_m = 0
            else:
                V_m, opt_m = CH_Port_Neg(p, V0, eta, u, d, r, alpha_m, epsilon_m, T, cc)
        
        diff = opt_p - lamb * opt_m
        if diff >= opt_diff:
            opt_diff = diff
            opt_V_p = V_p
            opt_V_m = V_m
            opt_A = c
            opt_Ac = cc   
            
    return opt_diff, opt_V_p, opt_V_m, opt_A, opt_Ac
