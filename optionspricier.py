# -*- coding: utf-8 -*-
"""
Author : Png Chin Yong

Description : 
Functions for valuation of options using 
1. Black Scholes
2. Monte Carlo
"""

from scipy import random,stats,math
import numpy as np
import time

def optionBS(spot,strike, vol, T, Rf):
    '''
    call option using Black Scholes
    '''
    denom = vol * math.sqrt(T)
    d1 = (math.log(spot/strike) + (Rf + 0.5*vol**2)*T) / denom
    d2 = d1 - denom
    
    call = spot * stats.norm.cdf(d1) - strike * math.exp(-Rf * T) * stats.norm.cdf(d2)
    put = strike * math.exp(-Rf * T) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1)
    
    return call, put


def calloptionMC(spot,strike,vol,T,Rf,M,alpha):
    '''
    call option using Monte Carlo simulation
    returns option price and lower and upper bounds
    '''
    S = np.zeros(M)
    S = spot * np.exp( (Rf - 0.5*vol**2)*T + vol * math.sqrt(T) * random.randn(M))
    Carray = np.exp(-Rf * T) * np.maximum(S - strike, 0)

    c = np.average(Carray)
    c_std = np.std(Carray) / math.sqrt(M)
    bounds = c + np.array([-1,1]) * stats.norm.ppf(0.5 + alpha/2) * c_std
    return c, bounds


spot = 100
strike = 100
vol = 0.3
duration = 1
riskfree = 0.04
simulation_runs = int(1E6)
alpha = 0.95

print (optionBS(spot, strike, vol, duration, riskfree))
print(calloptionMC(spot, strike, vol, duration, riskfree, simulation_runs,alpha))
