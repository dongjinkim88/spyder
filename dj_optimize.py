#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:14:39 2022

@author: djkim
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt

stocks = pd.read_csv('stocks.csv')
stocks_price = stocks[['date','symbol','close']].pivot(index='date',columns='symbol',values='close')

# calculate returns
stocks_rtn = stocks_price.pct_change()

sec_num = stocks_rtn.shape[1]

risk_free = 0.01
annual_multiple = 252

# objective function: minimize negative Sharpe ratio
rtn = lambda weights: np.sum(stocks_rtn.mean() * weights) * annual_multiple
vol = lambda weights: np.sqrt(np.dot(weights.T, np.dot(stocks_rtn.cov() * annual_multiple, weights)))
fun = lambda weights: -1 * (rtn(weights) - risk_free) / vol(weights)# sharpe ratio

# objective function: minimize negative Sharpe ratio
# def obj_fun(weights):
#     # anunalized expected return and volatility
#     rtn = np.sum(stocks_rtn.mean() * weights) * annual_multiple
#     vol = np.sqrt(np.dot(weights.T, np.dot(stocks_rtn.cov() * annual_multiple, weights)))
#     risk_free = 0.01
#     return -1 * (rtn - risk_free) / vol # sharpe ratio

# constraints: sum of weights = 1
constraints = ({'type':'eq','fun': lambda weights: np.sum(weights) - 1})

# bounds: weight in (0,1) i.e. long only 
bounds = [(0,1) for i in range(sec_num)]

# initial weights
weights = np.array([1/sec_num] * sec_num)

# results = opt.minimize(obj_fun, weights, method='SLSQP', bounds=bounds, constraints=constraints)
results = opt.minimize(fun, weights, method='SLSQP', bounds=bounds, constraints=constraints)

print(results)

print(f'stocks = {list(stocks_rtn.columns)}')
print(f'optimal weights = {results.x}')
print(f'optimal return = {rtn(results.x)}')
print(f'optimal risk = {vol(results.x)}')
print(f'Sharpe ratio = {-1*results.fun}')


# Monte Carlo
np.random.seed(0)

N = 1000
w_mat = np.zeros((N, sec_num))
rtn_vec = np.zeros(N)
vol_vec = np.zeros(N)
sharpe_vec = np.zeros(N)
for i in range(N):
    wi = np.random.rand(sec_num)
    wi = np.array(wi / np.sum(wi))
    w_mat[i,:] = wi
    rtn_vec[i] = rtn(wi)
    vol_vec[i] = vol(wi)
    sharpe_vec[i] = (rtn_vec[i] - risk_free) / vol_vec[i]
    
mc_df = pd.DataFrame([rtn_vec, vol_vec, sharpe_vec, w_mat]).T
mc_df.columns=['Return','Risk','Sharpe','Weights']
mc_df = mc_df.infer_objects()

mc_optimal = mc_df.loc[mc_df['Sharpe'].idxmax()]

print(f'mc optimal weights = {mc_optimal.Weights}')
print(f'mc optimal return = {mc_optimal.Return}')
print(f'mc optimal risk = {mc_optimal.Risk}')
print(f'mc Sharpe ratio = {mc_optimal.Sharpe}')

