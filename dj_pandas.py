#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:40:21 2020

@author: djkim
"""

import os
import sys
import time
import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm, t

# Read control file
control = pd.read_csv('./control.csv')

# Pull out control parameters
run_date = control.loc[control.parameter=='run_date', 'value'].values[0]
n_mc = int(control.loc[control.parameter=='num_of_mc_states', 'value'].values[0])
seed = int(control.loc[control.parameter=='random_seed', 'value'].values[0])
which_dist = control.loc[control.parameter=='sample_dist', 'value'].values[0]

# Print title
run_date = dt.datetime.strptime(run_date, '%Y%m%d')
print('Monte Carlo simulation with {} samples in {}-distribution as of {}'.format(
        n_mc, which_dist, run_date.strftime('%m/%d/%Y')))

# Sampling
np.random.seed(seed)

if which_dist == 't':
    dof1, dof2 = 3, 3
    #assume rf ~ t
    rf1_t = t.rvs(size=n_mc, df=dof1)
    rf2_t = t.rvs(size=n_mc, df=dof2)
    
    #t -> CDF -> normal
    rf1 = norm.ppf(t.cdf(rf1_t, df=dof1))
    rf2 = norm.ppf(t.cdf(rf2_t, df=dof2))

else:    
    #assume rf ~ normal
    rf1 = norm.rvs(size=n_mc)
    rf2 = norm.rvs(size=n_mc)

# multivariate normal sampling
mu = np.array([rf1.mean(), rf2.mean()])
vcv = np.cov(rf1, rf2)
mc = np.random.multivariate_normal(mu, vcv, size=(n_mc,))

# Change back to initial distributed sampling
if which_dist == 't':
    #normal -> CDF -> t
    mc1 = t.ppf(norm.cdf(mc)[:,0], df=dof1)
    mc2 = t.ppf(norm.cdf(mc)[:,1], df=dof2)
else:
    mc1 = mc[:,0]
    mc2 = mc[:,1]
   
# Report
from scipy.stats import kurtosis
print('initial rf_1 : mu= {:5.4f}, std={:5.4f}, kurtosis={:5.4f}'.format(rf1.mean(),rf1.std(),kurtosis(rf1)))
print('initial rf_2 : mu= {:5.4f}, std={:5.4f}, kurtosis={:5.4f}'.format(rf2.mean(),rf2.std(),kurtosis(rf2)))

print('final rf_1 : mu= {:5.4f}, std={:5.4f}, kurtosis={:5.4f}'.format(mc1.mean(),mc1.std(),kurtosis(mc1)))
print('final rf_2 : mu= {:5.4f}, std={:5.4f}, kurtosis={:5.4f}'.format(mc2.mean(),mc2.std(),kurtosis(mc2)))

import matplotlib.pyplot as plt
plt.figure(1, figsize=(20,10))
plt.style.use('classic')
plt.subplot(2,2,1)
plt.scatter(rf1, mc1)
plt.subplot(2,2,2)
plt.scatter(rf2, mc2)
plt.show()
 