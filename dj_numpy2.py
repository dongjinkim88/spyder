#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numpy random
5/15/2021
djkim
"""

import numpy as np
from scipy import stats
from scipy.stats import norm, skewnorm
import matplotlib.pyplot as plt

np.random.seed(1)
n=10000
m=100
a=4.0

# sn = np.random.standard_normal(n)
# sn_l= sn[sn<=0]; sn_r = sn[sn>0]; sn_skewed = np.concatenate((sn_l, sn_r*a))
sn = norm.rvs(size=n)
sn_skewed = skewnorm.rvs(a, size=n)

# skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)
# skewnorm.pdf(x, a, loc, scale) = skewnorm.pdf((x-loc)/scale, a) / scale
# mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')

#plot
fig = plt.figure()
# fig, ax = plt.subplots(1, 1)
# plt.figure(figsize=(20,10))
plt.style.use('classic')
plt.title('Standard Normal vs. Rightly Skewed Normal')
plt.hist(sn, m, density=True, histtype='stepfilled', alpha=0.2)
plt.hist(sn_skewed, m, density=True, histtype='stepfilled', alpha=0.2)
plt.legend(('standard normal','rightly skewed'))
plt.show()

#stats
print('standard normal dist  : mu= {:5.4f}, std={:5.4f}, skewness={:5.4f}, ex-kurtosis={:5.4f}'.format(sn.mean(),sn.std(),stats.skew(sn),stats.kurtosis(sn)))
print('rightly skewed normal : mu= {:5.4f}, std={:5.4f}, skewness={:5.4f}, ex-kurtosis={:5.4f}'.format(sn_skewed.mean(),sn_skewed.std(),stats.skew(sn_skewed),stats.kurtosis(sn_skewed)))
# print(stats.describe(sn))
# print(stats.describe(sn_skewed))

# from statsmodels.distributions.empirical_distribution import ECDF
# l = [3,3,1,4]
# dj = ECDF(l)
# dj([3,55,0.5,1.5])

